#!/usr/bin/env python3
"""API 精排（3.3.0，可选、默认关闭）：本地粗筛 Top 20 的低清拼图送多模态模型。

规矩（逐条对应规格）：
  - 只有用户本次手动勾选“本次启用 API 精排”才会被调用；默认完全离线。
  - 只把本地 Top 20 的 200x200 缩略图拼成 2 张带 [#1]…[#20] 标签的黑底
    拼图发出，绝不一次上传 20 张独立原图。
  - 底层只写一套 OpenAI 兼容 HTTP 代码（urllib.request，
    POST {base_url}/chat/completions）；厂商差异只在预设的 base_url/model。
  - 超时锁死 9 秒（规格要求 8~10 秒）。超时/失败/无 Key 一律抛给上层
    退回本地结果，不卡界面、不崩溃。
  - API 返回的 matched_ids 必须落在候选 1..20 内，未知 id 丢弃；
    未匹配的保持本地顺序。
  - Key 不写进代码、明文配置、前端 DOM 或日志：Windows 上用 DPAPI
    （ctypes 调 CryptProtectData，绑定当前 Windows 账户）加密后落盘；
    其它平台没有 DPAPI，就只放进程内存、退出即失效，并向用户说明。
"""
import base64
import ctypes
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request

try:
    import app_paths
except Exception:
    class _AppPaths(object):
        @staticmethod
        def writable_path(*parts):
            return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                *parts)
    app_paths = _AppPaths()

REQUEST_TIMEOUT = 9.0          # 秒，规格要求 8~10 秒
MAX_CANDIDATES = 20
GRID_CELL = 200
GRID_COLS = 5
GRID_GAP = 6
GRID_LABEL_H = 24

PROVIDERS = [
    {"id": "qwen", "label": "通义千问 VL（推荐）",
     "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
     "model": "qwen-vl-plus",
     "note": "阿里云百炼 DashScope 的 OpenAI 兼容模式。"},
    {"id": "doubao", "label": "豆包视觉（火山方舟）",
     "base_url": "https://ark.cn-beijing.volces.com/api/v3",
     "model": "doubao-1.5-vision-pro-32k-250115",
     "note": "方舟若要求填接入点 ID，请改用“自定义”并把 model 填成接入点。"},
    {"id": "gemini", "label": "Gemini（OpenAI 兼容）",
     "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
     "model": "gemini-2.0-flash",
     "note": "Google 的 OpenAI 兼容端点。"},
    {"id": "custom", "label": "自定义 OpenAI 兼容接口",
     "base_url": "", "model": "",
     "note": "填写 base_url（到 /v1 为止）与模型名，走同一套 chat/completions。"},
]
PROVIDER_MAP = {p["id"]: p for p in PROVIDERS}
DEFAULT_PROVIDER = "qwen"

_SETTINGS_LOCK = threading.Lock()
_MEM_KEY = {"value": ""}       # 非 Windows 平台只在内存里留一份
_KEY_FILE = "api_key.bin"
_SETTINGS_FILE = "settings.json"


def _settings_path():
    return app_paths.writable_path("semantic", _SETTINGS_FILE)


def _key_path():
    return app_paths.writable_path("semantic", _KEY_FILE)


def _ensure_dir(path):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def _read_json(path, default):
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else dict(default)
    except Exception:
        return dict(default)


def _write_json_atomic(path, data):
    _ensure_dir(path)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


# ---------------------------------------------------------------- 设置

def get_settings():
    with _SETTINGS_LOCK:
        data = _read_json(_settings_path(), {})
    provider = data.get("provider") if data.get("provider") in PROVIDER_MAP \
        else DEFAULT_PROVIDER
    return {
        "provider": provider,
        "base_url": data.get("base_url", ""),
        "model": data.get("model", ""),
        "privacy_ack": bool(data.get("privacy_ack")),
    }


def update_settings(payload):
    """只接受白名单字段；Key 单独走 set_key，绝不写进 settings.json。"""
    with _SETTINGS_LOCK:
        data = _read_json(_settings_path(), {})
        if "provider" in payload:
            provider = payload.get("provider")
            if provider not in PROVIDER_MAP:
                raise ValueError("未知的 API 厂商：%s" % provider)
            data["provider"] = provider
        if "base_url" in payload:
            data["base_url"] = str(payload.get("base_url") or "").strip()[:500]
        if "model" in payload:
            data["model"] = str(payload.get("model") or "").strip()[:200]
        if "privacy_ack" in payload:
            data["privacy_ack"] = bool(payload.get("privacy_ack"))
        _write_json_atomic(_settings_path(), data)
    return public_settings()


def active_endpoint(settings=None):
    settings = settings or get_settings()
    provider = PROVIDER_MAP.get(settings["provider"], PROVIDER_MAP[DEFAULT_PROVIDER])
    base_url = (settings.get("base_url") or provider["base_url"]).rstrip("/")
    model = settings.get("model") or provider["model"]
    if settings["provider"] == "custom":
        base_url = (settings.get("base_url") or "").rstrip("/")
        model = settings.get("model") or ""
    if base_url.endswith("/chat/completions"):
        url = base_url
    else:
        url = base_url + "/chat/completions"
    return {"provider": settings["provider"],
            "label": provider["label"], "base_url": base_url,
            "model": model, "url": url}


# ---------------------------------------------------------------- Key 存储

_DPAPI_ENTROPY = b"Shiying.SemanticRerank.Key.v1"


class _DataBlob(ctypes.Structure):
    _fields_ = [("cbData", ctypes.c_uint32),
                ("pbData", ctypes.POINTER(ctypes.c_byte))]


def _dpapi_available():
    return os.name == "nt" and hasattr(ctypes, "windll")


def _declare_dpapi():
    """P0-1 修复：Windows 64 位下必须显式声明 DPAPI 函数的 argtypes/restype，
    否则 ctypes 默认把指针当 32 位 int 传，高地址内存会 ArgumentError/崩溃。
    Linux 下 ctypes 没有 windll，此函数不做任何事。"""
    if not _dpapi_available():
        return
    try:
        import ctypes.wintypes as wintypes
        crypt32 = ctypes.windll.crypt32
        kernel32 = ctypes.windll.kernel32
        blob_p = ctypes.POINTER(_DataBlob)
        crypt32.CryptProtectData.argtypes = [
            blob_p, ctypes.c_wchar_p, blob_p, ctypes.c_void_p,
            ctypes.c_void_p, wintypes.DWORD, blob_p]
        crypt32.CryptProtectData.restype = wintypes.BOOL
        # Unprotect 的第 2 个参数是 LPWSTR*（二级指针），与 Protect 不同，
        # 调用时恒传 None（NULL），不要传 c_wchar_p。
        crypt32.CryptUnprotectData.argtypes = [
            blob_p, ctypes.POINTER(ctypes.c_wchar_p), blob_p,
            ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD, blob_p]
        crypt32.CryptUnprotectData.restype = wintypes.BOOL
        kernel32.LocalFree.argtypes = [ctypes.c_void_p]
        kernel32.LocalFree.restype = ctypes.c_void_p
    except Exception:
        pass


def _dpapi_call(func, data, description=None):
    """CryptProtectData / CryptUnprotectData 的薄封装。"""
    _declare_dpapi()
    buffer = ctypes.create_string_buffer(data, len(data))
    blob_in = _DataBlob(len(data),
                        ctypes.cast(buffer, ctypes.POINTER(ctypes.c_byte)))
    blob_out = _DataBlob()
    entropy_buf = ctypes.create_string_buffer(_DPAPI_ENTROPY,
                                              len(_DPAPI_ENTROPY))
    entropy = _DataBlob(len(_DPAPI_ENTROPY),
                        ctypes.cast(entropy_buf,
                                    ctypes.POINTER(ctypes.c_byte)))
    desc = ctypes.c_wchar_p(description) if description else None
    ok = func(ctypes.byref(blob_in), desc, ctypes.byref(entropy), None, None,
              0x1, ctypes.byref(blob_out))  # CRYPTPROTECT_UI_FORBIDDEN
    if not ok:
        raise OSError("DPAPI 调用失败（错误码 %s）"
                      % ctypes.get_last_error())
    try:
        return ctypes.string_at(blob_out.pbData, blob_out.cbData)
    finally:
        ctypes.windll.kernel32.LocalFree(blob_out.pbData)


def _protect(key):
    return _dpapi_call(ctypes.windll.crypt32.CryptProtectData,
                       key.encode("utf-8"), "Shiying API Key")


def _unprotect(blob):
    raw = _dpapi_call(ctypes.windll.crypt32.CryptUnprotectData, blob)
    return raw.decode("utf-8")


def set_key(key):
    """保存 Key；Windows 走 DPAPI，其它平台只放内存（不落明文）。"""
    key = (key or "").strip()
    if not key:
        raise ValueError("Key 不能为空")
    with _SETTINGS_LOCK:
        if _dpapi_available():
            blob = _protect(key)
            path = _key_path()
            _ensure_dir(path)
            tmp = path + ".tmp"
            with open(tmp, "wb") as f:
                f.write(base64.b64encode(blob))
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
            _MEM_KEY["value"] = ""   # 落盘成功就不在内存留明文
        else:
            _MEM_KEY["value"] = key
    return {"ok": True, "has_key": True, "masked_key": mask_key(key),
            "storage": key_storage_note()}


def clear_key():
    with _SETTINGS_LOCK:
        _MEM_KEY["value"] = ""
        try:
            os.unlink(_key_path())
        except OSError:
            pass
    return {"ok": True, "has_key": False, "masked_key": ""}


def get_key():
    """仅供 api_rerank 内部取用；任何接口都不返回明文。"""
    with _SETTINGS_LOCK:
        if _MEM_KEY["value"]:
            return _MEM_KEY["value"]
        if not _dpapi_available():
            return ""
        path = _key_path()
        if not os.path.isfile(path):
            return ""
        try:
            with open(path, "rb") as f:
                blob = base64.b64decode(f.read())
            return _unprotect(blob)
        except Exception:
            return ""


def has_key():
    return bool(get_key())


def mask_key(key):
    if not key:
        return ""
    if len(key) <= 8:
        return "****"
    return key[:3] + "****" + key[-4:]


def key_storage_note():
    if _dpapi_available():
        return "Windows DPAPI 加密保存（绑定当前 Windows 账户，换账户无法解密）"
    return ("当前平台没有 Windows DPAPI：Key 只保存在本次运行的内存里，"
            "关闭软件即失效，不落明文文件")


def public_settings():
    settings = get_settings()
    endpoint = active_endpoint(settings)
    key = get_key()
    return {
        "provider": settings["provider"],
        "providers": [{k: p[k] for k in ("id", "label", "base_url", "model",
                                          "note")} for p in PROVIDERS],
        "base_url": endpoint["base_url"],
        "model": endpoint["model"],
        "custom_base_url": settings.get("base_url", ""),
        "custom_model": settings.get("model", ""),
        "has_key": bool(key),
        "masked_key": mask_key(key),
        "key_storage": key_storage_note(),
        "privacy_ack": settings.get("privacy_ack", False),
        "timeout_seconds": REQUEST_TIMEOUT,
    }


def redact(text):
    """日志/错误里过滤 Key：先用真实 Key 做字面量替换（覆盖 Gemini AIzaSy、
    豆包 UUID 等非 sk- 格式），再用正则过滤 Authorization/Bearer/sk- 形式。
    真实 Key 只在内存里比对，绝不写进返回文本或日志。"""
    text = str(text or "")
    try:
        real_key = get_key()
    except Exception:
        real_key = ""
    if real_key and len(real_key) >= 6:
        text = text.replace(real_key, "***")
    text = re.sub(r"(?i)(authorization\s*[:=]\s*)(\S+)", r"\1***", text)
    text = re.sub(r"(?i)\bbearer\s+[A-Za-z0-9._\-]{6,}", "Bearer ***", text)
    text = re.sub(r"\bsk-[A-Za-z0-9._\-]{6,}", "sk-***", text)
    text = re.sub(r"\bAIza[0-9A-Za-z_\-]{20,}", "AIza***", text)
    return text


# ---------------------------------------------------------------- 拼图

def _load_font(size):
    from PIL import ImageFont
    for name in ("arial.ttf", "Arial.ttf", "DejaVuSans.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                 "C:/Windows/Fonts/arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    try:
        return ImageFont.load_default(size=size)
    except Exception:
        return ImageFont.load_default()


def build_grids(candidates):
    """把候选缩略图拼成 2 张图（每张最多 10 格），返回 [(jpeg_bytes, 起始序号)]。"""
    from PIL import Image, ImageDraw
    font = _load_font(15)
    cell = GRID_CELL
    block_h = cell + GRID_LABEL_H
    width = GRID_COLS * cell + (GRID_COLS + 1) * GRID_GAP
    grids = []
    per_grid = GRID_COLS * 2
    for start in range(0, len(candidates), per_grid):
        chunk = candidates[start:start + per_grid]
        rows = (len(chunk) + GRID_COLS - 1) // GRID_COLS
        height = rows * block_h + (rows + 1) * GRID_GAP
        canvas = Image.new("RGB", (width, height), (0, 0, 0))
        draw = ImageDraw.Draw(canvas)
        for offset, cand in enumerate(chunk):
            idx = start + offset + 1
            col = offset % GRID_COLS
            row = offset // GRID_COLS
            x = GRID_GAP + col * (cell + GRID_GAP)
            y = GRID_GAP + row * (block_h + GRID_GAP)
            try:
                with Image.open(cand["path"]) as im:
                    # JPEG 用 draft 先大致降采样，少解一遍大图，省内存也更快。
                    try:
                        im.draft("RGB", (cell * 2, cell * 2))
                    except Exception:
                        pass
                    thumb = im.convert("RGB")
                    thumb.thumbnail((cell, cell), Image.LANCZOS)
            except Exception:
                thumb = Image.new("RGB", (cell, cell), (24, 24, 24))
            # P1-3 修复：序号标签画在每格顶部 24px 标题栏里，缩略图贴在
            # 标题栏下方，不再盖住图片左上角，底部也不再留黑条。
            tile = Image.new("RGB", (cell, cell), (12, 12, 12))
            tile.paste(thumb, ((cell - thumb.width) // 2,
                               (cell - thumb.height) // 2))
            canvas.paste(tile, (x, y + GRID_LABEL_H))
            label = "[#%d]" % idx
            draw.rectangle([x, y, x + cell, y + GRID_LABEL_H], fill=(0, 0, 0))
            draw.text((x + 5, y + 3), label, fill=(255, 255, 255), font=font)
        buf = _jpeg_bytes(canvas)
        grids.append((buf, start + 1))
    return grids


def _jpeg_bytes(im):
    import io
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=80, optimize=True)
    return buf.getvalue()


def _data_url(jpeg_bytes):
    return "data:image/jpeg;base64," + base64.b64encode(jpeg_bytes).decode()


# ---------------------------------------------------------------- HTTP

def _prompt(query, count, grid_count):
    if grid_count >= 2:
        layout = ("下面 2 张图是候选缩略图拼图：第 1 张是 #1-#%d，"
                  "第 2 张是 #%d-#%d；每格顶部标题栏有 [#序号] 标签。"
                  % (min(10, count), 11, count))
    else:
        layout = ("下面 1 张图是候选缩略图拼图：#1-#%d，"
                  "每格顶部标题栏有 [#序号] 标签。" % count)
    return (
        "你是照片搜索结果排序助手。用户搜索词：「%s」。\n%s\n"
        "请判断哪些候选的画面内容确实符合搜索词，并按相关度从高到低排序。\n"
        "只输出 JSON，不要解释，格式：{\"matched_ids\":[3,7]}；"
        "没有符合的就输出 {\"matched_ids\":[]}。"
        % (query, layout))


def _extract_text(resp_json):
    try:
        choice = resp_json["choices"][0]
        message = choice.get("message") or {}
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict):
                    if isinstance(part.get("text"), str):
                        parts.append(part["text"])
                elif isinstance(part, str):
                    parts.append(part)
            return "\n".join(parts)
        if isinstance(choice.get("text"), str):
            return choice["text"]
    except Exception:
        pass
    return ""


def _parse_matched(text):
    text = (text or "").strip()
    if not text:
        return []
    candidates = []
    fenced = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", text, re.S)
    if fenced:
        candidates.append(fenced.group(1))
    obj = re.search(r"\{.*?\}", text, re.S)
    if obj:
        candidates.append(obj.group(0))
    arr = re.search(r"\[[^\[\]]*\]", text, re.S)
    if arr:
        candidates.append(arr.group(0))
    for raw in candidates:
        try:
            data = json.loads(raw)
        except Exception:
            continue
        if isinstance(data, dict):
            data = data.get("matched_ids") or data.get("ids") or []
        if isinstance(data, list):
            out = []
            for item in data:
                try:
                    out.append(int(item))
                except (TypeError, ValueError):
                    continue
            return out
    return []


def rerank(query, candidates, settings=None, key=None):
    """调用一次多模态 API；任何异常都转成 {ok: False, error}。"""
    base = get_settings()
    if settings:
        base.update({k: v for k, v in settings.items() if v not in (None, "")})
    settings = base
    endpoint = active_endpoint(settings)
    api_key = key if key is not None else get_key()
    if not api_key:
        return {"ok": False, "error": "没有配置 API Key"}
    if not endpoint["url"] or not endpoint["model"]:
        return {"ok": False, "error": "API 地址或模型名为空，请先在设置里填写"}
    items = list(candidates)[:MAX_CANDIDATES]
    if not items:
        return {"ok": False, "error": "没有候选图片"}
    try:
        grids = build_grids(items)
    except Exception as e:
        return {"ok": False, "error": redact("拼图生成失败：%s" % e)}
    if not grids:
        return {"ok": False, "error": "候选缩略图都读不出来"}

    content = [{"type": "text", "text": _prompt(query, len(items), len(grids))}]
    for jpeg, _start in grids:
        content.append({"type": "image_url",
                        "image_url": {"url": _data_url(jpeg)}})
    payload = {
        "model": endpoint["model"],
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "max_tokens": 200,
    }
    req = urllib.request.Request(
        endpoint["url"],
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + api_key,
            "Accept": "application/json",
        },
        method="POST",
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = e.read().decode("utf-8", "replace")[:300]
        except Exception:
            pass
        return {"ok": False,
                "error": redact("API 返回 HTTP %s：%s" % (e.code, detail))}
    except urllib.error.URLError as e:
        return {"ok": False,
                "error": redact("API 连接失败：%s" % getattr(e, "reason", e))}
    except Exception as e:
        return {"ok": False,
                "error": redact("API 请求失败（%s）：%s"
                                % (type(e).__name__, e))}
    latency = round(time.monotonic() - started, 2)
    try:
        data = json.loads(raw.decode("utf-8"))
    except Exception as e:
        return {"ok": False, "error": redact("API 返回不是 JSON：%s" % e)}
    text = _extract_text(data)
    if not text:
        return {"ok": False, "error": "API 返回里没有可读内容"}
    matched_raw = _parse_matched(text)
    id_to_candidate = {i + 1: c["id"] for i, c in enumerate(items)}
    matched, dropped = [], []
    for num in matched_raw:
        if num in id_to_candidate:
            cid = id_to_candidate[num]
            if cid not in matched:
                matched.append(cid)
        else:
            dropped.append(num)
    return {"ok": True, "matched_ids": matched, "dropped": dropped,
            "latency": latency, "provider": endpoint["label"],
            "raw_text": text[:300]}
