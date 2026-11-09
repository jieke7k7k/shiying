#!/usr/bin/env python3
"""本地语义搜图（3.4.0）：Chinese-CLIP 图像/文本向量 + 可续算索引。

设计要点（与 3.3.0 规格逐条对应）：

  模型
  ----
  - 默认模型是中文 CLIP（CN-CLIP）。规格首选 RN50 INT8 ONNX，但截至
    2026-10-31 实测：官方 OFA-Sys/chinese-clip-rn50 只发布 PyTorch 权重
    （clip_cn_rn50.pt，308 MB），HuggingFace / ModelScope 上都没有可用的
    RN50 ONNX 发布版；本模块因此改用“当前体积最小的中文 CLIP INT8 ONNX”
    —— Xenova/chinese-clip-vit-base-patch16 的 INT8 量化版（ViT-B/16，
    190,842,404 字节，512 维）。绝没有悄悄换成英文 MobileCLIP。
  - 模型不随主包捆绑：只有用户点“建立语义索引”且本地缺模型时才下载，
    下载到可写目录的 models/ 下（程序目录可写即程序目录/models）。
    下载走 HTTPS，先写 .part 临时文件，校验 SHA-256 后再 os.replace。
  - 用 onnxruntime 跑模型，不用 cv2.dnn（对 Transformer 算子兼容差）。

  索引
  ----
  - 主键 = 文件内容 sha256（scan.py 已算），不用相对路径；同一内容多份
    文件共享一条向量。
  - 单文件 semantic_index.npz，位于 app_paths 可写目录的 semantic/ 下。
    文件头（meta）记录 model_id 与 dim，模型不一致时判为不兼容、要求重建。
  - 写入：同目录临时文件 + fsync + os.replace 原子替换；索引线程每算完
    一批就落盘一次，暂停/取消时也会落盘，下次按 sha256 差集续算。
  - 查询只在“当前扫描结果的 sha256 集合”内计算；失效记录超过 20% 时
    做一次清理；删除成功后由 server.py 调 forget_images() 立即剔除向量。

  置信度校准（实测依据，见 README 3.3.0 章节）
  ----
  - 原始余弦不能直接当百分比：在 25 张演示合成图上跑 18 个中文词
    （450 个图文对），余弦分布为 min 0.299 / 中位数 0.336 /
    p95 0.386 / max 0.411，动态范围只有约 0.11。
  - 因此做线性映射：score = (cos - 0.30) / (0.42 - 0.30) * 100，
    截断到 0~100。0.30 取实测最小值附近（留一点余量），0.42 取实测
    最大值上方，避免分数顶格。
  - 三档阈值（映射后的 0~100 分）：宽松 ≥ 35（cos ≈ 0.342）、
    标准 ≥ 55（cos ≈ 0.366）、严格 ≥ 72（cos ≈ 0.386，约等于实测
    p95）。演示图上“标准”能把海滩/食物/城市/人像/截图等明确命中留下，
    “严格”只留最像的少数几张，“宽松”会带进相邻场景（如公园）。
  - 重要限定：演示图是生成器画的纯色合成图，不是真实照片；这套阈值
    只能证明映射逻辑可用，真实照片的分布仍需 Windows 真机复测后再校准。
"""
import hashlib
import io
import json
import math
import os
import re
import threading
import time
import unicodedata
import urllib.error
import urllib.request

try:
    import app_paths
except Exception:  # 极端单文件场景
    class _AppPaths(object):
        @staticmethod
        def writable_path(*parts):
            return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                *parts)

        @staticmethod
        def resource_dir():
            return os.path.dirname(os.path.abspath(__file__))
    app_paths = _AppPaths()

# ---- 模型登记（实际使用的型号，不冒充 RN50）----
MODEL_ID = "chinese-clip-vit-base-patch16-int8"
MODEL_LABEL = "Chinese-CLIP ViT-B/16 INT8（ONNX）"
MODEL_FILE = "chinese_clip_vit_b16_int8.onnx"
VOCAB_FILE = "chinese_clip_vit_b16_vocab.txt"
MODEL_BYTES = 190842404
MODEL_SHA256 = ("a2c2849329eafb24100acec5a9421a39"
                "f8cb445296444c5c98855b69e5e7a751")
VOCAB_BYTES = 109540
VOCAB_SHA256 = ("45bbac6b341c319adc98a532532882e91"
                "a9cefc0329aa57bac9ae761c27b291c")
EMBED_DIM = 512
IMAGE_SIZE = 224
# CLIP 图像归一化参数（与模型 preprocessor_config.json 一致）
IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)
MAX_TEXT_TOKENS = 52

_HF_REPO = "Xenova/chinese-clip-vit-base-patch16"
_DOWNLOAD_HOSTS = ("huggingface.co", "hf-mirror.com")


def _download_urls(remote_path):
    return ["https://%s/%s/resolve/main/%s" % (host, _HF_REPO, remote_path)
            for host in _DOWNLOAD_HOSTS]


MODEL_URLS = _download_urls("onnx/model_quantized.onnx")
VOCAB_URLS = _download_urls("vocab.txt")

# ---- 置信度校准（数值来自实测，见模块 docstring）----
# P1-6 说明：以下常数基于 25 张合成演示图实测，不是拍脑袋；但真机照片
# 的不相关图文余弦通常更低（约 0.15~0.22）、相关约 0.26~0.36，0.30 在真机
# 上可能偏严。没有真实照片分布数据前不擅自下调常数，留待真机实测验证
# 后再调；真机若发现标准档偏严，优先验证后再把 FLOOR 下调到 0.22~0.26。
CALIB_FLOOR = 0.30   # 实测最低余弦 0.299
CALIB_CEIL = 0.42    # 实测最高余弦 0.411
BANDS = {"loose": 35, "standard": 55, "strict": 72}
BAND_LABELS = {"loose": "宽松", "standard": "标准", "strict": "严格"}
DEFAULT_BAND = "standard"
MAX_RESULTS = 200
INDEX_PRUNE_RATIO = 0.20  # 旧值保留作参考；P1-1 后搜词不再自动清理，不再使用

_RENDER_LOCK = threading.Lock()


def model_dir():
    """模型目录：可写目录/models（程序目录可写时即程序目录/models）。"""
    path = app_paths.writable_path("models")
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        pass
    return path


def model_path(name):
    return os.path.join(model_dir(), name)


def semantic_dir():
    path = app_paths.writable_path("semantic")
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        pass
    return path


def index_path():
    return os.path.join(semantic_dir(), "semantic_index.npz")


def _file_size(path):
    try:
        return os.path.getsize(path)
    except OSError:
        return -1


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def model_status():
    """模型文件是否就绪。只查体积；SHA-256 在下载完成时校验一次。"""
    model_ok = _file_size(model_path(MODEL_FILE)) == MODEL_BYTES
    vocab_ok = _file_size(model_path(VOCAB_FILE)) == VOCAB_BYTES
    missing = []
    if not model_ok:
        missing.append(MODEL_FILE)
    if not vocab_ok:
        missing.append(VOCAB_FILE)
    try:
        import onnxruntime  # noqa: F401
        runtime_ok, runtime_note = True, "onnxruntime 已安装"
    except Exception as e:
        runtime_ok, runtime_note = False, "onnxruntime 未安装：%s" % e
    return {
        "present": model_ok and vocab_ok,
        "model_id": MODEL_ID,
        "model_label": MODEL_LABEL,
        "dim": EMBED_DIM,
        "model_bytes": MODEL_BYTES,
        "missing": missing,
        "dir": model_dir(),
        "runtime_ok": runtime_ok,
        "runtime_note": runtime_note,
        "download_hint": ("首次建立索引时自动下载 %s（约 %.0f MB），"
                          "需要联网；失败可重试。"
                          % (MODEL_LABEL, MODEL_BYTES / 1024.0 / 1024.0)),
    }


# ---------------------------------------------------------------- 下载

class DownloadError(Exception):
    pass


def _download_one(urls, dest, expected_bytes, expected_sha256,
                  progress, cancel_event=None):
    """按候选镜像逐个尝试；成功后原子替换到 dest。"""
    last_error = None
    for url in urls:
        part = dest + ".part"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Shiying/3.4.0"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                total = int(resp.headers.get("Content-Length") or 0) or expected_bytes
                done = 0
                with open(part, "wb") as f:
                    while True:
                        if cancel_event is not None and cancel_event.is_set():
                            raise DownloadError("已取消下载")
                        chunk = resp.read(1024 * 256)
                        if not chunk:
                            break
                        f.write(chunk)
                        done += len(chunk)
                        if progress:
                            progress(done, total)
                    f.flush()
                    os.fsync(f.fileno())
            if _file_size(part) != expected_bytes:
                raise DownloadError("文件体积不对（%d 字节，应为 %d）"
                                    % (_file_size(part), expected_bytes))
            if _sha256_file(part) != expected_sha256:
                raise DownloadError("SHA-256 校验失败，文件可能损坏")
            os.replace(part, dest)
            return
        except Exception as e:
            last_error = e
            try:
                os.unlink(dest + ".part")
            except OSError:
                pass
            if cancel_event is not None and cancel_event.is_set():
                raise DownloadError("已取消下载")
            continue
    raise DownloadError("下载失败：%s（可换网络后重试）"
                        % _readable_error(last_error))


def _readable_error(e):
    if isinstance(e, urllib.error.HTTPError):
        return "服务器返回 HTTP %s" % e.code
    if isinstance(e, urllib.error.URLError):
        return "网络连接失败（%s）" % getattr(e, "reason", e)
    if isinstance(e, DownloadError):
        return str(e)
    return "%s: %s" % (type(e).__name__, e)


def ensure_model(progress=None, cancel_event=None):
    """确保模型与词表就绪；缺什么下什么。progress(阶段, 已完成, 总量)。"""
    status = model_status()
    if status["present"]:
        return
    if not status["runtime_ok"]:
        raise DownloadError(status["runtime_note"]
                            + "。请先运行 pip install onnxruntime 再重试。")
    tasks = []
    if _file_size(model_path(MODEL_FILE)) != MODEL_BYTES:
        tasks.append((MODEL_FILE, MODEL_URLS, MODEL_BYTES, MODEL_SHA256))
    if _file_size(model_path(VOCAB_FILE)) != VOCAB_BYTES:
        tasks.append((VOCAB_FILE, VOCAB_URLS, VOCAB_BYTES, VOCAB_SHA256))
    total_all = sum(t[2] for t in tasks)
    base = 0
    for name, urls, size, sha in tasks:
        if progress:
            progress("download", base, total_all)
        _download_one(
            urls, model_path(name), size, sha,
            (lambda done, _total, _base=base: progress(
                "download", _base + done, total_all)) if progress else None,
            cancel_event)
        base += size


# ---------------------------------------------------------------- 分词

class BertChineseTokenizer(object):
    """极简 BERT WordPiece 分词（中文 CLIP 用的就是 BERT 词表）。

    与 HuggingFace BertTokenizer(do_lower_case=True, tokenize_chinese_chars
    =True) 对齐：控制字符清理 → 汉字逐字加空格 → 小写/去重音 → 标点拆分
    → WordPiece 贪心最长匹配。词表 21128 行，[CLS]=101、[SEP]=102。
    """

    def __init__(self, vocab_path):
        self.vocab = {}
        with open(vocab_path, encoding="utf-8", errors="replace") as f:
            for idx, line in enumerate(f):
                self.vocab[line.rstrip("\n")] = idx
        self.cls_id = self.vocab.get("[CLS]", 101)
        self.sep_id = self.vocab.get("[SEP]", 102)
        self.unk_id = self.vocab.get("[UNK]", 100)

    @staticmethod
    def _is_cjk(ch):
        cp = ord(ch)
        return (0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF
                or 0x20000 <= cp <= 0x2A6DF or 0xF900 <= cp <= 0xFAFF
                or 0x2F800 <= cp <= 0x2FA1F)

    @classmethod
    def _basic_tokens(cls, text):
        text = "".join((" " + c + " ") if cls._is_cjk(c) else c
                       for c in text)
        text = "".join(c for c in text
                       if not (unicodedata.category(c).startswith("C")
                               or c in "\uFFFD"))
        out = []
        for token in text.split():
            token = unicodedata.normalize("NFD", token.lower())
            token = "".join(c for c in token
                            if unicodedata.category(c) != "Mn")
            cur = ""
            for ch in token:
                if unicodedata.category(ch).startswith("P") or ch in "`":
                    if cur:
                        out.append(cur)
                        cur = ""
                    out.append(ch)
                else:
                    cur += ch
            if cur:
                out.append(cur)
        return out

    def _wordpiece(self, token):
        if len(token) > 100:
            return [self.unk_id]
        ids, start = [], 0
        while start < len(token):
            end = len(token)
            found = None
            while start < end:
                piece = token[start:end]
                if start > 0:
                    piece = "##" + piece
                if piece in self.vocab:
                    found = piece
                    break
                end -= 1
            if found is None:
                return [self.unk_id]
            ids.append(self.vocab[found])
            start = end
        return ids

    def encode(self, text):
        ids = [self.cls_id]
        for token in self._basic_tokens(text or ""):
            ids.extend(self._wordpiece(token))
        ids = ids[:MAX_TEXT_TOKENS - 1] + [self.sep_id]
        return ids


# ---------------------------------------------------------------- 编码器

_SESSION_LOCK = threading.Lock()
_SESSION = {"sess": None, "tokenizer": None, "model_id": MODEL_ID}
_TEXT_CACHE = {}
_TEXT_CACHE_ORDER = []
_TEXT_CACHE_MAX = 64


def _load_session():
    """线程安全的全局单例：文本编码与图像编码共用同一个会话。"""
    with _SESSION_LOCK:
        if _SESSION["sess"] is not None:
            return _SESSION["sess"], _SESSION["tokenizer"]
        status = model_status()
        if not status["present"]:
            raise RuntimeError("语义模型还没下载：请先点“建立语义索引”")
        if not status["runtime_ok"]:
            raise RuntimeError(status["runtime_note"]
                               + "。请先运行 pip install onnxruntime。")
        try:
            import onnxruntime as ort
        except Exception as e:
            raise RuntimeError("onnxruntime 不可用：%s" % e)
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess = ort.InferenceSession(model_path(MODEL_FILE),
                                    sess_options=opts,
                                    providers=["CPUExecutionProvider"])
        tokenizer = BertChineseTokenizer(model_path(VOCAB_FILE))
        _SESSION.update({"sess": sess, "tokenizer": tokenizer,
                         "model_id": MODEL_ID})
        return sess, tokenizer


def _dummy_image():
    import numpy as np
    return np.zeros((1, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)


def _text_ids(tokenizer, text):
    import numpy as np
    ids = tokenizer.encode(text)
    return (np.array([ids], dtype=np.int64),
            np.ones((1, len(ids)), dtype=np.int64))


def encode_text(text):
    """返回已 L2 归一化的 float32 向量（512,）。文本向量带小缓存。"""
    key = (text or "").strip()
    if not key:
        raise ValueError("搜索词为空")
    if key in _TEXT_CACHE:
        return _TEXT_CACHE[key].copy()
    import numpy as np
    sess, tokenizer = _load_session()
    input_ids, attention = _text_ids(tokenizer, key)
    # 组合模型要求同时给 pixel_values：这里喂一张全零图，只取 text_embeds。
    out = sess.run(["text_embeds"], {
        "input_ids": input_ids,
        "attention_mask": attention,
        "pixel_values": _dummy_image(),
    })[0][0]
    out = out.astype(np.float32)
    norm = float(np.linalg.norm(out))
    if norm > 0:
        out /= norm
    with _SESSION_LOCK:
        _TEXT_CACHE[key] = out
        _TEXT_CACHE_ORDER.append(key)
        while len(_TEXT_CACHE_ORDER) > _TEXT_CACHE_MAX:
            old = _TEXT_CACHE_ORDER.pop(0)
            _TEXT_CACHE.pop(old, None)
    return out.copy()


def _load_image_rgb(path):
    from PIL import Image, ImageOps
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        return im.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BICUBIC)


def _preprocess(images):
    import numpy as np
    mean = np.array(IMAGE_MEAN, dtype=np.float32)
    std = np.array(IMAGE_STD, dtype=np.float32)
    out = []
    for im in images:
        arr = np.asarray(im, dtype=np.float32) / 255.0
        arr = (arr - mean) / std
        out.append(arr.transpose(2, 0, 1))
    return np.stack(out, axis=0)


def encode_image_paths(paths, batch_size=8, cancel_event=None,
                       pause_event=None):
    """逐批编码图片；生成 (path, vector) —— 便于边算边落盘。

    暂停只在批次边界生效：当前批次已经算完的向量会全部产出并落盘，
    不会白算半批；取消则立即停止。
    """
    sess, _tokenizer = _load_session()
    import numpy as np
    for start in range(0, len(paths), batch_size):
        if cancel_event is not None and cancel_event.is_set():
            return
        if pause_event is not None and pause_event.is_set():
            return
        batch = paths[start:start + batch_size]
        imgs, keep = [], []
        for path in batch:
            try:
                imgs.append(_load_image_rgb(path))
                keep.append(path)
            except Exception:
                continue
        if not imgs:
            continue
        pixel_values = _preprocess(imgs)
        feats = sess.run(["image_embeds"], {
            "input_ids": np.zeros((1, 2), dtype=np.int64),
            "attention_mask": np.ones((1, 2), dtype=np.int64),
            "pixel_values": pixel_values,
        })[0]
        feats = feats.astype(np.float32)
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        feats = feats / norms
        for path, vec in zip(keep, feats):
            yield path, vec


# ---------------------------------------------------------------- 索引存储

def _atomic_write(path, writer):
    """同目录临时文件 + fsync + os.replace，进程崩溃也不留半截文件。"""
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        writer(f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


class IndexStore(object):
    """semantic_index.npz 的内存视图（sha256 -> 向量）。"""

    def __init__(self, path=None):
        self.path = path or index_path()
        self.lock = threading.RLock()
        self.shas = []
        self.vectors = None
        self.meta = {}
        self.incompatible = False
        self.incompatible_reason = ""
        self._loaded = False

    def load(self):
        with self.lock:
            if self._loaded:
                return
            self._loaded = True
            if not os.path.isfile(self.path):
                return
            try:
                import numpy as np
                with np.load(self.path, allow_pickle=False) as data:
                    self.meta = json.loads(str(data["meta"]))
                    self.shas = [str(s) for s in data["sha"].tolist()]
                    self.vectors = np.asarray(data["vec"], dtype="float32")
            except Exception as e:
                self.meta = {}
                self.shas = []
                self.vectors = None
                self.incompatible = True
                self.incompatible_reason = "索引文件损坏或格式不兼容：%s" % e
                return
            if (self.meta.get("model_id") != MODEL_ID
                    or int(self.meta.get("dim") or 0) != EMBED_DIM):
                self.incompatible = True
                self.incompatible_reason = (
                    "索引由其他模型建立（%s / %s 维），当前模型是 %s / %s 维，"
                    "需要重建索引。"
                    % (self.meta.get("model_id"), self.meta.get("dim"),
                       MODEL_ID, EMBED_DIM))

    def save(self):
        with self.lock:
            import numpy as np
            if self.vectors is None or not self.shas:
                try:
                    os.unlink(self.path)
                except OSError:
                    pass
                self._loaded = True
                return
            meta = {
                "format": 1,
                "model_id": MODEL_ID,
                "dim": EMBED_DIM,
                "count": len(self.shas),
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            self.meta = meta
            shas = np.array(self.shas, dtype="U64")
            vectors = np.asarray(self.vectors, dtype="float32")

            def writer(f):
                np.savez(f, sha=shas, vec=vectors,
                         meta=np.array(json.dumps(meta), dtype="U"))

            _atomic_write(self.path, writer)
            self._loaded = True

    def entries(self):
        with self.lock:
            if self.vectors is None:
                return [], None
            return list(self.shas), self.vectors

    def count(self):
        with self.lock:
            return len(self.shas)

    def upsert(self, pairs):
        """pairs: [(sha, vector)]，已存在的 sha 跳过。"""
        if not pairs:
            return
        import numpy as np
        with self.lock:
            existing = set(self.shas)
            new = [(s, v) for s, v in pairs if s not in existing]
            if not new:
                return
            new_vecs = np.stack([v for _s, v in new]).astype("float32")
            if self.vectors is None or not self.shas:
                self.vectors = new_vecs
                self.shas = [s for s, _v in new]
            else:
                self.vectors = np.concatenate([self.vectors, new_vecs], axis=0)
                self.shas.extend(s for s, _v in new)
            self._loaded = True

    def forget(self, shas):
        drop = set(shas)
        if not drop:
            return 0
        with self.lock:
            keep = [i for i, s in enumerate(self.shas) if s not in drop]
            removed = len(self.shas) - len(keep)
            if not removed:
                return 0
            self.shas = [self.shas[i] for i in keep]
            if self.vectors is not None and keep:
                self.vectors = self.vectors[keep]
            else:
                self.vectors = None
            self.save()
            return removed

    def prune(self, valid_shas):
        with self.lock:
            keep = [i for i, s in enumerate(self.shas) if s in valid_shas]
            removed = len(self.shas) - len(keep)
            if removed:
                self.shas = [self.shas[i] for i in keep]
                self.vectors = (self.vectors[keep] if keep else None)
                self.save()
            return removed


_STORE = IndexStore()


def store():
    _STORE.load()
    return _STORE


# ---------------------------------------------------------------- 索引线程

class _IndexState(object):
    def __init__(self):
        self.lock = threading.Lock()
        self.thread = None
        self.running = False
        self.paused = False
        self.cancelled = False
        self.phase = "idle"
        self.done = 0
        self.total = 0
        self.downloaded = 0
        self.download_total = 0
        self.error = ""
        self.message = ""
        self.skipped = 0
        self.pause_event = threading.Event()
        self.cancel_event = threading.Event()

    def snapshot(self):
        with self.lock:
            st = store()
            return {
                "running": self.running,
                "paused": self.paused,
                "phase": self.phase,
                "done": self.done,
                "total": self.total,
                "downloaded": self.downloaded,
                "download_total": self.download_total,
                "error": self.error,
                "message": self.message,
                "skipped": self.skipped,
                "index_count": st.count(),
                "model_present": model_status()["present"],
            }


_STATE = _IndexState()


def progress():
    return _STATE.snapshot()


def _items_for_scan(scan_result, root):
    """从扫描结果里取 (sha256, 绝对路径)；同一内容只保留一条。"""
    seen = {}
    for im in (scan_result or {}).get("images", []):
        if "error" in im or not im.get("sha256"):
            continue
        sha = im["sha256"]
        if sha in seen:
            continue
        rel = im.get("relative_path") or im.get("id")
        if not rel:
            continue
        seen[sha] = os.path.join(root, rel)
    return seen


def _index_worker(items):
    st = _STATE
    store_obj = store()
    try:
        if not model_status()["present"]:
            with st.lock:
                st.phase = "download"
                st.message = "正在下载语义模型（约 %.0f MB）……" % (
                    MODEL_BYTES / 1024.0 / 1024.0)
                st.downloaded = 0
                st.download_total = MODEL_BYTES

            def on_progress(_phase, done, total):
                with st.lock:
                    st.downloaded = done
                    st.download_total = total

            ensure_model(on_progress, st.cancel_event)
            if st.cancel_event.is_set():
                with st.lock:
                    st.phase = "idle"
                    st.message = "已取消下载"
                return
        with st.lock:
            st.phase = "index"
            st.message = "模型就绪，正在建立索引……"

        existing = set(store_obj.entries()[0])
        pending = [(sha, path) for sha, path in items.items()
                   if sha not in existing]
        with st.lock:
            st.total = len(pending)
            st.done = 0
            st.skipped = 0
            st.message = ("需要为 %d 张图计算向量" % len(pending)
                          if pending else "索引已是最新")
        if not pending:
            with st.lock:
                st.phase = "idle"
                st.message = "索引已是最新，无需重复计算"
            return

        paths = [p for _s, p in pending]
        sha_of = {p: s for s, p in pending}
        buffer = []
        last_save = time.monotonic()
        for path, vec in encode_image_paths(paths, 8, st.cancel_event,
                                            st.pause_event):
            if st.cancel_event.is_set():
                break
            sha = sha_of.get(path)
            if not sha:
                continue
            buffer.append((sha, vec))
            with st.lock:
                st.done += 1
            if len(buffer) >= 8:
                store_obj.upsert(buffer)
                buffer = []
                # 每 64 张或每 10 秒落盘一次；暂停/结束/取消时立即落盘。
                if st.done % 64 == 0 or time.monotonic() - last_save > 10:
                    store_obj.save()
                    last_save = time.monotonic()
        if buffer:
            store_obj.upsert(buffer)
        # 结束/暂停/取消都立即落盘，保证“算好的向量不白算”。
        store_obj.save()
        if st.cancel_event.is_set():
            with st.lock:
                st.phase = "idle"
                st.message = "已取消（已算好的向量已保存，下次可续算）"
            return
        if st.pause_event.is_set():
            with st.lock:
                st.paused = True
                st.phase = "paused"
                st.message = ("已暂停：%d/%d 已保存，下次按 sha256 差集续算"
                              % (st.done, st.total))
            return
        with st.lock:
            st.phase = "idle"
            st.message = "索引完成：%d 张已写入" % store_obj.count()
    except Exception as e:
        if st.cancel_event.is_set():
            with st.lock:
                st.phase = "idle"
                st.message = "已取消（已算好的向量已保存）"
        else:
            with st.lock:
                st.error = _readable_error(e)
                st.phase = "error"
                st.message = "建立索引失败：%s" % st.error
    finally:
        with st.lock:
            st.running = False
            if st.phase not in ("error", "paused"):
                st.paused = False


def start_index(scan_result, root):
    """启动后台索引线程；已在跑时直接返回当前状态。"""
    st = _STATE
    started = False
    with st.lock:
        if not st.running:
            store_obj = store()
            if store_obj.incompatible:
                # 模型不一致/索引损坏：旧向量不可用，清空后重建。
                store_obj.shas = []
                store_obj.vectors = None
                store_obj.incompatible = False
                store_obj.incompatible_reason = ""
                try:
                    os.unlink(store_obj.path)
                except OSError:
                    pass
            st.running = True
            st.paused = False
            st.error = ""
            st.message = "准备建立索引……"
            st.phase = "prepare"
            st.done = 0
            st.total = 0
            st.cancel_event.clear()
            st.pause_event.clear()
            started = True
    if not started:
        return st.snapshot()
    items = _items_for_scan(scan_result, root)
    thread = threading.Thread(target=_index_worker, args=(items,),
                              name="semantic-index", daemon=True)
    with st.lock:
        st.thread = thread
    thread.start()
    return st.snapshot()


def pause_index():
    with _STATE.lock:
        running = _STATE.running
        if running:
            _STATE.pause_event.set()
            _STATE.message = "正在暂停：算完当前这批就保存……"
    return _STATE.snapshot()


def cancel_index():
    with _STATE.lock:
        running = _STATE.running
        if running:
            _STATE.cancel_event.set()
            _STATE.pause_event.set()
            _STATE.message = "正在取消：已算好的向量会保存……"
    return _STATE.snapshot()


# ---------------------------------------------------------------- 查询

def calibrate(cos):
    """余弦 → 0~100 分（线性映射，常数与实测依据见模块 docstring）。"""
    score = (float(cos) - CALIB_FLOOR) / (CALIB_CEIL - CALIB_FLOOR) * 100.0
    return max(0, min(100, int(round(score))))


def band_threshold(band):
    return BANDS.get(band, BANDS[DEFAULT_BAND])


def index_info():
    st = store()
    return {
        "count": st.count(),
        "model_id": MODEL_ID,
        "dim": EMBED_DIM,
        "path": index_path(),
        "incompatible": st.incompatible,
        "incompatible_reason": st.incompatible_reason,
        "meta": st.meta,
    }


def scan_coverage(scan_result):
    """当前扫描里有多少张（按 sha 去重）已有向量。"""
    shas = {im["sha256"] for im in (scan_result or {}).get("images", [])
            if "error" not in im and im.get("sha256")}
    st = store()
    have = set(st.entries()[0])
    return {"total": len(shas), "indexed": len(shas & have)}


def search(scan_result, root, query, band=DEFAULT_BAND, top_k=MAX_RESULTS,
           use_api=False, api_opts=None):
    """本地语义检索；可选对 Top 20 做 API 精排。"""
    import numpy as np
    st = store()
    if st.incompatible:
        return {"ok": False, "error": st.incompatible_reason,
                "need_rebuild": True}
    shas, vectors = st.entries()
    if not shas or vectors is None:
        return {"ok": False, "error": "还没有语义索引，请先点“建立语义索引”",
                "need_index": True}

    scan_shas = {im["sha256"] for im in (scan_result or {}).get("images", [])
                 if "error" not in im and im.get("sha256")}
    if not scan_shas:
        return {"ok": False, "error": "当前扫描结果里没有可检索的图片"}

    index_of = {s: i for i, s in enumerate(shas)}
    valid_idx = [i for i, s in enumerate(shas) if s in scan_shas]
    # P1-1 修复：删掉搜词里的自动 prune。查询本就只抽当前文件夹的向量，
    # 自动清理会误删另一文件夹的全部向量（先建 A 文件夹 1000 条、再切到
    # 演示 22 条搜词，失效 98% 就把 A 删光）。已删照片由 /api/delete 的
    # forget_images() 精确剔除，不靠搜词时批量清理。
    pruned = 0
    if not valid_idx:
        return {"ok": False,
                "error": "当前文件夹的图片还没建立语义索引",
                "need_index": True, "pruned": pruned}

    text_vec = encode_text(query)
    matrix = np.asarray(vectors[valid_idx], dtype=np.float32)
    cos = matrix @ text_vec
    order = np.argsort(-cos)

    # sha -> 当前扫描里的所有图片 id（精确重复共享同一向量）
    ids_by_sha = {}
    for im in scan_result.get("images", []):
        if "error" in im or not im.get("sha256"):
            continue
        if im["sha256"] not in scan_shas:
            continue
        ids_by_sha.setdefault(im["sha256"], []).append(im["id"])

    ranked = []
    for pos in order:
        sha = shas[valid_idx[pos]]
        c = float(cos[pos])
        score = calibrate(c)
        for image_id in ids_by_sha.get(sha, []):
            ranked.append({"id": image_id, "sha256": sha, "cos": round(c, 4),
                           "score": score})
    # 同一分数按 id 稳定排序，避免同分抖动
    ranked.sort(key=lambda x: (-x["cos"], x["id"]))
    if top_k:
        ranked = ranked[:int(top_k)]

    threshold = band_threshold(band)
    local_top = ranked[:20]

    api_meta = {"requested": bool(use_api), "used": False, "matched": [],
                "error": "", "notice": ""}
    if use_api:
        api_meta = _apply_api_rerank(query, local_top, root, api_opts or {})
        matched = set(api_meta.get("matched", []))
        if matched:
            api_order = [r for r in local_top if r["id"] in matched]
            rest = [r for r in ranked if r["id"] not in matched]
            ranked = api_order + rest
            for r in ranked:
                r["api_matched"] = r["id"] in matched

    results = [r for r in ranked
               if r["score"] >= threshold or r.get("api_matched")]
    for r in results:
        r["band"] = band
        r["band_label"] = BAND_LABELS.get(band, band)

    return {
        "ok": True,
        "query": query,
        "band": band,
        "band_label": BAND_LABELS.get(band, band),
        "threshold": threshold,
        "score_mapping": {
            "type": "linear",
            "cos_floor": CALIB_FLOOR,
            "cos_ceil": CALIB_CEIL,
            "bands": BANDS,
            "note": "score=(cos-%.2f)/(%.2f-%.2f)*100；阈值来自"
                    "25 张演示合成图、18 个中文词的实测分布，真实照片需真机复测"
                    % (CALIB_FLOOR, CALIB_CEIL, CALIB_FLOOR),
        },
        "results": results,
        "local_top": [r["id"] for r in local_top],
        "total_ranked": len(ranked),
        "pruned": pruned,
        "api": api_meta,
    }


def _apply_api_rerank(query, local_top, root, api_opts):
    """API 精排：只把本地 Top 20 的 200x200 缩略图拼成 2 张图发出去。"""
    meta = {"requested": True, "used": False, "matched": [], "error": "",
            "notice": "", "provider": api_opts.get("provider", "")}
    try:
        import api_rerank
    except Exception as e:
        meta["error"] = "API 精排模块不可用：%s" % e
        meta["notice"] = "已自动退回本地结果"
        return meta
    if not api_rerank.has_key():
        meta["error"] = "没有配置 API Key"
        meta["notice"] = "未配置 Key，已自动退回本地搜索结果"
        return meta
    candidates = []
    for item in local_top:
        path = os.path.join(root, item["id"]) if root else None
        if path and os.path.isfile(path):
            candidates.append({"id": item["id"], "path": path,
                               "score": item["score"]})
    if len(candidates) < 2:
        meta["notice"] = "本地候选不足 2 张，已跳过 API 精排"
        return meta
    out = api_rerank.rerank(query, candidates, api_opts)
    if not out.get("ok"):
        meta["error"] = out.get("error", "API 精排失败")
        meta["notice"] = "API 不可用，已自动退回本地结果"
        return meta
    known = {c["id"] for c in candidates}
    matched = [i for i in out.get("matched_ids", []) if i in known]
    meta.update({"used": True, "matched": matched,
                 "latency": out.get("latency"),
                 "provider": out.get("provider", meta["provider"])})
    if not matched:
        meta["notice"] = "API 没有给出有效匹配，保持本地排序"
    return meta


def forget_images(shas, remaining_shas):
    """删除成功后剔除向量；同内容还有别的文件在时保留。"""
    drop = [s for s in set(shas) if s not in set(remaining_shas)]
    return store().forget(drop)


def rebuild_needed():
    st = store()
    return st.incompatible, st.incompatible_reason
