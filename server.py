#!/usr/bin/env python3
"""Shiying-3.4.0 本地服务：把扫描与界面串起来，本地优先。

  - 只监听 127.0.0.1（本机回环），不对局域网/外网开放。
  - 页面、样式、脚本全部内联在 index.html 里，不加载任何 CDN、字体、
    统计脚本；搜索与索引默认完全离线。联网只有两种、都由用户主动触发：
    首次建立语义索引时的模型下载，以及勾选“本次启用 API 精排”时的厂商请求。唯一的联网动作是用户主动点
    “建立语义索引”时的模型下载，以及用户本次勾选 API 精排时的厂商请求。
  - 接口：
      GET  /                 -> index.html
      GET  /api/health       -> {"ok", "version", "data_dir", "data_dir_fallback"}
      GET  /api/faces        -> 人脸尝试报告（诚实版，见 face_clustering.py）
      GET  /api/file?rel=xx  -> 本次扫描根目录内的原图（防目录穿越）
      GET  /api/thumb?rel=xx -> 本次扫描根目录内图片的 200px 动态缩略图
      GET  /api/pick_folder  -> 弹出系统文件夹选择框（子进程 + 超时）
      GET  /api/ping?s=xx    -> 窗口心跳（app_native 判断窗口是否还活着）
      POST /api/scan         -> {"demo": true} 或 {"path": "文件夹路径"}
                                 （全局扫描锁，正在扫描时返回 409 busy）

  3.4.0 复核表接口（见 face_review.py；只改单脸归属，绝不删照片）：
      GET  /api/review/list          -> 三卡片 + 人物 + 计数 + 阈值说明
      GET  /api/review/history       -> 持久撤销面板用的操作历史
      GET  /api/review/crop?k=&size= -> 单脸裁剪小图（动态裁剪 + 留白）
      POST /api/review/act           -> {"action","keys","person_id","name"}
      POST /api/review/undo          -> {"txn_id","keys"?,"force"?}
      POST /api/review/names         -> {"person_id","name"} 人名写本机文件
      POST /api/review/migrate_names -> {"names": {...}} 一次性迁移旧名

      POST /api/delete       -> {"items": [{"id", "source_dir"}],
                                 "confirmed": true}
                                 （必须二次确认；只走回收站或隔离区，
                                 见 trash_util.py；绝不直接删除）
      POST /api/client_closed -> 窗口关闭信号（pagehide 时 sendBeacon）
      POST /api/shutdown     -> 干净停止服务（页面上的“停止服务”按钮用）

  3.3.0 语义搜图接口（见 semantic_search.py / api_rerank.py）：
      GET  /api/semantic/status?source_dir=xx
      POST /api/index/build     {"source_dir": "...", "resume": true}
      GET  /api/index_progress
      POST /api/index/pause
      POST /api/index/cancel
      POST /api/semantic/search {"query", "band", "use_api", "source_dir"}
      GET  /api/settings        -> 厂商/模型/掩码 Key（绝不返回明文 Key）
      POST /api/settings        -> {"provider","base_url","model","api_key",
                                    "clear_key","privacy_ack"}
      POST /api/semantic/privacy_ack
  - 关闭协议（P0-3）：/api/shutdown 先回包，再置 SHUTDOWN_EVENT；
    app_native 的主线程监听该事件，销毁 pywebview 窗口或结束它启动的
    浏览器进程，然后停服务。不再靠“服务线程停了但窗口还挂着”。
  - 扫描是只读的：scan.py 不写被扫描文件夹、不删除、不移动任何文件。
    唯一会移走文件的动作是用户在待删篓二次确认后的 /api/delete，
    且只移进回收站或工具包内 .trash_backup 隔离区。

用法：
  python3 server.py [--port 8765] [--no-open]
"""
import argparse
import json
import mimetypes
import os
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

VERSION = "3.4.0"

TOOL_DIR = os.path.dirname(os.path.abspath(__file__))
if TOOL_DIR not in sys.path:
    sys.path.insert(0, TOOL_DIR)
try:
    import app_paths
    RESOURCE_DIR = app_paths.resource_dir()
except Exception:
    app_paths = None
    RESOURCE_DIR = TOOL_DIR
for _p in (RESOURCE_DIR, TOOL_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)
DEMO_DIR = os.path.join(RESOURCE_DIR, "demo_images")
DEMO_MANIFEST = os.path.join(RESOURCE_DIR, "faces_manifest.json")
INDEX_HTML = os.path.join(RESOURCE_DIR, "index.html")
APP_NATIVE_PY = os.path.join(TOOL_DIR, "app_native.py")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MAX_BODY = 1024 * 1024
PICK_TIMEOUT = 600  # 选文件夹对话框最长等 10 分钟，超时就干净放弃

STATE = {"root": None, "result": None, "review_vectors": {}}
STATE_LOCK = threading.Lock()
# 3.4.0：复核表的写入串行化（读文件 -> 改 -> 写回），避免两个请求互相覆盖。
REVIEW_LOCK = threading.Lock()
# 单脸裁剪小图的进程内缓存：key -> (bytes, meta)
_CROP_CACHE = {}
_CROP_CACHE_MAX = 400
SCAN_LOCK = threading.Lock()      # P1-1：同一时刻只允许一轮扫描
PICK_LOCK = threading.Lock()      # 防止连点“浏览”弹出多个对话框
SHUTDOWN_EVENT = threading.Event()  # P0-3：app_native 主线程监听它退出

_CLIENT_LOCK = threading.Lock()
_CLIENT = {"token": "", "last_seen": 0.0, "closed_at": 0.0}


def note_client_ping(token):
    with _CLIENT_LOCK:
        if token:
            _CLIENT["token"] = token
        _CLIENT["last_seen"] = time.monotonic()
        _CLIENT["closed_at"] = 0.0  # 刷新页面时 pagehide 会先来，ping 到了就撤销


def note_client_closed(token):
    with _CLIENT_LOCK:
        if token and _CLIENT["token"] and token != _CLIENT["token"]:
            return
        _CLIENT["closed_at"] = time.monotonic()


def client_state(token=None):
    """给 app_native 轮询用：窗口是否来过、最后心跳时间、关闭信号时间。"""
    with _CLIENT_LOCK:
        seen = _CLIENT["last_seen"] > 0.0
        if token and _CLIENT["token"] and token != _CLIENT["token"]:
            seen = False
        return {"token": _CLIENT["token"], "seen": seen,
                "last_seen": _CLIENT["last_seen"],
                "closed_at": _CLIENT["closed_at"]}


def data_dir_info():
    if app_paths is None:
        return {"data_dir": TOOL_DIR, "data_dir_fallback": False,
                "data_dir_reason": ""}
    info = app_paths.writable_info()
    return {"data_dir": info["dir"],
            "data_dir_fallback": bool(info["fallback"]),
            "data_dir_reason": info.get("reason", "")}


def find_manifest(root):
    for cand in (os.path.join(os.path.dirname(os.path.abspath(root)),
                              "faces_manifest.json"),
                 os.path.join(root, "faces_manifest.json")):
        if os.path.exists(cand):
            return cand
    return None


# ---------------------------------------------------------------- 3.4.0 复核表

def review_paths():
    """复核数据文件与特征缓存文件都放在可写目录（与日志/隔离区同一处）。"""
    base = app_paths.writable_dir() if app_paths is not None else TOOL_DIR
    return (os.path.join(base, "face_review.json"),
            os.path.join(base, "face_embed_cache.json"))


def load_review_store(face_review):
    store_path, _cache = review_paths()
    return face_review.load_store(store_path), store_path


def current_review(face_review, store, note=None):
    """用当前扫描结果 + 本机记录生成一份复核表视图。"""
    with STATE_LOCK:
        result = STATE["result"]
        vectors = dict(STATE["review_vectors"])
    if not result:
        return None
    return face_review.build_review(store, result, vectors, note_extra=note)


def _refresh_review_clusters(store):
    """P1-2：忽略/撤销忽略后立刻重算自动簇，不用等下次全盘重扫。

    只在真实特征向量链路（embedding_dbscan）下做：演示假数据的簇是
    按固定计划造的，重算会改簇 id；演示下忽略只走复核表状态过滤，
    簇等下次重扫再更新。返回 True 表示已重算。
    """
    with STATE_LOCK:
        result = STATE.get("result")
        vectors = dict(STATE.get("review_vectors") or {})
    if not result or result.get("face_method") != "embedding_dbscan":
        return False
    try:
        import face_clustering
        import face_review
    except Exception:
        return False
    ignore = face_review.ignored_keys(store)
    seen = set()
    faces = []
    for face in result.get("face_faces") or []:
        key = face.get("key")
        if not key or key in ignore or key in seen:
            continue
        seen.add(key)
        faces.append(face)
    try:
        clusters = face_clustering._cluster_face_keys(faces, vectors)
    except Exception:
        return False
    groups = []
    for cluster in clusters:
        cid = cluster["cluster_id"]
        groups.append({
            "id": cid,
            "label": "人物 %s" % cid[3:],
            "method": "embedding_dbscan",
            "note": result.get("face_note") or "",
            "members": list(cluster["image_ids"]),
            "anchor_key": cluster.get("anchor_key"),
            "face_count": cluster.get("size"),
        })
    with STATE_LOCK:
        if STATE.get("result") is result:
            result["face_clusters"] = clusters
            result["face_groups"] = groups
    return True


def _touches_ignored(items):
    """一次复核操作有没有动过“忽略”状态（忽略或撤销忽略都算）。"""
    for it in items or []:
        before = it.get("before") or {}
        after = it.get("after") or {}
        if before.get("state") == "ignored" or after.get("state") == "ignored":
            return True
    return False


def review_payload(face_review, store, extra=None):
    body = current_review(face_review, store) or {
        "ok": False, "error": "请先扫描一个文件夹"}
    body["history"] = face_review.history_view(store)
    store_file, cache_file = review_paths()
    body["store_file"] = store_file
    body["cache_file"] = cache_file
    body["migration"] = store.get("migration") or {}
    if extra:
        body.update(extra)
    return body


def find_face(result, key):
    for face in result.get("face_faces") or []:
        if face.get("key") == key:
            return face
    return None


def resolve_in_root(root, rel):
    """把相对路径解析为根目录内的绝对路径；穿越根目录时返回 None。"""
    if rel is None or rel == "":
        return None
    # 拒绝绝对路径与 .. 穿越
    if os.path.isabs(rel):
        return None
    abs_root = os.path.abspath(root)
    abs_path = os.path.abspath(os.path.join(abs_root, rel))
    try:
        if os.path.commonpath([abs_root, abs_path]) != abs_root:
            return None
    except ValueError:
        return None
    return abs_path


def _norm_scope(path):
    """文件夹身份归一化：Windows 忽略大小写与短路径差异。"""
    try:
        return os.path.normcase(os.path.realpath(os.path.abspath(path)))
    except Exception:
        return os.path.normcase(os.path.abspath(path))


_PICKER_SNIPPET = r"""
import json, sys
try:
    import tkinter as tk
    from tkinter import filedialog
except Exception as e:
    print(json.dumps({"ok": False, "gui_available": False,
                      "reason": "tkinter 不可用：%s" % e}, ensure_ascii=False))
    sys.exit(0)
try:
    root = tk.Tk()
    root.withdraw()
    try:
        root.attributes("-topmost", True)
    except Exception:
        pass
    path = filedialog.askdirectory(title="选择要扫描的照片文件夹")
    root.destroy()
except Exception as e:
    print(json.dumps({"ok": False, "gui_available": False,
                      "reason": "无法弹出文件夹选择框：%s" % e},
                     ensure_ascii=False))
    sys.exit(0)
if not path:
    print(json.dumps({"ok": False, "cancelled": True, "gui_available": True,
                      "reason": "已取消选择"}, ensure_ascii=False))
else:
    print(json.dumps({"ok": True, "gui_available": True, "path": path},
                     ensure_ascii=False))
"""


def _picker_command():
    """P1-1：绝不在 HTTP 子线程里直接建 Tk，改为独立子进程。"""
    if getattr(sys, "frozen", False):
        return [sys.executable, "--pick-folder"]
    if os.path.isfile(APP_NATIVE_PY):
        return [sys.executable, APP_NATIVE_PY, "--pick-folder"]
    return [sys.executable, "-c", _PICKER_SNIPPET]


def pick_folder():
    """弹系统文件夹选择框，带超时；失败时前端平滑降级为手输路径。"""
    if not PICK_LOCK.acquire(blocking=False):
        return {"ok": False, "gui_available": True, "busy": True,
                "reason": "已经有一个选择窗口在等你了，请先处理它"}
    try:
        env = dict(os.environ)
        env.setdefault("PYTHONUTF8", "1")
        kwargs = {}
        if os.name == "nt":
            # 不额外弹一个黑色控制台窗口。
            kwargs["creationflags"] = getattr(subprocess,
                                              "CREATE_NO_WINDOW", 0x08000000)
        try:
            proc = subprocess.run(
                _picker_command(), cwd=TOOL_DIR, env=env, timeout=PICK_TIMEOUT,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                encoding="utf-8", errors="replace", **kwargs)
        except subprocess.TimeoutExpired:
            return {"ok": False, "gui_available": True, "timeout": True,
                    "reason": "选择文件夹超时（%d 分钟），已放弃"
                              % (PICK_TIMEOUT // 60)}
        except Exception as e:
            return {"ok": False, "gui_available": False,
                    "reason": "无法启动选择窗口：%s" % e}
        for line in reversed((proc.stdout or "").splitlines()):
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                data = json.loads(line)
            except Exception:
                continue
            if isinstance(data, dict):
                return data
        detail = (proc.stderr or "").strip().splitlines()
        return {"ok": False, "gui_available": False,
                "reason": "选择窗口没有返回结果：%s"
                          % (detail[-1] if detail else "未知原因")}
    finally:
        PICK_LOCK.release()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass  # 本地工具保持安静，只在启动/停止时打印

    # ---- helpers ----
    def send_json(self, obj, status=200):
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def read_json(self):
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            return None
        if length <= 0 or length > MAX_BODY:
            return None
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception:
            return None

    def _query_token(self, parsed):
        qs = parse_qs(parsed.query)
        return (qs.get("s") or [""])[0]

    # ---- GET ----
    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        if path in ("/", "/index.html"):
            try:
                with open(INDEX_HTML, "rb") as f:
                    data = f.read()
            except FileNotFoundError:
                self.send_json({"error": "index.html 缺失"}, 500)
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)
            return
        if path == "/api/health":
            body = {"ok": True, "version": VERSION}
            body.update(data_dir_info())
            self.send_json(body)
            return
        if path == "/api/ping":
            token = self._query_token(parsed)
            note_client_ping(token)
            self.send_json({"ok": True, "shutdown": SHUTDOWN_EVENT.is_set()})
            return
        if path == "/api/faces":
            try:
                import face_clustering
                report = face_clustering.attempt_report()
            except Exception as e:
                report = {"detail": "face_clustering 模块不可用：%s" % e}
            with STATE_LOCK:
                if STATE["result"]:
                    report["scan_face_method"] = STATE["result"]["face_method"]
                    report["scan_face_note"] = STATE["result"]["face_note"]
            self.send_json(report)
            return
        if path == "/api/pick_folder":
            self.send_json(pick_folder())
            return
        if path == "/api/review/list":
            self._handle_review_list()
            return
        if path == "/api/review/history":
            self._handle_review_history()
            return
        if path == "/api/review/crop":
            self._handle_review_crop(parsed)
            return
        if path == "/api/semantic/status":
            self._handle_semantic_status(parsed)
            return
        if path == "/api/index_progress":
            try:
                import semantic_search
                body = semantic_search.progress()
                with STATE_LOCK:
                    result = STATE["result"]
                body["coverage"] = (semantic_search.scan_coverage(result)
                                    if result else {"total": 0, "indexed": 0})
                self.send_json(body)
            except Exception as e:
                self.send_json({"error": "读取索引进度失败：%s" % e}, 500)
            return
        if path == "/api/settings":
            try:
                import api_rerank
                self.send_json(api_rerank.public_settings())
            except Exception as e:
                self.send_json({"error": "读取设置失败：%s" % e}, 500)
            return
        if path == "/api/file":
            qs = parse_qs(parsed.query)
            rel = (qs.get("rel") or [None])[0]
            with STATE_LOCK:
                root = STATE["root"]
            if not root:
                self.send_json({"error": "请先扫描一个文件夹"}, 400)
                return
            full = resolve_in_root(root, rel)
            if full is None or not os.path.isfile(full):
                self.send_json({"error": "文件不存在或不在本次扫描范围内"}, 404)
                return
            if os.path.splitext(full)[1].lower() not in IMAGE_EXTS:
                self.send_json({"error": "不是支持的图片格式"}, 400)
                return
            try:
                with open(full, "rb") as f:
                    data = f.read()
            except OSError as e:
                self.send_json({"error": "读取失败：%s" % e}, 500)
                return
            ctype, _ = mimetypes.guess_type(full)
            self.send_response(200)
            self.send_header("Content-Type", ctype or "application/octet-stream")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)
            return
        if path == "/api/thumb":
            qs = parse_qs(parsed.query)
            rel = (qs.get("rel") or [None])[0]
            with STATE_LOCK:
                root = STATE["root"]
            if not root:
                self.send_json({"error": "请先扫描一个文件夹"}, 400)
                return
            full = resolve_in_root(root, rel)
            if full is None or not os.path.isfile(full):
                self.send_json({"error": "文件不存在或不在本次扫描范围内"}, 404)
                return
            if os.path.splitext(full)[1].lower() not in IMAGE_EXTS:
                self.send_json({"error": "不是支持的图片格式"}, 400)
                return
            try:
                import io as _io
                from PIL import Image
                with Image.open(full) as im:
                    thumb = im.convert("RGB").copy()
                    thumb.thumbnail((200, 200), Image.LANCZOS)
                    buf = _io.BytesIO()
                    thumb.save(buf, "JPEG", quality=70)
                    data = buf.getvalue()
            except Exception as e:
                self.send_json({"error": "缩略图生成失败：%s" % e}, 500)
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)
            return
        self.send_json({"error": "未知接口"}, 404)

    # ---- POST ----
    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/api/scan":
            body = self.read_json()
            if body is None:
                self.send_json({"error": "请求体不是合法 JSON"}, 400)
                return
            if body.get("demo"):
                root = DEMO_DIR
                manifest = DEMO_MANIFEST if os.path.exists(DEMO_MANIFEST) else None
            else:
                root = body.get("path")
                if not root or not isinstance(root, str):
                    self.send_json({"error": "请提供文件夹路径"}, 400)
                    return
                root = os.path.abspath(os.path.expanduser(root))
                if not os.path.isdir(root):
                    self.send_json({"error": "文件夹不存在：%s" % root}, 400)
                    return
                manifest = find_manifest(root)
            # P1-1：全局扫描锁，正在扫描时明确返回 409，不并发跑多轮。
            if not SCAN_LOCK.acquire(blocking=False):
                self.send_json({"error": "正在扫描中，请等当前扫描完成后再试",
                                "busy": True}, 409)
                return
            try:
                started = time.monotonic()
                import scan as scan_mod
                import face_review
                store_path, cache_path = review_paths()
                store = face_review.load_store(store_path)
                vectors = {}
                result = scan_mod.scan(
                    root, 85, manifest,
                    face_cache_path=cache_path,
                    ignore_keys=face_review.ignored_keys(store),
                    demo=bool(body.get("demo")),
                    vectors_out=vectors)
                result["scan_seconds"] = round(time.monotonic() - started, 2)
                result["tool_version"] = VERSION
                face_review.sync_scan(store, result.get("face_faces") or [])
                face_review.save_store(store_path, store)
            except Exception as e:
                self.send_json({"error": "扫描失败：%s" % e}, 500)
                return
            finally:
                SCAN_LOCK.release()
            with STATE_LOCK:
                STATE["root"] = root
                STATE["result"] = result
                STATE["review_vectors"] = vectors
            _CROP_CACHE.clear()  # 重扫后脸框可能变了，裁剪缓存整批作废
            self.send_json(result)
            return
        if path == "/api/delete":
            self._handle_delete()
            return
        if path == "/api/review/act":
            self._handle_review_act()
            return
        if path == "/api/review/undo":
            self._handle_review_undo()
            return
        if path == "/api/review/names":
            self._handle_review_names()
            return
        if path == "/api/review/migrate_names":
            self._handle_review_migrate_names()
            return
        if path in ("/api/index/build", "/api/index/pause", "/api/index/cancel"):
            self._handle_index(path)
            return
        if path == "/api/semantic/search":
            self._handle_semantic_search()
            return
        if path == "/api/settings":
            self._handle_settings()
            return
        if path == "/api/semantic/privacy_ack":
            try:
                import api_rerank
                self.read_json()
                self.send_json(api_rerank.update_settings(
                    {"privacy_ack": True}))
            except Exception as e:
                self.send_json({"error": "记录隐私确认失败：%s" % e}, 500)
            return
        if path == "/api/client_closed":
            note_client_closed(self._query_token(parsed))
            self.send_json({"ok": True})
            return
        if path == "/api/shutdown":
            self.send_json({"ok": True, "message": "服务正在停止"})
            SHUTDOWN_EVENT.set()
            threading.Thread(target=self.server.shutdown, daemon=True).start()
            return
        self.send_json({"error": "未知接口"}, 404)

    # ---- 3.3.0 语义搜图 ----
    def _scan_matches(self, source_dir):
        with STATE_LOCK:
            root, result = STATE["root"], STATE["result"]
        if not root or not result:
            return None, None, "请先扫描一个文件夹"
        if source_dir and _norm_scope(source_dir) != _norm_scope(root):
            return None, None, "当前扫描的是另一个文件夹，请先重新扫描再试"
        return root, result, None

    def _handle_semantic_status(self, parsed):
        try:
            import semantic_search
        except Exception as e:
            self.send_json({"error": "语义搜图模块不可用：%s" % e}, 500)
            return
        qs = parse_qs(parsed.query)
        source_dir = (qs.get("source_dir") or [""])[0]
        try:
            body = {
                "ok": True,
                "model": semantic_search.model_status(),
                "index": semantic_search.index_info(),
                "progress": semantic_search.progress(),
                "bands": semantic_search.BANDS,
                "band_labels": semantic_search.BAND_LABELS,
                "calibration": {
                    "cos_floor": semantic_search.CALIB_FLOOR,
                    "cos_ceil": semantic_search.CALIB_CEIL,
                    "note": "线性映射 (cos-%.2f)/(%.2f-%.2f)，三档阈值 "
                            "宽松/标准/严格 = %s；依据 25 张演示合成图的实测"
                            "分布校准，真实照片需真机复测。"
                            % (semantic_search.CALIB_FLOOR,
                               semantic_search.CALIB_CEIL,
                               semantic_search.CALIB_FLOOR,
                               [semantic_search.BANDS[k] for k in
                                ("loose", "standard", "strict")]),
                },
            }
            with STATE_LOCK:
                result = STATE["result"]
                root = STATE["root"]
            if result and (not source_dir
                           or _norm_scope(source_dir) == _norm_scope(root)):
                body["coverage"] = semantic_search.scan_coverage(result)
            else:
                body["coverage"] = {"total": 0, "indexed": 0}
                body["scan_mismatch"] = bool(result and source_dir)
            self.send_json(body)
        except Exception as e:
            self.send_json({"error": "读取语义索引状态失败：%s" % e}, 500)

    def _handle_index(self, path):
        try:
            import semantic_search
        except Exception as e:
            self.send_json({"error": "语义搜图模块不可用：%s" % e}, 500)
            return
        if path == "/api/index/pause":
            self.send_json(semantic_search.pause_index())
            return
        if path == "/api/index/cancel":
            self.send_json(semantic_search.cancel_index())
            return
        body = self.read_json()
        if body is None:
            self.send_json({"error": "请求体不是合法 JSON"}, 400)
            return
        root, result, err = self._scan_matches(body.get("source_dir"))
        if err:
            self.send_json({"error": err}, 409)
            return
        try:
            state = semantic_search.start_index(result, root)
            state["coverage"] = semantic_search.scan_coverage(result)
            self.send_json(state)
        except Exception as e:
            self.send_json({"error": "启动索引失败：%s" % e}, 500)

    def _handle_semantic_search(self):
        body = self.read_json()
        if body is None:
            self.send_json({"error": "请求体不是合法 JSON"}, 400)
            return
        query = str(body.get("query") or "").strip()
        if not query:
            self.send_json({"error": "搜索词为空"}, 400)
            return
        if len(query) > 80:
            self.send_json({"error": "搜索词太长（最多 80 字）"}, 400)
            return
        band = body.get("band") or "standard"
        try:
            import semantic_search
        except Exception as e:
            self.send_json({"error": "语义搜图模块不可用：%s" % e}, 500)
            return
        if band not in semantic_search.BANDS:
            self.send_json({"error": "未知的置信度档位：%s" % band}, 400)
            return
        root, result, err = self._scan_matches(body.get("source_dir"))
        if err:
            self.send_json({"error": err}, 409)
            return
        try:
            top_k = int(body.get("top_k") or semantic_search.MAX_RESULTS)
        except (TypeError, ValueError):
            top_k = semantic_search.MAX_RESULTS
        top_k = max(1, min(semantic_search.MAX_RESULTS, top_k))
        try:
            import api_rerank
            api_opts = {"provider": api_rerank.get_settings()["provider"]}
        except Exception:
            api_opts = {}
        try:
            out = semantic_search.search(
                result, root, query, band=band, top_k=top_k,
                use_api=bool(body.get("use_api")), api_opts=api_opts)
            out["source_dir"] = os.path.abspath(root)
            self.send_json(out)
        except Exception as e:
            self.send_json({"error": "语义搜索失败：%s" % e}, 500)

    def _handle_settings(self):
        body = self.read_json()
        if body is None:
            self.send_json({"error": "请求体不是合法 JSON"}, 400)
            return
        try:
            import api_rerank
        except Exception as e:
            self.send_json({"error": "API 精排模块不可用：%s" % e}, 500)
            return
        try:
            if body.get("clear_key"):
                api_rerank.clear_key()
            api_key = body.get("api_key")
            if isinstance(api_key, str) and api_key.strip():
                api_rerank.set_key(api_key)
            patch = {k: body[k] for k in
                     ("provider", "base_url", "model", "privacy_ack")
                     if k in body}
            if patch:
                api_rerank.update_settings(patch)
            self.send_json(api_rerank.public_settings())
        except ValueError as e:
            self.send_json({"error": str(e)}, 400)
        except Exception as e:
            self.send_json({"error": "保存设置失败：%s" % e}, 500)

    # ---- 3.4.0 复核表 ----
    def _review_store(self):
        import face_review
        return face_review

    def _handle_review_list(self):
        try:
            face_review = self._review_store()
        except Exception as e:
            self.send_json({"error": "复核模块不可用：%s" % e}, 500)
            return
        store, _path = load_review_store(face_review)
        body = review_payload(face_review, store)
        if not body.get("ok"):
            self.send_json({"error": body.get("error") or "请先扫描一个文件夹"},
                           400)
            return
        self.send_json(body)

    def _handle_review_history(self):
        try:
            face_review = self._review_store()
        except Exception as e:
            self.send_json({"error": "复核模块不可用：%s" % e}, 500)
            return
        store, _path = load_review_store(face_review)
        self.send_json({"ok": True, "history": face_review.history_view(store),
                        "migration": store.get("migration") or {}})

    def _handle_review_crop(self, parsed):
        qs = parse_qs(parsed.query)
        key = (qs.get("k") or [""])[0]
        try:
            size = int((qs.get("size") or ["168"])[0])
        except (TypeError, ValueError):
            size = 168
        size = max(48, min(400, size))
        with STATE_LOCK:
            result, root = STATE["result"], STATE["root"]
        if not result or not root:
            self.send_json({"error": "请先扫描一个文件夹"}, 400)
            return
        face = find_face(result, key)
        if face is None:
            self.send_json({"error": "找不到这张脸（可能已重扫）"}, 404)
            return
        cache_key = (key, size)
        hit = _CROP_CACHE.get(cache_key)
        if hit is not None:
            data, _meta = hit
        else:
            import face_review
            full = resolve_in_root(root, face.get("relative_path"))
            if full is None or not os.path.isfile(full):
                self.send_json({"error": "原图不存在或不在本次扫描范围内"}, 404)
                return
            try:
                data, meta = face_review.crop_face(full, face.get("box"),
                                                   size=size)
            except Exception as e:
                self.send_json({"error": "裁剪失败：%s" % e}, 500)
                return
            if len(_CROP_CACHE) > _CROP_CACHE_MAX:
                _CROP_CACHE.clear()
            _CROP_CACHE[cache_key] = (data, meta)
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "private, max-age=120")
        self.end_headers()
        self.wfile.write(data)

    def _review_common(self):
        """act/undo/names 的公共前置：取 store、当前扫描结果与向量。"""
        import face_review
        store, store_path = load_review_store(face_review)
        with STATE_LOCK:
            result = STATE["result"]
            vectors = dict(STATE["review_vectors"])
        if not result:
            return None, None, None, None, "请先扫描一个文件夹"
        face_review.sync_scan(store, result.get("face_faces") or [])
        return face_review, store, store_path, (result, vectors), None

    def _handle_review_act(self):
        body = self.read_json()
        if body is None:
            self.send_json({"error": "请求体不是合法 JSON"}, 400)
            return
        action = body.get("action")
        keys = body.get("keys")
        if not isinstance(keys, list):
            keys = [body.get("key")] if body.get("key") else []
        with REVIEW_LOCK:
            face_review, store, store_path, payload, err = self._review_common()
            if err:
                self.send_json({"error": err}, 400)
                return
            result, vectors = payload
            try:
                out = face_review.apply_action(
                    store, action, keys, result, vectors,
                    person_id=body.get("person_id"), name=body.get("name"))
            except ValueError as e:
                self.send_json({"error": str(e)}, 400)
                return
            except Exception as e:
                self.send_json({"error": "复核操作失败：%s" % e}, 500)
                return
            if _touches_ignored((out.get("txn") or {}).get("items")):
                _refresh_review_clusters(store)
            face_review.save_store(store_path, store)
            body_out = review_payload(face_review, store, {
                "txn": out.get("txn"),
                "created_persons": out.get("created_persons") or [],
                "unresolved": out.get("unresolved") or [],
            })
        self.send_json(body_out)

    def _handle_review_undo(self):
        body = self.read_json()
        if body is None:
            self.send_json({"error": "请求体不是合法 JSON"}, 400)
            return
        txn_id = body.get("txn_id")
        if not txn_id:
            self.send_json({"error": "缺少 txn_id"}, 400)
            return
        keys = body.get("keys")
        only_keys = keys if isinstance(keys, list) and keys else None
        with REVIEW_LOCK:
            face_review, store, store_path, payload, err = self._review_common()
            if err:
                self.send_json({"error": err}, 400)
                return
            try:
                out = face_review.undo(store, txn_id, only_keys=only_keys,
                                       force=bool(body.get("force")))
            except ValueError as e:
                self.send_json({"error": str(e)}, 400)
                return
            except Exception as e:
                self.send_json({"error": "撤销失败：%s" % e}, 500)
                return
            if out.get("conflict"):
                # 不静默覆盖：把冲突清单交回前端，让用户先处理最新记录。
                self.send_json(out, 409)
                return
            txn = None
            for item in store.get("history") or []:
                if item.get("txn_id") == txn_id:
                    txn = item
                    break
            if txn is not None and _touches_ignored(txn.get("items")):
                _refresh_review_clusters(store)
            face_review.save_store(store_path, store)
            body_out = review_payload(face_review, store, {"undo": out})
        self.send_json(body_out)

    def _handle_review_names(self):
        body = self.read_json()
        if body is None:
            self.send_json({"error": "请求体不是合法 JSON"}, 400)
            return
        person_id = body.get("person_id")
        if not person_id:
            self.send_json({"error": "缺少 person_id"}, 400)
            return
        with REVIEW_LOCK:
            face_review, store, store_path, payload, err = self._review_common()
            if err:
                self.send_json({"error": err}, 400)
                return
            result, vectors = payload
            try:
                out = face_review.apply_action(
                    store, "rename", [], result, vectors,
                    person_id=person_id, name=body.get("name"))
            except ValueError as e:
                self.send_json({"error": str(e)}, 400)
                return
            face_review.save_store(store_path, store)
            body_out = review_payload(face_review, store,
                                      {"txn": out.get("txn")})
        self.send_json(body_out)

    def _handle_review_migrate_names(self):
        body = self.read_json()
        if body is None:
            self.send_json({"error": "请求体不是合法 JSON"}, 400)
            return
        names = body.get("names")
        if not isinstance(names, dict):
            self.send_json({"error": "names 必须是对象"}, 400)
            return
        with REVIEW_LOCK:
            # 人名迁移不依赖当前扫描：哪怕还没扫描/扫描刚被另一个请求占用，
            # 也要能把浏览器里的旧名安全搬进本机数据文件，不能丢。
            face_review = self._review_store()
            store, store_path = load_review_store(face_review)
            with STATE_LOCK:
                result = STATE["result"] or {}
            counts = {}
            for cluster in result.get("face_clusters") or []:
                counts[cluster["cluster_id"]] = cluster.get("size") or 0
            for group in result.get("face_groups") or []:
                counts.setdefault(group["id"], len(group.get("members") or []))
            try:
                out = face_review.migrate_names(store, names, counts)
            except Exception as e:
                self.send_json({"error": "迁移失败：%s" % e}, 500)
                return
            face_review.save_store(store_path, store)
            body_out = review_payload(face_review, store, {"migration_result": out})
        self.send_json(body_out)

    def _handle_delete(self):
        body = self.read_json()
        if body is None:
            self.send_json({"error": "请求体不是合法 JSON"}, 400)
            return
        if body.get("confirmed") is not True:
            self.send_json({"error": "需要二次确认（confirmed 必须为 true）"},
                           400)
            return

        items = body.get("items")
        if items is None:
            ids = body.get("ids")
            if not isinstance(ids, list) or not ids:
                self.send_json({"error": "ids 为空"}, 400)
                return
            # 旧请求没有 source_dir：不给它们“按文件名猜文件夹”的机会。
            items = [{"id": i, "source_dir": None} for i in ids]
        if not isinstance(items, list) or not items:
            self.send_json({"error": "items 为空"}, 400)
            return

        with STATE_LOCK:
            root = STATE["root"]
            result = STATE["result"]
        if not root or not result:
            self.send_json({"error": "请先扫描一个文件夹"}, 400)
            return

        known = {im["id"] for im in result["images"] if "error" not in im}
        root_key = _norm_scope(root)
        results = [None] * len(items)
        jobs = []  # (index, rel, full)
        for idx, item in enumerate(items):
            if not isinstance(item, dict):
                results[idx] = {"ok": False, "error": "请求条目格式错误"}
                continue
            rel = item.get("id")
            src = item.get("source_dir")
            if not isinstance(rel, str) or not rel:
                results[idx] = {"ok": False, "error": "缺少文件 id"}
                continue
            if rel not in known:
                results[idx] = {"ok": False, "id": rel,
                                "error": "不在本次扫描结果内，拒绝删除"}
                continue
            if not src or not isinstance(src, str):
                results[idx] = {
                    "ok": False, "id": rel,
                    "error": "旧版待删篓条目未绑定文件夹，已拒绝删除；"
                             "请从 PK 台重新加入待删篓"}
                continue
            if _norm_scope(src) != root_key:
                results[idx] = {
                    "ok": False, "id": rel, "source_dir": src,
                    "error": "该条目属于其他文件夹（%s），换文件夹后拒绝"
                             "删除当前文件夹里的同名文件" % src}
                continue
            full = resolve_in_root(root, rel)
            if full is None or not os.path.isfile(full):
                results[idx] = {"ok": False, "id": rel,
                                "error": "文件不存在或不在本次扫描范围内"}
                continue
            jobs.append((idx, rel, full))

        if jobs:
            try:
                import trash_util
            except Exception as e:
                self.send_json({"error": "删除模块不可用：%s" % e}, 500)
                return
            try:
                outs = trash_util.delete_many([j[2] for j in jobs])
            except Exception as e:
                outs = [{"ok": False, "error": "删除失败：%s" % e}
                        for _ in jobs]
            for (idx, rel, _full), out in zip(jobs, outs):
                out["id"] = rel
                out["source_dir"] = items[idx].get("source_dir")
                results[idx] = out

        # 3.3.0：删除成功后同步剔除语义向量，避免已删图继续出现在搜图结果里。
        deleted = [r.get("id") for r in results if r and r.get("ok")]
        if deleted:
            try:
                import semantic_search
                by_id = {im["id"]: im.get("sha256")
                         for im in result["images"] if "error" not in im}
                deleted_shas = {by_id.get(i) for i in deleted} - {None}
                gone = set(deleted)
                remaining = {im.get("sha256") for im in result["images"]
                             if "error" not in im and im["id"] not in gone}
                removed = semantic_search.forget_images(deleted_shas,
                                                         remaining)
                if removed:
                    for r in results:
                        if r:
                            r["semantic_vectors_removed"] = removed
            except Exception:
                # 索引剔除失败不影响删除结果（下次搜索会按 sha 集合自动清理）
                pass

        self.send_json({"results": results})


def pick_port(preferred):
    if preferred:
        return preferred
    for p in range(8765, 8775):
        s = socket.socket()
        try:
            s.bind(("127.0.0.1", p))
            s.close()
            return p
        except OSError:
            continue
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Shiying-3.4.0 本地服务（仅监听 127.0.0.1）")
    ap.add_argument("--port", type=int, default=0,
                    help="指定端口（默认自动在 8765~8774 间选空闲端口）")
    ap.add_argument("--no-open", action="store_true",
                    help="不自动打开浏览器（供自测用）")
    args = ap.parse_args(argv)

    # P1-4：独立运行 server.py 时也遵守单实例，不和 app_native 抢端口。
    lock = None
    lock_path = None
    if app_paths is not None:
        from instance_lock import InstanceLock, read_info
        lock_path = app_paths.writable_path("instance.lock")
        lock = InstanceLock(lock_path)
        if not lock.acquire():
            info = read_info(lock_path)
            print("已有实例在运行：%s" % (info.get("url") or
                                          "（拿不到实例地址）"))
            print("请直接使用已打开的窗口；本进程不再开新端口。")
            return 2

    port = pick_port(args.port)
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    actual = server.server_address[1]
    url = "http://127.0.0.1:%d/" % actual
    if lock_path is not None:
        from instance_lock import write_info
        write_info(lock_path, {"pid": os.getpid(), "url": url,
                               "version": VERSION})
    print("Shiying-%s 本地服务已启动：%s" % (VERSION, url))
    print("只监听本机（127.0.0.1），不对局域网/外网开放；"
          "联网仅限你主动触发的模型下载与可选 API 精排。")
    info = data_dir_info()
    if info["data_dir_fallback"]:
        print("注意：程序目录不可写，日志与隔离区已回退到 %s"
              % info["data_dir"])
    print("关闭方式：页面点“停止服务”，或本窗口按 Ctrl+C。")
    if not args.no_open:
        try:
            import webbrowser
            webbrowser.open(url)
        except Exception as e:
            print("自动打开浏览器失败（请手动打开上面的地址）：%s" % e)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        if lock_path is not None:
            from instance_lock import clear_info
            clear_info(lock_path)
        if lock is not None:
            lock.release()
        print("服务已停止，无后台残留。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
