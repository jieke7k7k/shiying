#!/usr/bin/env python3
"""Shiying-3.2.0 本地服务：把扫描与界面串起来，纯本地运行。

  - 只监听 127.0.0.1（本机回环），不对局域网/外网开放。
  - 页面、样式、脚本全部内联在 index.html 里，零外部请求
    （不加载任何 CDN、字体、统计脚本）。
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
      POST /api/delete       -> {"items": [{"id", "source_dir"}],
                                 "confirmed": true}
                                 （必须二次确认；只走回收站或隔离区，
                                 见 trash_util.py；绝不直接删除）
      POST /api/client_closed -> 窗口关闭信号（pagehide 时 sendBeacon）
      POST /api/shutdown     -> 干净停止服务（页面上的“停止服务”按钮用）
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

VERSION = "3.2.0"

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

STATE = {"root": None, "result": None}
STATE_LOCK = threading.Lock()
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
                result = scan_mod.scan(root, 85, manifest)
                result["scan_seconds"] = round(time.monotonic() - started, 2)
                result["tool_version"] = VERSION
            except Exception as e:
                self.send_json({"error": "扫描失败：%s" % e}, 500)
                return
            finally:
                SCAN_LOCK.release()
            with STATE_LOCK:
                STATE["root"] = root
                STATE["result"] = result
            self.send_json(result)
            return
        if path == "/api/delete":
            self._handle_delete()
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
        description="Shiying-3.2.0 本地服务（仅监听 127.0.0.1）")
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
    print("只监听本机（127.0.0.1），不对局域网/外网开放，零外部请求。")
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
