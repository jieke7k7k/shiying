#!/usr/bin/env python3
"""Shiying-3.2.0 原生窗口启动入口：像桌面软件一样打开，不显示地址栏。

启动顺序：
  1) 首选 pywebview（Windows 上即系统 WebView2）创建原生窗口；
  2) pywebview 不可用时，回退到 Chrome / Edge 的 --app 应用窗口模式
     （--app 没有地址栏、没有标签栏，看起来就是独立窗口）；
  3) 两者都不可用时明确提示安装方式并退出，绝不退化成带地址栏的普通浏览器。

3.2.0 生命周期（P0-2 / P0-3）：
  - 服务跑在本进程的后台线程，只监听 127.0.0.1；
  - --app 回退不再用 proc.wait() 直接判断窗口关闭：
      * 使用可写目录下持久化的浏览器配置目录 browser_profile/，
        不每次新建临时目录再删除——否则 localStorage 里的人物改名
        和待删篓会被清掉；
      * 页面带一次性 session 参数，每 5 秒 /api/ping 心跳，关闭窗口时
        sendBeacon /api/client_closed；主线程用「进程退出 + 心跳 + 关闭
        信号」三条线索轮询判断，浏览器把请求转交给已在运行的实例而
        自身秒退时，不会误把服务一起关掉；
      * --disable-background-mode，窗口关掉后浏览器进程跟着退出。
  - 点“停止服务”后：/api/shutdown 置 SHUTDOWN_EVENT，主线程收到后
    销毁 pywebview 窗口或结束自己启动的浏览器进程，再停 HTTP 服务；
    服务停掉后不会留下一个死窗口。
  - P1-4：可写目录下 instance.lock 做单实例锁（fcntl / msvcrt），
    第二个实例只打印“已有实例在运行”并退出，不开新端口、不开新窗口。
  - P1-5：app.log 按 1 MB × 3 个备份轮转；所有计时用 time.monotonic()。
  - P1-3：app.log / 浏览器配置 / 隔离区 / 单实例锁都走 app_paths 的
    可写目录；程序目录不可写时回退到系统用户数据目录并在日志里明说。

用法：
  python3 app_native.py                 # 正常打开原生窗口
  python3 app_native.py --port 8765     # 指定端口
  python3 app_native.py --no-window     # 只起服务（调试/无 GUI 环境）
  python3 app_native.py --self-test     # 起服务、自检接口、再干净退出
  python3 app_native.py --pick-folder   # 只弹文件夹选择框（供服务端子进程调用）
  python3 app_native.py --version
"""
import io
import os
import sys
import threading

# ---- P0：必须在任何可能 print 的代码之前完成 stdout/stderr 兜底 ----


def _real_dir():
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


try:
    import app_paths
except Exception:  # 极端的单文件拷贝场景：退回本文件目录
    class _AppPaths(object):
        @staticmethod
        def app_dir():
            return _real_dir()

        @staticmethod
        def resource_dir():
            return _real_dir()

        @staticmethod
        def writable_dir():
            return _real_dir()

        @staticmethod
        def writable_path(*parts):
            return os.path.join(_real_dir(), *parts)

        @staticmethod
        def writable_info():
            return {"dir": _real_dir(), "fallback": False, "reason": "",
                    "app_dir": _real_dir()}
    app_paths = _AppPaths()


class _RotatingStream(object):
    """app.log 轮转写入器：1 MB 一个文件，保留 3 个备份（P1-5）。

    既是文件包装器，也可以直接顶替 sys.stdout / sys.stderr；轮转后会把
    dup2 过的 fd 重新指向新文件，无控制台模式下重定向不会断。
    """

    def __init__(self, path, max_bytes=1024 * 1024, backups=3):
        self.path = path
        self.max_bytes = max_bytes
        self.backups = backups
        self._lock = threading.Lock()
        self._fds = []
        self._f = None
        self._open()

    def _open(self):
        parent = os.path.dirname(os.path.abspath(self.path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._f = open(self.path, "a", encoding="utf-8", buffering=1)

    def attach_fd(self, fd):
        try:
            os.dup2(self._f.fileno(), fd)
            if fd not in self._fds:
                self._fds.append(fd)
        except Exception:
            pass

    def write(self, text):
        if not isinstance(text, str):
            text = str(text)
        with self._lock:
            try:
                self._f.write(text)
                self._f.flush()
                if self._f.tell() >= self.max_bytes:
                    self._rotate_locked()
            except Exception:
                pass
        return len(text)

    def _rotate_locked(self):
        try:
            self._f.close()
            for idx in range(self.backups - 1, 0, -1):
                src = "%s.%d" % (self.path, idx)
                dst = "%s.%d" % (self.path, idx + 1)
                if os.path.exists(src):
                    os.replace(src, dst)
            if os.path.exists(self.path):
                os.replace(self.path, self.path + ".1")
        except Exception:
            pass
        try:
            self._open()
        except Exception:
            try:
                self._f = open(os.devnull, "w", encoding="utf-8")
            except Exception:
                self._f = io.StringIO()
        for fd in list(self._fds):
            try:
                os.dup2(self._f.fileno(), fd)
            except Exception:
                pass

    def flush(self):
        try:
            self._f.flush()
        except Exception:
            pass

    def isatty(self):
        return False

    def fileno(self):
        return self._f.fileno()


_ORIG_STDOUT = sys.stdout
_ORIG_STDERR = sys.stderr
_LOG_STREAM = None
LOG_PATH = None


def _bootstrap_streams():
    """始终保留 app.log；无控制台时把它接到 stdout/stderr，绝不留 None。"""
    global _LOG_STREAM, LOG_PATH
    try:
        LOG_PATH = app_paths.writable_path("app.log")
        _LOG_STREAM = _RotatingStream(LOG_PATH)
    except Exception:
        _LOG_STREAM = None

    if sys.stdout is not None:
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    if sys.stderr is not None:
        try:
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    if sys.stdout is None:
        sys.stdout = _LOG_STREAM or io.StringIO()
        if _LOG_STREAM is not None:
            _LOG_STREAM.attach_fd(1)
    if sys.stderr is None:
        sys.stderr = _LOG_STREAM or io.StringIO()
        if _LOG_STREAM is not None:
            _LOG_STREAM.attach_fd(2)


_bootstrap_streams()

import argparse
import json
import shutil
import signal
import subprocess
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
import uuid
from http.server import ThreadingHTTPServer


VERSION = "3.2.0"
WINDOW_TITLE = "拾影 Shiying-3.2.0"
_SESSION = uuid.uuid4().hex
_SERVER_MOD = None


def _excepthook(exc_type, exc, tb):
    try:
        traceback.print_exception(exc_type, exc, tb)
        sys.stderr.flush()
    except Exception:
        pass


sys.excepthook = _excepthook


def log(msg):
    line = "[%s] %s\n" % (time.strftime("%Y-%m-%d %H:%M:%S"), msg)
    if _LOG_STREAM is not None:
        _LOG_STREAM.write(line)
    if _ORIG_STDOUT is not None:
        try:
            _ORIG_STDOUT.write(line)
            _ORIG_STDOUT.flush()
        except Exception:
            pass


def _prepare_import_path():
    for path in (app_paths.resource_dir(), app_paths.app_dir(), _real_dir()):
        if path and path not in sys.path:
            sys.path.insert(0, path)


def _start_server(port):
    """在后台线程里启动 server.py 的本地服务，只绑定 127.0.0.1。"""
    global _SERVER_MOD
    _prepare_import_path()
    import server as server_mod
    _SERVER_MOD = server_mod

    bind_port = server_mod.pick_port(port)
    httpd = ThreadingHTTPServer(("127.0.0.1", bind_port), server_mod.Handler)
    httpd.daemon_threads = True
    thread = threading.Thread(target=httpd.serve_forever, name="photo-http",
                              daemon=True)
    thread.start()
    return httpd, "http://127.0.0.1:%d/" % httpd.server_address[1]


def _stop_server(httpd):
    try:
        httpd.shutdown()
    except Exception:
        pass
    try:
        httpd.server_close()
    except Exception:
        pass


def _install_signal_handlers():
    def handler(signum, _frame):
        log("收到退出信号 %s，正在停止服务……" % signum)
        if _SERVER_MOD is not None:
            _SERVER_MOD.SHUTDOWN_EVENT.set()

    for sig in (getattr(signal, "SIGINT", None), getattr(signal, "SIGTERM", None)):
        if sig is None:
            continue
        try:
            signal.signal(sig, handler)
        except Exception:
            pass


def _shutdown_requested():
    return _SERVER_MOD is not None and _SERVER_MOD.SHUTDOWN_EVENT.is_set()


def _client_window_gone(st):
    """心跳/关闭信号判断窗口是否已关闭（用于浏览器进程常驻或被接管时）。"""
    now = time.monotonic()
    if not st.get("seen"):
        return False
    closed_at = st.get("closed_at") or 0.0
    last_seen = st.get("last_seen") or 0.0
    # pagehide 会先到、刷新后的 ping 会把 closed_at 清零；给 4 秒宽限。
    if closed_at and now - closed_at > 4.0 and now - last_seen > 4.0:
        return True
    # 浏览器后台会节流定时器（最慢约 1 次/分钟），所以兜底阈值放到 120 秒，
    # 避免用户把窗口切到后台就被误判成关闭。
    if last_seen and now - last_seen > 120.0:
        return True
    return False


def _run_pywebview(url):
    """原生窗口；SHUTDOWN_EVENT 或窗口关闭时干净退出。"""
    try:
        import webview
    except Exception as e:
        log("pywebview 不可用，尝试 Chrome/Edge 应用窗口：%s" % e)
        return False
    try:
        window = webview.create_window(
            WINDOW_TITLE, url + "?s=" + _SESSION,
            width=1280, height=860, min_size=(960, 640))
    except Exception as e:
        log("pywebview 创建窗口失败，尝试 Chrome/Edge 应用窗口：%s" % e)
        return False

    done = threading.Event()

    def watcher():
        while not done.wait(0.3):
            if _shutdown_requested():
                try:
                    window.destroy()
                except Exception:
                    pass
                return
            if _SERVER_MOD is not None:
                st = _SERVER_MOD.client_state(_SESSION)
                if _client_window_gone(st):
                    try:
                        window.destroy()
                    except Exception:
                        pass
                    return

    threading.Thread(target=watcher, name="photo-window-watch",
                     daemon=True).start()
    started = time.monotonic()
    try:
        webview.start()
    except Exception as e:
        log("pywebview 启动失败，尝试 Chrome/Edge 应用窗口：%s" % e)
        done.set()
        return False
    finally:
        done.set()
    # 瞬间返回且页面从未发过心跳：多半是 WebView2 运行时缺失导致的假启动。
    if time.monotonic() - started < 1.5:
        st = _SERVER_MOD.client_state(_SESSION) if _SERVER_MOD else {}
        if not st.get("seen"):
            log("pywebview 窗口瞬间退出且没有页面心跳，改试 Chrome/Edge。")
            return False
    return True


def _browser_candidates():
    cands = []
    if sys.platform.startswith("win"):
        roots = [os.environ.get("PROGRAMFILES"),
                 os.environ.get("PROGRAMFILES(X86)"),
                 os.environ.get("LOCALAPPDATA")]
        rels = (("Google", "Chrome", "Application", "chrome.exe"),
                ("Microsoft", "Edge", "Application", "msedge.exe"))
        for root in roots:
            if not root:
                continue
            for rel in rels:
                cands.append(os.path.join(root, *rel))
        for name in ("chrome.exe", "msedge.exe"):
            found = shutil.which(name)
            if found:
                cands.append(found)
    elif sys.platform == "darwin":
        cands.extend([
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
        ])
    else:
        for name in ("google-chrome", "google-chrome-stable", "chromium",
                     "chromium-browser", "microsoft-edge",
                     "microsoft-edge-stable"):
            found = shutil.which(name)
            if found:
                cands.append(found)

    out = []
    seen = set()
    for path in cands:
        if not path or not os.path.isfile(path):
            continue
        key = os.path.normcase(os.path.abspath(path))
        if key not in seen:
            seen.add(key)
            out.append(path)
    return out


def _browser_profile_dir(exe):
    """持久化浏览器配置目录（P0-2）：localStorage 的改名与待删篓靠它留存。

    Chrome 与 Edge 的配置格式不通用，按浏览器分目录，但同一浏览器
    始终复用同一个目录（不会每次新建临时目录后删除）。
    """
    low = os.path.basename(exe or "").lower()
    if "edge" in low or "msedge" in low:
        family = "edge"
    elif "chrom" in low:
        family = "chrome"
    else:
        family = "browser"
    path = app_paths.writable_path("browser_profile", family)
    os.makedirs(path, exist_ok=True)
    return path


def _terminate(proc):
    if proc is None:
        return
    if proc.poll() is None:
        try:
            proc.terminate()
        except Exception:
            pass
        try:
            proc.wait(timeout=5)
        except Exception:
            try:
                proc.kill()
                proc.wait(timeout=3)
            except Exception:
                pass


def _monitor_browser(proc):
    """轮询窗口状态，返回 closed / shutdown / no-client。

    三条线索：自己启动的进程是否退出、页面心跳、窗口 pagehide 信号。
    浏览器把启动请求转交给已在运行的实例时，进程会秒退（<5 秒），
    这时不能当成“窗口关了”，要继续等心跳与关闭信号。
    """
    started = time.monotonic()
    seen_at = None
    exit_at = None
    while True:
        if _shutdown_requested():
            return "shutdown"
        st = _SERVER_MOD.client_state(_SESSION)
        now = time.monotonic()
        if st.get("seen") and seen_at is None:
            seen_at = now
        if proc.poll() is not None and exit_at is None:
            exit_at = now

        if seen_at is None:
            # 30 秒内没有任何页面心跳：窗口没起来（或地址打不开）。
            if now - started > 30:
                return "no-client"
            if exit_at is not None and now - exit_at > 12:
                return "no-client"
        else:
            # 进程正常活了 5 秒以上才退出，基本可判定是窗口真的关了。
            if exit_at is not None and (exit_at - started) > 5:
                return "closed"
            # 秒退（被已有实例接管）或进程常驻时，改看心跳/关闭信号。
            if _client_window_gone(st):
                return "closed"
        time.sleep(0.25)


def _run_app_browser(url):
    """Chrome/Edge --app 回退；返回 True 表示窗口跑过并已结束。"""
    candidates = _browser_candidates()
    if not candidates:
        return False
    app_url = url + "?s=" + _SESSION
    for exe in candidates:
        profile = _browser_profile_dir(exe)
        args = [
            exe,
            "--app=" + app_url,
            "--window-size=1280,860",
            "--user-data-dir=" + profile,
            "--no-first-run",
            "--no-default-browser-check",
            "--disable-background-mode",
            "--disable-session-crashed-bubble",
        ]
        try:
            log("尝试应用窗口模式：%s（配置目录：%s）" % (exe, profile))
            proc = subprocess.Popen(args, cwd=_real_dir(),
                                    stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL)
        except Exception as e:
            log("应用窗口启动异常：%s，换下一个候选。" % e)
            continue
        try:
            outcome = _monitor_browser(proc)
        except BaseException:
            # Ctrl+C / 异常退出时也要收掉自己启动的浏览器进程。
            _terminate(proc)
            raise
        _terminate(proc)
        if outcome in ("closed", "shutdown"):
            return True
        log("应用窗口没有等到页面心跳（%s），换下一个候选。" % outcome)
    return False


def _http_get(url, timeout=20):
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.status, resp.read()


def _http_post_json(url, payload, timeout=180):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))


def _emit_picker_result(obj):
    """把选择结果写到真正的 stdout。

    打包成窗口程序后 sys.stdout 可能是 None（被日志流顶替），但父进程
    server.py 用 subprocess 捕获的是 fd 1 管道，所以这里直接写 fd 1。
    """
    line = json.dumps(obj, ensure_ascii=False) + "\n"
    try:
        os.write(1, line.encode("utf-8"))
        return
    except Exception:
        pass
    try:
        print(line, end="")
        sys.stdout.flush()
    except Exception:
        pass


def _pick_folder_cli():
    """子进程对话框：只弹框、输出一行 JSON、立刻退出。"""
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as e:
        _emit_picker_result({"ok": False, "gui_available": False,
                             "reason": "tkinter 不可用：%s" % e})
        return 0
    root = None
    try:
        root = tk.Tk()
        root.withdraw()
        try:
            root.attributes("-topmost", True)
        except Exception:
            pass
        path = filedialog.askdirectory(title="选择要扫描的照片文件夹")
    except Exception as e:
        _emit_picker_result({"ok": False, "gui_available": False,
                             "reason": "无法弹出文件夹选择框（无显示环境）：%s"
                                       % e})
        return 0
    finally:
        try:
            if root is not None:
                root.destroy()
        except Exception:
            pass
    if not path:
        _emit_picker_result({"ok": False, "cancelled": True,
                             "gui_available": True, "reason": "已取消选择"})
    else:
        _emit_picker_result({"ok": True, "gui_available": True, "path": path})
    return 0


def _self_test(port):
    """不打开窗口，起服务 -> 校验页面/接口/演示扫描/二次确认 -> 干净退出。"""
    httpd, url = _start_server(port)
    try:
        status, html = _http_get(url)
        ok_html = (status == 200 and b"<title>" in html
                   and b'id="gallery"' in html)

        status, body = _http_get(url + "api/health")
        health = json.loads(body.decode("utf-8"))
        ok_health = (health.get("ok") is True
                     and health.get("version") == VERSION
                     and bool(health.get("data_dir")))

        status, ping = _http_get(url + "api/ping?s=selftest")
        ok_ping = status == 200 and json.loads(ping.decode("utf-8")).get("ok")

        status, scan = _http_post_json(url + "api/scan", {"demo": True})
        required = ("stats", "images", "exact_groups", "similar_groups",
                    "burst_groups", "screenshot_ids", "no_exif_ids",
                    "face_groups", "recommendations")
        missing = [k for k in required if k not in scan]

        # 演示扫描完成后再校验缩略图（该接口依赖本次扫描根目录）。
        status, thumb = _http_get(
            url + "api/thumb?rel=" + urllib.parse.quote("base_beach.jpg"))
        ok_thumb = status == 200 and len(thumb) > 0

        # 未二次确认时必须拒绝删除，且不得动任何文件。
        delete_status = None
        try:
            _http_post_json(url + "api/delete",
                            {"ids": ["base_beach.jpg"], "confirmed": False})
        except urllib.error.HTTPError as e:
            delete_status = e.code

        # P1-7：来源文件夹不匹配 / 旧版没有 source_dir 的条目必须拒绝。
        status, mismatch = _http_post_json(url + "api/delete", {
            "items": [{"id": "base_beach.jpg",
                       "source_dir": os.path.join(os.sep, "not", "this")}],
            "confirmed": True})
        mismatch_ok = (status == 200
                       and mismatch.get("results")
                       and mismatch["results"][0].get("ok") is False)
        status, legacy = _http_post_json(url + "api/delete", {
            "items": [{"id": "base_beach.jpg"}], "confirmed": True})
        legacy_ok = (status == 200 and legacy.get("results")
                     and legacy["results"][0].get("ok") is False)

        s = scan.get("stats", {})
        baseline = (s.get("total") == 25 and s.get("errors") == 0
                    and s.get("exact_groups") == 2
                    and s.get("similar_groups") == 3
                    and s.get("burst_groups") == 2
                    and s.get("screenshots") == 2
                    and s.get("no_exif") == 4)
        log("自检：index.html=%s health=%s ping=%s 缩略图=%s"
            % (ok_html, ok_health, ok_ping, ok_thumb))
        log("自检：演示扫描 total=%s errors=%s exact=%s similar=%s burst=%s "
            "screenshot=%s no_exif=%s faces=%s 推荐组=%s 耗时=%ss"
            % (s.get("total"), s.get("errors"), s.get("exact_groups"),
               s.get("similar_groups"), s.get("burst_groups"),
               s.get("screenshots"), s.get("no_exif"),
               len(scan.get("face_groups", [])),
               len(scan.get("recommendations", [])),
               scan.get("scan_seconds")))
        log("自检：未二次确认的删除请求返回 %s（应为 400）" % delete_status)
        log("自检：换文件夹/旧版篓子删除均被拒绝=%s/%s"
            % (mismatch_ok, legacy_ok))
        log("自检：数据目录=%s（回退=%s）"
            % (health.get("data_dir"), health.get("data_dir_fallback")))

        passed = (ok_html and ok_health and ok_ping and ok_thumb
                  and not missing and delete_status == 400
                  and mismatch_ok and legacy_ok and baseline)
        if missing:
            log("自检：缺少字段 %s" % missing)
        if not baseline:
            log("自检：演示数字偏离基线（25/0/2/3/2/2/4）")
        log("自检%s" % ("通过" if passed else "失败"))
        return 0 if passed else 1
    except Exception as e:
        log("自检失败：%s" % e)
        traceback.print_exc()
        return 1
    finally:
        _stop_server(httpd)


def _instance_message(lock_path):
    from instance_lock import read_info
    info = read_info(lock_path)
    url = info.get("url") or ""
    alive = False
    if url:
        try:
            status, _body = _http_get(url + "api/health", timeout=2)
            alive = status == 200
        except Exception:
            alive = False
    if alive:
        return "已有实例在运行：%s\n请直接使用已打开的窗口；本次启动不再开新端口、新窗口。" % url
    return ("已有实例在运行（单实例锁被占用），但实例地址暂时无法确认。\n"
            "请先看已经打开的窗口；若界面已关闭，请等几秒后重试。")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Shiying-3.2.0 原生窗口入口（照片整理与去重）")
    ap.add_argument("--port", type=int, default=0,
                    help="指定端口（默认自动在 8765~8774 间选空闲端口）")
    ap.add_argument("--no-window", action="store_true",
                    help="只启动本地服务，不打开窗口（调试用）")
    ap.add_argument("--self-test", action="store_true",
                    help="起服务并自检页面/接口/演示扫描，然后干净退出")
    ap.add_argument("--pick-folder", action="store_true",
                    help=argparse.SUPPRESS)
    ap.add_argument("--version", action="store_true",
                    help="打印版本号后退出")
    args = ap.parse_args(argv)

    if args.version:
        print("Shiying-%s（app_native.py）" % VERSION)
        return 0
    if args.pick_folder:
        return _pick_folder_cli()
    # 自检用独立临时端口，不参与单实例锁，方便在有窗口运行时复跑验证。
    if args.self_test:
        return _self_test(args.port)

    lock_path = app_paths.writable_path("instance.lock")
    from instance_lock import InstanceLock, clear_info, write_info
    lock = InstanceLock(lock_path)
    if not lock.acquire():
        print(_instance_message(lock_path))
        log("第二个实例被单实例锁挡下，未开新端口/新窗口。")
        return 2

    httpd = None
    try:
        _install_signal_handlers()
        httpd, url = _start_server(args.port)
        write_info(lock_path, {"pid": os.getpid(), "url": url,
                               "version": VERSION})
        log("Shiying-%s 本地服务已启动：%s" % (VERSION, url))
        log("只监听 127.0.0.1，关闭窗口或点“停止服务”即停止，不驻留后台。")
        info = app_paths.writable_info()
        log("日志：%s" % (LOG_PATH or "(不可用)"))
        if info["fallback"]:
            log("注意：程序目录不可写（%s），日志/隔离区/浏览器配置已回退到 %s"
                % (info.get("reason", ""), info["dir"]))

        if args.no_window:
            log("--no-window：按 Ctrl+C 或结束进程即停止服务。")
            while not _shutdown_requested():
                time.sleep(0.5)
            return 0

        try:
            if _run_pywebview(url):
                return 0
            if _run_app_browser(url):
                return 0
        except KeyboardInterrupt:
            log("收到 Ctrl+C，正在停止服务……")
            return 0

        log("无法打开原生窗口：未安装 pywebview，且未找到 Chrome/Edge。")
        log("请任选其一后重试：pip install pywebview，或安装 Chrome/Edge。")
        return 1
    except Exception as e:
        log("启动失败：%s" % e)
        traceback.print_exc()
        return 1
    finally:
        if httpd is not None:
            _stop_server(httpd)
        clear_info(lock_path)
        lock.release()
        log("服务已停止，无后台残留。")


if __name__ == "__main__":
    sys.exit(main())
