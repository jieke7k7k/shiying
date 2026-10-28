#!/usr/bin/env python3
"""统一的资源目录 / 可写目录解析（源码运行与 PyInstaller 打包后通用）。

3.2.0 起的三层约定（P1-3）：

  - 资源（index.html、demo_images、models 等）：只读。打包后优先用
    sys._MEIPASS（PyInstaller 解包目录），否则用本文件所在真实目录。
    不能写进 _MEIPASS——程序退出即被清空。
  - 可写（app.log、浏览器配置目录、单实例锁、.trash_backup 等）：
    先尝试程序所在目录（便携模式，U 盘/桌面直接跑时数据跟着走）；
    该目录不可写（例如装在 Program Files、只读盘）时，自动回退到
    系统用户数据目录：
      Windows : %LOCALAPPDATA%\\Shiying
      macOS   : ~/Library/Application Support/Shiying
      Linux   : $XDG_DATA_HOME/Shiying 或 ~/.local/share/Shiying
    回退结果与原因通过 writable_info() 暴露，启动日志与界面上都会明说。
  - 全部走 os.path，不依赖当前工作目录，中文路径可直接使用。
"""
import os
import sys
import tempfile

_CACHE = {"dir": None, "fallback": False, "reason": "", "app_dir": None}


def app_dir():
    """程序真实所在目录（打包后是 exe 所在目录，源码运行时是本文件目录）。"""
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


def resource_dir():
    """只读资源目录（打包后是 _MEIPASS，源码运行时同 app_dir）。"""
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        return os.path.abspath(meipass)
    return app_dir()


def resource_path(*parts):
    return os.path.join(resource_dir(), *parts)


def user_data_dir():
    """系统用户数据目录（跨平台），并确保目录存在。"""
    if os.name == "nt":
        base = (os.environ.get("LOCALAPPDATA")
                or os.environ.get("APPDATA")
                or os.path.expanduser("~"))
    elif sys.platform == "darwin":
        base = os.path.join(os.path.expanduser("~"), "Library",
                            "Application Support")
    else:
        base = (os.environ.get("XDG_DATA_HOME")
                or os.path.join(os.path.expanduser("~"), ".local", "share"))
    path = os.path.join(base, "Shiying")
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        pass
    return path


def _probe_writable(path):
    """真正建一个临时文件再删掉，别只信 os.access（Windows 上会骗人）。"""
    tmp = None
    try:
        os.makedirs(path, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".write-probe-", dir=path)
        try:
            os.write(fd, b"ok")
        finally:
            os.close(fd)
        os.unlink(tmp)
        return True, ""
    except Exception as e:
        if tmp:
            try:
                os.unlink(tmp)
            except Exception:
                pass
        return False, "%s: %s" % (type(e).__name__, e)


def writable_info():
    """返回 {"dir", "fallback", "reason", "app_dir"}；结果进程内缓存。"""
    if _CACHE["dir"] is not None:
        return dict(_CACHE)
    base = app_dir()
    ok, reason = _probe_writable(base)
    if ok:
        _CACHE.update({"dir": base, "fallback": False, "reason": "",
                       "app_dir": base})
    else:
        fallback_dir = user_data_dir()
        _CACHE.update({"dir": fallback_dir, "fallback": True,
                       "reason": reason, "app_dir": base})
    return dict(_CACHE)


def writable_dir():
    """可写根目录：程序目录可写就用它，否则回退系统用户数据目录。"""
    return writable_info()["dir"]


def writable_path(*parts):
    return os.path.join(writable_dir(), *parts)
