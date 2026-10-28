#!/usr/bin/env python3
"""跨平台单实例锁（3.2.0，P1-4）。

原理：对可写目录下的 instance.lock 加操作系统的非阻塞排他锁。
  - POSIX  : fcntl.flock(LOCK_EX | LOCK_NB)
  - Windows: msvcrt.locking(LK_NBLCK)
锁由操作系统在进程退出/崩溃时自动释放，不会留下“永久占用”的死锁；
锁文件本身可以一直保留，不需要删除。

第二个实例拿不到锁时不会开新端口、不会开新窗口，只读取
instance.json 里上一次写下的地址，尝试 /api/health 确认实例还活着，
然后打印明确提示并退出。
"""
import json
import os
import tempfile


class InstanceLock(object):
    def __init__(self, path):
        self.path = path
        self._fh = None

    def acquire(self):
        """拿到锁返回 True；已被别的实例占用返回 False。"""
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.path)),
                        exist_ok=True)
            fh = open(self.path, "a+b")
        except Exception:
            return False
        try:
            fh.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except Exception:
            try:
                fh.close()
            except Exception:
                pass
            return False
        self._fh = fh
        return True

    def release(self):
        if self._fh is None:
            return
        try:
            self._fh.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self._fh.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass
        try:
            self._fh.close()
        except Exception:
            pass
        self._fh = None


def info_path(lock_path):
    return os.path.join(os.path.dirname(os.path.abspath(lock_path)),
                        "instance.json")


def read_info(lock_path):
    try:
        with open(info_path(lock_path), encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def write_info(lock_path, payload):
    """原子写 instance.json，供第二个实例提示“已有实例在运行”。"""
    path = info_path(lock_path)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".instance.", suffix=".tmp",
                                   dir=os.path.dirname(path))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        except Exception:
            try:
                os.unlink(tmp)
            except Exception:
                pass
            raise
        return True
    except Exception:
        return False


def clear_info(lock_path):
    try:
        os.unlink(info_path(lock_path))
    except Exception:
        pass
