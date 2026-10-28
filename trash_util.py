#!/usr/bin/env python3
"""安全删除：只走回收站或隔离区，绝不直接 os.remove 彻底抹除（3.2.0）。

规则：
  1) 优先用 send2trash 库把文件送进系统回收站（Windows 回收站 /
     macOS 废纸篓 / Linux Trash 规范目录）。系统回收站本身支持跨盘：
     移动硬盘上的文件会进那块盘自己的回收站，不会先拷到 C 盘。
  2) 若 send2trash 不可用（未安装、或系统不支持），则把文件移进
     工具包内的 .trash_backup/<时间戳>_<文件名> 隔离文件夹，
     并写一条删除日志（.trash_backup/deletion_log.json）。
     隔离区与源文件不在同一磁盘时**不再慢速拷贝**：直接报错，提示
     安装 send2trash 或把工具放到源文件所在磁盘，避免把移动硬盘的
     几十 GB 照片慢慢搬进 C 盘。
  3) 本模块是唯一允许“移走”文件的入口；调用方（server.py）必须
     在用户二次确认后才调用它，且只能删除本次扫描根目录内的文件。
  4) 本模块绝不调用 os.remove / os.unlink 删除用户文件。
     （隔离区里的文件由用户自己在回收站或隔离文件夹里手动清空，
     工具不会替用户彻底抹除。）

P1-2：删除日志改为原子写 + 批量记账
  - 先写同目录临时文件、flush + fsync，再 os.replace 原子替换；
    中途断电/崩溃不会留下半截 JSON。
  - 解析失败时不清空历史：把损坏文件改名备份为
    deletion_log.corrupt-<时间戳>.json，再另起新日志。
  - delete_many() 整批只读一次、只写一次，不再每张图全量读写。

用法：
  delete_to_trash(path) -> {"ok", "method", "destination", "original"}
  delete_many(paths)    -> [{"ok", "id"?, "method", "destination", ...}, ...]
"""
import json
import os
import shutil
import tempfile
import threading
from datetime import datetime

TOOL_DIR = os.path.dirname(os.path.abspath(__file__))
try:
    from app_paths import writable_path
    TRASH_BACKUP_DIR = writable_path(".trash_backup")
except Exception:
    TRASH_BACKUP_DIR = os.path.join(TOOL_DIR, ".trash_backup")
DELETION_LOG = os.path.join(TRASH_BACKUP_DIR, "deletion_log.json")
_LOG_LOCK = threading.Lock()


def _now_iso():
    return datetime.now().isoformat(timespec="seconds")


def _same_volume(path_a, path_b):
    """判断两个路径是否在同一磁盘/文件系统（跨盘隔离会退化成慢速拷贝）。"""
    if os.name == "nt":
        drive_a = os.path.splitdrive(os.path.abspath(path_a))[0].lower()
        drive_b = os.path.splitdrive(os.path.abspath(path_b))[0].lower()
        if drive_a != drive_b:
            return False
    try:
        return os.stat(path_a).st_dev == os.stat(path_b).st_dev
    except OSError:
        # 判断不了时按同盘处理（Windows 上已有盘符判断兜底）。
        return True


def _unique_backup_path(filename):
    os.makedirs(TRASH_BACKUP_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base, ext = os.path.splitext(filename)
    cand = os.path.join(TRASH_BACKUP_DIR, "%s_%s%s" % (stamp, base, ext))
    n = 1
    while os.path.exists(cand):
        cand = os.path.join(TRASH_BACKUP_DIR,
                            "%s_%s_%d%s" % (stamp, base, n, ext))
        n += 1
    return cand


# ---------------- 删除日志：原子写 + 损坏备份 + 批量记账 ----------------

def _load_log():
    """读取历史日志。损坏时不丢历史：改名备份后返回空列表。"""
    if not os.path.exists(DELETION_LOG):
        return []
    try:
        with open(DELETION_LOG, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        raise ValueError("日志不是 JSON 数组")
    except Exception:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = "%s.corrupt-%s.json" % (DELETION_LOG, stamp)
        try:
            # 保留旧文件：整份改名备份，而不是清空重写。
            os.replace(DELETION_LOG, backup)
        except Exception:
            pass
        return []


def _atomic_write_log(entries):
    """同目录临时文件 + fsync + os.replace，绝不留半截 JSON。"""
    os.makedirs(TRASH_BACKUP_DIR, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".deletion_log.", suffix=".tmp",
                               dir=TRASH_BACKUP_DIR)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(entries, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, DELETION_LOG)
    except Exception:
        try:
            os.unlink(tmp)
        except Exception:
            pass
        raise


def _commit_entries(new_entries):
    """一次读、一次写，把整批删除记录写进日志（进程内串行化）。"""
    if not new_entries:
        return
    with _LOG_LOCK:
        log = _load_log()
        log.extend(new_entries)
        _atomic_write_log(log)


# ---------------- 单个文件的删除计划（不落日志） ----------------

def _plan_one(abspath, force_backup):
    """把文件送进回收站/隔离区，返回 (result, log_entry|None, error|None)。"""
    if not os.path.exists(abspath):
        return None, None, "文件不存在：%s" % abspath
    if os.path.isdir(abspath):
        return None, None, "拒绝删除目录：%s" % abspath

    reason = ""
    if not force_backup:
        try:
            import send2trash
            send2trash.send2trash(abspath)
            entry = {"original": abspath, "method": "send2trash",
                     "destination": "系统回收站", "time": _now_iso()}
            return ({"ok": True, "method": "send2trash",
                     "destination": "系统回收站", "original": abspath},
                    entry, None)
        except ImportError:
            reason = "send2trash 未安装"
        except Exception as e:
            reason = "%s: %s" % (type(e).__name__, str(e)[:300])

    # 隔离区兜底：同盘才移，跨盘拒绝慢速拷贝。
    os.makedirs(TRASH_BACKUP_DIR, exist_ok=True)
    if not _same_volume(abspath, TRASH_BACKUP_DIR):
        return None, None, (
            "文件所在磁盘与隔离区不在同一块盘，已停止以避免把大文件慢速"
            "拷贝到系统盘。请安装 send2trash（pip install send2trash）"
            "后重试，让文件进它所在磁盘的系统回收站。%s"
            % ("（上次原因：%s）" % reason if reason else ""))

    dest = _unique_backup_path(os.path.basename(abspath))
    try:
        shutil.move(abspath, dest)
    except Exception as e:
        return None, None, "移入隔离区失败：%s" % e

    method = "backup" if force_backup else "backup_fallback"
    entry = {"original": abspath, "method": method, "destination": dest,
             "time": _now_iso()}
    if reason:
        entry["reason"] = reason
    result = {"ok": True, "method": method, "destination": dest,
              "original": abspath}
    if reason:
        result["reason"] = reason
    return result, entry, None


def delete_many(paths, _force_backup=False):
    """批量删除：整批只写一次日志。返回与 paths 等长的结果列表。"""
    results = []
    entries = []
    for path in paths:
        try:
            abspath = os.path.abspath(path)
            result, entry, error = _plan_one(abspath, _force_backup)
        except Exception as e:
            abspath = os.path.abspath(path)
            result, entry, error = None, None, "删除失败：%s" % e
        if error:
            results.append({"ok": False, "id": path, "original": abspath,
                            "error": error})
            continue
        results.append(result)
        if entry:
            entries.append(entry)
    if entries:
        try:
            _commit_entries(entries)
        except Exception as e:
            # 文件已经移走，日志失败不能反向报成删除失败；明确提示日志异常。
            note = "文件已移走，但删除日志写入失败：%s" % e
            for r in results:
                if r.get("ok"):
                    r["log_warning"] = note
    return results


def delete_to_trash(path, _force_backup=False):
    """单个文件删除（兼容旧调用）。返回 method/destination/original。"""
    result = delete_many([path], _force_backup=_force_backup)[0]
    if not result.get("ok"):
        raise RuntimeError(result.get("error") or "删除失败")
    return result
