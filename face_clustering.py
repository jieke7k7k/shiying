#!/usr/bin/env python3
"""人脸分组：YuNet 检测 + SFace 特征向量 + 余弦距离 DBSCAN（零编译路线，3.2.0）。

dlib 路线已彻底放弃：本模块不再导入、不再检测 face_recognition /
dlib / InsightFace。真实人脸聚类只走 OpenCV 4.x/5.x 原生内置的
cv2.FaceDetectorYN（YuNet）+ cv2.FaceRecognizerSF（SFace），两者
都是预编译轮子 + ONNX 模型，纯 Python 调用，零编译。

模型文件（放在 tool/models/ 里，随工具包一起走，不用联网现下）：
  - face_detection_yunet_2023mar.onnx（约 0.23 MB）
  - face_recognition_sface_2021dec.onnx（约 38.7 MB，fp32 全精度）
  另有官方 INT8 量化版（YuNet 约 0.1 MB / SFace 约 9.9 MB），3.2.0 起
  优先加载 INT8 版，FP32 留作回退。

关于 INT8 的如实说明（P1-6 更正）：
  - INT8 已在 OpenCV 5.0 下实测创建成功（YuNet 与 SFace 均能正常
    create，见本机验证记录），不再沿用“SFace INT8 解析失败、不能用
    int8”的旧说法。
  - 但**真实人脸聚类尚未在 Windows 真机实测**：本机只在 Linux +
    OpenCV 5.0 上验证了模型可创建、公开测试人像可聚类；Windows
    真机、用户真实照片上的效果仍须实测，不能宣称已通过。

中文路径（P0-1）：
  - 图片读取一律 np.fromfile + cv2.imdecode，不使用 cv2.imread。
    Windows 下 cv2.imread 走窄字符 fopen，含中文/非 GBK 字符的路径
    会直接返回 None；np.fromfile 走 Python 的文件对象（Unicode 路径
    安全），再把字节交给 cv2.imdecode 解码。
  - 模型路径含非 ASCII 时，OpenCV 的 ONNX 读取同样会失败。这里优先
    尝试 Windows 8.3 短路径；不行就把模型复制到纯 ASCII 的临时目录
    再交给 OpenCV（见 _ascii_model_path），进程退出时清理临时副本。

实测定下的阈值（2026-10-26，本机 Linux + OpenCV 5.0，公开测试人像）：
  - YuNet 检测分数阈值 0.7：默认 0.9 会漏检——同一张 Lena 照缩放
    / 调亮后的检测分数为 0.89/0.89，0.9 阈值下直接漏掉；0.7 时全部检出。
  - SFace 余弦相似度阈值 0.6（距离 eps=0.4）：同一人不同版本
    （缩放、调亮）相似度 0.83 / 0.87，不同人（Lena vs Obama）为 -0.13，
    0.6 的分界留有充分余量。
  - DBSCAN 取 min_samples=1：此时每个点都是核心点，聚类等价于
    按阈值的连通分量；单张脸自成一组，不会被当成噪声丢掉。

诚实边界（与旧版一致，继续遵守）：
  - Haar 级联只能框出脸，做不了“认出同一个人”，绝不冒充。
  - 合成卡通头像检不出属正常：卡通画的脸不是真实人脸，不能拿
    卡通图当验证，更不能反推“模型坏了”。
  - 本机无 cv2 或模型文件缺失时，cluster_faces() 返回 []，
    由调用方（scan.py）改用测试分组兜底并在界面与报告里明说。

本模块提供：
  1) attempt_report()：如实汇报 cv2、两个模型文件是否就绪，
     以及真实聚类路线是否可用（embedding_available）。
  2) detect_faces_haar(path)：仅返回检测框，method=detection_only，
     绝不用于身份分组。
  3) cluster_faces(images, root)：YuNet 检测 -> SFace 特征向量 ->
     余弦距离 DBSCAN 聚类；不可用或未检出时返回 []。
"""
import atexit
import os
import shutil
import tempfile

TOOL_DIR = os.path.dirname(os.path.abspath(__file__))
try:
    from app_paths import resource_path
    MODEL_DIR = resource_path("models")
except Exception:
    # 单文件拷贝运行时的兜底：仍按本文件所在目录找 models。
    MODEL_DIR = os.path.join(TOOL_DIR, "models")

# 模型文件名：3.2.0 起优先用官方 INT8 量化版（已在 OpenCV 5.0 下实测能创建），
# FP32 原版留作回退。是否可用以实际 create 成功为准，见 attempt_report。
YUNET_CANDIDATES = (
    "face_detection_yunet_2023mar_int8.onnx",
    "face_detection_yunet_2023mar.onnx",
    "yunet.onnx",
)
SFACE_CANDIDATES = (
    "face_recognition_sface_2021dec_int8.onnx",
    "face_recognition_sface_2021dec.onnx",
    "sface.onnx",
)

# 实测定下的阈值（见模块 docstring）。
YUNET_SCORE_THRESHOLD = 0.7
YUNET_NMS_THRESHOLD = 0.3
YUNET_TOP_K = 5000
# 余弦相似度 >= 0.6 判为同一人（等价于余弦距离 <= 0.4）。
COSINE_SIMILARITY_THRESHOLD = 0.6
COSINE_DISTANCE_EPS = 1.0 - COSINE_SIMILARITY_THRESHOLD

_ASCII_MODEL_CACHE = {}
_ASCII_TEMP_DIRS = []


def is_ascii_path(path):
    try:
        return str(path).isascii()
    except AttributeError:  # Python < 3.7
        return all(ord(ch) < 128 for ch in str(path))


def _windows_short_path(path):
    """Windows 8.3 短路径（纯 ASCII），卷上未启用 8.3 时返回 None。"""
    if os.name != "nt":
        return None
    try:
        import ctypes
        from ctypes import wintypes
        get_short = ctypes.windll.kernel32.GetShortPathNameW
        get_short.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR,
                              wintypes.DWORD]
        get_short.restype = wintypes.DWORD
        buf = ctypes.create_unicode_buffer(1024)
        n = get_short(path, buf, len(buf))
        if 0 < n < len(buf):
            short = buf.value
            if short and is_ascii_path(short):
                return short
    except Exception:
        pass
    return None


def _ascii_roots():
    roots = []
    candidates = [tempfile.gettempdir()]
    if os.name == "nt":
        candidates += [
            os.environ.get("LOCALAPPDATA"),
            os.environ.get("TEMP"),
            os.environ.get("TMP"),
            "C:\\ProgramData",
            "C:\\Users\\Public",
        ]
    else:
        candidates += ["/tmp", os.path.expanduser("~")]
    seen = set()
    for root in candidates:
        if not root or not is_ascii_path(root):
            continue
        key = os.path.normcase(os.path.abspath(root))
        if key in seen:
            continue
        seen.add(key)
        if os.path.isdir(root):
            roots.append(root)
    return roots


def _cleanup_ascii_temps():
    for path in list(_ASCII_TEMP_DIRS):
        shutil.rmtree(path, ignore_errors=True)


atexit.register(_cleanup_ascii_temps)


def _ascii_model_path(path):
    """把非 ASCII 的模型路径换成 OpenCV 能打开的 ASCII 路径。

    顺序：原路径已是 ASCII -> 直接用；Windows 短路径 -> 免拷贝；
    否则复制到纯 ASCII 临时目录（进程内复用、退出时清理）。
    """
    if not path:
        return path
    if is_ascii_path(path):
        return path
    cached = _ASCII_MODEL_CACHE.get(path)
    if cached and os.path.isfile(cached):
        return cached

    short = _windows_short_path(path)
    if short:
        _ASCII_MODEL_CACHE[path] = short
        return short

    for root in _ascii_roots():
        tmp_dir = None
        try:
            tmp_dir = tempfile.mkdtemp(prefix="po-model-", dir=root)
            dest = os.path.join(tmp_dir, "model" + os.path.splitext(path)[1])
            shutil.copyfile(path, dest)
            if not is_ascii_path(dest) or not os.path.isfile(dest):
                raise OSError("临时路径仍含非 ASCII 字符")
        except Exception:
            if tmp_dir:
                shutil.rmtree(tmp_dir, ignore_errors=True)
            continue
        _ASCII_TEMP_DIRS.append(tmp_dir)
        _ASCII_MODEL_CACHE[path] = dest
        return dest
    # 实在找不到可写的 ASCII 目录：返回原路径，让调用方按失败处理并如实报告。
    return path


def read_image(path, flags=None):
    """Unicode 安全的图片读取：np.fromfile + cv2.imdecode。

    Windows 下 cv2.imread 用的是窄字符 fopen，中文/emoji 路径会返回
    None；np.fromfile 走 Python 文件对象（Unicode 路径安全），再由
    cv2.imdecode 解码内存字节，效果与 imread 完全一致。
    """
    try:
        import cv2
        import numpy as np
    except Exception:
        return None
    if flags is None:
        flags = cv2.IMREAD_COLOR
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except Exception:
        return None
    if data is None or data.size == 0:
        return None
    try:
        return cv2.imdecode(data, flags)
    except Exception:
        return None


def _find_model(candidates):
    for name in candidates:
        path = os.path.join(MODEL_DIR, name)
        if os.path.isfile(path):
            return path
    return None


def _try_import(name):
    try:
        mod = __import__(name)
        return True, getattr(mod, "__version__", "unknown")
    except Exception as e:
        return False, "%s: %s" % (type(e).__name__, str(e)[:200])


def attempt_report():
    cv2_ok, cv2_info = _try_import("cv2")
    yunet_path = _find_model(YUNET_CANDIDATES)
    sface_path = _find_model(SFACE_CANDIDATES)
    embedding_ok = bool(cv2_ok and yunet_path and sface_path)
    detail_parts = []
    detail_parts.append("opencv: %s (%s)"
                        % ("可用" if cv2_ok else "不可用", cv2_info))
    detail_parts.append("YuNet 模型：%s"
                        % (yunet_path if yunet_path else "缺失（应放在 "
                           "tool/models/face_detection_yunet_2023mar.onnx）"))
    detail_parts.append("SFace 模型：%s"
                        % (sface_path if sface_path else "缺失（应放在 "
                           "tool/models/face_recognition_sface_2021dec.onnx）"))
    detail_parts.append("路径处理：图片 np.fromfile+cv2.imdecode；"
                        "模型路径含非 ASCII 时复制到 ASCII 临时目录")
    detail_parts.append("INT8 模型：已在 OpenCV 5.0 下实测创建成功；"
                        "真实人脸聚类尚未在 Windows 真机实测")
    detail_parts.append("dlib 路线：已彻底放弃，不再检测、不再使用")
    return {
        # 旧键保留给调用方兼容：dlib 路线已放弃，恒为 False。
        "face_recognition_available": False,
        "dlib_available": False,
        "insightface_available": False,
        # 新路线：只有 cv2 + 两个模型文件齐全时才为 True。
        "embedding_available": embedding_ok,
        "yunet_available": bool(yunet_path),
        "sface_available": bool(sface_path),
        "models_present": bool(yunet_path and sface_path),
        "yunet_model": yunet_path,
        "sface_model": sface_path,
        "opencv_available": cv2_ok,
        "sklearn_available": False,
        "detail": "；".join(detail_parts),
        "honest_note": "Haar 只能检测不能认人；身份分组必须走特征向量 + "
        "聚类。本路线为 YuNet + SFace（零编译），无 cv2 或无模型文件时"
        "返回空并由调用方用测试分组兜底，绝不冒充。",
    }


def detect_faces_haar(path):
    """只返回检测框。method 永远是 detection_only，不用于身份分组。"""
    try:
        import cv2
    except Exception as e:
        return {"method": "detection_only", "available": False,
                "boxes": [], "note": "cv2 不可用：%s" % e}
    try:
        cascade_path = os.path.join(cv2.data.haarcascades,
                                    "haarcascade_frontalface_default.xml")
        cascade = cv2.CascadeClassifier(cascade_path)
        if cascade.empty():
            return {"method": "detection_only", "available": False,
                    "boxes": [], "note": "Haar 级联文件加载失败。"}
        img = read_image(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return {"method": "detection_only", "available": False,
                    "boxes": [], "note": "图片读取失败。"}
        faces = cascade.detectMultiScale(img, scaleFactor=1.1,
                                         minNeighbors=5, minSize=(60, 60))
        boxes = [[int(x), int(y), int(w), int(h)]
                 for (x, y, w, h) in faces]
        return {"method": "detection_only", "available": True,
                "boxes": boxes,
                "note": "仅检测出脸的位置，不能认出是谁；绝不用于身份分组。"}
    except Exception as e:
        return {"method": "detection_only", "available": False,
                "boxes": [], "note": "检测失败：%s" % e}


def _dbscan_cosine(vectors, eps):
    """余弦距离 DBSCAN（min_samples=1，即按阈值的连通分量）。

    vectors：已 L2 归一化的特征向量列表。
    返回：与 vectors 等长的簇标签列表（0, 1, 2, ...）。
    """
    n = len(vectors)
    labels = [-1] * n
    # 余弦距离 = 1 - 点积（向量已归一化）。
    neighbors = []
    for i in range(n):
        nb = []
        for j in range(n):
            dist = 1.0 - float(vectors[i].dot(vectors[j]))
            if dist <= eps:
                nb.append(j)
        neighbors.append(nb)
    cluster = 0
    for i in range(n):
        if labels[i] != -1:
            continue
        labels[i] = cluster
        stack = [i]
        while stack:
            cur = stack.pop()
            for nb in neighbors[cur]:
                if labels[nb] == -1:
                    labels[nb] = cluster
                    stack.append(nb)
        cluster += 1
    return labels


def cluster_faces(images, root):
    """YuNet + SFace 真实聚类。不可用或未检出时返回 []（由调用方兜底）。

    images：scan.py 的 ok 列表（每项含 relative_path）。
    返回：[{id, label, method, members}]，method 恒为 embedding_dbscan。
    一张图里有多张脸时，每张脸单独参与聚类，图片可能同时出现在
    多个分组里（这是正确的：合影里确实有多个不同的人）。
    """
    try:
        import cv2
    except Exception:
        return []
    yunet_path = _find_model(YUNET_CANDIDATES)
    sface_path = _find_model(SFACE_CANDIDATES)
    if not yunet_path or not sface_path:
        return []
    try:
        import numpy as np
    except Exception:
        return []
    try:
        # P0-1：模型路径含中文时 OpenCV 读不了，先换成 ASCII 路径。
        detector = cv2.FaceDetectorYN_create(
            _ascii_model_path(yunet_path), "", (320, 320),
            YUNET_SCORE_THRESHOLD, YUNET_NMS_THRESHOLD, YUNET_TOP_K)
        recognizer = cv2.FaceRecognizerSF_create(
            _ascii_model_path(sface_path), "")
    except Exception:
        return []
    instances = []  # (image_id, 归一化特征向量)
    for item in images:
        full = os.path.join(root, item["relative_path"])
        try:
            img = read_image(full)
            if img is None:
                continue
            h, w = img.shape[:2]
            detector.setInputSize((w, h))
            _ret, faces = detector.detect(img)
            if faces is None:
                continue
            for face in faces:
                try:
                    aligned = recognizer.alignCrop(img, face)
                    feat = recognizer.feature(aligned).flatten().astype(float)
                except Exception:
                    continue
                norm = float(np.linalg.norm(feat))
                if norm <= 0:
                    continue
                instances.append((item["id"], feat / norm))
        except Exception:
            continue
    if not instances:
        return []
    vecs = [v for (_iid, v) in instances]
    labels = _dbscan_cosine(vecs, COSINE_DISTANCE_EPS)
    groups = {}
    for (iid, _v), lab in zip(instances, labels):
        groups.setdefault(int(lab), set()).add(iid)
    out = []
    for n, (_lab, members) in enumerate(sorted(groups.items())):
        out.append({
            "id": "face_%d" % (n + 1),
            "label": "人物 %d" % (n + 1),
            "method": "embedding_dbscan",
            "note": "YuNet 检测（分数阈值 0.7）+ SFace 特征向量 + "
                    "余弦距离 DBSCAN（相似度阈值 0.6，min_samples=1）。",
            "members": sorted(members),
        })
    return out
