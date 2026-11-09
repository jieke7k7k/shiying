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
import json
import os
import shutil
import tempfile
from collections import defaultdict

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
    # 修复（2026-11-09 小马自查发现）：analyze_faces 存进 vectors 的是
    # list（_l2_normalize 返回 list），这里直接调 .dot() 会在 2 张脸
    # 以上时报 AttributeError，整条真实聚类链路静默退回测试分组。
    # 先统一转成 ndarray 再算点积。
    import numpy as np
    vectors = [np.asarray(v, dtype=float) for v in vectors]
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


# ---------------------------------------------------------------- 3.4.0 单脸
# 3.4.0 复核表要按“单张脸”工作，而不是按“某组照片”。为此本模块新增：
#   - face_key()：单脸主键 = 照片内容哈希前 16 位 + 脸序号；
#   - analyze_faces()：一次检测出所有单脸，保留框、检测分与 128 维特征；
#   - 特征落盘缓存（单独的 JSON 缓存文件），重扫直接复用，不重复算；
#   - 脸序号按框中心确定性排序（先上后下、同一行再左到右），不按检测分；
#   - 已忽略的脸在聚类前就被剔除，不参与聚类，免得把两个簇“桥接”起来。
ALGO_TAG = "yunet070-sface128-bandrow-v1"
EMBED_DIM = 128
FACE_KEY_HEX = 16


def face_key(sha256, index):
    """单脸主键：照片内容哈希前 16 位 + 脸序号（同内容照片共享同一身份）。"""
    hexpart = (sha256 or "")[:FACE_KEY_HEX].lower()
    if not hexpart:
        hexpart = "0" * FACE_KEY_HEX
    return "f%s-%d" % (hexpart, int(index))


def parse_face_key(key):
    """把主键拆回 (哈希前缀, 脸序号)；格式不对时返回 (None, None)。"""
    try:
        body = str(key)[1:]
        hexpart, idx = body.split("-", 1)
        return hexpart, int(idx)
    except Exception:
        return None, None


def cluster_id_for(anchor_key):
    """自动簇 id：锚点脸哈希前缀（不再用“人物 1/2/3”顺序编号）。"""
    hexpart, idx = parse_face_key(anchor_key)
    if not hexpart:
        return "af_unknown"
    return "af_%s-%s" % (hexpart[:8], idx)


def _l2_normalize(vec):
    norm = 0.0
    for v in vec:
        norm += float(v) * float(v)
    norm = norm ** 0.5
    if norm <= 0:
        return None
    return [float(v) / norm for v in vec]


def _cosine(a, b):
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    s = 0.0
    for i in range(n):
        s += a[i] * b[i]
    return s


def _box_area(box):
    try:
        return max(0.0, float(box[2])) * max(0.0, float(box[3]))
    except Exception:
        return 0.0


def _order_faces(faces, img_h):
    """按框中心确定性排序：先上后下（按行带），同一行再左到右。

    行带高度取“中位脸高 × 0.6”，避免同一排人脸因为十几像素的高低
    差被拆成两行；全部为 0 或异常时回退到严格 (cy, cx)。绝不按检测
    分数排——检测分随光照抖动，用它排序会让脸序号在重扫后跳变。
    """
    items = []
    for face in faces:
        x, y, w, h = (float(face[0]), float(face[1]),
                      float(face[2]), float(face[3]))
        # YuNet 每行是 [x, y, w, h, 置信度, 10 个关键点坐标]：置信度在
        # 下标 4，绝不能取 face[-1]（那是最后一个关键点的 y 坐标，
        # 数值几十上百，当检测分会把“检测分 0.95”写成“65.0”，还会
        # 带偏按检测分选代表脸的排序）。2026-11-09 小马自查修正。
        items.append((x + w / 2.0, y + h / 2.0, x, y, w, h, float(face[4])))
    if not items:
        return []
    heights = sorted(it[5] for it in items)
    median_h = heights[len(heights) // 2]
    band = max(24.0, median_h * 0.6)
    if img_h:
        band = min(band, max(24.0, img_h * 0.25))
    base_y = min(it[1] for it in items)
    items.sort(key=lambda it: (int((it[1] - base_y) // band), it[0], it[1]))
    return items


# ---- 特征缓存（单独文件，重扫复用）----

def _load_embed_cache(path):
    cache = {"algo": ALGO_TAG, "entries": {}}
    if not path:
        return cache
    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
    except Exception:
        return cache
    if not isinstance(raw, dict) or raw.get("algo") != ALGO_TAG:
        # 算法/阈值变了就整体作废，绝不复用可能错位的旧框。
        return cache
    entries = raw.get("entries")
    if isinstance(entries, dict):
        cache["entries"] = entries
    return cache


def _save_embed_cache(path, cache):
    if not path:
        return False
    try:
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False)
        os.replace(tmp, path)
        return True
    except Exception:
        return False


def _vec_ok(vec):
    return (isinstance(vec, list) and len(vec) == EMBED_DIM
            and all(isinstance(v, (int, float)) for v in vec))


def analyze_faces(images, root, ignore_keys=None, cache_path=None):
    """检测 + 提特征 + 聚类，返回“单脸级”结果（3.4.0 复核表的数据底座）。

    images：scan.py 的 ok 列表（每项须含 id / relative_path / sha256）。
    ignore_keys：已人工忽略的单脸主键集合，这些脸在聚类前就被剔除。
    cache_path：128 维特征落盘缓存文件；命中即复用，不重算。

    返回：
      {available, reason, method, faces:[...], clusters:[...],
       vectors:{key: [128 floats]}, stats:{...}}
    faces 每项不含向量（向量单独放 vectors，扫描响应 JSON 里绝不塞
    上百个浮点数），含 key/index/box/det_score/sha256/image_id/w/h。
    """
    report = attempt_report()
    out = {"available": False, "reason": report.get("detail", ""),
           "method": "none", "faces": [], "clusters": [], "vectors": {},
           "stats": {"images_with_faces": 0, "faces": 0, "ignored": 0,
                     "cache_hits": 0, "cache_misses": 0}}
    if not report.get("embedding_available"):
        return out
    try:
        import cv2
        import numpy as np
    except Exception as e:
        out["reason"] = "cv2/numpy 不可用：%s" % e
        return out
    yunet_path = _find_model(YUNET_CANDIDATES)
    sface_path = _find_model(SFACE_CANDIDATES)
    if not yunet_path or not sface_path:
        return out
    try:
        detector = cv2.FaceDetectorYN_create(
            _ascii_model_path(yunet_path), "", (320, 320),
            YUNET_SCORE_THRESHOLD, YUNET_NMS_THRESHOLD, YUNET_TOP_K)
        recognizer = cv2.FaceRecognizerSF_create(
            _ascii_model_path(sface_path), "")
    except Exception as e:
        out["reason"] = "模型创建失败：%s" % e
        return out

    ignore_keys = set(ignore_keys or ())
    cache = _load_embed_cache(cache_path)
    entries = cache["entries"]
    dirty = False
    vectors = {}
    faces_out = []
    seen_keys = {}  # P0-1：同内容照片（同 sha256）共享同一单脸，只留一条记录
    images_with_faces = 0
    stats = out["stats"]

    for item in images:
        rel = item.get("relative_path")
        sha = item.get("sha256") or ""
        if not rel or not sha:
            continue
        try:
            img = read_image(os.path.join(root, rel))
            if img is None:
                continue
            h, w = img.shape[:2]
            detector.setInputSize((int(w), int(h)))
            _ret, detected = detector.detect(img)
            if detected is None or len(detected) == 0:
                continue
            ordered = _order_faces(list(detected), h)
            images_with_faces += 1
            for idx, (cx, cy, x, y, bw, bh, score) in enumerate(ordered):
                key = face_key(sha, idx)
                if key in ignore_keys:
                    stats["ignored"] += 1
                    continue
                box = [int(round(x)), int(round(y)),
                       int(round(bw)), int(round(bh))]
                cached = entries.get(key)
                vec = None
                if cached and _vec_ok(cached.get("vec")):
                    vec = [float(v) for v in cached["vec"]]
                    stats["cache_hits"] += 1
                if vec is None:
                    feature = None
                    for row in detected:
                        rx, ry, rw, rh = (float(row[0]), float(row[1]),
                                          float(row[2]), float(row[3]))
                        if (abs(rx - x) < 1.5 and abs(ry - y) < 1.5
                                and abs(rw - bw) < 1.5 and abs(rh - bh) < 1.5):
                            feature = row
                            break
                    if feature is None:
                        feature = np.array([x, y, bw, bh, 0, 0, 0, 0, 0, 0,
                                            0, 0, 0, 0, score], dtype=float)
                    try:
                        aligned = recognizer.alignCrop(img, feature)
                        raw = recognizer.feature(aligned).flatten().astype(float)
                    except Exception:
                        continue
                    vec = _l2_normalize(raw)
                    if vec is None:
                        continue
                    stats["cache_misses"] += 1
                    entries[key] = {"vec": [round(v, 6) for v in vec],
                                    "box": box, "det": round(score, 4),
                                    "w": int(w), "h": int(h),
                                    "image_id": item.get("id") or rel}
                    dirty = True
                vectors[key] = vec
                face_id = item.get("id") or rel
                if key in seen_keys:
                    # 同内容重复照片的同一张脸：只复核一次，把这张照片
                    # 记进同一单脸的 image_ids，不再追加第二条记录。
                    existing = seen_keys[key]
                    if face_id not in existing.get("image_ids", []):
                        existing["image_ids"].append(face_id)
                    continue
                face_rec = {
                    "key": key,
                    "sha256": sha,
                    "image_id": face_id,
                    "image_ids": [face_id],
                    "relative_path": rel,
                    "index": idx,
                    "box": box,
                    "det_score": round(score, 4),
                    "image_w": int(w),
                    "image_h": int(h),
                    "cache_hit": bool(cached and _vec_ok(cached.get("vec"))),
                }
                seen_keys[key] = face_rec
                faces_out.append(face_rec)
        except Exception:
            continue

    if dirty:
        _save_embed_cache(cache_path, cache)

    stats["images_with_faces"] = images_with_faces
    stats["faces"] = len(faces_out)
    out["available"] = True
    out["method"] = "embedding_dbscan"
    out["reason"] = ("YuNet 检测（分数阈值 %.1f）+ SFace %d 维特征 + "
                     "余弦距离 DBSCAN（相似度阈值 %.2f，min_samples=1）。"
                     % (YUNET_SCORE_THRESHOLD, EMBED_DIM,
                        COSINE_SIMILARITY_THRESHOLD))
    out["faces"] = faces_out
    out["vectors"] = vectors
    out["clusters"] = _cluster_face_keys(faces_out, vectors)
    return out


def representative_key(member_keys, faces_by_key):
    """组内代表脸（头像用）：检测分高、框大者优先，同分按主键定序。"""
    def sort_key(k):
        face = faces_by_key.get(k) or {}
        return (-(face.get("det_score") or 0.0),
                -_box_area(face.get("box") or [0, 0, 0, 0]), k)
    keys = [k for k in member_keys if k in faces_by_key] or list(member_keys)
    if not keys:
        return None
    return sorted(keys, key=sort_key)[0]


def _cluster_face_keys(faces_out, vectors):
    if not faces_out:
        return []
    faces_by_key = {f["key"]: f for f in faces_out}
    keys = [f["key"] for f in faces_out if f["key"] in vectors]
    if not keys:
        return []
    labels = _dbscan_cosine([vectors[k] for k in keys], COSINE_DISTANCE_EPS)
    grouped = defaultdict(list)
    for key, lab in zip(keys, labels):
        grouped[int(lab)].append(key)
    clusters = []
    for lab in sorted(grouped):
        members = sorted(grouped[lab])
        anchor = representative_key(members, faces_by_key)
        cluster = {
            "cluster_id": cluster_id_for(anchor),
            "anchor_key": anchor,
            "member_keys": members,
            "image_ids": sorted({faces_by_key[k]["image_id"] for k in members}),
            "size": len(members),
        }
        clusters.append(cluster)
    clusters.sort(key=lambda c: c["cluster_id"])
    return clusters


def cluster_faces(images, root, ignore_keys=None, cache_path=None):
    """兼容旧调用：返回按组汇总的照片 id 列表（3.4.0 起 id 用锚点哈希）。

    新的复核表走 analyze_faces() 拿单脸级数据；本函数只保留给
    scan.py 的老字段 face_groups 用，语义与 3.2.0 一致。
    """
    res = analyze_faces(images, root, ignore_keys=ignore_keys,
                        cache_path=cache_path)
    if not res.get("available") or not res.get("clusters"):
        return []
    out = []
    for cluster in res["clusters"]:
        cid = cluster["cluster_id"]
        out.append({
            "id": cid,
            "label": "人物 %s" % cid[3:],
            "method": "embedding_dbscan",
            "note": ("YuNet 检测（分数阈值 %.1f）+ SFace 特征向量 + "
                     "余弦距离 DBSCAN（相似度阈值 %.2f，min_samples=1）；"
                     "簇 id 用锚点脸哈希前缀，不再用顺序编号。"
                     % (YUNET_SCORE_THRESHOLD, COSINE_SIMILARITY_THRESHOLD)),
            "members": list(cluster["image_ids"]),
            "anchor_key": cluster["anchor_key"],
            "face_count": cluster["size"],
        })
    return out
