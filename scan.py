#!/usr/bin/env python3
"""本地扫描脚本（3.2.0）：扫描指定文件夹，输出 scan_result.json。

本文件是阶段二 scan.py 的阶段三修订版（阶段二原文件未动，修订版只活在
tool/ 里）。相对阶段二，阶段三按 Argon 审查结论改了以下 5 项：

3.2.0（P0-1 中文路径）：cv2 读图一律 np.fromfile + cv2.imdecode，不再用
       cv2.imread。Windows 下 cv2.imread 走窄字符 fopen，路径含中文/
       emoji 时直接返回 None（表现为“人脸检测不可用”）；PIL 读图本身
       是 Unicode 安全的，保持不变。扫描只读、不加全局状态，配合
       server.py 的扫描锁后不会并发跑多轮。

  P0-1 连拍规则：时间窗口由 1~15 秒改为 0~5 秒，并用三条条件区分
       “0 秒真连拍”与“0 秒修图变体”：
       (a) 文件名变体特征：含 crop / edit / copy / filter / mmexport /
           wx_camera / compressed / bright 等后缀的是修图变体，不是连拍；
       (b) 原始分辨率一致性：连拍帧共享同一分辨率，取组内众数分辨率，
           偏离众数的成员不进连拍组；
       (c) 色彩被置灰的修图副本排除：灰度图不进连拍组。
       另要求链式相似度（3.2.0 收官补丁 1，见下方）且 >= 3 张成组。
       说明：棕褐变体不靠单图猜测（实测暖色食物场景 R-B 差达 77，
       单图棕褐启发式会误伤正常暖色照片），棕褐/黑白变体改靠
       P1-2 的成对 RGB 均值差标注——有原图可对照时才可靠。
  3.2.0 收官补丁 1（连拍长序列不断链）：组内相似度判定由“全部与首帧
       比不低于 80”改为“与链内前一张相邻帧相似度不低于 80”；相邻帧
       相似度 < 80 时断链，剩余帧另起新链（>= 3 张才成组）。连续拍
       10 张运动画面时，后半截与第 1 张位移过大也不再漏检。
  P0-2 真实缩略图：每张图输出 200x200 以内的轻量 Base64 缩略图字段
       thumbnail（data URI），画廊与 PK 台直接渲染真实画面。
  P1-1 推荐规则：多维评分——像素总面积最大优先、EXIF 完整度次之、
       彩色优先（P1-2 的一部分）再次之、文件名干净再次之、文件体积最后。
       裁切版在生成器里不再 resize 放大回原尺寸，故母本靠像素胜出。
  P1-2 色彩差异：dHash 先转灰度，对黑白/棕褐变体有 100% 相似度盲区；
       补充 RGB 通道均值差：dHash 为 100% 但 SHA-256 不同、且 RGB 均值
       欧氏距离 >= 阈值（默认 10）时，标注 color_variant（色彩变体），
       并在推荐评分里优先保留彩色原图。
  P1-3 安全删除：本脚本只读不写被扫描文件夹，绝不删除、移动任何文件。
       真正删除只发生在本地服务的 /api/delete 里，且只走回收站或隔离区
       （见 trash_util.py），本脚本没有任何删除代码。

人脸（诚实处理）：Haar 级联只能做人脸检测（框出脸），做不了人脸聚类
（认出同一个人）。身份分组必须走特征向量 + 聚类路线（见
face_clustering.py）。本脚本先尝试特征向量聚类；不可用时用
faces_manifest.json 的测试分组兜底，并在输出中明确标注
method=test_grouping（不是真实人脸识别）。

用法:
  python3 scan.py <文件夹> [--output scan_result.json]
      [--similar-threshold 85] [--faces-manifest faces_manifest.json]

本脚本只读不写被扫描文件夹，绝不删除、移动任何文件。
"""
import argparse
import base64
import hashlib
import io
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime

from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
EXIF_TAGS = (36867, 36868, 306)  # DateTimeOriginal, DateTimeDigitized, DateTime

# ---- P0-1：连拍 vs 修图变体的区分条件 ----
# 文件名里出现这些 token 即判定为修图/副本变体，不是连拍帧。
# 注意：不要把 "burst" 放进这里——含 BURST 是连拍的正向特征。
VARIANT_TOKENS = (
    "crop", "cropped", "edit", "edited", "copy",
    "filter", "filtered", "mmexport", "wx_camera", "wechat",
    "compressed", "compress", "bright", "brightness",
    "enhance", "enhanced", "adjust", "adjusted",
    "duplicate", "thumb", "version", "v2", "final",
)
# ---- P1-2：色彩变体判定阈值（RGB 均值欧氏距离，0~441）----
COLOR_VARIANT_THRESHOLD = 10.0
# 灰度判定：RGB 均值极差小于该值即视为灰度图。
GRAY_SPREAD_THRESHOLD = 8.0


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dhash_of(img):
    """标准 8x8 dHash：灰度 -> 9x8 -> 相邻像素比较 -> 64 位整数与 hex。"""
    gray = img.convert("L").resize((9, 8), Image.LANCZOS)
    px = list(gray.getdata())
    bits = 0
    for y in range(8):
        for x in range(8):
            bits = (bits << 1) | (1 if px[y * 9 + x] > px[y * 9 + x + 1] else 0)
    return bits, "%016x" % bits


def hamming(a, b):
    return bin(a ^ b).count("1")


def similarity(a, b):
    return round((1 - hamming(a, b) / 64.0) * 100, 1)


def read_exif_time(img):
    try:
        exif = img._getexif()
    except Exception:
        return None
    if not exif:
        return None
    for tag in EXIF_TAGS:
        raw = exif.get(tag)
        if raw:
            try:
                return datetime.strptime(str(raw).strip(),
                                         "%Y:%m:%d %H:%M:%S")
            except ValueError:
                continue
    return None


def white_fraction(img):
    small = img.convert("RGB").resize((100, 100), Image.LANCZOS)
    px = list(small.getdata())
    white = sum(1 for r, g, b in px if r > 240 and g > 240 and b > 240)
    return white / max(len(px), 1)


def rgb_mean_of(img):
    """RGB 通道均值（P1-2 用）：缩到 64x64 后求均值，四舍五入到 1 位小数。"""
    small = img.convert("RGB").resize((64, 64), Image.LANCZOS)
    px = list(small.getdata())
    n = max(len(px), 1)
    r = round(sum(p[0] for p in px) / n, 1)
    g = round(sum(p[1] for p in px) / n, 1)
    b = round(sum(p[2] for p in px) / n, 1)
    return [r, g, b]


def is_grayscale_rgb(rgb):
    return (max(rgb) - min(rgb)) < GRAY_SPREAD_THRESHOLD


def color_score_of(rgb):
    """P1-2 推荐用色彩分：彩色 2 分，灰度 0 分。

    不设单图棕褐分：暖色照片与棕褐滤镜在单图上无法可靠区分
    （实测食物场景 R-B 差 77，若设棕褐启发式会误伤正常暖色照片），
    棕褐变体只在成对比较时用 RGB 均值差标注（见 similar_pairs）。
    """
    return 0 if is_grayscale_rgb(rgb) else 2


def color_distance(a, b):
    return round(math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b))), 1)


def thumbnail_data_uri(img):
    """P0-2：200x200 以内的轻量 JPEG 缩略图（data URI）。"""
    thumb = img.convert("RGB").copy()
    thumb.thumbnail((200, 200), Image.LANCZOS)
    buf = io.BytesIO()
    thumb.save(buf, "JPEG", quality=70)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def is_variant_name(name):
    low = name.lower()
    return any(tok in low for tok in VARIANT_TOKENS)


def is_clean_name(name):
    return not is_variant_name(name)


def sequence_number(name):
    """取文件名中扩展名前的尾随数字（如 burst_001 -> 1，IMG_1234 -> 1234）。"""
    m = re.search(r"(\d+)(?=\.[^.]+$)", name)
    return int(m.group(1)) if m else None


def try_haar(images, root):
    """返回 (detected_count, available)。只做可行性探测，不用于身份分组。

    Haar 只能框出脸，做不了“认出同一个人”——这一点在报告与界面里
    都会明说，不会把 Haar 检出数冒充成人脸聚类结果。
    """
    try:
        import cv2
    except Exception:
        return 0, False
    try:
        cascade_path = os.path.join(cv2.data.haarcascades,
                                    "haarcascade_frontalface_default.xml")
        cascade = cv2.CascadeClassifier(cascade_path)
        if cascade.empty():
            return 0, False
        count = 0
        try:
            import numpy as np
        except Exception:
            return 0, False
        for item in images:
            # P0-1：中文路径下 cv2.imread 会返回 None，改走
            # np.fromfile（Unicode 安全）+ cv2.imdecode。
            try:
                data = np.fromfile(os.path.join(root, item["relative_path"]),
                                   dtype=np.uint8)
                gray = (cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
                        if data is not None and data.size else None)
            except Exception:
                gray = None
            if gray is None:
                continue
            faces = cascade.detectMultiScale(gray, scaleFactor=1.1,
                                             minNeighbors=5,
                                             minSize=(60, 60))
            count += len(faces)
        return count, True
    except Exception:
        return 0, False


def scan(root, similar_threshold, manifest_path):
    images = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for name in sorted(filenames):
            ext = os.path.splitext(name)[1].lower()
            if ext not in IMAGE_EXTS or name.startswith("."):
                continue
            full = os.path.join(dirpath, name)
            rel = os.path.relpath(full, root)
            try:
                with Image.open(full) as im:
                    im.load()
                    width, height = im.size
                    exif_time = read_exif_time(im)
                    dhash_int, dhash_hex = dhash_of(im)
                    wf = white_fraction(im)
                    rgb = rgb_mean_of(im)
                    # 3.2.0 复审新 P1-B：不再无条件生成 Base64 缩略图，
                    # 前端改走 /api/thumb 动态接口；thumbnail 留空字符串
                    # 仅作无相对路径时的回退键兼容。
                    thumb = ""
            except Exception as e:
                images.append({"id": rel, "filename": name,
                               "relative_path": rel, "error": str(e)})
                continue
            images.append({
                "id": rel,
                "filename": name,
                "relative_path": rel,
                "size_bytes": os.path.getsize(full),
                "width": width,
                "height": height,
                "pixels": width * height,
                "sha256": sha256_of(full),
                "dhash": dhash_hex,
                "_dhash_int": dhash_int,
                "exif_time": exif_time.isoformat() if exif_time else None,
                "has_exif": exif_time is not None,
                "_exif_dt": exif_time,
                "rgb_mean": rgb,
                "is_grayscale": is_grayscale_rgb(rgb),
                "thumbnail": thumb,
                "is_screenshot": ("screenshot" in name.lower()
                                  or "截屏" in name
                                  or (height / max(width, 1) >= 1.4
                                      and wf >= 0.45)),
            })
    ok = [im for im in images if "error" not in im]

    # 精确重复组：SHA-256 相同
    by_sha = defaultdict(list)
    for im in ok:
        by_sha[im["sha256"]].append(im["id"])
    exact_groups = sorted([sorted(v) for v in by_sha.values() if len(v) > 1])
    exact_members = {m for g in exact_groups for m in g}
    rep_of_exact = {}
    for g in exact_groups:
        for m in g[1:]:
            rep_of_exact[m] = g[0]

    # ---- P0-1 连拍组（新规则）----
    # 1) 先按 EXIF 时间排序，把相邻间隔 0~5 秒的切成时间簇。
    #    0 秒必须放行：真机高速连拍（1 秒多张）的秒级 EXIF 时间差恰恰是 0 秒。
    # 2) 簇内逐一排除修图变体：文件名含变体后缀、灰度图都不进组。
    #    （棕褐不靠单图排除，原因见模块 docstring；成对标注见 P1-2。）
    # 3) 分辨率一致性：取簇内众数分辨率，偏离众数的不进组。
    # 4) 相似度：链式判定，成员与链内前一张相邻帧 dHash 相似度 >= 80
    #    即可接续，< 80 则断链另起新链；每条链 >= 3 张才成组。
    # 这样：0 秒高速连拍（序列名、同分辨率、彩色）能检出；0 秒修图变体
    # （文件名带变体后缀、或被置灰）即使时间戳相同也不会误判为连拍。
    dated = [im for im in ok if im["_exif_dt"] is not None
             and im["id"] not in rep_of_exact
             and not im["is_screenshot"]]
    dated.sort(key=lambda im: (im["_exif_dt"], im["id"]))
    burst_groups = []
    cur = []
    clusters = []
    for im in dated:
        if not cur:
            cur = [im]
            continue
        gap = (im["_exif_dt"] - cur[-1]["_exif_dt"]).total_seconds()
        if 0 <= gap <= 5:
            cur.append(im)
        else:
            clusters.append(cur)
            cur = [im]
    if cur:
        clusters.append(cur)
    for cluster in clusters:
        eligible = [im for im in cluster
                    if not is_variant_name(im["filename"])
                    and not im["is_grayscale"]]
        if len(eligible) < 3:
            continue
        # 分辨率众数过滤
        mode_res, _ = Counter((im["width"], im["height"])
                              for im in eligible).most_common(1)[0]
        eligible = [im for im in eligible
                    if (im["width"], im["height"]) == mode_res]
        if len(eligible) < 3:
            continue
        # 4) 链式相似度（3.2.0 收官补丁 1）：成员只需与链内前一张相邻帧
        #    dHash 相似度 >= 80，不再要求全部与首帧比；相邻帧 < 80 时断链，
        #    剩余帧另起新链，>= 3 张才成组。连续运动画面后半截与首帧位移
        #    过大时也不再漏检。
        chain = [eligible[0]]
        for im in eligible[1:]:
            if similarity(im["_dhash_int"], chain[-1]["_dhash_int"]) >= 80:
                chain.append(im)
            else:
                if len(chain) >= 3:
                    burst_groups.append([m["id"] for m in chain])
                chain = [im]
        if len(chain) >= 3:
            burst_groups.append([m["id"] for m in chain])
    burst_members = {m for g in burst_groups for m in g}

    screenshot_ids = sorted(im["id"] for im in ok if im["is_screenshot"])
    no_exif_ids = sorted(im["id"] for im in ok if not im["has_exif"])

    # 疑似重复：只在“候选池”内计算，避免链式误并。
    # 候选池排除：精确重复的副本（只留首张）、连拍组成员（受保护）、截图。
    by_id = {im["id"]: im for im in ok}
    pool = [im for im in ok if im["id"] not in rep_of_exact
            and im["id"] not in burst_members
            and not im["is_screenshot"]]
    similar_pairs = []
    for i in range(len(pool)):
        for j in range(i + 1, len(pool)):
            sim = similarity(pool[i]["_dhash_int"], pool[j]["_dhash_int"])
            if sim >= similar_threshold:
                a, b = pool[i], pool[j]
                cd = color_distance(a["rgb_mean"], b["rgb_mean"])
                # P1-2：dHash 100% 但 SHA 不同，且 RGB 均值差够大，
                # 说明 100% 是灰度盲区（黑白/棕褐变体），不是真的同一张。
                is_color_variant = (
                    sim == 100.0
                    and a["sha256"] != b["sha256"]
                    and (cd >= COLOR_VARIANT_THRESHOLD
                         or a["is_grayscale"] != b["is_grayscale"])
                )
                similar_pairs.append({
                    "a": a["id"], "b": b["id"], "similarity": sim,
                    "color_diff": cd,
                    "color_variant": is_color_variant,
                    "color_note": ("色彩变体（黑白/滤镜）"
                                   if is_color_variant else None),
                })
    similar_pairs.sort(key=lambda p: -p["similarity"])
    parent = {im["id"]: im["id"] for im in pool}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for p in similar_pairs:
        ra, rb = find(p["a"]), find(p["b"])
        if ra != rb:
            parent[rb] = ra
    comp = defaultdict(list)
    for im in pool:
        comp[find(im["id"])].append(im["id"])
    similar_groups = []
    for members in comp.values():
        if len(members) > 1:
            members = sorted(members)
            pairs = [pp for pp in similar_pairs
                     if pp["a"] in members and pp["b"] in members]
            similar_groups.append({"members": members, "pairs": pairs})
    similar_groups.sort(key=lambda g: g["members"])

    # ---- 人脸分组（诚实处理）----
    # 顺序：特征向量聚类（可用且检出分组时）> 测试分组兜底 > 无分组。
    # Haar 只做可行性探测，检出数不用于身份分组。
    haar_count, haar_available = try_haar(ok, root)
    face_attempt = {"face_recognition_available": False,
                    "embedding_available": False,
                    "yunet_available": False,
                    "sface_available": False,
                    "models_present": False,
                    "dlib_available": False,
                    "insightface_available": False,
                    "opencv_available": haar_available,
                    "detail": "未尝试特征向量聚类（face_clustering 模块不可用）。"}
    embedding_groups = []
    try:
        import face_clustering
        face_attempt = face_clustering.attempt_report()
        # 3.2.0：dlib 路线已彻底放弃；只有 YuNet+SFace 特征向量
        # 路线可用（embedding_available）时才尝试真实聚类。
        if face_attempt.get("embedding_available"):
            embedding_groups = face_clustering.cluster_faces(ok, root)
    except Exception as e:
        face_attempt = {"face_recognition_available": False,
                        "embedding_available": False,
                        "yunet_available": False,
                        "sface_available": False,
                        "models_present": False,
                        "dlib_available": False,
                        "insightface_available": False,
                        "opencv_available": haar_available,
                        "detail": "特征向量聚类尝试失败：%s" % e}
    manifest = {}
    if manifest_path and os.path.exists(manifest_path):
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f).get("groups", {})
    id_set = {im["id"] for im in ok}
    face_groups = []
    if embedding_groups:
        face_groups = embedding_groups
        face_method = "embedding_dbscan"
        face_note = ("真实人脸特征向量聚类（YuNet 检测 + SFace 特征向量 + "
                     "余弦距离 DBSCAN 聚类），非 Haar 检测冒充。")
    elif manifest:
        for gid in sorted(manifest):
            info = manifest[gid]
            members = [m for m in info["members"]
                       if m in id_set and m not in rep_of_exact]
            if members:
                face_groups.append({
                    "id": gid,
                    "label": info["label"],
                    "method": "test_grouping",
                    "note": "测试分组（非真实人脸识别）：生成时已知的人像身份，"
                            "仅用于演练分组逻辑。",
                    "members": sorted(members),
                })
        face_method = "test_grouping"
        face_note = ("特征向量聚类不可用（见 face_attempt），故身份分组改用"
                     "测试分组兜底；Haar 只能检测不能认人，未用于身份分组。")
    else:
        face_method = "none"
        face_note = ("特征向量聚类不可用，且无测试分组 sidecar，"
                     "故无人脸分组；Haar 只能检测不能认人，未用于身份分组。")

    # ---- P1-1 推荐保留：多维评分 ----
    # 顺序：像素总面积最大 > EXIF 完整 > 彩色优先（P1-2）> 文件名干净 >
    # 文件体积最大 > 文件名（决胜，避免不稳定排序）。
    def pick_keep(members):
        def key(m):
            im = by_id[m]
            return (-im["pixels"],
                    -int(im["has_exif"]),
                    -color_score_of(im["rgb_mean"]),
                    -int(is_clean_name(im["filename"])),
                    -im["size_bytes"],
                    m)
        return sorted(members, key=key)[0]

    recommendations = []
    reclaimable = 0
    for g in exact_groups:
        keep = pick_keep(g)
        discard = [m for m in sorted(g) if m != keep]
        recommendations.append({
            "group_type": "exact", "members": sorted(g), "keep": keep,
            "discard": discard,
            "reason": "SHA-256 完全相同，字节级一模一样；按像素/EXIF/"
                      "色彩/文件名干净度/体积多维评分保留最干净的一张。",
        })
        reclaimable += sum(by_id[m]["size_bytes"] for m in discard)
    for g in similar_groups:
        members = g["members"]
        if len(set(by_id[m]["sha256"] for m in members)) == 1:
            continue
        keep = pick_keep(members)
        discard = []
        for m in members:
            if m == keep:
                continue
            sim = similarity(by_id[m]["_dhash_int"],
                             by_id[keep]["_dhash_int"])
            cd = color_distance(by_id[m]["rgb_mean"],
                                by_id[keep]["rgb_mean"])
            discard.append({"id": m, "similarity_to_keep": sim,
                            "color_diff_to_keep": cd})
        recommendations.append({
            "group_type": "similar", "members": members, "keep": keep,
            "discard": discard,
            "reason": "感知哈希相似度高，按像素/EXIF/色彩/文件名干净度/"
                      "体积多维评分保留最佳一张，其余进待删确认篓。",
        })
        reclaimable += sum(by_id[d["id"]]["size_bytes"] for d in discard)
    for g in burst_groups:
        recommendations.append({
            "group_type": "burst_protected", "members": g, "keep": None,
            "discard": [],
            "reason": "连拍组默认保护，不建议删除；仅分组展示，不进待删篓。",
        })

    for im in images:
        im.pop("_dhash_int", None)
        im.pop("_exif_dt", None)
        if im.get("is_screenshot"):
            im["source_tag"] = "screenshot"
        elif im["id"] in burst_members:
            im["source_tag"] = "burst"
        elif not im.get("has_exif", True):
            im["source_tag"] = "no_exif"
        else:
            im["source_tag"] = "dated"

    result = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_dir": os.path.abspath(root),
        "similar_threshold": similar_threshold,
        "similar_scope_note": "疑似重复分组只在候选池内计算：排除精确重复副本、"
        "连拍组成员与截图；池内用并查集合并（池已干净，不再链式误并）。",
        "burst_rule": "连拍窗口 0~5 秒；簇内排除文件名变体后缀与灰度图；"
        "取众数分辨率；链式判定（成员与链内前一张相邻帧 dHash 相似度 "
        ">= 80 即可接续，< 80 断链另起新链），每条链 >= 3 张成组。"
        "（棕褐变体不靠单图排除，改由 P1-2 成对 RGB 均值差标注。）",
        "recommendation_rule": "多维评分：像素总面积 > EXIF 完整度 > 彩色优先 > "
        "文件名干净度 > 文件体积 > 文件名。",
        "color_variant_threshold": COLOR_VARIANT_THRESHOLD,
        "face_method": face_method,
        "face_note": face_note,
        "face_attempt": face_attempt,
        "haar_available": haar_available,
        "haar_faces_detected": haar_count,
        "thumbnail_note": "缩略图改走 /api/thumb 动态接口加懒加载，"
        "扫描响应里不再内嵌 Base64 缩略图，大图库不再因此变大变慢。",
        "stats": {
            "total": len(ok),
            "errors": len(images) - len(ok),
            "exact_groups": len(exact_groups),
            "exact_extra_copies": sum(len(g) - 1 for g in exact_groups),
            "similar_groups": len(similar_groups),
            "burst_groups": len(burst_groups),
            "screenshots": len(screenshot_ids),
            "no_exif": len(no_exif_ids),
            "dated": sum(1 for im in ok if im["has_exif"]),
            "total_bytes": sum(im["size_bytes"] for im in ok),
            "reclaimable_bytes": reclaimable,
        },
        "images": sorted(images, key=lambda im: im["id"]),
        "exact_groups": exact_groups,
        "similar_groups": similar_groups,
        "similar_pairs": sorted(similar_pairs,
                                key=lambda p: -p["similarity"]),
        "burst_groups": burst_groups,
        "screenshot_ids": screenshot_ids,
        "no_exif_ids": no_exif_ids,
        "face_groups": face_groups,
        "recommendations": recommendations,
    }
    return result


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="要扫描的图片文件夹")
    ap.add_argument("--output", default="scan_result.json")
    ap.add_argument("--similar-threshold", type=float, default=85)
    ap.add_argument("--faces-manifest", default=None)
    args = ap.parse_args(argv)
    manifest = args.faces_manifest
    if manifest is None:
        for cand in (os.path.join(os.path.dirname(
                        os.path.abspath(args.input)), "faces_manifest.json"),
                     os.path.join(args.input, "faces_manifest.json")):
            if os.path.exists(cand):
                manifest = cand
                break
    result = scan(args.input, args.similar_threshold, manifest)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    s = result["stats"]
    print("images:", s["total"], "errors:", s["errors"])
    print("exact_groups:", s["exact_groups"],
          "extra_copies:", s["exact_extra_copies"])
    print("similar_groups:", s["similar_groups"])
    print("burst_groups:", s["burst_groups"])
    print("screenshots:", s["screenshots"], "no_exif:", s["no_exif"])
    print("face_method:", result["face_method"],
          "haar_available:", result["haar_available"],
          "haar_faces:", result["haar_faces_detected"])
    print("wrote:", os.path.abspath(args.output))
    return 0


if __name__ == "__main__":
    sys.exit(main())
