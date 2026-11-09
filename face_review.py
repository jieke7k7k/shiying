#!/usr/bin/env python3
"""复核表（3.4.0）：单脸状态、三卡片分流、批量事务、持久撤销、人名落盘。

设计依据（2026-11-04 定版 + Argon 架构建议，逐条落地）：

  1. 已确认人物与待复核单脸分开：只有人工拍板（采纳/分到某人/新建人物）
     的单脸才写持久记录并移出复核表；忽略的脸写 ignored 状态。
  2. 单脸主键 = 照片内容哈希前 16 位 + 脸序号（face_key），同一份内容
     的多张重复照片共享同一单脸身份，只复核一次；框、检测分、与各组的
     相似度都在记录里，128 维特征落在独立的缓存文件（不塞进扫描 JSON）。
  3. 打分不用组平均中心：用组内已有成员的最近邻（前 3 名均值），
     头像用组内代表脸（检测分高、框大者）。
  4. 三卡片阈值 0.60 / 0.40；Top1 与 Top2 差 < 0.06 时即使 >= 0.60 也
     降级进“可疑”，并同展两个候选；单脸孤立簇不会因为跟“自己”相似
     而进建议（算分时先把自己排除），只能按与其他组的最高分分流。
  5. 撤销走单个本机数据文件里的“人物 + 每张脸当前状态 + 操作历史”；
     批量操作记一条含多项快照的事务，可整批撤销，也可单脸再改；撤销
     旧记录时若该脸后来又被改过，先返回冲突让用户处理最新记录，不静默
     覆盖；新建人物被撤销时，仅当名下已无脸且没被改名才清理。
  6. 人名从 localStorage 挪到本机数据文件（这里选：与本文件的撤销记录
     同一个文件 face_review.json，人物记录的 name 字段即人名），首次
     加载一次性迁移旧名，本机人工命名优先，冲突按已确认脸更多的一方。

不含任何删除逻辑：复核操作只改这张脸的归属状态，绝不删照片，也绝不
碰删照片的二次确认与待删篓（那是 trash_util / /api/delete 那一套）。
"""
import hashlib
import io
import json
import math
import os
import threading
from datetime import datetime

STORE_VERSION = 1
SUGGEST_MIN = 0.60          # 建议：与某人相似度 >= 0.60
DOUBT_MIN = 0.40            # 可疑：0.40 ~ 0.60；未匹配：< 0.40
GAP_DEGRADE = 0.06          # Top1 与 Top2 差小于此值即降级进可疑
TOP_K = 3                   # 组内最近邻取前几名均值
MAX_ROWS = 2000             # 单次返回上限，防超大图库把响应撑爆
MAX_HISTORY = 60            # 历史最多保留条数
EMBED_DIM = 128

_STORE_LOCK = threading.RLock()


def now_iso():
    return datetime.now().isoformat(timespec="seconds")


def empty_store():
    return {
        "version": STORE_VERSION,
        "updated_at": now_iso(),
        "persons": {},
        "faces": {},
        "history": [],
        "legacy_names": {},
        "migration": {},
    }


# ---------------------------------------------------------------- 存取

def load_store(path):
    store = empty_store()
    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
    except Exception:
        return store
    if not isinstance(raw, dict):
        return store
    for key in ("persons", "faces", "legacy_names", "migration"):
        if isinstance(raw.get(key), dict):
            store[key] = raw[key]
    if isinstance(raw.get("history"), list):
        store["history"] = raw["history"]
    store["version"] = raw.get("version") or STORE_VERSION
    return store


def save_store(path, store):
    with _STORE_LOCK:
        store["updated_at"] = now_iso()
        try:
            parent = os.path.dirname(os.path.abspath(path))
            if parent:
                os.makedirs(parent, exist_ok=True)
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(store, f, ensure_ascii=False, indent=1)
            os.replace(tmp, path)
            return True
        except Exception:
            return False


# ---------------------------------------------------------------- 向量与打分

def cosine(a, b):
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    s = 0.0
    for i in range(n):
        s += float(a[i]) * float(b[i])
    return s


def topk_mean(vec, member_vecs, k=TOP_K):
    """组内最近邻打分：取与成员相似度最高的前 k 名求均值。

    不用组平均中心——平均中心会被组内长相差异拉偏，最近邻更贴近
    “这张脸像组里哪个人”的直觉。
    """
    sims = sorted((cosine(vec, mv) for mv in member_vecs), reverse=True)
    if not sims:
        return 0.0
    top = sims[:max(1, k)]
    return round(sum(top) / len(top), 4)


# ---------------------------------------------------------------- 扫描同步

def _new_face_entry(face):
    return {
        "state": "pending",
        "person_id": None,
        "sha256": face.get("sha256", ""),
        "index": int(face.get("index") or 0),
        "image_id": face.get("image_id") or "",
        "relative_path": face.get("relative_path") or "",
        "box": face.get("box") or [0, 0, 0, 0],
        "det_score": face.get("det_score") or 0.0,
        "image_w": face.get("image_w") or 0,
        "image_h": face.get("image_h") or 0,
        "first_seen_at": now_iso(),
        "last_seen_at": now_iso(),
        "updated_at": now_iso(),
        "last_txn": None,
    }


def sync_scan(store, faces):
    """把本次扫描看到的单脸登记进 store；顺手清掉已消失的 pending 记录。

    已拍板（person）与已忽略（ignored）的记录永远保留——它们是人工
    结论，重扫后必须继续生效；pending 只是“还没拍板”，扫描结果本身
    就能推导，消失的清掉即可，文件不会无限长大。
    """
    seen = set()
    for face in faces or []:
        key = face.get("key")
        if not key:
            continue
        seen.add(key)
        entry = store["faces"].get(key)
        if entry is None:
            store["faces"][key] = _new_face_entry(face)
            continue
        for field in ("sha256", "image_id", "relative_path", "box",
                      "det_score", "image_w", "image_h"):
            if face.get(field) not in (None, ""):
                entry[field] = face[field]
        entry["last_seen_at"] = now_iso()
    for key in list(store["faces"].keys()):
        entry = store["faces"][key]
        if entry.get("state") == "pending" and key not in seen:
            del store["faces"][key]
    return store


def ignored_keys(store):
    return {k for k, v in store["faces"].items()
            if v.get("state") == "ignored"}


def confirmed_by_person(store):
    out = {}
    for key, entry in store["faces"].items():
        if entry.get("state") == "person" and entry.get("person_id"):
            out.setdefault(entry["person_id"], []).append(key)
    return out


# ---------------------------------------------------------------- 三卡片分流

def _person_label(person):
    return person.get("name") or ("人物 %s" % person.get("anchor_key", "")[:6])


def _score_candidates(face_key_value, vectors, persons, confirmed, clusters,
                      faces_by_key):
    """算这张待复核脸与每个“可归属目标”的相似度。

    目标有两类：
      - 已确认人物（有 >= 1 张已拍板的脸）→ 只跟已确认成员比；
      - 还没人拍板的自动簇 → 跟簇内其他待复核脸比（排除自己，所以
        孤立簇算不出分，也就不会因为“跟自己 1.0 相似”进建议）。
    已经被某个簇覆盖的人物优先：目标成员有交集时丢掉簇候选，避免
    “同一个人既是人物又是簇”重复出现两个候选。
    """
    candidates = []
    person_members = set()
    for pid, members in confirmed.items():
        member_vecs = [vectors[k] for k in members if k in vectors]
        if not member_vecs:
            continue
        person_members.update(members)
        person = persons.get(pid) or {"id": pid, "name": ""}
        candidates.append({
            "target_id": pid,
            "kind": "person",
            "label": _person_label(person),
            "score": topk_mean(vectors.get(face_key_value), member_vecs),
            "member_count": len(member_vecs),
        })
    for cluster in clusters:
        members = [k for k in cluster.get("member_keys", [])
                   if k != face_key_value and k in vectors]
        if not members:
            continue
        # 与某个人物的已确认成员有交集 -> 这个人已经代表该簇，跳过。
        if person_members.intersection(cluster.get("member_keys", [])):
            continue
        candidates.append({
            "target_id": cluster["cluster_id"],
            "kind": "cluster",
            "label": "簇 %s（未命名）" % cluster["cluster_id"][3:],
            "score": topk_mean(vectors.get(face_key_value),
                               [vectors[k] for k in members]),
            "member_count": len(members),
            "anchor_key": cluster.get("anchor_key"),
        })
    candidates.sort(key=lambda c: (-c["score"], c["target_id"]))
    return candidates


def classify(candidates):
    """按 0.60 / 0.40 分三档；Top1 与 Top2 差 < 0.06 时降级为可疑。"""
    if not candidates:
        return "unmatched", None, None
    best = candidates[0]
    second = candidates[1] if len(candidates) > 1 else None
    gap_small = bool(second and (best["score"] - second["score"]) < GAP_DEGRADE)
    if best["score"] >= SUGGEST_MIN:
        if gap_small:
            return "doubtful", best, second
        return "suggested", best, None
    if best["score"] >= DOUBT_MIN:
        return "doubtful", best, (second if gap_small else None)
    return "unmatched", best, None


def build_review(store, result, vectors, note_extra=None):
    """把扫描结果 + 本机记录拼成复核表（三张卡片 + 人物 + 计数）。"""
    faces = result.get("face_faces") or []
    clusters = result.get("face_clusters") or []
    images = {im["id"]: im for im in result.get("images") or []
              if "error" not in im}
    faces_by_key = {f["key"]: f for f in faces}
    persons = store["persons"]
    confirmed = confirmed_by_person(store)
    sha_count = {}
    for im in images.values():
        sha = im.get("sha256")
        if sha:
            sha_count[sha] = sha_count.get(sha, 0) + 1

    cards = {"suggested": [], "doubtful": [], "unmatched": []}
    pending = 0
    seen_keys = set()  # P0-1 兜底：同一单脸主键只出一次卡片行
    for face in faces:
        key = face["key"]
        if key in seen_keys:
            continue
        seen_keys.add(key)
        entry = store["faces"].get(key) or {}
        state = entry.get("state") or "pending"
        if state != "pending":
            continue  # 已拍板 / 已忽略的脸不进复核表
        pending += 1
        image = images.get(face.get("image_id")) or {}
        candidates = _score_candidates(key, vectors, persons, confirmed,
                                       clusters, faces_by_key)
        bucket, best, second = classify(candidates)
        row = {
            "key": key,
            "sha256": face.get("sha256"),
            "image_id": face.get("image_id"),
            "relative_path": face.get("relative_path"),
            "filename": image.get("filename") or face.get("image_id"),
            "exif_time": image.get("exif_time"),
            "index": int(face.get("index") or 0),
            "index_label": "第 %d 张脸" % (int(face.get("index") or 0) + 1),
            "box": face.get("box"),
            "det_score": face.get("det_score"),
            "image_w": face.get("image_w"),
            "image_h": face.get("image_h"),
            "bucket": bucket,
            "best": best,
            "second": second,
            "candidates": candidates[:5],
            "dup_photos": sha_count.get(face.get("sha256"), 1),
            "image_ids": list(face.get("image_ids")
                              or [face.get("image_id")]),
            "has_crop": bool(face.get("box")),
        }
        cards[bucket].append(row)

    for bucket in cards:
        cards[bucket].sort(key=lambda r: (
            -(r["best"]["score"] if r["best"] else -1.0), r["filename"], r["index"]))
        if len(cards[bucket]) > MAX_ROWS:
            cards[bucket] = cards[bucket][:MAX_ROWS]

    # P0-2：已确认的人名要回到前端，画廊卡片与顶部“按人名搜照片”才找得到。
    sha_images = {}
    for im in images.values():
        sha = im.get("sha256")
        if sha:
            sha_images.setdefault(sha, []).append(im["id"])
    confirmed_faces = []
    for face in faces:
        key = face["key"]
        entry = store["faces"].get(key) or {}
        if (entry.get("state") or "pending") != "person":
            continue
        pid = entry.get("person_id")
        if not pid:
            continue
        for img_id in sha_images.get(face.get("sha256"), []):
            confirmed_faces.append({"key": key, "image_id": img_id,
                                    "person_id": pid})

    person_list = []
    for pid, person in persons.items():
        members = confirmed.get(pid, [])
        photos = {store["faces"][k].get("sha256") for k in members}
        # P0-2：人物名下全部照片 id（同内容重复照片也算在内），供点人物看照片用。
        person_image_ids = sorted({img_id for sha in photos if sha
                                   for img_id in sha_images.get(sha, [])})
        avatar = None
        for key in members:
            if key in faces_by_key:
                avatar = key
                break
        if avatar is None and members:
            avatar = members[0]
        person_list.append({
            "id": pid,
            "name": person.get("name") or "",
            "source": person.get("source") or "manual",
            "anchor_key": person.get("anchor_key"),
            "face_count": len(members),
            "photo_count": len([p for p in photos if p]),
            "image_ids": person_image_ids,
            "avatar_key": avatar,
            "created_at": person.get("created_at"),
        })
    person_list.sort(key=lambda p: (-p["face_count"], p["name"], p["id"]))

    ignored_total = sum(1 for v in store["faces"].values()
                        if v.get("state") == "ignored")
    note = ("打分规则：组内最近邻（前 %d 名均值），建议 ≥ %.2f、可疑 %.2f–%.2f、"
            "未匹配 < %.2f；Top1 与 Top2 差 < %.2f 时即使够 0.60 也进可疑，"
            "并同展两个候选。忽略只移出复核表，绝不删照片。"
            % (TOP_K, SUGGEST_MIN, DOUBT_MIN, SUGGEST_MIN, DOUBT_MIN,
               GAP_DEGRADE))
    return {
        "ok": True,
        "generated_at": now_iso(),
        "cards": cards,
        "counts": {
            "suggested": len(cards["suggested"]),
            "doubtful": len(cards["doubtful"]),
            "unmatched": len(cards["unmatched"]),
            "pending": pending,
            "ignored": ignored_total,
            "persons": len(person_list),
        },
        "persons": person_list,
        "confirmed_faces": confirmed_faces,
        "clusters": [{"cluster_id": c["cluster_id"], "size": c.get("size"),
                      "image_ids": c.get("image_ids")} for c in clusters],
        "thresholds": {"suggest": SUGGEST_MIN, "doubt": DOUBT_MIN,
                       "gap": GAP_DEGRADE, "top_k": TOP_K},
        "face_method": result.get("face_method"),
        "note": note,
        "extra_note": note_extra or result.get("face_review_note") or "",
        "migration": store.get("migration") or {},
    }


# ---------------------------------------------------------------- 人物

def _random_person_id(store):
    for _ in range(200):
        pid = "p_" + hashlib.sha1(os.urandom(16)).hexdigest()[:8]
        if pid not in store["persons"]:
            return pid
    return "p_%08x" % (len(store["persons"]) + 1)


def _legacy_name_for_cluster(store, cluster, result):
    """给新人物找旧名（迁移自 localStorage 的浏览器旧名）。

    旧名是按“旧分组 id”存的，新的簇用锚点哈希，两者不一定同名，所以按
    三档从宽到窄找：簇 id → 锚点哈希前缀 → 当前扫描里“成员照片有交集”
    的旧分组 id。命中即标记 used_by，同一旧名不会套给两个人物。

    P0-3：一个新簇可能同时命中多个旧名（比如旧分组 A 有 5 张脸叫
    “妈妈”，旧分组 B 有 1 张脸叫“阿姨”），这时不按列表顺序先碰到
    谁就用谁，严格按旧名对应的已确认脸数（旧分组照片数）降序择优，
    脸数相同时才按探查顺序；落败的候选记进迁移冲突备查。
    """
    legacy = store.get("legacy_names") or {}
    anchor = cluster.get("anchor_key") or ""
    probes = [cluster.get("cluster_id"), anchor[1:9] if anchor else None]
    group_sizes = {}
    for group in result.get("face_groups") or []:
        members = set(group.get("members") or [])
        group_sizes[group.get("id")] = len(members)
        if members.intersection(cluster.get("image_ids") or []):
            probes.append(group.get("id"))
    candidates = []  # (脸数, 探查顺序, 旧名记录, 探查键)
    seen_probes = set()
    for order, probe in enumerate(probes):
        if not probe or probe in seen_probes:
            continue
        seen_probes.add(probe)
        hit = legacy.get(probe)
        if not hit or not hit.get("name") or hit.get("used_by"):
            continue
        faces = int(hit.get("faces") or 0)
        if probe in group_sizes:
            faces = max(faces, group_sizes[probe])
        candidates.append((faces, -order, hit, probe))
    if not candidates:
        return None, None
    candidates.sort(key=lambda c: (c[0], c[1]), reverse=True)
    best_faces, _neg_order, best_hit, best_probe = candidates[0]
    migration = store.get("migration") or {}
    conflicts = migration.get("conflicts")
    if conflicts is not None and len(candidates) > 1:
        for faces, _o, hit, probe in candidates[1:]:
            conflicts.append({
                "key": probe, "kept": best_hit.get("name"),
                "dropped": hit.get("name"),
                "reason": "同簇多旧名按脸数择优（%d 张对 %d 张）"
                          % (best_faces, faces)})
    return best_hit, best_probe


def _person_for_cluster(store, cluster, txn_id, created, result=None):
    """把自动簇提升成人物：id 用锚点脸哈希前缀，绝不按顺序编号。"""
    anchor = cluster.get("anchor_key") or cluster.get("cluster_id")
    hexpart = anchor[1:9] if anchor and anchor.startswith("f") else "unknown"
    pid = "p_%s" % hexpart[:8]
    base = pid
    n = 2
    while pid in store["persons"]:
        pid = "%s_%d" % (base, n)
        n += 1
    name = ""
    hit, probe = _legacy_name_for_cluster(store, cluster, result or {})
    if hit:
        name = hit["name"]
        hit["used_by"] = pid
        hit["used_for"] = probe
    if not name:
        # 自动簇默认名就是锚点脸的哈希前缀（不是“人物 1/2/3”顺序编号）。
        name = "人物 %s" % hexpart[:6]
    person = {
        "id": pid,
        "name": name,
        "source": "auto",
        "anchor_key": cluster.get("anchor_key"),
        "cluster_id": cluster.get("cluster_id"),
        "created_at": now_iso(),
        "created_txn": txn_id,
        "manual_name": False,
    }
    store["persons"][pid] = person
    created.append({"id": pid, "name": name, "anchor_key": person["anchor_key"],
                    "cluster_id": person["cluster_id"]})
    return pid


def _find_cluster(clusters, target_id):
    for cluster in clusters or []:
        if cluster.get("cluster_id") == target_id:
            return cluster
    return None


def _apply(store, key, state, person_id, txn_id, items):
    entry = store["faces"].setdefault(key, _new_face_entry({}))
    before = {"state": entry.get("state") or "pending",
              "person_id": entry.get("person_id")}
    after = {"state": state, "person_id": person_id}
    if before == after:
        return None
    entry["state"] = state
    entry["person_id"] = person_id
    entry["updated_at"] = now_iso()
    entry["last_txn"] = txn_id
    item = {"key": key, "before": before, "after": after}
    items.append(item)
    return item


ACTION_TITLES = {
    "adopt": "采纳建议",
    "assign": "分到指定人物",
    "new_person": "新建人物并分入",
    "ignore": "忽略这张脸",
}


def apply_action(store, action, keys, result, vectors, person_id=None,
                 name=None):
    """执行一次复核操作（单脸或批量），整体记为一条事务。

    返回 {"ok", "txn", "created_persons"}；参数错误抛 ValueError。
    """
    keys = [str(k) for k in (keys or []) if k]
    if action not in ACTION_TITLES and action != "rename":
        raise ValueError("未知操作：%s" % action)
    if not keys and action != "rename":
        raise ValueError("没有选中任何脸")
    txn_id = "t%d-%s" % (int(datetime.now().timestamp() * 1000),
                         hashlib.sha1(os.urandom(8)).hexdigest()[:4])
    clusters = result.get("face_clusters") or []
    faces = {f["key"]: f for f in (result.get("face_faces") or [])}
    persons = store["persons"]
    created = []
    items = []

    if action == "rename":
        if not person_id or person_id not in persons:
            raise ValueError("人物不存在")
        person = persons[person_id]
        before = {"name": person.get("name") or "",
                  "manual_name": bool(person.get("manual_name"))}
        person["name"] = (name or "").strip()
        person["manual_name"] = True
        person["renamed_at"] = now_iso()
        txn = {"txn_id": txn_id, "at": now_iso(), "action": "rename",
               "title": "改名为「%s」" % (person["name"] or "（清空）"),
               "items": [], "renames": [{"person_id": person_id,
                                         "before": before,
                                         "after": {"name": person["name"],
                                                   "manual_name": True}}],
               "persons_created": [], "undone": False}
        _push_history(store, txn)
        return {"ok": True, "txn": txn, "created_persons": []}

    if action == "new_person":
        pid = _random_person_id(store)
        nm = (name or "").strip() or ("新人物 %s" % pid[-4:])
        persons[pid] = {
            "id": pid, "name": nm, "source": "manual", "anchor_key": None,
            "cluster_id": None, "created_at": now_iso(),
            "created_txn": txn_id, "manual_name": True,
        }
        created.append({"id": pid, "name": nm, "anchor_key": None,
                        "cluster_id": None})
        person_id = pid
        action_label = "new_person"
    else:
        action_label = action

    unresolved = []
    for key in keys:
        face = faces.get(key)
        if face is None:
            # 不在本次扫描里（比如换了文件夹）：按已存记录判断是否存在。
            if key not in store["faces"]:
                unresolved.append(key)
                continue
        if action_label == "ignore":
            _apply(store, key, "ignored", None, txn_id, items)
            continue
        if action_label == "assign" or action_label == "new_person":
            if not person_id or person_id not in persons:
                raise ValueError("必须先指定分到哪个人物")
            _apply(store, key, "person", person_id, txn_id, items)
            continue
        # adopt：按这张脸自己的最佳候选落地（簇候选顺带提升为人物）
        candidates = _score_candidates(key, vectors, persons,
                                       confirmed_by_person(store), clusters,
                                       faces)
        target = candidates[0] if candidates else None
        if target is None:
            unresolved.append(key)
            continue
        if target["kind"] == "person":
            _apply(store, key, "person", target["target_id"], txn_id, items)
        else:
            cluster = _find_cluster(clusters, target["target_id"])
            if cluster is None:
                unresolved.append(key)
                continue
            pid = None
            for probe in persons.values():
                if probe.get("cluster_id") == cluster["cluster_id"]:
                    pid = probe["id"]
                    break
            if pid is None:
                pid = _person_for_cluster(store, cluster, txn_id, created, result)
            _apply(store, key, "person", pid, txn_id, items)

    if not items and not created:
        raise ValueError("没有可执行的内容（可能是这些脸已被处理或缺少候选）")

    if action_label == "new_person":
        title = "新建人物「%s」并分入 %d 张脸" % (
            persons[person_id]["name"], len(items))
    elif action_label == "ignore":
        title = "忽略 %d 张脸" % len(items)
    elif action_label == "assign":
        title = "分到「%s」%d 张脸" % (
            _person_label(persons[person_id]), len(items))
    else:
        target_names = sorted({_person_label(persons[i["after"]["person_id"]])
                               for i in items
                               if i["after"].get("person_id") in persons})
        title = "采纳建议 %d 张脸%s" % (
            len(items), ("（%s）" % "、".join(target_names)) if target_names else "")
    if len(keys) > 1:
        title = "批量：" + title
    txn = {"txn_id": txn_id, "at": now_iso(), "action": action_label,
           "title": title, "items": items, "persons_created": created,
           "unresolved": unresolved, "undone": False}
    _push_history(store, txn)
    return {"ok": True, "txn": txn, "created_persons": created,
            "unresolved": unresolved}


def _push_history(store, txn):
    store["history"].insert(0, txn)
    if len(store["history"]) > MAX_HISTORY:
        del store["history"][MAX_HISTORY:]


# ---------------------------------------------------------------- 撤销

def _find_txn(store, txn_id):
    for txn in store["history"]:
        if txn.get("txn_id") == txn_id:
            return txn
    return None


def _mark_key_undone_in_later_txns(store, txn_id, key):
    """P1-1：强行撤销旧记录时，把 key 在其他事务里的那一项同步记成已撤销。

    不这么做的话，后来事务 T2 的 undone 还是 False，可它对这张脸的
    改动已经被强行抹掉——用户再去撤 T2 时会因 last_txn 对不上报 409 死结。
    """
    for other in store.get("history") or []:
        if other.get("txn_id") == txn_id:
            continue
        items = other.get("items") or []
        if not any(it.get("key") == key for it in items):
            continue
        undone = set(other.get("undone_keys") or [])
        if key in undone:
            continue
        undone.add(key)
        other["undone_keys"] = sorted(undone)
        if len(other["undone_keys"]) >= len(items):
            other["undone"] = True
            other["undone_at"] = now_iso()


def undo(store, txn_id, only_keys=None, force=False):
    """撤销一条记录；可整批撤销，也可只撤其中几张脸。

    若某张脸在这条记录之后又被改过（last_txn 变了，或当前状态已经不是
    该记录的“操作后状态”），默认不静默覆盖，返回冲突清单交给前端提示
    用户先处理最新记录；force=True 时才连最新改动一起覆盖。
    """
    txn = _find_txn(store, txn_id)
    if txn is None:
        raise ValueError("找不到这条撤销记录")
    if txn.get("undone"):
        raise ValueError("这条记录已经撤销过了")
    only = set(only_keys or []) or None
    done_before = set(txn.get("undone_keys") or [])
    targets = [it for it in txn.get("items", [])
               if it.get("key") not in done_before
               and (only is None or it.get("key") in only)]
    renames = (txn.get("renames") or []) if not done_before else []
    if not targets and not renames:
        raise ValueError("这条记录里选中的部分已经撤销过了")

    conflicts = []
    for item in targets:
        entry = store["faces"].get(item["key"])
        if entry is None:
            conflicts.append({"key": item["key"], "reason": "记录已不存在"})
            continue
        if entry.get("last_txn") != txn_id:
            conflicts.append({"key": item["key"],
                              "reason": "这张脸之后又被改过",
                              "latest_txn": entry.get("last_txn")})
            continue
        cur = {"state": entry.get("state") or "pending",
               "person_id": entry.get("person_id")}
        if cur != item["after"]:
            conflicts.append({"key": item["key"],
                              "reason": "当前状态已与操作时不同",
                              "latest_txn": entry.get("last_txn")})
    if conflicts and not force:
        return {"ok": False, "conflict": True, "txn_id": txn_id,
                "conflicts": conflicts,
                "message": "有 %d 张脸在这条记录之后又被改过，"
                           "请先处理最新记录，或选择只撤销未被改过的部分。"
                           % len(conflicts)}

    restored = 0
    # force=True 是用户明确选择“连最新改动一起覆盖”，这时不能再按冲突跳过。
    blocked = set() if force else {c["key"] for c in conflicts if c.get("key")}
    for item in targets:
        if item["key"] in blocked:
            continue
        entry = store["faces"].get(item["key"])
        if entry is None:
            continue
        before = item["before"]
        pid = before.get("person_id")
        if pid and pid not in store["persons"]:
            pid = None
            before = {"state": "pending", "person_id": None}
        entry["state"] = before.get("state") or "pending"
        entry["person_id"] = pid
        entry["updated_at"] = now_iso()
        entry["last_txn"] = None
        restored += 1
        if force:
            _mark_key_undone_in_later_txns(store, txn_id, item["key"])

    for rename in renames:
        person = store["persons"].get(rename.get("person_id"))
        if not person:
            continue
        if person.get("name") != rename["after"].get("name"):
            conflicts.append({"person_id": rename.get("person_id"),
                              "reason": "名字之后又被改过"})
            if not force:
                continue
        person["name"] = rename["before"].get("name") or ""
        person["manual_name"] = bool(rename["before"].get("manual_name"))

    removed_persons = []
    for person in txn.get("persons_created") or []:
        pid = person.get("id")
        current = store["persons"].get(pid)
        if not current:
            continue
        still_has = any(v.get("person_id") == pid and v.get("state") == "person"
                        for v in store["faces"].values())
        renamed = bool(current.get("manual_name")) and (
            current.get("name") != (person.get("name") or ""))
        if still_has:
            continue
        if renamed:
            # 被改过名的人物不清理：留着名字比删掉更重要。
            continue
        del store["persons"][pid]
        removed_persons.append(pid)
        # 人名迁移来的旧名如果套在这个人物身上，人物被清掉时要把旧名释放，
        # 否则这把旧名会被永久“占用”，再也套不到别的人物上。
        for entry in (store.get("legacy_names") or {}).values():
            if entry.get("used_by") == pid:
                entry["used_by"] = None
                entry.pop("used_for", None)

    done_keys = {it["key"] for it in targets} - blocked
    txn["undone_keys"] = sorted(done_before | done_keys)
    if len(txn["undone_keys"]) >= len(txn.get("items", [])) and txn.get("items"):
        txn["undone"] = True
        txn["undone_at"] = now_iso()
    return {"ok": True, "txn_id": txn_id, "restored": restored,
            "conflicts": conflicts, "removed_persons": removed_persons,
            "undone": bool(txn.get("undone"))}


def history_view(store, limit=12):
    out = []
    for txn in store["history"][:limit]:
        out.append({
            "txn_id": txn.get("txn_id"),
            "at": txn.get("at"),
            "action": txn.get("action"),
            "title": txn.get("title"),
            "count": len(txn.get("items") or []),
            "undone": bool(txn.get("undone")),
            "undone_at": txn.get("undone_at"),
            "undone_keys": txn.get("undone_keys") or [],
            "keys": [it.get("key") for it in (txn.get("items") or [])],
        })
    return out


# ---------------------------------------------------------------- 人名迁移

def migrate_names(store, mapping, faces_count=None):
    """一次性迁移 localStorage 里的旧人名到本机数据文件。

    - 本机已有的人工命名优先，不被浏览器旧名覆盖；
    - 冲突时以“已确认脸更多”的一方为准（浏览器旧名没有确认脸，
      计 0，所以人工命名天然胜出，冲突都记进 migration.conflicts 备查）；
    - 迁移只做一次，记录 names_imported_at，之后前端不再读 localStorage。
    """
    if store.get("migration", {}).get("names_imported_at"):
        return {"ok": True, "already_done": True,
                "migrated": 0,
                "imported_at": store["migration"]["names_imported_at"]}
    mapping = mapping or {}
    imported = 0
    conflicts = []
    applied = []
    faces_count = faces_count or {}
    for old_key, name in mapping.items():
        if not isinstance(old_key, str) or not isinstance(name, str):
            continue
        name = name.strip()
        if not name:
            continue
        store["legacy_names"][old_key] = {
            "name": name,
            "faces": int(faces_count.get(old_key) or 0),
            "imported_at": now_iso(),
            "used_by": None,
        }
        imported += 1
        person = store["persons"].get(old_key)
        if person is None:
            continue
        if person.get("name") and person.get("manual_name"):
            # P0-3：按已确认脸数裁决——旧名只是浏览器里没确认过的名字，
            # 确认脸数按 0 计；本机人工命名的确认脸数 ≥ 0，自然胜出。
            # 脸数要真的算出来、写进冲突记录，不能只凭一句话默认。
            local_faces = sum(
                1 for v in store["faces"].values()
                if v.get("state") == "person"
                and v.get("person_id") == person.get("id"))
            conflicts.append({"key": old_key, "kept": person.get("name"),
                              "dropped": name,
                              "reason": "本机人工命名优先（已确认 %d 张脸，"
                                        "旧名确认 0 张）" % local_faces})
            continue
        person["name"] = name
        person["manual_name"] = True
        store["legacy_names"][old_key]["used_by"] = person["id"]
        applied.append({"key": old_key, "person_id": person["id"],
                        "name": name})
    store["migration"] = {
        "names_imported_at": now_iso(),
        "migrated": imported,
        "applied": applied,
        "conflicts": conflicts,
        "target_file": "face_review.json",
        "note": "人名已从浏览器 localStorage 迁到本机数据文件；"
                "之后不再从 localStorage 读取。",
    }
    return {"ok": True, "already_done": False, "migrated": imported,
            "applied": applied, "conflicts": conflicts,
            "imported_at": store["migration"]["names_imported_at"]}


# ---------------------------------------------------------------- 裁剪

def crop_face(image_path, box, size=168, margin=0.28, border=0.0):
    """按脸框裁小图：外扩留出五官余地，内容不贴边。

    四周的留白边不再焊死在 JPEG 里（纯白 (255,255,255) 在深色模式下
    会刺眼）：border 默认 0，留白由前端 .revcrop 的 CSS padding 按
    当前主题给（浅色 #FAFAFA、深色 #141414）。margin 外扩本身已保证
    脸框完整落在裁剪矩形内，见 face_fully_inside 校验。

    返回 (jpeg_bytes, meta)；meta 里带 crop_rect 与 inside 校验结果，
    自测用它证明“裁图逻辑完整”（脸框必须完整落在裁剪矩形内）。
    """
    from PIL import Image
    with Image.open(image_path) as im:
        img = im.convert("RGB")
        W, H = img.size
        try:
            x, y, w, h = [float(v) for v in box]
        except Exception:
            x, y, w, h = 0.0, 0.0, float(W), float(H)
        if w <= 1 or h <= 1:
            x, y, w, h = 0.0, 0.0, float(W), float(H)
        cx, cy = x + w / 2.0, y + h / 2.0
        side = max(w, h) * (1.0 + 2.0 * margin)
        left = max(0, int(round(cx - side / 2.0)))
        top = max(0, int(round(cy - side / 2.0)))
        right = min(W, int(round(cx + side / 2.0)))
        bottom = min(H, int(round(cy + side / 2.0)))
        if right - left < 4 or bottom - top < 4:
            left, top, right, bottom = 0, 0, W, H
        crop = img.crop((left, top, right, bottom))
        cv_w, cv_h = crop.size
        pad = max(1, int(round(max(cv_w, cv_h) * border)))
        canvas = Image.new("RGB", (cv_w + pad * 2, cv_h + pad * 2),
                           (255, 255, 255))
        canvas.paste(crop, (pad, pad))
        canvas.thumbnail((size, size), Image.LANCZOS)
        buf = io.BytesIO()
        canvas.save(buf, "JPEG", quality=86)
        inside = (left <= x and top <= y
                  and right >= x + w and bottom >= y + h)
        meta = {"crop_rect": [left, top, right, bottom],
                "pad": pad, "face_box": [x, y, w, h],
                "source_size": [W, H], "out_size": list(canvas.size),
                "face_fully_inside": bool(inside),
                "clamped": not inside}
        return buf.getvalue(), meta


def image_path_for(root, relative_path):
    if not root or not relative_path:
        return None
    return os.path.join(root, relative_path)


# ---------------------------------------------------------------- 演示假数据

_FIXTURE_PLAN = (
    # (文件名, 身份, 框（相对宽高的比例）, 检测分)
    # 演示图是 800x600 合成图：A/B 两组各有一张“人像”和一张“场景里的脸”，
    # 这里的框按生成器里 draw_face 的实际坐标取，裁剪出来的确是那张脸；
    # C（树冠）、D（楼窗）、E（盘子）故意是不像人的圆形物体，用来演示
    # “误检进可疑 / 未匹配”这条真实业务路径（假数据，仅演练界面）。
    ("base_park.jpg", "A", (0.1375, 0.3833, 0.1750, 0.2333), 0.94),
    ("base_park.jpg", "C", (0.3750, 0.2500, 0.2375, 0.3000), 0.88),
    ("portrait_a2.jpg", "A", (0.2375, 0.2833, 0.2750, 0.3667), 0.96),
    ("base_beach.jpg", "B", (0.1625, 0.3000, 0.1750, 0.2333), 0.93),
    ("portrait_b2.jpg", "B", (0.5438, 0.3583, 0.2125, 0.2833), 0.95),
    ("base_city.jpg", "D", (0.3000, 0.4000, 0.1125, 0.1500), 0.87),
    ("base_food.jpg", "E", (0.4125, 0.3833, 0.1750, 0.2333), 0.90),
)
_FIXTURE_TARGETS = {
    "C": (0.47, 0.445),
    "D": (0.28, 0.30),
    "E": (0.33, 0.20),
}


def _det_unit_vector(seed, dim=EMBED_DIM):
    """由种子确定地生成单位向量（不依赖随机数状态，重跑结果一致）。"""
    vals = []
    counter = 0
    while len(vals) < dim:
        digest = hashlib.sha256(("%s|%d" % (seed, counter)).encode()).digest()
        for i in range(0, len(digest), 4):
            if len(vals) >= dim:
                break
            raw = int.from_bytes(digest[i:i + 4], "big") / 4294967295.0
            vals.append(raw * 2.0 - 1.0)
        counter += 1
    norm = math.sqrt(sum(v * v for v in vals)) or 1.0
    return [v / norm for v in vals]


def _orthogonalize(vec, basis):
    out = list(vec)
    for b in basis:
        dot = sum(x * y for x, y in zip(out, b))
        out = [x - dot * y for x, y in zip(out, b)]
    return out


def _normalize(vec):
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def demo_fixture(images, note="", ignore_keys=None):
    """缺模型时给演示图造一份“假数据”单脸表，只用于演练复核界面。

    绝不冒充真实人脸识别：method 恒为 demo_fixture，界面与报告都会
    明说这是假数据。向量只在内存里用，不写进任何缓存文件，避免污染
    真实特征缓存。
    """
    by_name = {}
    for im in images or []:
        name = im.get("filename") or im.get("relative_path")
        if name:
            by_name[name] = im
    base_a = _det_unit_vector("fixture-identity-A")
    # 让 A、B 两个基准几乎不相似（0.05），贴近“不同人”。
    b_perp = _normalize(_orthogonalize(_det_unit_vector("fixture-identity-B"),
                                       [base_a]))
    base_b = _normalize([b + 0.05 * a for a, b in zip(base_a, b_perp)])
    dot_ab = sum(x * y for x, y in zip(base_a, base_b))

    identity_vec = {
        "A": base_a,
        "B": base_b,
    }
    for ident, (ca, cb) in _FIXTURE_TARGETS.items():
        # 每个“陌生人”用各自的正交轴，避免两个陌生人的向量互相撞车
        # （那样会被误聚成一组，卡片就不对了）。
        z_axis = _normalize(_orthogonalize(_det_unit_vector("fixture-z-" + ident),
                                          [base_a, b_perp]))
        x = (cb - ca * dot_ab) / max(1e-6, math.sqrt(max(0.0, 1 - dot_ab ** 2)))
        y2 = max(0.0, 1.0 - ca * ca - x * x)
        vec = [ca * a + x * b + math.sqrt(y2) * z
               for a, b, z in zip(base_a, b_perp, z_axis)]
        identity_vec[ident] = _normalize(vec)

    ignore_keys = set(ignore_keys or ())
    faces = []
    vectors = {}
    per_image_index = {}
    for name, ident, box_rel, det in _FIXTURE_PLAN:
        im = by_name.get(name)
        if not im:
            continue
        w_img = int(im.get("width") or 0)
        h_img = int(im.get("height") or 0)
        if not w_img or not h_img:
            continue
        x = int(round(box_rel[0] * w_img))
        y = int(round(box_rel[1] * h_img))
        bw = int(round(box_rel[2] * w_img))
        bh = int(round(box_rel[3] * h_img))
        idx = per_image_index.get(name, 0)
        per_image_index[name] = idx + 1
        key = "f%s-%d" % ((im.get("sha256") or "0" * 16)[:16], idx)
        if key in ignore_keys:
            continue  # 已忽略的脸在聚类前就剔除，跟真实链路一致
        jitter = 0.10
        noise = _det_unit_vector("fixture-face-%s" % key)
        vec = _normalize([base * (1.0 - jitter) + n * jitter
                          for base, n in zip(identity_vec[ident], noise)])
        vectors[key] = vec
        faces.append({
            "key": key,
            "sha256": im.get("sha256"),
            "image_id": im.get("id"),
            "relative_path": im.get("relative_path"),
            "index": idx,
            "box": [x, y, bw, bh],
            "det_score": det,
            "image_w": w_img,
            "image_h": h_img,
            "cache_hit": False,
            "fixture_identity": ident,
        })
    # 聚类：同一身份（A/B 各两张）成组，C/D/E 各自孤立。
    clusters = []
    groups = {}
    for face in faces:
        groups.setdefault(face["fixture_identity"], []).append(face)
    for ident in sorted(groups):
        members = groups[ident]
        anchor = sorted(members, key=lambda f: (-f["det_score"], f["key"]))[0]
        clusters.append({
            "cluster_id": "af_%s-%d" % (anchor["key"][1:9], anchor["index"]),
            "anchor_key": anchor["key"],
            "member_keys": sorted(f["key"] for f in members),
            "image_ids": sorted({f["image_id"] for f in members}),
            "size": len(members),
        })
    clusters.sort(key=lambda c: c["cluster_id"])
    return {
        "available": True,
        "method": "demo_fixture",
        "reason": note or ("演示假数据：基线包不含人脸模型，这里用固定比例的"
                           "假脸框与假特征演练复核表（不是真实人脸识别）。"),
        "faces": faces,
        "clusters": clusters,
        "vectors": vectors,
        "stats": {"images_with_faces": len(per_image_index),
                  "faces": len(faces), "ignored": 0,
                  "cache_hits": 0, "cache_misses": 0},
    }
