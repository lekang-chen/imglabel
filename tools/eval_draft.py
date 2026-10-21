# -*- coding: utf-8 -*-
"""
草稿质量评估：用 COCO 官方真值衡量"模型预标注"的质量。

为什么需要这个工具：
  任务书要求回答"修正率是多少、迭代有没有效果"。修正率的分母是
  人工要改的对象总数，而这个数只有拿真值做参照才有意义。
  光看"模型吐了多少个框"是看不出好坏的——多标和漏标都可能。

三个核心指标（都按 IoU=0.5 判定匹配）：
  精确率 P = TP / (TP + FP)   模型标出来的是对的占比（多标少标都扣分）
  召回率 R  = TP / (TP + FN)   真值里被找到的占比（漏检扣分）
  F1        = 2PR / (P + R)

以及两个诊断指标：
  同类重复对  同一类里 IoU>0.6 的对象对数（模型把一个东西标两遍）
  多边形顶点数  每个对象的顶点数是否超过 180 上限

用法：
  python tools/eval_draft.py                # 评估当前 data/annotations 里的草稿
  python tools/eval_draft.py --json out.json  # 同时存成 json 供前端展示
"""
import argparse
import json
import os
import sys
import zipfile
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANNO_DIR = os.path.join(BASE_DIR, "data", "annotations")
COCO_JSON = os.path.join(BASE_DIR, "data", "coco", "instances_val2017.json")
# 我们把 COCO 图片重命名成了 000000.jpg 这种顺序名，真值要靠这张映射表还原
SOURCE_MAP = os.path.join(BASE_DIR, "data", "data_source.txt")

# COCO 的 80 类 id -> 我们的十类名。COCO 的 id 是不连续的，
# 必须靠这张表映射，不能用"我们的 id - 1"去猜。
# 只保留任务书要求的十类，其余类别直接忽略。
COCO_ID_TO_NAME = {
    1: "person",        # person
    62: "chair",        # chair
    67: "dining table", # dining table
    63: "couch",        # couch
    72: "tv",           # tv
    73: "laptop",       # laptop
    47: "cup",          # cup
    44: "bottle",       # bottle
    64: "potted plant", # potted plant
    84: "book",         # book
}

# 任务书规定的十类（不可更改）
OUR_CLASSES = ["person", "chair", "dining table", "couch", "tv",
               "laptop", "cup", "bottle", "potted plant", "book"]

# COCO 里 "dining table" 的 id 是 67，"tv" 是 72，"book" 是 84
COCO_NAME_TO_ID = {
    "person": 1, "bicycle": 2, "car": 3, "motorcycle": 4, "airplane": 5,
    "bus": 6, "train": 7, "truck": 8, "boat": 9, "traffic light": 10,
    "dining table": 67, "tv": 72, "book": 84,
}

IOU_MATCH = 0.5
DUP_IOU = 0.6


def _build_id_map(coco):
    """从 COCO 的 categories 动态构造 id -> 类名 映射。

    为什么不用手写常量：COCO 的 category_id 是不连续的
    （chair=62、cup=47、bottle=44……），手写极易记错，
    记错的表现很隐蔽——真值能加载出来但某一类恒为 0，
    看起来像"模型没检测到"，其实是映射表错了。
    """
    id2name = {}
    for cat in coco.get("categories", []):
        id2name[cat["id"]] = cat["name"]
    return id2name


def _bbox_of_seg(ann):
    """COCO 多边形标注 -> 外接框 [x, y, w, h]"""
    seg = ann.get("segmentation")
    if not seg:
        return ann["bbox"]
    if isinstance(seg, list) and seg and isinstance(seg[0], list):
        xs = seg[0][0::2]
        ys = seg[0][1::2]
        x0, y0 = min(xs), min(ys)
        return [x0, y0, max(xs) - x0, max(ys) - y0]
    return ann["bbox"]


def load_truth(coco_path):
    """读 COCO 真值，返回 {COCO原名: [(cls_name, bbox), ...]}

    bbox 一律换算成"我们压缩后图片的像素坐标"，
    因为我们的多边形是按压缩后尺寸归一化到 0~1 的。
    """
    with open(coco_path, "r", encoding="utf-8") as f:
        coco = json.load(f)
    id2name = _build_id_map(coco)
    id2file = {im["id"]: im["file_name"] for im in coco["images"]}
    id2size = {im["id"]: (im["width"], im["height"]) for im in coco["images"]}
    out = defaultdict(list)
    for ann in coco["annotations"]:
        if ann.get("iscrowd", 0):
            continue
        cname = id2name.get(ann["category_id"])
        if cname not in OUR_CLASSES:
            continue
        fname = id2file.get(ann["image_id"])
        if not fname:
            continue
        w, h = id2size.get(ann["image_id"], (0, 0))
        out[fname].append((cname, _bbox_of_seg(ann), w, h))
    return out


def _iou(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _box_from_poly(poly):
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return [min(xs), min(ys), max(xs), max(ys)]


def load_name_map(source_map):
    """读 data_source.txt 里的对应关系：{我们的名字: COCO 原名}

    数据准备阶段把 COCO 图片重命名成 000000.jpg 这种顺序名，
    评估时必须还原回原名才能对上官方真值。
    """
    mapping = {}
    if not os.path.exists(source_map):
        return mapping
    with open(source_map, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if "<-" not in line:
                continue
            left, right = line.split("<-", 1)
            ours = left.strip()
            theirs = right.strip().split("(")[0].strip()
            if ours and theirs:
                mapping[ours] = theirs
    return mapping


def evaluate(anno_dir, coco_path, source_map=None):
    truth = load_truth(coco_path)
    name_map = load_name_map(source_map) if source_map else {}

    tp = fp = fn = 0
    total_draft = 0
    total_truth = 0
    dup_pairs = 0
    max_pts = 0
    over_limit = 0
    per_class = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "draft": 0, "truth": 0})
    files = sorted(f for f in os.listdir(anno_dir) if f.endswith(".json"))

    for fname in files:
        img_name = os.path.splitext(fname)[0] + ".jpg"
        coco_name = name_map.get(img_name, img_name)
        with open(os.path.join(anno_dir, fname), "r", encoding="utf-8") as f:
            data = json.load(f)
        objs = data.get("objects", [])
        total_draft += len(objs)

        gt = truth.get(coco_name, [])
        total_truth += len(gt)

        # 我们的图片被压缩过（原图长边可能 >960），所以真值 bbox 要按比例缩放
        our_w = data.get("width") or 0
        our_h = data.get("height") or 0
        scale = 1.0
        if gt and our_w and our_h:
            ow, oh = gt[0][2], gt[0][3]
            if ow and oh:
                scale = min(our_w / ow, our_h / oh)

        # 类别级贪心匹配：每个预测只能配一个真值，一个真值只能配一个预测
        used_gt = set()
        matched_pred = set()
        for pi, o in enumerate(objs):
            cname = o.get("class") or o.get("class_name", "")
            poly = o.get("polygon") or []
            if poly:
                # 归一化坐标 -> 我们图片的像素坐标
                pb = _box_from_poly([[p[0] * our_w, p[1] * our_h]
                                     for p in poly]) if our_w and our_h \
                    else _box_from_poly(poly)
                npts = len(poly)
            elif o.get("bbox"):
                x, y, w, h = o["bbox"]
                pb = [x, y, x + w, y + h]
                npts = 0
            else:
                continue
            if npts > max_pts:
                max_pts = npts
            if npts > 180:
                over_limit += 1

            best_j, best_iou = -1, IOU_MATCH
            for gi, (gc, gb, _gw, _gh) in enumerate(gt):
                if gi in used_gt or gc != cname:
                    continue
                gbox = [gb[0] * scale, gb[1] * scale,
                        (gb[0] + gb[2]) * scale, (gb[1] + gb[3]) * scale]
                v = _iou(pb, gbox)
                if v >= best_iou:
                    best_iou, best_j = v, gi
            if best_j >= 0:
                used_gt.add(best_j)
                matched_pred.add(pi)
                tp += 1
                per_class[cname]["tp"] += 1
            else:
                fp += 1
                per_class[cname]["fp"] += 1
            per_class[cname]["draft"] += 1

        miss = len(gt) - len(used_gt)
        fn += miss
        for gi, (gc, _gb, _gw, _gh) in enumerate(gt):
            per_class[gc]["truth"] += 1
            if gi not in used_gt:
                per_class[gc]["fn"] += 1

        # 同类重复诊断
        for i in range(len(objs)):
            for j in range(i + 1, len(objs)):
                oi, oj = objs[i], objs[j]
                if (oi.get("class") or oi.get("class_name")) != (oj.get("class") or oj.get("class_name")):
                    continue
                pi_, pj = oi.get("polygon") or [], oj.get("polygon") or []
                if not pi_ or not pj:
                    continue
                if _iou(_box_from_poly([[p[0] * our_w, p[1] * our_h]
                                        for p in pi_]),
                        _box_from_poly([[p[0] * our_w, p[1] * our_h]
                                        for p in pj])) > DUP_IOU:
                    dup_pairs += 1

    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

    return {
        "files": len(files),
        "total_draft": total_draft,
        "total_truth": total_truth,
        "tp": tp, "fp": fp, "fn": fn,
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "dup_pairs": dup_pairs,
        "max_poly_points": max_pts,
        "over_limit_objects": over_limit,
        "per_class": {k: v for k, v in sorted(per_class.items())},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anno-dir", default=ANNO_DIR)
    ap.add_argument("--coco", default=COCO_JSON)
    ap.add_argument("--source", default=SOURCE_MAP,
                    help="data_source.txt，含我们的文件名 <-> COCO 原名 映射")
    ap.add_argument("--json", default=None, help="额外写出一份 json")
    args = ap.parse_args()

    if not os.path.exists(args.coco):
        print("找不到 COCO 真值:", args.coco)
        print("请先运行 python tools/prepare_data.py --fetch-url 生成数据集")
        sys.exit(1)

    r = evaluate(args.anno_dir, args.coco, args.source)

    print("=" * 58)
    print("草稿质量评估（对照 COCO 官方真值，IoU=%.1f 判定匹配）" % IOU_MATCH)
    print("=" * 58)
    print("图片数        : %d" % r["files"])
    print("草稿对象数    : %d    真值对象数: %d" % (r["total_draft"], r["total_truth"]))
    print("TP=%d  FP=%d  FN=%d" % (r["tp"], r["fp"], r["fn"]))
    print("精确率 P      : %.3f" % r["precision"])
    print("召回率 R      : %.3f" % r["recall"])
    print("F1            : %.3f" % r["f1"])
    print("同类重复对    : %d  (IoU>%.1f)" % (r["dup_pairs"], DUP_IOU))
    print("多边形顶点    : 最大 %d，超 180 上限的对象 %d 个" %
          (r["max_poly_points"], r["over_limit_objects"]))
    print("-" * 58)
    print("%-14s %6s %6s %6s %8s %8s" % ("类别", "草稿", "真值", "TP", "精确率", "召回率"))
    for cname, v in r["per_class"].items():
        p = v["tp"] / (v["tp"] + v["fp"]) if (v["tp"] + v["fp"]) else 0
        rr = v["tp"] / (v["tp"] + v["fn"]) if (v["tp"] + v["fn"]) else 0
        print("%-14s %6d %6d %6d %8.3f %8.3f" %
              (cname, v["draft"], v["truth"], v["tp"], p, rr))

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(r, f, ensure_ascii=False, indent=2)
        print("\n已写出:", args.json)


if __name__ == "__main__":
    main()
