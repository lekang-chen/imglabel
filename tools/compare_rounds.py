# -*- coding: utf-8 -*-
"""
R1 / R2 修正率对比：回答"迭代闭环到底有没有用"。

任务书里的核心指标：
    修正率 = 修正对象数 / 总对象数
    迭代效果 = R2 的修正率是否低于 R1

判定"一个对象需要人工修正"的规则（用 COCO 官方真值做参照）：
    预测框与真值框 IoU >= 0.5  ->  匹配成功，不用改
    没匹配上                   ->  要改（要么多标了，要么漏检了）

注意 R1 和 R2 的分母不同、含义也不同，这里说清楚：
    R1 草稿 = 官方预训练权重吐的对象，人要去核对每一个
    R2 草稿 = 自训权重回标吐的对象，理论上应该更准、人要改的更少
  所以 R2 修正率更低 = 迭代有效。

用法：
  python tools/compare_rounds.py
  python tools/compare_rounds.py --json data/round_compare.json
"""
import argparse
import json
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "tools"))

from eval_draft import (COCO_JSON, SOURCE_MAP, OUR_CLASSES, IOU_MATCH,  # noqa: E402
                        load_truth, load_name_map, _bbox_of_seg, _iou,
                        _box_from_poly)

R1_DIR = os.path.join(BASE_DIR, "data", "annotations")
R2_DIR = os.path.join(BASE_DIR, "data", "annotations_r2")
OUT_JSON = os.path.join(BASE_DIR, "data", "round_compare.json")


def _round_stats(anno_dir, truth, name_map):
    """统计某轮的：对象数、需要修正数、TP/FP/FN、重复对。"""
    if not os.path.isdir(anno_dir):
        return None
    total = need_fix = 0
    tp = fp = fn = 0
    dup = 0
    per_class = {}
    files = sorted(f for f in os.listdir(anno_dir) if f.endswith(".json"))

    for fname in files:
        with open(os.path.join(anno_dir, fname), "r", encoding="utf-8") as f:
            data = json.load(f)
        objs = data.get("objects", [])
        total += len(objs)

        img_name = os.path.splitext(fname)[0] + ".jpg"
        gt = truth.get(name_map.get(img_name, img_name), [])
        our_w, our_h = data.get("width") or 0, data.get("height") or 0
        scale = 1.0
        if gt and our_w and our_h:
            ow, oh = gt[0][2], gt[0][3]
            if ow and oh:
                scale = min(our_w / ow, our_h / oh)

        used_gt = set()
        for o in objs:
            cname = o.get("class") or o.get("class_name", "")
            poly = o.get("polygon") or []
            if not poly:
                need_fix += 1
                fp += 1
                d = per_class.setdefault(cname, {"total": 0, "need_fix": 0})
                d["total"] += 1
                d["need_fix"] += 1
                continue

            pb = _box_from_poly([[p[0] * our_w, p[1] * our_h] for p in poly]) \
                if our_w and our_h else _box_from_poly(poly)

            best_j, best_iou = -1, IOU_MATCH
            for gi, (gc, gb, _gw, _gh) in enumerate(gt):
                if gi in used_gt or gc != cname:
                    continue
                gbox = [gb[0] * scale, gb[1] * scale,
                        (gb[0] + gb[2]) * scale, (gb[1] + gb[3]) * scale]
                v = _iou(pb, gbox)
                if v >= best_iou:
                    best_iou, best_j = v, gi

            d = per_class.setdefault(cname, {"total": 0, "need_fix": 0})
            d["total"] += 1
            if best_j >= 0:
                used_gt.add(best_j)
                tp += 1
            else:
                need_fix += 1
                fp += 1
                d["need_fix"] += 1

        fn += len(gt) - len(used_gt)

        # 同类重复
        for i in range(len(objs)):
            for j in range(i + 1, len(objs)):
                oi, oj = objs[i], objs[j]
                ci = oi.get("class") or oi.get("class_name")
                cj = oj.get("class") or oj.get("class_name")
                if ci != cj:
                    continue
                pi_, pj = oi.get("polygon") or [], oj.get("polygon") or []
                if not pi_ or not pj:
                    continue
                if _iou(_box_from_poly([[p[0] * our_w, p[1] * our_h] for p in pi_]),
                        _box_from_poly([[p[0] * our_w, p[1] * our_h]
                                        for p in pj])) > 0.6:
                    dup += 1

    rate = need_fix / total if total else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    return {
        "files": len(files),
        "total_objects": total,
        "need_fix": need_fix,
        "fix_rate": round(rate, 4),
        "tp": tp, "fp": fp, "fn": fn,
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "dup_pairs": dup,
        "per_class": per_class,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r1", default=R1_DIR)
    ap.add_argument("--r2", default=R2_DIR)
    ap.add_argument("--json", default=OUT_JSON)
    args = ap.parse_args()

    with open(COCO_JSON, "r", encoding="utf-8") as f:
        coco = json.load(f)
    id2name = {c["id"]: c["name"] for c in coco["categories"]}
    id2file = {im["id"]: im["file_name"] for im in coco["images"]}
    id2size = {im["id"]: (im["width"], im["height"]) for im in coco["images"]}

    truth = {}
    for ann in coco["annotations"]:
        if ann.get("iscrowd", 0):
            continue
        cname = id2name.get(ann["category_id"])
        if cname not in OUR_CLASSES:
            continue
        fn = id2file.get(ann["image_id"])
        if not fn:
            continue
        w, h = id2size.get(ann["image_id"], (0, 0))
        truth.setdefault(fn, []).append((cname, _bbox_of_seg(ann), w, h))

    name_map = load_name_map(SOURCE_MAP)

    r1 = _round_stats(args.r1, truth, name_map)
    r2 = _round_stats(args.r2, truth, name_map)

    print("=" * 66)
    print("R1 / R2 迭代效果对比（判定阈值 IoU >= %.1f）" % IOU_MATCH)
    print("=" * 66)
    if not r1 or not r2:
        print("缺少标注目录，无法对比。")
        print("  R1:", args.r1, "->", "有" if r1 else "无")
        print("  R2:", args.r2, "->", "有" if r2 else "无")
        sys.exit(1)

    rows = [
        ("图片数", r1["files"], r2["files"], ""),
        ("草稿对象数", r1["total_objects"], r2["total_objects"], "个"),
        ("需人工修正数", r1["need_fix"], r2["need_fix"], "个"),
        ("修正率", r1["fix_rate"], r2["fix_rate"], ""),
        ("精确率", r1["precision"], r2["precision"], ""),
        ("召回率", r1["recall"], r2["recall"], ""),
        ("同类重复对", r1["dup_pairs"], r2["dup_pairs"], "对"),
    ]
    print("%-14s %10s %10s %12s" % ("指标", "R1 官方权重", "R2 自训权重", "变化"))
    for name, a, b, unit in rows:
        if name == "修正率":
            delta = "↓ %.1f%%" % ((a - b) * 100) if b < a else "↑ %.1f%%" % ((b - a) * 100)
            print("%-14s %10.4f %10.4f %12s" % (name, a, b, delta))
        elif isinstance(a, int):
            d = b - a
            print("%-14s %10d %10d %12s" % (name, a, b, ("%+d %s" % (d, unit)) if d else "持平"))
        else:
            print("%-14s %10.4f %10.4f %12s" % (name, a, b, "%+.4f" % (b - a)))

    print("-" * 66)
    print("逐类修正率：")
    print("%-14s %8s %10s %8s %10s %10s" % ("类别", "R1对象", "R1修正率",
                                            "R2对象", "R2修正率", "变化"))
    for cname in OUR_CLASSES:
        d1 = r1["per_class"].get(cname, {"total": 0, "need_fix": 0})
        d2 = r2["per_class"].get(cname, {"total": 0, "need_fix": 0})
        q1 = d1["need_fix"] / d1["total"] if d1["total"] else 0
        q2 = d2["need_fix"] / d2["total"] if d2["total"] else 0
        mark = ""
        if d1["total"] and d2["total"]:
            if q2 < q1 - 0.01:
                mark = "↓ 改善"
            elif q2 > q1 + 0.01:
                mark = "↑ 变差"
        print("%-14s %8d %10.3f %8d %10.3f %10s" %
              (cname, d1["total"], q1, d2["total"], q2, mark))

    out = {"r1": r1, "r2": r2}
    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\n已写出:", os.path.relpath(args.json, BASE_DIR))


if __name__ == "__main__":
    main()
