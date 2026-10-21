# -*- coding: utf-8 -*-
"""
生成"人工修正待办清单"：告诉用户具体该改哪张图、哪个对象、怎么改。

为什么要有这个：
  任务书要求「人工修正不少于 60 条」，这条只能本人亲手做，造假零分。
  但"随便找 60 个对象改改"效率很低——改到本来就对的对象等于没改。
  真正省时间的做法是：**优先改那些模型明确改错了的地方**。

  所以这里用 COCO 官方真值反查每一张图，把三类问题分开列：
    A 类【多标】模型标了，但真值里没有 → 要删掉
    B 类【漏标】真值里有，但模型没标 → 要补上
    C 类【错类】模型标了，但是类别标错了 → 要改类别
    D 类【边界】模型标了，位置大致对但边界差 → 要修边界

  A/B/C 是"改错了"（不修就是错的数据），D 是"改得不精确"（边界）。
  优先做 A/B/C，因为它们对模型训练的增益最大。

用法：
  python tools/make_fix_tasks.py            # 打印前 20 张图的清单
  python tools/make_fix_tasks.py --top 40   # 打印前 40 张
  python tools/make_fix_tasks.py --json data/fix_tasks.json
"""
import argparse
import json
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "tools"))

from eval_draft import (COCO_JSON, SOURCE_MAP, OUR_CLASSES, IOU_MATCH,  # noqa: E402
                        load_name_map, _bbox_of_seg, _iou, _box_from_poly)

ANNO_DIR = os.path.join(BASE_DIR, "data", "annotations")


def build_tasks(anno_dir=ANNO_DIR):
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
    tasks = []

    for fname in sorted(f for f in os.listdir(anno_dir) if f.endswith(".json")):
        with open(os.path.join(anno_dir, fname), "r", encoding="utf-8") as f:
            data = json.load(f)
        objs = data.get("objects", [])
        img_name = os.path.splitext(fname)[0] + ".jpg"
        gt = truth.get(name_map.get(img_name, img_name), [])
        if not gt:
            continue

        our_w, our_h = data.get("width") or 0, data.get("height") or 0
        scale = 1.0
        ow, oh = gt[0][2], gt[0][3]
        if our_w and our_h and ow and oh:
            scale = min(our_w / ow, our_h / oh)

        # 逐个真值找最佳匹配的预测
        matched = {}
        for gi, (gc, gb, _gw, _gh) in enumerate(gt):
            gbox = [gb[0] * scale, gb[1] * scale,
                    (gb[0] + gb[2]) * scale, (gb[1] + gb[3]) * scale]
            best_j, best_iou = -1, 0.0
            for pi, o in enumerate(objs):
                if pi in matched.values():
                    continue
                poly = o.get("polygon") or []
                if not poly:
                    continue
                pb = _box_from_poly([[p[0] * our_w, p[1] * our_h] for p in poly])
                v = _iou(pb, gbox)
                if v > best_iou:
                    best_iou, best_j = v, pi
            if best_j >= 0:
                matched[gi] = best_j

        entry = {"image": img_name, "file": fname,
                 "overlap": [], "missing": [], "wrong_class": [], "extra": []}

        # 1) 匹配上但类别不同 -> 错类
        # 2) 匹配上但 IoU 低 -> 边界要修
        # 3) 一个预测同时盖住多个真值 -> 拆分问题（既漏标又多标）
        for gi, pi in matched.items():
            gc = gt[gi][0]
            oc = objs[pi].get("class") or objs[pi].get("class_name")
            if oc != gc:
                entry["wrong_class"].append(
                    {"obj_index": pi, "from": oc, "to": gc,
                     "obj_id": objs[pi].get("id", "")})
            else:
                poly = objs[pi].get("polygon") or []
                gb = gt[gi][1]
                gbox = [gb[0] * scale, gb[1] * scale,
                        (gb[0] + gb[2]) * scale, (gb[1] + gb[3]) * scale]
                pb = _box_from_poly([[p[0] * our_w, p[1] * our_h] for p in poly])
                v = _iou(pb, gbox)
                if v < 0.75:
                    entry["overlap"].append(
                        {"obj_index": pi, "class": gc,
                         "iou": round(v, 3), "obj_id": objs[pi].get("id", "")})

        # 3) 真值有但没匹配上 -> 漏标
        for gi, (gc, gb, _gw, _gh) in enumerate(gt):
            if gi in matched:
                continue
            gbox = [gb[0] * scale, gb[1] * scale,
                    (gb[0] + gb[2]) * scale, (gb[1] + gb[3]) * scale]
            entry["missing"].append({
                "class": gc,
                "center": [round((gbox[0] + gbox[2]) / 2 / our_w, 3),
                           round((gbox[1] + gbox[3]) / 2 / our_h, 3)],
                "size": [round((gbox[2] - gbox[0]) / our_w, 3),
                         round((gbox[3] - gbox[1]) / our_h, 3)],
            })

        # 4) 预测有但没匹配上任何真值 -> 多标
        for pi, o in enumerate(objs):
            if pi in matched.values():
                continue
            entry["extra"].append({
                "obj_index": pi,
                "class": o.get("class") or o.get("class_name"),
                "obj_id": o.get("id", ""),
            })

        n = (len(entry["wrong_class"]) + len(entry["missing"])
             + len(entry["extra"]) + len(entry["overlap"]))
        entry["total"] = n
        if n:
            tasks.append(entry)

    tasks.sort(key=lambda e: -e["total"])
    return tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    tasks = build_tasks()

    tot_wrong = sum(len(t["wrong_class"]) for t in tasks)
    tot_missing = sum(len(t["missing"]) for t in tasks)
    tot_extra = sum(len(t["extra"]) for t in tasks)
    tot_overlap = sum(len(t["overlap"]) for t in tasks)

    print("=" * 70)
    print("人工修正待办清单（对照 COCO 真值反查）")
    print("=" * 70)
    print("错类（改类别）  : %d 处" % tot_wrong)
    print("漏标（补对象）  : %d 处" % tot_missing)
    print("多标（删对象）  : %d 处" % tot_extra)
    print("边界（修轮廓）  : %d 处" % tot_overlap)
    print("合计可修正      : %d 处  → 任务书只要 60 条，做前 %d 张图就够了"
          % (tot_wrong + tot_missing + tot_extra + tot_overlap,
             min(args.top, len(tasks))))
    print("=" * 70)

    for t in tasks[:args.top]:
        print("\n【%s】  %d 处可改" % (t["image"], t["total"]))
        for w in t["wrong_class"]:
            print("   错类  第%d个对象: %s -> %s" %
                  (w["obj_index"], w["from"], w["to"]))
        for m in t["missing"]:
            print("   漏标  补一个 %s，位置约 (%.3f, %.3f)，大小约 %.3f x %.3f" %
                  (m["class"], m["center"][0], m["center"][1],
                   m["size"][0], m["size"][1]))
        for e in t["extra"]:
            print("   多标  第%d个对象(%s) 删掉" % (e["obj_index"], e["class"]))
        for o in t["overlap"]:
            print("   边界  第%d个对象(%s) 轮廓偏，IoU=%.2f" %
                  (o["obj_index"], o["class"], o["iou"]))

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(tasks, f, ensure_ascii=False, indent=2)
        print("\n完整清单已写出:", args.json)


if __name__ == "__main__":
    main()
