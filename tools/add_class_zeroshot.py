#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""用开放词表模型给「新加进词表的类别」补标注（零样本冷启动）。

【为什么需要这个脚本】
词表里加一个类，预标注立刻能找它——但预标注只处理**还没有标注**的图。
现有的 169 张全都标过了，点「预标注」只会弹「全部已有标注，确定重做吗」，
重做会把人工修正全冲掉。于是新类永远拿不到样本，训练也就永远学不会它。

这个脚本只做一件事：对**已有标注**的图跑一遍开放词表检测，把**指定类别**
的检测结果追加进去，别的对象一个字节都不动。相当于只给训练集「补一列」，
不改原有的列。

【为什么只追加、不覆盖】
已有对象是人工确认过的（62 条修正都在里面），重跑会把这些全丢掉。
代价是新补的框是模型原始输出，质量不如人工——但新类此刻一个样本都没有，
零个样本和「模型说的」之间，只能先选后者，这就是零样本冷启动。

【为什么要做 IoU 过滤】
床和沙发在 COCO 里本来就容易混：一张图若已经把这块标成 couch，
再叠一个 bed 上去，训练时同一块区域有两个类，模型会学出两个类都响应。
所以与已有对象重合度超过 --iou-skip 的检测直接丢掉——宁可少，不要打架。

用法：
  python tools/add_class_zeroshot.py --cls bed              # 只看，不写
  python tools/add_class_zeroshot.py --cls bed --apply       # 真的写进标注
"""
import argparse
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import core  # noqa: E402
import prelabel  # noqa: E402
from core import Annotation, IMAGE_EXTS, polygon_bbox  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE_DIR = os.path.join(ROOT, "data", "images")
ANNO_DIR = os.path.join(ROOT, "data", "annotations")


def bbox_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = ((ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter)
    return inter / union if union > 0 else 0.0


def load_anno(path):
    with io.open(path, encoding="utf-8") as f:
        return Annotation.from_dict(json.load(f))


def save_anno(ann, path):
    tmp = path + ".tmp"
    with io.open(tmp, "w", encoding="utf-8") as f:
        json.dump(ann.to_dict(), f, ensure_ascii=False)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cls", required=True, help="要补的类别名（必须在当前词表里）")
    ap.add_argument("--conf", type=float, default=0.15,
                    help="该类的检测阈值，零样本偏低，默认 0.15")
    ap.add_argument("--iou-skip", type=float, default=0.5,
                    help="与已有对象重合超过这个值就跳过，默认 0.5")
    ap.add_argument("--apply", action="store_true", help="真的写进标注文件")
    args = ap.parse_args()

    class_names = core.get_classes()
    if args.cls not in class_names:
        print("词表里没有「%s」，先加进词表再跑。当前：%s"
              % (args.cls, ", ".join(class_names)))
        return 2

    seg_model = prelabel._load_seg_model()
    world_model = prelabel._load_world_model()
    world_model.set_classes(list(class_names))

    files = sorted(f for f in os.listdir(ANNO_DIR) if f.endswith(".json"))
    print("扫描 %d 张已标注的图，目标类别 = %s，conf = %.2f，%s"
          % (len(files), args.cls, args.conf, "写入" if args.apply else "只看不写"))

    added = 0
    touched = []
    skipped_conflict = 0
    empty_after = 0

    for i, fn in enumerate(files, 1):
        path = os.path.join(ANNO_DIR, fn)
        try:
            ann = load_anno(path)
        except Exception as e:                     # noqa: BLE001
            print("  跳过坏文件 %s (%s)" % (fn, e))
            continue
        img_path = os.path.join(IMAGE_DIR, ann.image_name)
        if not os.path.exists(img_path):
            continue

        try:
            objs = prelabel._prelabel_one(
                seg_model, world_model, img_path, class_names,
                args.conf, 0.5, thr_map={})
        except Exception as e:                     # noqa: BLE001
            print("  推理失败 %s (%s)" % (ann.image_name, e))
            continue

        cands = [o for o in objs if o.class_name == args.cls]
        if not cands:
            continue

        have = [polygon_bbox(o.polygon) for o in ann.objects]
        keep = []
        for o in cands:
            box = polygon_bbox(o.polygon)
            if any(bbox_iou(box, h) > args.iou_skip for h in have):
                skipped_conflict += 1
                continue
            keep.append(o)

        if not keep:
            continue
        if args.apply:
            ann.objects.extend(keep)
            save_anno(ann, path)
        added += len(keep)
        touched.append((ann.image_name, len(keep)))
        if i % 20 == 0:
            print("  ... %d/%d，已补 %d 个" % (i, len(files), added))

    for name, n in touched:
        print("  + %-28s %d" % (name, n))
    print("\n结论：%d 张图补了 %d 个 %s 对象；因与已有对象重合跳过 %d 个。"
          % (len(touched), added, args.cls, skipped_conflict))
    if not args.apply:
        print("（只是预览，没写文件。加 --apply 才会写。）")

    # 补完检查：整个数据集里这个类现在有多少
    total = 0
    for fn in os.listdir(ANNO_DIR):
        if not fn.endswith(".json"):
            continue
        try:
            a = load_anno(os.path.join(ANNO_DIR, fn))
        except Exception:                          # noqa: BLE001
            continue
        total += sum(1 for o in a.objects if o.class_name == args.cls)
    if total == 0:
        empty_after += 1
    print("全库「%s」对象总数：%d%s"
          % (args.cls, total, "（一个都没有，训练学不会这个类）" if not total else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
