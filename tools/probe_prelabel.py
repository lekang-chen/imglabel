# -*- coding: utf-8 -*-
"""预标注诊断：查清「有多少对象来自掩码、多少来自检测框兜底」。

为什么需要这个脚本
------------------
界面上几乎所有对象都显示为矩形框，一度怀疑是描边代码画错了。
实测发现data/annotations/ 里83% 的多边形是「4 点轴对齐矩形」，
而且是逐像素推导出来的纯矩形（x 只有两个值、y 只有两个值）。
追代码锁定在 _prelabel_one 的兜底分支：
seg 模型没匹配上的检测框，被直接造了一个矩形多边形。

本脚本只读不写：在内存里跑一遍预标注，打印来源统计，
用来量化「兜底分支到底占了多少」，从而判断该改代码还是改阈值。
"""
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prelabel import _load_seg_model, _load_world_model, mask_to_polygon, iou  # noqa: E402

CLASSES = ["person", "chair", "dining table", "couch", "tv",
           "laptop", "cup", "bottle", "potted plant", "book"]


def is_axis_rect(poly):
    """判断多边形是不是纯粹的轴对齐矩形（4 点、只有两个 x 和两个 y）。"""
    if len(poly) != 4:
        return False
    xs = sorted(set(round(p[0], 6) for p in poly))
    ys = sorted(set(round(p[1], 6) for p in poly))
    return len(xs) == 2 and len(ys) == 2


def probe(img_path, seg_model, world_model, conf=0.25, iou_thresh=0.55):
    """对单图跑一遍完整配对逻辑，返回各类计数。"""
    stat = dict(masks=0, dets=0, matched=0, fallback=0,
                dropped_poly=0, unmatched_by_class=0)

    seg_out = seg_model.predict(img_path, conf=conf, verbose=False)
    masks = []
    if seg_out and len(seg_out):
        r = seg_out[0]
        if r.masks is not None:
            for i, m in enumerate(r.masks.data):
                masks.append({"mask": m.cpu().numpy(), "cls": int(r.boxes.cls[i])})
    stat["masks"] = len(masks)

    world_out = world_model.predict(img_path, conf=conf, verbose=False)
    dets = []
    if world_out and len(world_out):
        r = world_out[0]
        if r.boxes is not None:
            names = r.names if isinstance(r.names, dict) else {}
            for i in range(len(r.boxes)):
                cn = names.get(int(r.boxes.cls[i]), "")
                if cn not in CLASSES:
                    continue
                x1, y1, x2, y2 = r.boxes.xyxy[i].tolist()
                dets.append({"box": [x1, y1, x2, y2], "cls": cn,
                             "w": r.orig_shape[1], "h": r.orig_shape[0]})
    stat["dets"] = len(dets)

    from core import polygon_bbox

    used = set()
    for m in masks:
        poly = mask_to_polygon(m["mask"])
        if not poly:
            stat["dropped_poly"] += 1
            continue
        mb = polygon_bbox(poly)
        best, bi = None, 0.0
        for j, d in enumerate(dets):
            if j in used:
                continue
            v = iou(mb, d["box"])
            if v > bi:
                bi, best = v, j
        if best is not None and bi >= iou_thresh:
            used.add(best)
            stat["matched"] += 1
        else:
            stat["unmatched_by_class"] += 1

    stat["fallback"] = len(dets) - len(used)
    return stat


if __name__ == "__main__":
    n_img = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    seg = _load_seg_model()
    world = _load_world_model()
    world.set_classes(CLASSES)

    imgs = sorted(glob.glob("data/images/*.jpg"))[:n_img]
    if not imgs:
        print("找不到 data/images/*.jpg")
        sys.exit(1)

    total = dict(masks=0, dets=0, matched=0, fallback=0,
                 dropped_poly=0, unmatched_by_class=0)
    print("逐图统计（阈值 conf=0.25, iou_thresh=0.55）")
    print("-" * 72)
    for p in imgs:
        s = probe(p, seg, world)
        for k in total:
            total[k] += s[k]
        print("%-16s掩码%3d检测框%3d  匹配%3d  兜底矩形%3d  掩码被丢%2d  掩码无类别%2d"
              % (os.path.basename(p), s["masks"], s["dets"],
                 s["matched"], s["fallback"], s["dropped_poly"],
                 s["unmatched_by_class"]))
    print("-" * 72)
    print("合计: 掩码 %d, 检测框 %d, 匹配 %d, 兜底矩形 %d, 掩码被丢 %d, 掩码无类别 %d"
          % (total["masks"], total["dets"], total["matched"],
             total["fallback"], total["dropped_poly"], total["unmatched_by_class"]))

    denom = total["matched"] + total["fallback"]
    if denom:
        print()
        print("最终对象里，矩形兜底占比: %.1f%%  (掩码匹配 %.1f%%)"
              % (100 * total["fallback"] / denom,
                 100 * total["matched"] / denom))
        print("（磁盘实测为83% 矩形 —— 若这里接近，说明成因确认为此）")