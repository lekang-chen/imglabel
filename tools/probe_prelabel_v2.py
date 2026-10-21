# -*- coding: utf-8 -*-
"""预标注方案 A（小范围验证）：以框为锚，借用掩码的形状。

对比新旧两套配对逻辑，输出「矩形占比 / 顶点数分布 / 对象总数」，
让你在决定是否全量重跑之前，先看到真实数字。

只读不写：全部在内存里跑，不碰data/annotations。

用法:
    .venv/Scripts/python.exe tools/probe_prelabel_v2.py [图片张数]
"""
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prelabel import (  # noqa: E402
    _load_seg_model, _load_world_model, mask_to_polygon,
    mask_box_coverage, _seg_class_names,
)

CLASSES = ["person", "chair", "dining table", "couch", "tv",
           "laptop", "cup", "bottle", "potted plant", "book"]

COVER_STRONG = 0.55
COVER_WEAK = 0.25


def is_axis_rect(poly):
    if len(poly) != 4:
        return False
    xs = sorted(set(round(p[0], 6) for p in poly))
    ys = sorted(set(round(p[1], 6) for p in poly))
    return len(xs) == 2 and len(ys) == 2


def run(img, seg, world, conf=0.25):
    """返回 (旧方案对象列表, 新方案对象列表, 明细统计)。"""
    # --- 取两个模型的原始输出 ---
    r = seg.predict(img, conf=conf, verbose=False)[0]
    masks = []
    if r.masks is not None:
        for i, m in enumerate(r.masks.data):
            masks.append({"mask": m.cpu().numpy(),
                          "cls": int(r.boxes.cls[i]),
                          "poly": mask_to_polygon(m.cpu().numpy())})

    w = world.predict(img, conf=conf, verbose=False)[0]
    dets = []
    if w.boxes is not None:
        names = w.names if isinstance(w.names, dict) else {}
        for i in range(len(w.boxes)):
            cn = names.get(int(w.boxes.cls[i]), "")
            if cn not in CLASSES:
                continue
            dets.append({"box_norm": w.boxes.xyxy[i].tolist(), "cls": cn,
                         "w": w.orig_shape[1], "h": w.orig_shape[0]})

    # --- 旧方案：IoU 配对 + 矩形兜底 ---
    old = []
    used = set()
    for m in masks:
        if len(m["poly"]) < 3:
            continue
        mb = [min(p[0] for p in m["poly"]), min(p[1] for p in m["poly"]),
              max(p[0] for p in m["poly"]), max(p[1] for p in m["poly"])]
        best, bi = None, 0.0
        for j, d in enumerate(dets):
            if j in used:
                continue
            ix1, iy1 = max(mb[0], d["box_norm"][0]), max(mb[1], d["box_norm"][1])
            ix2, iy2 = min(mb[2], d["box_norm"][2]), min(mb[3], d["box_norm"][3])
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            aa = (mb[2] - mb[0]) * (mb[3] - mb[1])
            bb = (d["box_norm"][2] - d["box_norm"][0]) * (d["box_norm"][3] - d["box_norm"][1])
            u = aa + bb - inter
            v = inter / u if u > 0 else 0.0
            if v > bi:
                bi, best = v, j
        if best is not None and bi >= 0.55:
            used.add(best)
            old.append(m["poly"])
    for j, d in enumerate(dets):
        if j in used:
            continue
        x1, y1, x2, y2 = d["box_norm"]
        old.append([[x1 / d["w"], y1 / d["h"]], [x2 / d["w"], y1 / d["h"]],
                    [x2 / d["w"], y2 / d["h"]], [x1 / d["w"], y2 / d["h"]]])

    # --- 新方案：覆盖率借形状 ---
    new = []
    used_m = set()
    n_strong = n_weak = n_rect = 0
    for d in dets:
        bi_i, bi_cov = -1, 0.0
        for i, m in enumerate(masks):
            if i in used_m:
                continue
            cov = mask_box_coverage(m["mask"], d["box_norm"], d["w"], d["h"])
            if cov > bi_cov:
                bi_cov, bi_i = cov, i
        if bi_i >= 0 and bi_cov >= COVER_STRONG and len(masks[bi_i]["poly"]) >= 3:
            used_m.add(bi_i)
            new.append(masks[bi_i]["poly"])
            n_strong += 1
            continue
        if bi_i >= 0 and bi_cov >= COVER_WEAK and len(masks[bi_i]["poly"]) >= 3:
            mp = masks[bi_i]["poly"]
            bx1, by1 = min(p[0] for p in mp), min(p[1] for p in mp)
            bx2, by2 = max(p[0] for p in mp), max(p[1] for p in mp)
            new.append([[bx1, by1], [bx2, by1], [bx2, by2], [bx1, by2]])
            n_weak += 1
            continue
        x1, y1, x2, y2 = d["box_norm"]
        new.append([[x1 / d["w"], y1 / d["h"]], [x2 / d["w"], y1 / d["h"]],
                    [x2 / d["w"], y2 / d["h"]], [x1 / d["w"], y2 / d["h"]]])
        n_rect += 1

    return old, new, dict(masks=len(masks), dets=len(dets),
                          strong=n_strong, weak=n_weak, rect=n_rect)


def main():
    n_img = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    seg = _load_seg_model()
    world = _load_world_model()
    world.set_classes(CLASSES)

    imgs = sorted(glob.glob("data/images/*.jpg"))[:n_img]
    if not imgs:
        print("找不到 data/images/*.jpg")
        return 1

    agg = dict(old=0, new=0, old_rect=0, new_rect=0, old_pts=0, new_pts=0)
    print("逐图对比（conf=0.25, 强覆盖>=0.55, 弱覆盖>=0.25）")
    print("-" * 78)
    print("%-16s %5s %5s | %-19s | %-19s" % (
        "图片", "掩码", "框", "旧方案 对象/矩形", "新方案 强/弱/矩形"))
    print("-" * 78)
    for p in imgs:
        old, new, s = run(p, seg, world)
        o_rect = sum(1 for q in old if is_axis_rect(q))
        n_rect2 = sum(1 for q in new if is_axis_rect(q))
        agg["old"] += len(old); agg["new"] += len(new)
        agg["old_rect"] += o_rect; agg["new_rect"] += n_rect2
        agg["old_pts"] += sum(len(q) for q in old)
        agg["new_pts"] += sum(len(q) for q in new)
        print("%-16s %5d %5d | %3d / %3d (%4.0f%%) | %2d/%2d/%2d 矩形%2d (%4.0f%%)"
              % (os.path.basename(p), s["masks"], s["dets"],
                 len(old), o_rect, 100 * o_rect / max(1, len(old)),
                 s["strong"], s["weak"], s["rect"], n_rect2,
                 100 * n_rect2 / max(1, len(new))))
    print("-" * 78)
    print()
    print("合计对象:  旧 %d  ->  新 %d" % (agg["old"], agg["new"]))
    print("矩形占比:  旧 %.1f%%  ->  新 %.1f%%"
          % (100 * agg["old_rect"] / max(1, agg["old"]),
             100 * agg["new_rect"] / max(1, agg["new"])))
    print("平均顶点数: 旧 %.1f  ->  新 %.1f"
          % (agg["old_pts"] / max(1, agg["old"]),
             agg["new_pts"] / max(1, agg["new"])))
    print()
    print("磁盘现状(data/annotations)实测: 83.0% 是矩形")
    return 0


if __name__ == "__main__":
    sys.exit(main())