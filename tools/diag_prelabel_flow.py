# -*- coding: utf-8 -*-
"""复现首轮预标注（world + seg）的单图决策过程，打印每个框的配对情况。"""
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from prelabel import (_load_seg_model, _load_world_model, mask_to_polygon,
                      mask_box_coverage, load_class_thresholds)
from core import CLASSES, polygon_bbox

img_rel = sys.argv[1] if len(sys.argv) > 1 else "data/images/2.jpg"
img_path = img_rel if os.path.isabs(img_rel) else os.path.join(BASE, img_rel)

thr = load_class_thresholds() or {}
conf_low = min(thr.values()) if thr else 0.15
print("conf_low = %.2f" % conf_low)

seg = _load_seg_model()
world = _load_world_model()
world.set_classes(list(CLASSES))

so = seg.predict(img_path, conf=conf_low, verbose=False)
wo = world.predict(img_path, conf=conf_low, verbose=False)

seg_names = seg.names if isinstance(seg.names, dict) else {}
masks = []
if so and so[0].masks is not None:
    r = so[0]
    for i, m in enumerate(r.masks.data):
        masks.append({
            "mask": m.cpu().numpy(),
            "poly": mask_to_polygon(m.cpu().numpy()),
            "cls": seg_names.get(int(r.boxes.cls[i]), str(int(r.boxes.cls[i]))),
            "conf": float(r.boxes.conf[i]),
            "box": [float(v) for v in r.boxes.xyxy[i]],
        })
print("\nSeg 模型给出 %d 个掩码:" % len(masks))
for i, m in enumerate(masks):
    xs = [p[0] for p in m["poly"]] if m["poly"] else [0]
    ys = [p[1] for p in m["poly"]] if m["poly"] else [0]
    print("  m%d %-14s conf=%.3f pts=%3d  norm_box=(%.3f,%.3f)-(%.3f,%.3f)" %
          (i, m["cls"], m["conf"], len(m["poly"]),
           min(xs), min(ys), max(xs), max(ys)))

dets = []
r = wo[0]
names = r.names if isinstance(r.names, dict) else {}
for i in range(len(r.boxes)):
    cls_name = names.get(int(r.boxes.cls[i]), "")
    if cls_name not in CLASSES:
        continue
    cf = float(r.boxes.conf[i])
    if cf < thr.get(cls_name, conf_low):
        continue
    x1, y1, x2, y2 = [float(v) for v in r.boxes.xyxy[i]]
    dets.append({"cls": cls_name, "conf": cf, "box": [x1, y1, x2, y2],
                 "w": r.orig_shape[1], "h": r.orig_shape[0]})

print("\nWorld 通过阈值的框 %d 个:" % len(dets))
used = set()
for d in dets:
    x1, y1, x2, y2 = d["box"]
    bw, bh = x2 - x1, y2 - y1
    best_i, best_cov = -1, 0.0
    for i, m in enumerate(masks):
        if i in used:
            continue
        cov = mask_box_coverage(m["mask"], d["box"], d["w"], d["h"])
        if cov > best_cov:
            best_cov, best_i = cov, i
    tag = "无掩码"
    if best_i >= 0:
        m = masks[best_i]
        mx1, my1, mx2, my2 = m["box"]
        fill = (mx2 - mx1) * (my2 - my1) / max(bw * bh, 1)
        tag = "m%d(%s) cov=%.2f 掩码框/检测框=%.2f" % (best_i, m["cls"], best_cov, fill)
        if best_cov >= 0.55:
            used.add(best_i)
            tag += " -> 用掩码形状"
        elif best_cov >= 0.25:
            tag += " -> 用掩码紧致框"
        else:
            tag += " -> 矩形兜底"
    print("  %-13s conf=%.3f box=(%.3f,%.3f)-(%.3f,%.3f)  %s" %
          (d["cls"], d["conf"], x1 / d["w"], y1 / d["h"], x2 / d["w"], y2 / d["h"], tag))
