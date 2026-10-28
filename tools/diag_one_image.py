# -*- coding: utf-8 -*-
"""诊断单张图的预标注质量：掩码连通块 / 多边形覆盖 / 重复框。

用法: python tools/diag_one_image.py data/images/2.jpg
产出: data/_diag/<name>_overlay.png （左边标注多边形，右边模型原始掩码）
"""
import os
import sys
import json
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

import cv2
from ultralytics import YOLO

from prelabel import load_class_thresholds, mask_to_polygon

WEIGHTS = os.path.join(BASE, "model", "best.pt")


def mask_components(mask):
    """返回掩码的连通块数、各块面积占比、块间是否被细缝切断。"""
    arr = np.asarray(mask)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    arr = (arr > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(arr, connectivity=8)
    total = int(arr.sum())
    comps = []
    for i in range(1, n):
        a = int(stats[i, cv2.CC_STAT_AREA])
        if a < max(1, total * 0.01):
            continue
        comps.append({
            "area": a,
            "ratio": a / max(total, 1),
            "box": [int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP]),
                    int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT])],
        })
    comps.sort(key=lambda c: -c["area"])
    return n - 1, total, comps


def main():
    if len(sys.argv) < 2:
        print("用法: python tools/diag_one_image.py <图片路径> [权重]")
        return
    img_path = sys.argv[1]
    if not os.path.isabs(img_path):
        img_path = os.path.join(BASE, img_path)
    weights = sys.argv[2] if len(sys.argv) > 2 else WEIGHTS

    thr = load_class_thresholds() or {}
    conf_low = min(thr.values()) if thr else 0.25
    print("阈值表最低值 conf_low = %.2f" % conf_low)

    model = YOLO(weights)
    r = model.predict(img_path, conf=conf_low, verbose=False)[0]
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    print("图片 %dx%d, 检出 %d 个实例" % (w, h, len(r.boxes)))

    names = r.names if isinstance(r.names, dict) else {}
    overlay = img.copy()
    raw = img.copy()

    rows = []
    for i in range(len(r.boxes)):
        cls = names.get(int(r.boxes.cls[i]), str(int(r.boxes.cls[i])))
        cf = float(r.boxes.conf[i])
        x1, y1, x2, y2 = [float(v) for v in r.boxes.xyxy[i]]
        th = thr.get(cls, 0.40)
        n_raw, area, comps = 0, 0, []
        poly = []
        if r.masks is not None and i < len(r.masks.data):
            m = r.masks.data[i].cpu().numpy()
            n_raw, area, comps = mask_components(m)
            poly = mask_to_polygon(m)
            # 原始掩码上色
            mm = cv2.resize(np.asarray(m).squeeze().astype(np.float32), (w, h))
            raw[mm > 0.5] = (0.45 * raw[mm > 0.5] + 0.55 * np.array([200, 120, 255])).astype(np.uint8)
        if len(poly) >= 3:
            pts = np.array([[int(px * (w - 1)), int(py * (h - 1))] for px, py in poly], np.int32)
            cv2.polylines(overlay, [pts], True, (0, 0, 255), 3)
        cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), (0, 200, 0), 2)
        rows.append((cls, cf, th, (x1 / w, y1 / h, x2 / w, y2 / h), n_raw, len(poly), comps))

    print("\n%-8s %6s %6s  %-28s %4s %4s  %s" %
          ("类", "conf", "阈值", "归一化框", "块", "点", "主要连通块占比"))
    for cls, cf, th, b, n_raw, npt, comps in rows:
        flag = "" if cf >= th else "  <- 低于阈值"
        ratio = ", ".join("%.2f" % c["ratio"] for c in comps[:3])
        print("%-8s %6.3f %6.2f  (%.3f,%.3f)-(%.3f,%.3f) %4d %4d  %s%s" %
              (cls, cf, th, b[0], b[1], b[2], b[3], n_raw, npt, ratio, flag))

    # 同类多框重叠检查
    print("\n同类框重叠检查：")
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            if rows[i][0] != rows[j][0]:
                continue
            a, b = rows[i][3], rows[j][3]
            ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
            iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
            inter = ix * iy
            ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
            print("  %d vs %d  IoU=%.3f" % (i + 1, j + 1, inter / ua if ua > 0 else 0))

    out_dir = os.path.join(BASE, "data", "_diag")
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(img_path))[0]
    side = np.hstack([overlay, raw])
    out = os.path.join(out_dir, "%s_overlay.png" % stem)
    cv2.imwrite(out, side)
    print("\n对比图: %s" % out)


if __name__ == "__main__":
    main()
