# -*- coding: utf-8 -*-
"""比较 retina_masks 开关对「掩码 -> 多边形」保真度的影响。

指标：
  comps   掩码连通块数
  cover   多边形面积 / 掩码像素面积（越接近 1 越好；<0.8 说明多边形丢了大块）
  selfx   多边形自相交次数（>0 画布填充会出现「只圈一半」）
"""
import os
import sys
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

import cv2
from ultralytics import YOLO
from prelabel import mask_to_polygon, load_class_thresholds
from core import polygon_area


def segments_cross(p1, p2, p3, p4):
    def d(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    d1, d2, d3, d4 = d(p3, p4, p1), d(p3, p4, p2), d(p1, p2, p3), d(p1, p2, p4)
    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))


def count_self_cross(poly):
    n = len(poly)
    if n < 4:
        return 0
    cnt = 0
    for i in range(n):
        a1, a2 = poly[i], poly[(i + 1) % n]
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue
            b1, b2 = poly[j], poly[(j + 1) % n]
            if segments_cross(a1, a2, b1, b2):
                cnt += 1
    return cnt


def main():
    img_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(BASE, "data/images/2.jpg")
    thr = load_class_thresholds() or {}
    conf_low = min(thr.values()) if thr else 0.25
    model = YOLO(os.path.join(BASE, "model", "best.pt"))
    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    for retina in (False, True):
        r = model.predict(img_path, conf=conf_low, retina_masks=retina, verbose=False)[0]
        print("\n===== retina_masks=%s =====" % retina)
        print("%-8s %5s %8s %5s %5s %7s %6s %6s" %
              ("类", "conf", "掩码尺寸", "块", "点", "cover", "selfx", "conf"))
        for i in range(len(r.boxes)):
            cls = r.names.get(int(r.boxes.cls[i]), "")
            if cls != "couch":
                continue
            m = r.masks.data[i].cpu().numpy()
            arr = (np.asarray(m).squeeze() > 0.5).astype(np.uint8)
            mh, mw = arr.shape
            n, labels, stats, _ = cv2.connectedComponentsWithStats(arr, connectivity=8)
            big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            keep = (labels == big).astype(np.uint8)
            mask_area = keep.sum() / (mh * mw)          # 归一化到 0~1 面积
            poly = mask_to_polygon(m)
            if len(poly) < 3:
                print("%-8s %5.3f %8s  多边形无效" % (cls, float(r.boxes.conf[i]), "%dx%d" % (mw, mh)))
                continue
            pa = abs(polygon_area(poly))
            cover = pa / mask_area if mask_area > 0 else 0
            print("%-8s %5.3f %8s %5d %5d %7.2f %6d" %
                  (cls, float(r.boxes.conf[i]), "%dx%d" % (mw, mh), n - 1,
                   len(poly), cover, count_self_cross(poly)))

    # 旧标注（177 点）是否自相交
    import json
    ap = os.path.join(BASE, "data/annotations/2.json")
    if os.path.exists(ap):
        j = json.load(open(ap, encoding="utf-8"))
        for o in j["objects"]:
            if o["class"] == "couch" and len(o["polygon"]) > 4:
                print("\n已有标注 couch %d 点：自相交 %d 处，面积 %.4f" %
                      (len(o["polygon"]), count_self_cross(o["polygon"]),
                       abs(polygon_area(o["polygon"]))))


if __name__ == "__main__":
    main()
