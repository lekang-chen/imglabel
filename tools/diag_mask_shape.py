# -*- coding: utf-8 -*-
"""
诊断：模型输出的掩码形状 vs 标注的形状。

【要回答的问题】
标注里 80% 是 4 点矩形。如果模型「学会了矩形」，那它的掩码填充率
(掩码面积 / 检测框面积) 会接近 1.0，跟框几乎一样；
如果模型输出的是真实轮廓，填充率会明显低于 1.0（椅子约 0.4）。

这个数字决定了下一步该做什么：
  填充率 ≈ 1.0  -> 模型被矩形标签带偏了，需要清洗标注
  填充率 << 1.0 -> 模型形状是对的，是 GT 太粗，指标被压住

只读，不写任何标注文件。
"""
import os
import sys
import zipfile

import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_DIR = os.path.join(BASE_DIR, "dataset", "images", "val")
LBL_DIR = os.path.join(BASE_DIR, "dataset", "labels", "val")

NAMES = ["person", "chair", "dining table", "couch", "tv",
         "laptop", "cup", "bottle", "potted plant", "book"]


def main():
    from ultralytics import YOLO

    weights = sys.argv[1] if len(sys.argv) > 1 else os.path.join(BASE_DIR, "model", "best.pt")
    model = YOLO(weights)

    expect = {}   # cls -> {"pred_fill": [], "iou": [], "gt_fill": []}
    files = sorted(f for f in os.listdir(IMG_DIR)
                   if f.lower().endswith((".jpg", ".jpeg", ".png")))

    for fn in files:
        img_path = os.path.join(IMG_DIR, fn)
        lbl_path = os.path.join(LBL_DIR, os.path.splitext(fn)[0] + ".txt")
        if not os.path.exists(lbl_path):
            continue

        r = model.predict(img_path, conf=0.40, verbose=False)[0]
        h, w = r.orig_shape
        if r.masks is None:
            continue
        pm = r.masks.data.cpu().numpy()
        pb = r.boxes.xyxy.cpu().numpy()
        pc = r.boxes.cls.cpu().numpy().astype(int)

        # 预测掩码可能小于原图（模型内部尺寸），缩放到原图
        if pm.shape[1] != h or pm.shape[2] != w:
            import cv2
            pm = np.stack([cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST) for m in pm])

        for k in range(len(pc)):
            c = int(pc[k])
            m = pm[k] > 0.5
            area = float(m.sum())
            x1, y1, x2, y2 = pb[k]
            box_area = max((x2 - x1) * (y2 - y1), 1.0)
            expect.setdefault(c, {"pred_fill": [], "gt_fill": []})
            expect[c]["pred_fill"].append(area / box_area)

    # 标注的填充率：多边形面积 / 外接矩形面积
    for fn in os.listdir(LBL_DIR):
        with open(os.path.join(LBL_DIR, fn), encoding="utf-8") as f:
            for ln in f:
                p = ln.split()
                if not p or (len(p) - 1) % 2:
                    continue
                c = int(p[0])
                xs = np.array([float(v) for v in p[1::2]])
                ys = np.array([float(v) for v in p[2::2]])
                bw, bh = xs.max() - xs.min(), ys.max() - ys.min()
                if bw <= 0 or bh <= 0:
                    continue
                # 鞋带公式算多边形面积
                x2 = np.roll(xs, -1)
                y2 = np.roll(ys, -1)
                poly = abs(np.sum(xs * y2 - x2 * ys)) / 2.0
                expect.setdefault(c, {"pred_fill": [], "gt_fill": []})
                expect[c]["gt_fill"].append(poly / (bw * bh))

    print("=" * 72)
    print("模型输出形状 vs 标注形状  (权重 %s)"
          % os.path.relpath(weights, BASE_DIR).replace("\\", "/"))
    print("=" * 72)
    print("%-14s %6s %12s %12s %8s" % ("类别", "样本", "标注填充率", "预测填充率", "差异"))
    print("-" * 72)
    for c in range(len(NAMES)):
        d = expect.get(c)
        if not d or not d["pred_fill"] or not d["gt_fill"]:
            continue
        gf = float(np.mean(d["gt_fill"]))
        pf = float(np.mean(d["pred_fill"]))
        print("%-14s %6d %11.2f %12.2f %8.2f"
              % (NAMES[c], min(len(d["gt_fill"]), len(d["pred_fill"])), gf, pf, pf - gf))
    print("-" * 72)
    allg, allp = [], []
    for d in expect.values():
        allg += d["gt_fill"]
        allp += d["pred_fill"]
    if allg and allp:
        print("%-14s %6d %11.2f %12.2f %8.2f"
              % ("全部", min(len(allg), len(allp)),
                 float(np.mean(allg)), float(np.mean(allp)),
                 float(np.mean(allp)) - float(np.mean(allg))))
    print()
    print("解读：预测填充率接近 1.0 = 模型输出的掩码几乎就是外接框（被矩形标注带偏）；")
    print("      预测填充率明显低于标注填充率 = 模型给出的是真实轮廓，是标注偏粗。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
