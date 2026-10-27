# -*- coding: utf-8 -*-
"""
按类别搜索置信度阈值。

【为什么统一阈值会吃亏】
现在预标注对所有类都用同一个 conf=0.40。但这十类的难度差得很远：
  - laptop 只有 19 个训练样本，分数天然偏低，0.40 会把它们全部砍掉
    （诊断结果：laptop 召回 0.333，掩码 mAP50 为 0）
  - person 有 128 个样本，模型很有信心，0.40 之下混着不少把靠垫、
    衣服堆当成人的假框
一个阈值同时服务这两头，必然有一头是错的。

【怎么定】
对每一类，在验证集上扫一遍阈值，取 F1 最高的那一点。
样本太少的类（< MIN_INSTANCES）不单独定——3 个样本搜出来的阈值
没有任何统计意义，直接沿用全局值。阈值还被限制在一个保守区间内，
免得为了 30 张验证集上的 0.01 收益把阈值推到一个普适性很差的位置。

用法：
  python tools/tune_threshold.py                        # 用 model/best.pt
  python tools/tune_threshold.py --weights runs/exp_iter3_long/weights/best.pt
  python tools/tune_threshold.py --dry                  # 只打印，不写文件
"""
import argparse
import json
import os
import sys

import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
OUT_JSON = os.path.join(BASE_DIR, "data", "class_thresholds.json")

NAMES = ["person", "chair", "dining table", "couch", "tv",
         "laptop", "cup", "bottle", "potted plant", "book"]

GRID = [round(0.10 + 0.025 * i, 3) for i in range(23)]   # 0.10 ~ 0.65
MIN_INSTANCES = 6         # 类实例少于这个数就不单独定阈值
LO, HI = 0.20, 0.55       # 阈值允许区间（保守，避免过拟合验证集）
GLOBAL_DEFAULT = 0.40     # 沿用现在 prelabel 的默认值


def load_gt(lbl_dir):
    """读 GT：每张图一个 list[(cls, box_xyxy_norm)]。"""
    gt = {}
    for fn in os.listdir(lbl_dir):
        if not fn.endswith(".txt"):
            continue
        items = []
        with open(os.path.join(lbl_dir, fn), encoding="utf-8") as f:
            for ln in f:
                p = ln.split()
                if not p or (len(p) - 1) < 4:
                    continue
                c = int(p[0])
                v = np.array([float(x) for x in p[1:]], dtype=float)
                xs, ys = v[0::2], v[1::2]
                items.append((c, (xs.min(), ys.min(), xs.max(), ys.max())))
        gt[fn[:-4]] = items
    return gt


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def match_at(preds, gts, thr):
    """在给定阈值下统计 TP/FP/FN（同类 + IoU>=0.5 + 一对一）。"""
    tp = fp = 0
    used = set()
    p = [(c, b, s) for c, b, s in preds if s >= thr]
    for c, b, _s in sorted(p, key=lambda x: -x[2]):
        best_i, best_iou = -1, 0.0
        for i, (gc, gb) in enumerate(gts):
            if i in used or gc != c:
                continue
            v = iou(b, gb)
            if v > best_iou:
                best_i, best_iou = i, v
        if best_iou >= 0.5:
            used.add(best_i)
            tp += 1
        else:
            fp += 1
    fn = len(gts) - len(used)
    return tp, fp, fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=os.path.join(BASE_DIR, "model", "best.pt"))
    ap.add_argument("--split", default="val", choices=["train", "val", "all"])
    ap.add_argument("--dry", action="store_true", help="只打印，不写文件")
    args = ap.parse_args()

    from ultralytics import YOLO

    splits = ["train", "val"] if args.split == "all" else [args.split]
    gt = {}
    for sp in splits:
        img_dir = os.path.join(DATASET_DIR, "images", sp)
        lbl_dir = os.path.join(DATASET_DIR, "labels", sp)
        gt.update(load_gt(lbl_dir))

    model = YOLO(args.weights)
    per_image = {}
    files = []
    for sp in splits:
        d = os.path.join(DATASET_DIR, "images", sp)
        files += [os.path.join(sp, f) for f in sorted(os.listdir(d))
                  if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    print("推理 %d 张（%s）..." % (len(files), args.split))
    for rel in files:
        sp, fn = os.path.split(rel)
        r = model.predict(os.path.join(DATASET_DIR, "images", sp, fn), conf=min(GRID), verbose=False)[0]
        h, w = r.orig_shape
        preds = []
        if r.boxes is not None and len(r.boxes):
            xyxy = r.boxes.xyxy.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy().astype(int)
            sc = r.boxes.conf.cpu().numpy()
            for k in range(len(cls)):
                x1, y1, x2, y2 = xyxy[k]
                preds.append((int(cls[k]),
                              (x1 / w, y1 / h, x2 / w, y2 / h),
                              float(sc[k])))
        per_image[os.path.splitext(fn)[0]] = preds

    print("")
    print("=" * 76)
    print("按类搜阈值   权重 %s" % os.path.relpath(args.weights, BASE_DIR).replace("\\", "/"))
    print("=" * 76)
    print("%-14s %6s %8s %10s %8s %8s %8s" %
          ("类别", "实例", "原阈值F1", "建议阈值", "P", "R", "F1"))
    print("-" * 76)

    out = {}
    for c in range(len(NAMES)):
        # 该类在验证集里的全部预测与全部 GT
        preds = []
        gts = []
        for key, items in gt.items():
            gts += [g for g in items if g[0] == c]
            preds += [p for p in per_image.get(key, []) if p[0] == c]
        n_gt = len(gts)
        if n_gt == 0:
            continue

        # 原阈值下的 F1
        tp0, fp0, fn0 = match_at(preds, gts, GLOBAL_DEFAULT)
        p0 = tp0 / max(tp0 + fp0, 1)
        r0 = tp0 / max(tp0 + fn0, 1)
        f0 = 2 * p0 * r0 / max(p0 + r0, 1e-9)

        if n_gt < MIN_INSTANCES:
            out[c] = {"name": NAMES[c], "threshold": GLOBAL_DEFAULT,
                      "reason": "实例仅 %d 个，沿用全局" % n_gt}
            print("%-14s %6d %8.3f %10s %8s %8s %8s"
                  % (NAMES[c], n_gt, f0, "沿用0.40", "-", "-", "-"))
            continue

        best = None
        for t in GRID:
            tp, fp, fn = match_at(preds, gts, t)
            p = tp / max(tp + fp, 1)
            r = tp / max(tp + fn, 1)
            f = 2 * p * r / max(p + r, 1e-9)
            if best is None or f > best[3] + 1e-9:
                best = (t, p, r, f)
        # 样本量收缩：实例越少的类，最优阈值越不可信，越要向全局值 0.40 靠。
        # 实例 >= 40 个完全信任搜索结果，22 个的 laptop 只信一半左右。
        shrink = min(1.0, n_gt / 40.0)
        t = shrink * best[0] + (1.0 - shrink) * GLOBAL_DEFAULT
        t = min(max(t, LO), HI)
        tp, fp, fn = match_at(preds, gts, t)
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        f = 2 * p * r / max(p + r, 1e-9)
        out[c] = {"name": NAMES[c], "threshold": round(t, 2),
                  "precision": round(p, 3), "recall": round(r, 3), "f1": round(f, 3)}
        print("%-14s %6d %8.3f %10.2f %8.3f %8.3f %8.3f"
              % (NAMES[c], n_gt, f0, t, p, r, f))

    print("-" * 76)
    # 合起来看：全用新阈值 vs 全用 0.40
    for label, use in (("统一 0.40", None), ("按类阈值", out)):
        TP = FP = FN = 0
        for c in range(len(NAMES)):
            preds, gts = [], []
            for key, items in gt.items():
                gts += [g for g in items if g[0] == c]
                preds += [p for p in per_image.get(key, []) if p[0] == c]
            t = GLOBAL_DEFAULT if use is None else use.get(c, {}).get("threshold", GLOBAL_DEFAULT)
            tp, fp, fn = match_at(preds, gts, t)
            TP += tp
            FP += fp
            FN += fn
        P = TP / max(TP + FP, 1)
        R = TP / max(TP + FN, 1)
        F = 2 * P * R / max(P + R, 1e-9)
        print("%-14s TP=%3d FP=%3d FN=%3d   P=%.3f R=%.3f F1=%.3f"
              % (label, TP, FP, FN, P, R, F))

    if not args.dry:
        with open(OUT_JSON, "w", encoding="utf-8") as f:
            json.dump({"weights": os.path.relpath(args.weights, BASE_DIR).replace("\\", "/"),
                       "split": args.split,
                       "global_default": GLOBAL_DEFAULT,
                       "per_class": out}, f, ensure_ascii=False, indent=2)
        print("\n已写入 %s" % OUT_JSON)
    return 0


if __name__ == "__main__":
    sys.exit(main())
