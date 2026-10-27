# -*- coding: utf-8 -*-
"""
用「人工要动多少东西」来选模型，而不是只看 mAP。

【为什么还要这个脚本】
mAP 是排序指标，不等于工作量。对一个预标注工具来说，人真正付出的成本是：
    误检的框要删掉      —— 每个 FP 一次点击
    漏掉的对象要手画    —— 每个 FN 一笔描边（最贵）
    框对了但掩码歪      —— 每个都要拖顶点（中等）
一个模型 mAP 高，但如果它精确率低、吐一堆假框，人反而更累。
所以这里直接统计这三类工作量，并且扫一遍置信度阈值，
找出「总工作量最小」的组合（模型 + 阈值）。

【判定口径】
  框匹配：同类且 IoU>=0.5 才算找到
  掩码偏差：配对成功的对象里，掩码 IoU < QUALITY_OK 视为"形状要修"
  配对成功后按班级联，一个真值只能被一个预测占用

【数据】
真值用 dataset/labels/val/*.txt —— 这是人工修正过 60 条之后的最终标签，
是当前项目里质量最高的参照。

用法：
  python tools/eval_model_draft.py
  python tools/eval_model_draft.py --split train
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from PIL import Image, ImageDraw

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

DATASET_DIR = os.path.join(BASE_DIR, "dataset")
OUT_JSON = os.path.join(BASE_DIR, "data", "draft_workload.json")

CLASSES = ["person", "chair", "dining table", "couch", "tv",
           "laptop", "cup", "bottle", "potted plant", "book"]

# 候选权重：现有线上模型 + 三组实验的最好结果
CANDIDATES = [
    "model/best.pt",
    "runs/exp_head/weights/best.pt",
    "runs/exp_head_long/weights/best.pt",
    "runs/exp_lowlr/weights/best.pt",
]

# 置信度扫描点。0.05 是"能吐的都吐"，0.5 是"很确信才吐"
THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]

IOU_MATCH = 0.5     # 框匹配门槛
QUALITY_OK = 0.75   # 掩码 IoU 高于这个就当"不用修形状"
TARGET_R = 0.65     # 同口径对比时统一的召回率点位
RASTER = 160        # 掩码栅格化分辨率（算 IoU 用，不用太高）


# ---------------------------------------------------------------- 真值读取
def load_gt(label_path):
    """YOLO 分割标签 -> [(cls_id, mask_bool[RASTER, RASTER], bbox_norm)]"""
    out = []
    if not os.path.exists(label_path):
        return out
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 7:          # 至少 3 个点才构成多边形
                continue
            cid = int(float(parts[0]))
            vals = [float(v) for v in parts[1:]]
            pts = list(zip(vals[0::2], vals[1::2]))
            if len(pts) < 3:
                continue
            mask = rasterize([pts])
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            out.append((cid, mask, (min(xs), min(ys), max(xs), max(ys))))
    return out


def rasterize(polys):
    """归一化多边形 -> 布尔掩码（RASTER x RASTER）"""
    img = Image.new("1", (RASTER, RASTER), 0)
    d = ImageDraw.Draw(img)
    for pts in polys:
        px = [(min(max(x, 0.0), 1.0) * (RASTER - 1),
               min(max(y, 0.0), 1.0) * (RASTER - 1)) for x, y in pts]
        if len(px) >= 3:
            d.polygon(px, fill=1)
    return np.array(img, dtype=bool)


# ---------------------------------------------------------------- 几何
def box_iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    u = aa + bb - inter
    return inter / u if u > 0 else 0.0


def mask_iou(m1, m2):
    inter = np.logical_and(m1, m2).sum()
    if inter == 0:
        return 0.0
    u = np.logical_or(m1, m2).sum()
    return float(inter) / float(u) if u else 0.0


# ---------------------------------------------------------------- 主评估
def evaluate(model, img_dir, lbl_dir, conf_floor):
    """跑一遍预测，把结果存起来（阈值扫描在内存里做，避免重复推理）"""
    names = sorted(os.listdir(img_dir))
    per_image = []
    for name in names:
        stem = os.path.splitext(name)[0]
        gt = load_gt(os.path.join(lbl_dir, stem + ".txt"))
        img_path = os.path.join(img_dir, name)
        r = model.predict(img_path, conf=conf_floor, imgsz=640,
                          verbose=False, device=0)[0]
        preds = []
        if r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            clss = r.boxes.cls.cpu().numpy().astype(int)
            h, w = r.orig_shape
            polys_all = r.masks.xy if r.masks is not None else None
            for i in range(len(xyxy)):
                x0, y0, x1, y1 = xyxy[i]
                bn = (x0 / w, y0 / h, x1 / w, y1 / h)
                if polys_all is not None and i < len(polys_all) and len(polys_all[i]) >= 3:
                    pts = [(p[0] / w, p[1] / h) for p in polys_all[i]]
                    mk = rasterize([pts])
                else:
                    mk = rasterize([[(bn[0], bn[1]), (bn[2], bn[1]),
                                     (bn[2], bn[3]), (bn[0], bn[3])]])
                preds.append((int(clss[i]), float(confs[i]),
                              tuple(float(v) for v in bn), mk))
        per_image.append((gt, preds))
    return per_image


def score_at(per_image, thr):
    """在给定阈值下统计工作量"""
    tp = fp = fn = 0
    shape_bad = 0
    iou_sum = 0.0
    for gt, preds in per_image:
        kept = [p for p in preds if p[1] >= thr]
        used = set()
        for cid, conf, bn, mk in kept:
            best_j, best_iou = -1, IOU_MATCH
            for gi, (gc, _gm, gb) in enumerate(gt):
                if gi in used or gc != cid:
                    continue
                v = box_iou(bn, gb)
                if v >= best_iou:
                    best_iou, best_j = v, gi
            if best_j >= 0:
                used.add(best_j)
                tp += 1
                iou_sum += best_iou
                mi = mask_iou(mk, gt[best_j][1])
                if mi < QUALITY_OK:
                    shape_bad += 1
            else:
                fp += 1
        fn += len(gt) - len(used)

    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    # 工作量：漏检按 2 倍计权（手画一笔比点一下删除贵得多）
    workload = fp + fn * 2 + shape_bad * 0.5
    return {
        "thr": thr, "tp": tp, "fp": fp, "fn": fn, "shape_bad": shape_bad,
        "precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4),
        "mean_box_iou": round(iou_sum / tp, 4) if tp else 0.0,
        "workload": round(workload, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["train", "val"])
    ap.add_argument("--imgs", nargs="+", default=None, help="候选权重，可传多个")
    args = ap.parse_args()

    from ultralytics import YOLO

    img_dir = os.path.join(DATASET_DIR, "images", args.split)
    lbl_dir = os.path.join(DATASET_DIR, "labels", args.split)
    paths = args.imgs or CANDIDATES
    paths = [p if os.path.isabs(p) else os.path.join(BASE_DIR, p) for p in paths]
    paths = [p for p in paths if os.path.exists(p)]

    print("评估集: %s（%d 张）" % (args.split, len(os.listdir(img_dir))))
    results = []
    for p in paths:
        rel = os.path.relpath(p, BASE_DIR).replace("\\", "/")
        print("")
        print("=" * 68)
        print("权重 %s" % rel)
        print("=" * 68)
        t0 = time.time()
        per_image = evaluate(YOLO(p), img_dir, lbl_dir, min(THRESHOLDS) / 2)
        rows = [score_at(per_image, t) for t in THRESHOLDS]
        print("  %5s %5s %5s %5s %6s %6s %6s %7s %9s" %
              ("conf", "TP", "FP", "FN", "P", "R", "F1", "形状待修", "工作量"))
        for r in rows:
            print("  %5.2f %5d %5d %5d %6.3f %6.3f %6.3f %7d %9.1f" %
                  (r["thr"], r["tp"], r["fp"], r["fn"], r["precision"],
                   r["recall"], r["f1"], r["shape_bad"], r["workload"]))
        best = min(rows, key=lambda r: r["workload"])
        print("  -> 工作量最小: conf=%.2f（FP=%d FN=%d 形状待修=%d，合计 %.1f，用时 %.0fs）" %
              (best["thr"], best["fp"], best["fn"], best["shape_bad"],
               best["workload"], time.time() - t0))
        results.append({"weights": rel, "rows": rows, "best": best})

    # 全局排名：谁的"最小工作量"最低
    print("")
    print("=" * 68)
    print("按「同一个模型的最优阈值」排序（工作量越低越好）")
    print("=" * 68)
    for row in sorted(results, key=lambda x: x["best"]["workload"]):
        b = row["best"]
        print("  %-38s 工作量 %6.1f  (conf=%.2f, P=%.3f R=%.3f F1=%.3f)"
              % (row["weights"], b["workload"], b["thr"],
                 b["precision"], b["recall"], b["f1"]))

    # 上面的"各自最优阈值"其实没法直接比：阈值不同，检出量也不同。
    # 这里补一个同口径对比——把所有模型都调到召回率接近 TARGET_R 的位置，
    # 此时谁的误检更少、形状更准，才是真的更好。
    print("")
    print("=" * 68)
    print("同口径对比：都调到召回率≈%.2f 时，谁更省人工" % TARGET_R)
    print("=" * 68)
    print("  %-38s %6s %6s %8s %7s" % ("权重", "FP", "FN", "形状待修", "工作量"))
    matched = []
    for row in results:
        cand = min(row["rows"], key=lambda r: abs(r["recall"] - TARGET_R))
        matched.append((cand["workload"], row["weights"], cand))
        print("  %-38s %6d %6d %8d %7.1f  (conf=%.2f R=%.3f)"
              % (row["weights"], cand["fp"], cand["fn"], cand["shape_bad"],
                 cand["workload"], cand["thr"], cand["recall"]))
    best_m = min(matched, key=lambda x: x[0])
    print("")
    print("  同口径最优: %s（工作量 %.1f）" % (best_m[1], best_m[0]))

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({"split": args.split, "target_recall": TARGET_R,
                   "results": results}, f, ensure_ascii=False, indent=2)
    print("")
    print("已写入 %s" % OUT_JSON)
    return 0


if __name__ == "__main__":
    sys.exit(main())
