# -*- coding: utf-8 -*-
"""
矩形标注自动细化建议生成器。

【背景】
诊断（tools/diag_perclass.py / diag_mask_shape.py）发现：799 个标注里
78-80% 是 4 顶点矩形——早期预标注没给出掩码时的矩形兜底。
掩码指标和画布质量的下一个突破点都在这里：把矩形换成真实轮廓。

【本工具做什么】
用 yolo26s-seg（官方 COCO 权重，形状知识最完整）对每张图重新分割，
为每个 4 点矩形标注找同类别、位置重合度最高的掩码：
  - 重合度（掩码中心落点 & IoU）达标 -> 生成一份「建议多边形」
  - 不达标 -> 该矩形维持原样，宁可不动也不瞎改

【重要边界：本工具只产出建议，绝不改 data/annotations/】
自动细化绕过了人工确认，等于让模型给自己出考卷（循环论证），
答辩时会被质疑「真值是模型生成的」。正确的用法是：
  1. 跑本工具生成建议清单；
  2. 人在界面上逐张对照，认可哪条就照哪条改（或只微调顶点）；
  3. 改完重新导出数据集、重训、重评。
建议清单本身是加快人工修改速度的，不是替代人工的。

【产出】
data/refine_suggestions.json + 控制台统计。

用法：
  python tools/refine_rect_masks.py                     # 全量扫描
  python tools/refine_rect_masks.py --iou 0.55          # 放宽配对门槛
"""
import argparse
import json
import os
import sys

import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
IMG_DIR = os.path.join(DATA_DIR, "images")
ANNO_DIR = os.path.join(DATA_DIR, "annotations")
OUT_JSON = os.path.join(DATA_DIR, "refine_suggestions.json")

# 我们的十类 -> COCO 类别名（yolo26s-seg 的词表）
CLS_TO_COCO = {
    "person": "person", "chair": "chair", "dining table": "dining table",
    "couch": "couch", "tv": "tv", "laptop": "laptop", "cup": "cup",
    "bottle": "bottle", "potted plant": "potted plant", "book": "book",
}


def bbox_of(poly):
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return min(xs), min(ys), max(xs), max(ys)


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def area(poly):
    x1, y1, x2, y2 = bbox_of(poly)
    return max((x2 - x1) * (y2 - y1), 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iou", type=float, default=0.6,
                    help="矩形与 s 掩码的最小 IoU")
    ap.add_argument("--min-area", type=float, default=0.30,
                    help="s 掩码面积 / 矩形面积 的下限（太瘦说明找错对象）")
    ap.add_argument("--max-area", type=float, default=0.95,
                    help="上限（比矩形还大说明是整幅图的大掩码）")
    ap.add_argument("--weights", default=None,
                    help="默认自动找 weights/yolo26s-seg.pt")
    args = ap.parse_args()

    from PIL import Image
    from ultralytics import YOLO

    weights = args.weights or os.path.join(BASE_DIR, "weights", "yolo26s-seg.pt")
    if not os.path.exists(weights):
        sys.exit("[错误] 找不到 %s（先从镜像下载，见 ai_log 第十一轮）" % weights)

    model = YOLO(weights)
    id2name = model.names
    want = {v: k for k, v in CLS_TO_COCO.items()}   # COCO 名 -> 我们的类名

    annos = sorted(f for f in os.listdir(ANNO_DIR) if f.endswith(".json"))
    suggestions = []
    n_rect = n_hit = 0
    print("扫描 %d 份标注（s 掩码配对门槛 IoU>=%.2f）..." % (len(annos), args.iou))
    for fn in annos:
        path = os.path.join(ANNO_DIR, fn)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        rects = [o for o in data.get("objects", []) if len(o.get("polygon", [])) <= 4]
        if not rects:
            continue
        n_rect += len(rects)

        img_name = data.get("image") or (fn[:-5] + ".jpg")
        img_path = os.path.join(IMG_DIR, img_name)
        if not os.path.exists(img_path):
            continue
        w, h = Image.open(img_path).size
        r = model.predict(img_path, conf=0.25, verbose=False)[0]
        if r.masks is None:
            continue

        # 把 s 模型的掩码转成归一化多边形 + 类别名索引
        import cv2
        masks = r.masks.data.cpu().numpy()
        cand = []
        for k in range(len(masks)):
            cls_name = id2name.get(int(r.boxes.cls[k]), "")
            if cls_name not in want:
                continue
            m = masks[k] > 0.5
            if m.shape != (h, w):
                m = cv2.resize(m.astype(np.uint8), (w, h),
                               interpolation=cv2.INTER_NEAREST) > 0
            ys, xs = np.nonzero(m)
            if len(xs) < 30:
                continue
            poly = [[round(float(x) / w, 5), round(float(y) / h, 5)]
                    for x, y in zip(xs[:: max(1, len(xs) // 120)], ys[:: max(1, len(ys) // 120)])]
            cand.append({"cls": want[cls_name], "poly": poly,
                         "conf": float(r.boxes.conf[k])})

        for o in rects:
            rb = bbox_of(o["polygon"])
            best, best_iou = None, 0.0
            for c in cand:
                if c["cls"] != o.get("class"):
                    continue
                v = iou(rb, bbox_of(c["poly"]))
                if v > best_iou:
                    best, best_iou = c, v
            if best is None or best_iou < args.iou:
                continue
            ratio = area(best["poly"]) / area(o["polygon"])
            if not (args.min_area <= ratio <= args.max_area):
                continue
            n_hit += 1
            suggestions.append({
                "image": img_name,
                "class": o["class"],
                "old_polygon": o["polygon"],
                "new_polygon": best["poly"],
                "match_iou": round(best_iou, 3),
                "area_ratio": round(ratio, 3),
                "s_conf": round(best["conf"], 3),
            })

    # 按类统计
    from collections import Counter
    per_cls = Counter(s["class"] for s in suggestions)
    out = {
        "weights": os.path.relpath(weights, BASE_DIR).replace("\\", "/"),
        "min_match_iou": args.iou,
        "total_rects": n_rect,
        "suggestions": suggestions,
        "per_class": dict(per_cls),
        "note": "仅建议清单，不写入 data/annotations/；人工在界面上确认后重新导出训练",
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("")
    print("=" * 64)
    print("矩形细化建议  共 %d 个矩形，可细化 %d 个 (%.0f%%)"
          % (n_rect, n_hit, n_hit * 100.0 / max(n_rect, 1)))
    print("=" * 64)
    for name, cnt in per_cls.most_common():
        print("  %-14s %4d 条" % (name, cnt))
    print("  已写入 %s" % os.path.relpath(OUT_JSON, BASE_DIR).replace("\\", "/"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
