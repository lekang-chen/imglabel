# -*- coding: utf-8 -*-
"""
评估推理端「掩码上采样」开关对掩码指标的影响。

【背景】
YOLO 分割模型的默认输出是一组低分辨率「原型掩码」(mask_ratio=4，
即 160x160 的 32 通道原型)，再与每个实例的系数相乘、上采样到原图尺寸。
这个上采样是双线性的，边界天然发糊。

ultralytics 给了一个开关 `retina_masks=True`：让掩码直接在网络的高分辨率
特征图上算，再做一次精确裁剪。代价是显存和耗时上升。

【为什么要单独测】
这是「不重新训练就能涨指标」的唯一手段。如果有效，画布上人工要修的多边形
顶点会明显变少——这是业务口径上真正省人力的地方，不只是刷 mAP。

用法：
  python tools/eval_retina.py                 # 用 model/best.pt
  python tools/eval_retina.py --weights runs/exp_aug/weights/best.pt
"""
import argparse
import json
import os
import sys
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
TMP_YAML = os.path.join(DATASET_DIR, "_retina.yaml")

NAMES = ["person", "chair", "dining table", "couch", "tv",
         "laptop", "cup", "bottle", "potted plant", "book"]


def write_yaml():
    lines = ["path: %s" % DATASET_DIR.replace("\\", "/"),
             "train: images/train", "val: images/val",
             "nc: %d" % len(NAMES), "names:"]
    lines += ["  %d: %s" % (i, c) for i, c in enumerate(NAMES)]
    with open(TMP_YAML, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return TMP_YAML


def measure(model, yaml, retina, imgsz=640):
    t0 = time.time()
    r = model.val(data=yaml, imgsz=imgsz, batch=8, retina_masks=retina,
                  plots=False, verbose=False)
    return {
        "retina_masks": retina,
        "box_map50": round(float(r.box.map50), 4),
        "mask_map50": round(float(r.seg.map50), 4),
        "box_map": round(float(r.box.map), 4),
        "mask_map": round(float(r.seg.map), 4),
        "seconds": round(time.time() - t0, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=os.path.join(BASE_DIR, "model", "best.pt"))
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    from ultralytics import YOLO

    yaml = write_yaml()
    try:
        model = YOLO(args.weights)
        print("=" * 64)
        print("权重: %s" % args.weights)
        print("=" * 64)
        rows = []
        for retina in (False, True):
            m = measure(model, yaml, retina, args.imgsz)
            rows.append(m)
            print("  retina_masks=%-5s 检测 mAP50 %.4f | 掩码 mAP50 %.4f | 掩码 mAP50-95 %.4f | %.1fs"
                  % (retina, m["box_map50"], m["mask_map50"], m["mask_map"], m["seconds"]))

        d50 = rows[1]["mask_map50"] - rows[0]["mask_map50"]
        d5095 = rows[1]["mask_map"] - rows[0]["mask_map"]
        db = rows[1]["box_map50"] - rows[0]["box_map50"]
        print("-" * 64)
        print("  开启后变化: 检测 %+.4f | 掩码 mAP50 %+.4f | 掩码 mAP50-95 %+.4f"
              % (db, d50, d5095))
        print("  耗时变化: %.1fs -> %.1fs (%.2fx)"
              % (rows[0]["seconds"], rows[1]["seconds"],
                 rows[1]["seconds"] / max(rows[0]["seconds"], 0.01)))

        out = os.path.join(BASE_DIR, "data", "retina_eval.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump({"weights": args.weights, "imgsz": args.imgsz, "rows": rows},
                      f, ensure_ascii=False, indent=2)
        print("  结果已写入 %s" % out)
    finally:
        if os.path.exists(TMP_YAML):
            os.remove(TMP_YAML)
    return 0


if __name__ == "__main__":
    sys.exit(main())
