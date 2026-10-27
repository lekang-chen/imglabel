# -*- coding: utf-8 -*-
"""
对比「同一份权重，开/关 TTA」与不同输入分辨率的验证指标。

【为什么要试这个】
训练侧能做的都做了（冻结主干把掩码 mAP 从 0.442 提到 0.538）。
推理侧还有一个不花钱的杠杆：TTA（Test-Time Augmentation，测试时增强）——
把同一张图翻转/多尺度各推理一遍再融合结果，通常能小幅提升 mAP，
代价是推理耗时成倍增加。

【本脚本报什么】
对每个候选权重，分别测：
  base    imgsz=640，不开 TTA（现有流程的口径）
  tta     imgsz=640，开 TTA
  r960    imgsz=960，不开 TTA

用法：
  python tools/eval_tta.py                       # 测默认候选
  python tools/eval_tta.py runs/exp_head_long/weights/best.pt
"""
import json
import os
import sys
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

DATASET_DIR = os.path.join(BASE_DIR, "dataset")
TMP_YAML = os.path.join(DATASET_DIR, "_eval.yaml")
OUT_JSON = os.path.join(BASE_DIR, "data", "eval_tta.json")

CANDIDATES = [
    "model/best.pt",
    "runs/exp_head/weights/best.pt",
    "runs/exp_lowlr/weights/best.pt",
    "runs/exp_head_long/weights/best.pt",
]


def write_yaml():
    names = ["person", "chair", "dining table", "couch", "tv",
             "laptop", "cup", "bottle", "potted plant", "book"]
    lines = ["path: %s" % DATASET_DIR.replace("\\", "/"),
             "train: images/train", "val: images/val",
             "nc: %d" % len(names), "names:"]
    lines += ["  %d: %s" % (i, c) for i, c in enumerate(names)]
    with open(TMP_YAML, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def measure(model, imgsz, augment):
    t0 = time.time()
    r = model.val(data=TMP_YAML, imgsz=imgsz, batch=8,
                  augment=augment, verbose=False)
    return {
        "box_map50": round(float(r.box.map50), 4),
        "mask_map50": round(float(r.seg.map50), 4),
        "box_map50_95": round(float(r.box.map), 4),
        "mask_map50_95": round(float(r.seg.map), 4),
        "seconds": round(time.time() - t0, 1),
    }


def main():
    from ultralytics import YOLO

    paths = sys.argv[1:] or CANDIDATES
    paths = [p if os.path.isabs(p) else os.path.join(BASE_DIR, p) for p in paths]
    paths = [p for p in paths if os.path.exists(p)]
    if not paths:
        print("没有可用的权重文件")
        return 1

    write_yaml()
    out = []
    try:
        for p in paths:
            rel = os.path.relpath(p, BASE_DIR).replace("\\", "/")
            print("")
            print("=" * 62)
            print("权重 %s" % rel)
            print("=" * 62)
            m = YOLO(p)
            row = {"weights": rel}
            for tag, imgsz, aug in [("base", 640, False),
                                    ("tta", 640, True),
                                    ("r960", 960, False)]:
                r = measure(m, imgsz, aug)
                row[tag] = r
                print("  %-5s imgsz=%-4d TTA=%-5s 检测 %.3f 掩码 %.3f (%.1fs)"
                      % (tag, imgsz, aug, r["box_map50"], r["mask_map50"],
                         r["seconds"]))
            out.append(row)
    finally:
        if os.path.exists(TMP_YAML):
            os.remove(TMP_YAML)

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("")
    print("结果已写入 %s" % OUT_JSON)

    # 排个名，按掩码 mAP50（分割任务的主指标）
    print("")
    print("按掩码 mAP50 排序：")
    flat = []
    for row in out:
        for tag in ("base", "tta", "r960"):
            flat.append((row[tag]["mask_map50"], row["weights"], tag))
    for mask, w, tag in sorted(flat, reverse=True):
        print("  %.3f  %-40s %s" % (mask, w, tag))
    return 0


if __name__ == "__main__":
    sys.exit(main())
