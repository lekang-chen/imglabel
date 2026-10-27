# -*- coding: utf-8 -*-
"""
逐类诊断：当前模型在哪几类上拖后腿。

【为什么需要】
整体 mAP 是十类的平均，一条 0.35 的 tv 会把 0.65 的平均硬拉下来。
不做逐类拆解，就不知道该补数据、该加增强、还是该调阈值。

同时输出「每类实例数」，用来判断低分是数据太少还是确实难学。

用法：
  python tools/diag_perclass.py
  python tools/diag_perclass.py --weights runs/exp_aug/weights/best.pt
"""
import argparse
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

NAMES = ["person", "chair", "dining table", "couch", "tv",
         "laptop", "cup", "bottle", "potted plant", "book"]


def write_yaml():
    path = os.path.join(DATASET_DIR, "_diag.yaml")
    lines = ["path: %s" % DATASET_DIR.replace("\\", "/"),
             "train: images/train",
             "val: images/val",
             "nc: %d" % len(NAMES),
             "names:"]
    lines += ['  %d: "%s"' % (i, n) for i, n in enumerate(NAMES)]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path


def count_train_instances():
    d = os.path.join(DATASET_DIR, "labels", "train")
    cnt = {i: 0 for i in range(len(NAMES))}
    if os.path.isdir(d):
        for fn in os.listdir(d):
            with open(os.path.join(d, fn), encoding="utf-8") as f:
                for ln in f:
                    parts = ln.split()
                    if parts:
                        cnt[int(parts[0])] = cnt.get(int(parts[0]), 0) + 1
    return cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=os.path.join(BASE_DIR, "model", "best.pt"))
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    from ultralytics import YOLO

    yaml = write_yaml()
    try:
        model = YOLO(args.weights)
        r = model.val(data=yaml, imgsz=args.imgsz, batch=8,
                      plots=False, verbose=False)
    finally:
        if os.path.exists(yaml):
            os.remove(yaml)

    n_train = count_train_instances()
    rows = []
    for i, name in enumerate(NAMES):
        rows.append((name, n_train.get(i, 0), float(r.box.p[i]), float(r.box.r[i]),
                     float(r.box.ap50[i]), float(r.seg.ap50[i]), float(r.seg.ap[i])))

    print("=" * 74)
    print("逐类诊断  权重: %s" % os.path.relpath(args.weights, BASE_DIR).replace("\\", "/"))
    print("=" * 74)
    print("%-14s %7s %8s %8s %8s %9s %10s" %
          ("类别", "训练数", "box_P", "box_R", "box50", "mask50", "mask50-95"))
    print("-" * 74)
    for name, n, p, rec, b50, m50, m5095 in rows:
        print("%-14s %7d %8.3f %8.3f %8.3f %9.3f %10.3f"
              % (name, n, p, rec, b50, m50, m5095))
    print("-" * 74)
    print("%-14s %7d %8.3f %8.3f %8.3f %9.3f %10.3f"
          % ("全部", sum(n_train.values()), float(r.box.mp), float(r.box.mr),
             float(r.box.map50), float(r.seg.map50), float(r.seg.map)))

    weakest = sorted(rows, key=lambda x: x[5])[:3]
    print("\n最弱三类（按 mask mAP50）：")
    for name, n, p, rec, b50, m50, m5095 in weakest:
        print("  %-14s mask50 %.3f  (训练样本仅 %d 个)" % (name, m50, n))
    return 0


if __name__ == "__main__":
    sys.exit(main())
