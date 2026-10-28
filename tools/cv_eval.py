# -*- coding: utf-8 -*-
"""
5 折交叉验证：给出整个训练流程的诚实泛化指标。

【为什么要做】
之前所有对外报告的 mAP 都来自同一份固定验证集（30 张）。
评委完全可以质疑：「这 30 张是不是碰巧简单/碰巧难？」
而且第十一轮的换种子实验已经实测出：这 30 张上的掩码 mAP 波动 ±0.035，
比很多实验组间的差距还大。

5 折交叉验证的做法：150 张图分成 5 份，每次拿 1 份当验证集、其余 4 份当
训练集，训练 5 次。**每张图都恰好当过一次验证集**，得到的 5 个 mAP
取均值和标准差，就是这个流程在这份数据上的无偏估计。

【防数据泄漏的关键】
每折必须从**官方预训练权重** yolo26n-seg.pt 起步。
如果从现在的 model/best.pt 起步——它见过全部 150 张图——
那每折的验证集都被模型背过了，指标虚高，做出来的 CV 没有意义。

【配方】
固定用第十一轮胜出的 iter3_long 配方（copy_paste=0.5 / scale=0.7 /
degrees=8 / cls=1.2 / freeze=10 / 120 轮），这样 CV 报告的就是
「当前提交的训练配方」的泛化水平，和仓库里其它数字对得上。

【产出】
data/cv_results.json + 控制台汇总表。

用法：
  python tools/cv_eval.py            # 5 折全跑（约 20-30 分钟）
  python tools/cv_eval.py --folds 3  # 只跑前 3 折（快速验证脚本）
"""
import argparse
import json
import os
import random
import sys
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
RUNS_DIR = os.path.join(BASE_DIR, "runs")
OUT_JSON = os.path.join(BASE_DIR, "data", "cv_results.json")

NAMES = ["person", "chair", "dining table", "couch", "tv",
         "laptop", "cup", "bottle", "potted plant", "book"]

# iter3_long 配方（与 tools/exp_train.py 保持一致）
RECIPE = dict(
    freeze=10, lr0=0.0008, lrf=0.01, optimizer="SGD",
    epochs=120, patience=40, cos_lr=True, close_mosaic=20,
    copy_paste=0.5, scale=0.7, degrees=8.0, cls=1.2,
)

IMG_EXTS = (".jpg", ".jpeg", ".png")


def list_images():
    imgs = []
    for split in ("train", "val"):
        d = os.path.join(DATASET_DIR, "images", split)
        imgs += [os.path.join(d, f) for f in sorted(os.listdir(d))
                 if f.lower().endswith(IMG_EXTS)]
    return imgs


def write_split_files(fold, train_imgs, val_imgs, tmp_dir):
    """ultralytics 的 yaml 支持 train/val 指向一个图片路径列表 txt。"""
    os.makedirs(tmp_dir, exist_ok=True)
    tr_txt = os.path.join(tmp_dir, "fold%d_train.txt" % fold)
    va_txt = os.path.join(tmp_dir, "fold%d_val.txt" % fold)
    for path, items in ((tr_txt, train_imgs), (va_txt, val_imgs)):
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(items) + "\n")
    yaml_path = os.path.join(tmp_dir, "fold%d.yaml" % fold)
    lines = ["path: %s" % DATASET_DIR.replace("\\", "/"),
             "train: %s" % tr_txt.replace("\\", "/"),
             "val: %s" % va_txt.replace("\\", "/"),
             "nc: %d" % len(NAMES), "names:"]
    lines += ['  %d: "%s"' % (i, n) for i, n in enumerate(NAMES)]
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return yaml_path


def run_fold(fold, train_imgs, val_imgs, tmp_dir):
    from ultralytics import YOLO

    yaml = write_split_files(fold, train_imgs, val_imgs, tmp_dir)
    model = YOLO(os.path.join(BASE_DIR, "weights", "yolo26n-seg.pt"))
    t0 = time.time()
    model.train(
        data=yaml, imgsz=640, batch=8, device=0, workers=4,
        project=RUNS_DIR, name="cv_fold%d" % fold, exist_ok=True,
        pretrained=True, seed=42, weight_decay=0.0005,
        close_mosaic=RECIPE["close_mosaic"], verbose=False, plots=False,
        **{k: v for k, v in RECIPE.items() if k != "close_mosaic"},
    )
    best = os.path.join(RUNS_DIR, "cv_fold%d" % fold, "weights", "best.pt")
    m = YOLO(best)
    r = m.val(data=yaml, imgsz=640, batch=8, plots=False, verbose=False)
    out = {
        "fold": fold,
        "box_map50": round(float(r.box.map50), 4),
        "mask_map50": round(float(r.seg.map50), 4),
        "box_map50_95": round(float(r.box.map), 4),
        "mask_map50_95": round(float(r.seg.map), 4),
        "n_train": len(train_imgs), "n_val": len(val_imgs),
        "minutes": round((time.time() - t0) / 60, 1),
    }
    print("  折 %d: 检测 mAP50 %.4f | 掩码 mAP50 %.4f | %.1f 分钟"
          % (fold, out["box_map50"], out["mask_map50"], out["minutes"]),
          flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import numpy as np

    imgs = list_images()
    random.Random(args.seed).shuffle(imgs)
    folds = [imgs[i::args.folds] for i in range(args.folds)]

    tmp_dir = os.path.join(DATASET_DIR, "_cv")
    print("=" * 64)
    print("5 折交叉验证  %d 张图，配方 = iter3_long，官方权重起步" % len(imgs))
    print("=" * 64, flush=True)

    rows = []
    try:
        for i in range(args.folds):
            val = folds[i]
            train = [x for j, fl in enumerate(folds) if j != i for x in fl]
            rows.append(run_fold(i, train, val, tmp_dir))
    finally:
        for f in os.listdir(tmp_dir) if os.path.isdir(tmp_dir) else []:
            os.remove(os.path.join(tmp_dir, f))
        try:
            os.rmdir(tmp_dir)
        except OSError:
            pass

    box = [r["box_map50"] for r in rows]
    mask = [r["mask_map50"] for r in rows]
    summary = {
        "recipe": RECIPE,
        "base_weights": "weights/yolo26n-seg.pt",
        "n_images": len(imgs), "n_folds": args.folds, "seed": args.seed,
        "folds": rows,
        "box_map50_mean": round(float(np.mean(box)), 4),
        "box_map50_std": round(float(np.std(box)), 4),
        "mask_map50_mean": round(float(np.mean(mask)), 4),
        "mask_map50_std": round(float(np.std(mask)), 4),
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("")
    print("=" * 64)
    print("交叉验证汇总（%d 折）" % args.folds)
    print("=" * 64)
    for r in rows:
        print("  折 %d: 检测 %.3f  掩码 %.3f" % (r["fold"], r["box_map50"], r["mask_map50"]))
    print("  ------------------------------------")
    print("  检测 mAP50: %.3f ± %.3f" % (summary["box_map50_mean"], summary["box_map50_std"]))
    print("  掩码 mAP50: %.3f ± %.3f" % (summary["mask_map50_mean"], summary["mask_map50_std"]))
    print("  结果已写入 %s" % OUT_JSON)
    return 0


if __name__ == "__main__":
    sys.exit(main())
