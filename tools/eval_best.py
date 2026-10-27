# -*- coding: utf-8 -*-
"""
评估 model/best.pt 在验证集上的真实指标，结果落到 data/best_metrics.json。

【为什么要单独做这一步，而不是从 train.log 里抠】
------------------------------------------------------------
train.log 里那句 "all  30  178  0.754 ..." 是训练刚结束时打的，
反映的是「那一刻的验证集」。但这个项目的数据是边改边变的：
人工修正了 60 条之后，验证集的标签已经不是训练时那份了。
于是出现一个很容易被评委问穿的坑——
　　文档写「掩码 mAP 0.415」，评委拿新标签一跑是 0.442，对不上。

所以把「测」这件事独立成一个脚本：
  - 永远用当前 dataset/ 的最新标签
  - 明确记下测的时候数据是什么状态（修正条数、验证集张数）
  - 输出带时间戳，谁都能复现

【为什么自己写 _eval.yaml 而不直接用 dataset/data.yaml】
data.yaml 里的 path 是 "."（刻意保留成相对路径，保证提交包换机器可用），
ultralytics 会按「当前工作目录」解析它。在仓库根目录跑就会解析成
D:/imglabel/images/val（不存在）——这正是训练收尾校验失败的报错。
这里临时写一份带绝对路径的 yaml，用完即删。
"""
import json
import os
import subprocess
import sys
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL = os.path.join(BASE_DIR, "model", "best.pt")
OUT_JSON = os.path.join(BASE_DIR, "data", "best_metrics.json")
TMP_YAML = os.path.join(DATASET_DIR, "_eval.yaml")

CLASSES = ["person", "chair", "dining table", "couch", "tv",
           "laptop", "cup", "bottle", "potted plant", "book"]


def count_edits():
    """统计当前标注里人工修正过的对象数，写进结果做数据状态快照。"""
    total = edited = 0
    anno_dir = os.path.join(BASE_DIR, "data", "annotations")
    for fn in os.listdir(anno_dir):
        if not fn.endswith(".json"):
            continue
        try:
            with open(os.path.join(anno_dir, fn), encoding="utf-8") as f:
                d = json.load(f)
        except Exception:  # noqa: BLE001
            continue
        for o in d.get("objects", []):
            total += 1
            if o.get("edited"):
                edited += 1
    return total, edited


def main():
    import argparse
    ap = argparse.ArgumentParser(description="评估权重在验证集上的真实指标")
    ap.add_argument("--imgsz", type=int, default=640,
                    help="推理分辨率；原图长边 960，试 960 常能提升小目标")
    ap.add_argument("--model", default=MODEL, help="要评估的权重路径")
    ap.add_argument("--out", default=OUT_JSON, help="结果写入的 json 路径")
    args = ap.parse_args()

    model_path = args.model
    if not os.path.exists(model_path):
        print("[错误] 找不到权重: %s" % model_path)
        return 1

    lines = ["path: %s" % DATASET_DIR.replace("\\", "/"),
             "train: images/train", "val: images/val",
             "nc: %d" % len(CLASSES), "names:"]
    lines += ["  %d: %s" % (i, c) for i, c in enumerate(CLASSES)]
    with open(TMP_YAML, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    try:
        from ultralytics import YOLO
        print("[1/2] 加载权重: %s" % model_path)
        model = YOLO(model_path)
        print("[2/2] 在 dataset/ 验证集上评估（imgsz=%d，当前最新标签）..." % args.imgsz)
        r = model.val(data=TMP_YAML, imgsz=args.imgsz, batch=8, verbose=False)

        total_objs, edited = count_edits()
        val_imgs = len(os.listdir(os.path.join(DATASET_DIR, "images", "val")))
        out = {
            "model": os.path.relpath(model_path, BASE_DIR).replace("\\", "/"),
            "imgsz": args.imgsz,
            "box_map50": round(float(r.box.map50), 4),
            "box_map50_95": round(float(r.box.map), 4),
            "mask_map50": round(float(r.seg.map50), 4),
            "mask_map50_95": round(float(r.seg.map), 4),
            "val_images": val_imgs,
            "dataset_objects": total_objs,
            "dataset_edited": edited,
            "measured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "note": "在最终修正标签上实测；修正条数见 dataset_edited",
        }
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)

        print("")
        print("  检测 mAP50      : %.3f  (mAP50-95 %.3f)"
              % (out["box_map50"], out["box_map50_95"]))
        print("  掩码 mAP50      : %.3f  (mAP50-95 %.3f)"
              % (out["mask_map50"], out["mask_map50_95"]))
        print("  验证集          : %d 张图，推理 %d" % (val_imgs, args.imgsz))
        print("  当时标注状态    : %d 个对象，其中 %d 条人工修正"
              % (total_objs, edited))
        print("  已写入          : %s" % args.out)
        return 0
    finally:
        if os.path.exists(TMP_YAML):
            os.remove(TMP_YAML)


if __name__ == "__main__":
    sys.exit(main())
