# -*- coding: utf-8 -*-
"""
R2 回标：用自训 best.pt 对全部图片重新预标注，并单独存一份。

为什么要单独存一份目录，而不是直接覆盖 data/annotations/：
  - R1 的草稿要留着给人工修正（任务书要求 ≥60 条修正记录），
    覆盖了就看不到"改之前长什么样"。
  - 修正率是 R1/R2 的核心对比指标：
        R1 修正率 = R1 草稿里需要人工改的对象数 / R1 草稿总对象数
        R2 修正率 = R2 草稿里需要人工改的对象数 / R2 草稿总对象数
    两个数要能同时算出来，草稿就必须分开存。

"需要人工改"怎么判定？用 COCO 官方真值做参照：
    预测框跟真值框 IoU < 0.5  -> 算多标（FP）或漏标（FN），都要改
    IoU >= 0.5                 -> 算对，不用改
  这是任务书里"修正率 = 修正对象数 / 总对象数"的可操作化定义。

用法：
  python tools/relabel_r2.py
  python tools/relabel_r2.py --conf 0.15
"""
import argparse
import json
import os
import shutil
import sys
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from prelabel import PrelabelJob, run_relabel_job   # noqa: E402
from core import Object                              # noqa: E402

IMAGE_DIR = os.path.join(BASE_DIR, "data", "images")
R1_DIR = os.path.join(BASE_DIR, "data", "annotations")
R2_DIR = os.path.join(BASE_DIR, "data", "annotations_r2")
WEIGHTS = os.path.join(BASE_DIR, "model", "best.pt")


def save_anno_to(dirpath, name, objs, w, h):
    """把对象列表写成标注文件（结构与 core.Annotation.to_dict 一致）。"""
    os.makedirs(dirpath, exist_ok=True)
    data = {
        "image": name,
        "width": w,
        "height": h,
        "objects": [o.to_dict() for o in objs],
    }
    with open(os.path.join(dirpath, os.path.splitext(name)[0] + ".json"),
              "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", type=float, default=0.15,
                    help="置信度阈值，和 R1 保持一致才能公平对比")
    ap.add_argument("--weights", default=WEIGHTS)
    ap.add_argument("--out", default=R2_DIR)
    args = ap.parse_args()

    if not os.path.exists(args.weights):
        print("找不到自训权重:", args.weights)
        print("请先运行 python train.py")
        sys.exit(1)
    if not os.path.isdir(IMAGE_DIR):
        print("找不到图片目录:", IMAGE_DIR)
        sys.exit(1)

    from PIL import Image

    job = PrelabelJob("relabel_r2", 0)
    print("R2 回标开始，权重=%s conf=%.2f" % (args.weights, args.conf))
    t0 = time.time()
    run_relabel_job(job, IMAGE_DIR, args.weights, conf=args.conf, round_tag="R2")
    print("模型推理完成，耗时 %.1f 秒" % (time.time() - t0))

    total = 0
    for name, res in job.results.items():
        objs = [Object.from_dict(o) for o in res.get("objects", [])]
        img_path = os.path.join(IMAGE_DIR, name)
        try:
            with Image.open(img_path) as im:
                w, h = im.size
        except Exception:  # noqa: BLE001
            w = h = 960
        save_anno_to(args.out, name, objs, w, h)
        total += len(objs)

    print("R2 草稿已写入", os.path.relpath(args.out, BASE_DIR))
    print("  图片 %d 张，对象 %d 个" % (len(job.results), total))

    # R1 做个简单对照，方便肉眼确认
    r1_files = [f for f in os.listdir(R1_DIR) if f.endswith(".json")] \
        if os.path.isdir(R1_DIR) else []
    r1_total = 0
    for f in r1_files:
        with open(os.path.join(R1_DIR, f), "r", encoding="utf-8") as fp:
            r1_total += len(json.load(fp).get("objects", []))
    if r1_files:
        print("  对照 R1：%d 张图，%d 个对象" % (len(r1_files), r1_total))


if __name__ == "__main__":
    main()
