# -*- coding: utf-8 -*-
"""
训练策略对比实验：验证「冻结主干 + 低学习率」能否缓解灾难性遗忘。

【要解决的问题】
历次训练都出现同一个反常现象：第 1 轮 mAP 最高，之后一路下降，
patience=50 跑满早停。诊断是灾难性遗忘——
基线 yolo26n-seg.pt 在 COCO 80 类上预训练，本来就会认 person/chair/cup/...
这十类（只是类别编号和我们的十类词表对不上，所以零样本 mAP 只有 0.075）。
第 1 轮训练主要在干「类别编号对齐」这一件事，干完之后，
再用 120 张图全量微调，反而把预训练里的通用视觉特征冲掉了。

【本脚本做什么】
固定同一份数据、同一个验证集，只改训练策略，各自训一次并报 mAP：

  head        freeze=10（冻结主干，只训 head + 分割分支），lr0=0.001
  lowlr       freeze=0，lr0=0.0003，余弦退火
  baseline    freeze=0，optimizer=auto（复现现有流程，作为对照）

【为什么每个都只报 val 的 mAP，不看训练 loss】
loss 一直在降而验证指标在掉，正是过拟合/遗忘的典型表现。
对外能说的只有验证集指标，训练 loss 好看没有意义。

用法：
  python tools/exp_train.py head
  python tools/exp_train.py lowlr
  python tools/exp_train.py baseline
  python tools/exp_train.py head lowlr      # 连续跑多个
"""
import json
import os
import sys
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

DATASET_DIR = os.path.join(BASE_DIR, "dataset")
RUNS_DIR = os.path.join(BASE_DIR, "runs")
BASE_WEIGHTS = os.path.join(BASE_DIR, "weights", "yolo26n-seg.pt")
TMP_YAML = os.path.join(DATASET_DIR, "_exp.yaml")
RESULT_JSON = os.path.join(BASE_DIR, "data", "exp_train.json")

# 各策略的超参。epochs 都取 40：再多也是早停，省时间。
CONFIGS = {
    "head": dict(
        freeze=10, lr0=0.001, lrf=0.01, optimizer="SGD",
        epochs=40, patience=12, cos_lr=False,
        desc="冻结主干（前 10 层），只训检测头与分割分支",
    ),
    "lowlr": dict(
        freeze=0, lr0=0.0003, lrf=0.01, optimizer="SGD",
        epochs=40, patience=12, cos_lr=True,
        desc="全量微调但学习率压到 0.0003 并余弦退火",
    ),
    "baseline": dict(
        freeze=0, lr0=0.01, lrf=0.01, optimizer="auto",
        epochs=40, patience=12, cos_lr=False,
        desc="对照：现有流程（auto 优化器）",
    ),
    # 第二轮：在 head 胜出的基础上再调。冻结主干后参数量少、不易过拟合，
    # 可以放心把训练轮次拉长，配余弦退火让 lr 平滑衰减到接近 0。
    "head_long": dict(
        freeze=10, lr0=0.0015, lrf=0.01, optimizer="SGD",
        epochs=80, patience=25, cos_lr=True, close_mosaic=15,
        desc="冻结主干 + 余弦退火 + 80 轮（放宽早停）",
    ),
    # 第三轮：head_long 掩码最好但召回偏低（0.581），lowlr 检测最高。
    # 提高 head 的学习率，让检测头学得更充分，试着把召回报回来。
    "head_fast": dict(
        freeze=10, lr0=0.003, lrf=0.01, optimizer="SGD",
        epochs=80, patience=25, cos_lr=True, close_mosaic=15,
        desc="冻结主干 + 学习率提到 0.003 + 余弦退火 80 轮",
    ),
    # 第四轮：小物体（cup/bottle/book）在 640 下掩码最差，
    # 提到 960 训练，让分割分支看到更细的边界。训练耗时约为 640 的 2 倍。
    "head_hi": dict(
        freeze=10, lr0=0.0015, lrf=0.01, optimizer="SGD",
        epochs=60, patience=20, cos_lr=True, close_mosaic=12,
        imgsz=960,
        desc="冻结主干 + 960 分辨率训练（改善小物体掩码）",
    ),
    # 第五轮：不再从官方权重起步，而是从「当前已经在用的 best.pt」继续微调。
    # 前面几组都是从零对齐类别，天然会牺牲一部分精确率；
    # 从现成权重起步、很小的学习率，目标是「只涨不跌」。
    "continue": dict(
        base="runs/old_best_r2.pt",
        freeze=10, lr0=0.0005, lrf=0.05, optimizer="SGD",
        epochs=50, patience=18, cos_lr=True, close_mosaic=12,
        desc="从现有 best.pt 继续微调（冻结主干，极小学习率）",
    ),
}


def write_yaml():
    names = ["person", "chair", "dining table", "couch", "tv",
             "laptop", "cup", "bottle", "potted plant", "book"]
    lines = ["path: %s" % DATASET_DIR.replace("\\", "/"),
             "train: images/train", "val: images/val",
             "nc: %d" % len(names), "names:"]
    lines += ["  %d: %s" % (i, c) for i, c in enumerate(names)]
    with open(TMP_YAML, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def run_one(tag):
    from ultralytics import YOLO

    cfg = dict(CONFIGS[tag])
    desc = cfg.pop("desc")
    # close_mosaic 允许各实验自己指定，这里取出来单独传，
    # 否则会与下面 train() 里的显式参数重名报错
    close_mosaic = cfg.pop("close_mosaic", 10)
    # 输入分辨率也允许各实验自己指定，评测时必须用同一个值
    imgsz = cfg.pop("imgsz", 640)
    # 起始权重也允许覆盖：有的实验从官方权重起步，有的从现成 best.pt 继续微调
    base = cfg.pop("base", None)
    if base and not os.path.isabs(base):
        base = os.path.join(BASE_DIR, base)
    if base and not os.path.exists(base):
        raise FileNotFoundError("起始权重不存在: %s" % base)
    print("")
    print("=" * 62)
    print("实验 %-10s %s" % (tag, desc))
    print("起始权重: %s" % (os.path.relpath(base, BASE_DIR).replace("\\", "/")
                          if base else "weights/yolo26n-seg.pt"))
    print("超参: %s" % json.dumps(cfg, ensure_ascii=False))
    print("=" * 62)

    t0 = time.time()
    model = YOLO(base or BASE_WEIGHTS)
    model.train(
        data=TMP_YAML,
        imgsz=imgsz,
        batch=8,
        device=0,
        workers=4,
        project=RUNS_DIR,
        name="exp_%s" % tag,
        exist_ok=True,
        pretrained=True,
        seed=42,
        weight_decay=0.0005,
        # 小数据集：关掉最后几轮的 mosaic，让模型在真实分布上收敛
        close_mosaic=close_mosaic,
        verbose=False,
        plots=False,
        **cfg
    )

    # 用 best.pt 在当前验证集上复评（训练中自带的 val 和这次口径一致，
    # 但重新评一次能确保用的是同一份最新标签）
    best = os.path.join(RUNS_DIR, "exp_%s" % tag, "weights", "best.pt")
    m = YOLO(best)
    r = m.val(data=TMP_YAML, imgsz=imgsz, batch=8, verbose=False)
    out = {
        "tag": tag,
        "desc": desc,
        "params": cfg,
        "box_map50": round(float(r.box.map50), 4),
        "mask_map50": round(float(r.seg.map50), 4),
        "box_map50_95": round(float(r.box.map), 4),
        "mask_map50_95": round(float(r.seg.map), 4),
        "imgsz": imgsz,
        "minutes": round((time.time() - t0) / 60, 1),
        "weights": os.path.relpath(best, BASE_DIR).replace("\\", "/"),
    }
    print("  -> 检测 mAP50 %.3f | 掩码 mAP50 %.3f | 耗时 %.1f 分钟"
          % (out["box_map50"], out["mask_map50"], out["minutes"]))
    return out


def main():
    tags = sys.argv[1:] or ["head", "lowlr"]
    bad = [t for t in tags if t not in CONFIGS]
    if bad:
        print("未知实验: %s；可选 %s" % (bad, list(CONFIGS)))
        return 1

    write_yaml()
    results = []
    try:
        for t in tags:
            results.append(run_one(t))
    finally:
        if os.path.exists(TMP_YAML):
            os.remove(TMP_YAML)

    # 与现有 best.pt 的基线对比
    baseline_file = os.path.join(BASE_DIR, "data", "best_metrics.json")
    cur = None
    if os.path.exists(baseline_file):
        with open(baseline_file, encoding="utf-8") as f:
            cur = json.load(f)

    print("")
    print("=" * 62)
    print("汇总（验证集 30 张，同一份标签）")
    print("=" * 62)
    if cur:
        print("  现有 model/best.pt      检测 %.3f  掩码 %.3f"
              % (cur["box_map50"], cur["mask_map50"]))
    for r in results:
        print("  %-22s 检测 %.3f  掩码 %.3f  (%.1f 分钟)"
              % (r["tag"], r["box_map50"], r["mask_map50"], r["minutes"]))

    old = []
    if os.path.exists(RESULT_JSON):
        try:
            with open(RESULT_JSON, encoding="utf-8") as f:
                old = json.load(f)
        except Exception:  # noqa: BLE001
            old = []
    old = [o for o in old if o["tag"] not in [r["tag"] for r in results]] + results
    with open(RESULT_JSON, "w", encoding="utf-8") as f:
        json.dump(old, f, ensure_ascii=False, indent=2)
    print("")
    print("  结果已追加到 %s" % RESULT_JSON)
    return 0


if __name__ == "__main__":
    sys.exit(main())
