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

    # ==================== 第六轮（R4）：增强 + 多轮迭代 ====================
    # 【为什么改走增强这条路】
    # 逐类诊断（tools/diag_perclass.py）暴露了两件事：
    #   1) 十类里有 80% 的标注是「4 点矩形」，也就是早期预标注没给出掩码时
    #      的矩形兜底。椅子的真实轮廓只占它外接框约四成面积，IoU 天然过不了
    #      0.5，所以 chair 的掩码 mAP50 卡在 0.20、laptop 卡在 0.00。
    #      这层天花板靠调超参突破不了。
    #   2) 能靠训练改善的是检测侧：person 召回 0.478、bottle 0.291、
    #      laptop 0.333，都是「框找不全」。120 张图的小数据集，提升检测
    #      最有效的手段是增强，而不是加轮数。
    # copy_paste 是实例分割专用增强：把一个实例的多边形裁剪下来贴到另一张图上，
    # 等于不额外采图就扩充了「同一类在不同背景/尺度下」的样本。
    "cont_aug": dict(
        base="runs/old_best_r2.pt",
        freeze=10, lr0=0.0005, lrf=0.05, optimizer="SGD",
        epochs=80, patience=25, cos_lr=True, close_mosaic=15,
        copy_paste=0.4, scale=0.6, degrees=5.0,
        desc="对照 continue：同超参 + copy_paste 增强（看增强本身的净收益）",
    ),
    # 在 cont_aug 的基础上放开主干：增强已经把「过拟合 120 张图」的风险压住了，
    # 让主干也跟着这批数据调整，检测头才有机会学到更贴合本项目的特征。
    "cont_aug_all": dict(
        base="runs/old_best_r2.pt",
        freeze=0, lr0=0.0003, lrf=0.02, optimizer="SGD",
        epochs=90, patience=30, cos_lr=True, close_mosaic=15,
        copy_paste=0.4, scale=0.6, degrees=5.0,
        desc="不冻结主干 + copy_paste 增强（靠增强抗遗忘）",
    ),
    # 用户要的「多训练几次」：从当前在用的 best.pt 再来一轮，同策略只降学习率。
    # 这条线的意义是验证迭代是否还能继续小涨，涨不动就说明已经收敛。
    "iter2": dict(
        base="model/best.pt",
        freeze=10, lr0=0.0003, lrf=0.05, optimizer="SGD",
        epochs=60, patience=20, cos_lr=True, close_mosaic=12,
        copy_paste=0.3, scale=0.5,
        desc="从当前 best.pt 迭代第二轮（降学习率 + 轻增强）",
    ),
    # 更激进的一档：强增强 + 长训练 + 提高分类损失权重。
    # cls 提权是为了压低「明明是椅子却判成沙发」这类跨类混淆，
    # 它同时能把 bottle / laptop 那些偏低的召回往上抬。
    "iter3_long": dict(
        base="model/best.pt",
        freeze=10, lr0=0.0008, lrf=0.01, optimizer="SGD",
        epochs=120, patience=40, cos_lr=True, close_mosaic=20,
        copy_paste=0.5, scale=0.7, degrees=8.0, cls=1.2,
        desc="强增强 + 120 轮 + 分类损失提权（冲检测召回）",
    ),

    # ==================== 第七轮（R5）：沿 iter3_long 继续迭代 ====================
    # iter3_long 是第一条「检测和掩码同时涨」的配置（0.733 / 0.583）。
    # 既然还在涨，就顺着它往下训几轮，看收益什么时候停。
    "iter4": dict(
        base="runs/exp_iter3_long/weights/best.pt",
        freeze=10, lr0=0.0005, lrf=0.01, optimizer="SGD",
        epochs=100, patience=35, cos_lr=True, close_mosaic=18,
        copy_paste=0.5, scale=0.7, degrees=8.0, cls=1.2,
        desc="沿 iter3_long 继续第三轮（同增强，学习率减半）",
    ),
    # 同一配置换一个随机种子重跑。验证集只有 30 张，mAP 的抖动本身就有
    # 一两个千分点，不做这次复现，就分不清 iter3_long 的涨幅是真提升还是运气。
    "iter3b": dict(
        base="model/best.pt", seed=7,
        freeze=10, lr0=0.0008, lrf=0.01, optimizer="SGD",
        epochs=120, patience=40, cos_lr=True, close_mosaic=20,
        copy_paste=0.5, scale=0.7, degrees=8.0, cls=1.2,
        desc="iter3_long 同配置换种子（检验涨幅是否稳定）",
    ),
    # 在 iter3_long 基础上把分类损失和增强都再推一档，
    # 目标是继续压 bottle / laptop 那几类的漏检。
    "iter3c": dict(
        base="runs/exp_iter3_long/weights/best.pt",
        freeze=10, lr0=0.0006, lrf=0.01, optimizer="SGD",
        epochs=100, patience=35, cos_lr=True, close_mosaic=18,
        copy_paste=0.7, scale=0.8, degrees=10.0, cls=1.5,
        desc="在 iter3_long 上加大增强与分类权重（冲漏检）",
    ),
    # 收敛式收尾：强制关掉 mosaic 跑满后段，让模型在真实分布上再稳一稳。
    "iter4_final": dict(
        base="runs/exp_iter3_long/weights/best.pt",
        freeze=10, lr0=0.0003, lrf=0.005, optimizer="SGD",
        epochs=80, patience=30, cos_lr=True, close_mosaic=40,
        copy_paste=0.35, scale=0.5, degrees=5.0, cls=1.2,
        desc="iter4 收尾版：早关 mosaic，最后 40 轮贴近真实分布",
    ),

    # ==================== 第八轮（通宵）：多种子重复 ====================
    # iter3_long 配方换 4 个新种子，与已有的 seed42/7 合计 6 个种子，
    # 用来算「均值 ± 标准差」，把 iter4 的选取建立在可靠的统计上。
    "ms1": dict(
        base="model/best.pt", seed=1,
        freeze=10, lr0=0.0008, lrf=0.01, optimizer="SGD",
        epochs=120, patience=40, cos_lr=True, close_mosaic=20,
        copy_paste=0.5, scale=0.7, degrees=8.0, cls=1.2,
        desc="iter3_long 配方 seed=1",
    ),
    "ms2": dict(
        base="model/best.pt", seed=2,
        freeze=10, lr0=0.0008, lrf=0.01, optimizer="SGD",
        epochs=120, patience=40, cos_lr=True, close_mosaic=20,
        copy_paste=0.5, scale=0.7, degrees=8.0, cls=1.2,
        desc="iter3_long 配方 seed=2",
    ),
    "ms3": dict(
        base="model/best.pt", seed=123,
        freeze=10, lr0=0.0008, lrf=0.01, optimizer="SGD",
        epochs=120, patience=40, cos_lr=True, close_mosaic=20,
        copy_paste=0.5, scale=0.7, degrees=8.0, cls=1.2,
        desc="iter3_long 配方 seed=123",
    ),
    "ms4": dict(
        base="model/best.pt", seed=2024,
        freeze=10, lr0=0.0008, lrf=0.01, optimizer="SGD",
        epochs=120, patience=40, cos_lr=True, close_mosaic=20,
        copy_paste=0.5, scale=0.7, degrees=8.0, cls=1.2,
        desc="iter3_long 配方 seed=2024",
    ),

    # ==================== 第九轮（通宵）：更大模型 s ====================
    # yolo26s-seg 参数量约为 n 的 3 倍。风险：120 张图喂不饱（过拟合）；
    # 收益：COCO 上学到的特征更丰富，小目标（cup/bottle）可能受益。
    # batch 从 8 降到 6 防 8GB 显存 OOM。
    "s_long": dict(
        base="weights/yolo26s-seg.pt",
        freeze=10, lr0=0.0008, lrf=0.01, optimizer="SGD",
        epochs=120, patience=40, cos_lr=True, close_mosaic=20,
        copy_paste=0.5, scale=0.7, degrees=8.0, cls=1.2, batch=6,
        desc="s 模型 + iter3_long 配方（官方权重起步）",
    ),
    "s_cont": dict(
        base="model/best.pt",
        freeze=10, lr0=0.0004, lrf=0.01, optimizer="SGD",
        epochs=80, patience=30, cos_lr=True, close_mosaic=18,
        copy_paste=0.4, scale=0.6, degrees=6.0, cls=1.2, batch=6,
        desc="从当前 n 权重「升维」继续训 s 结构（继承十类知识）",
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
    # 随机种子允许覆盖，用来做「同配置换种子」的复现实验
    seed = cfg.pop("seed", 42)
    # batch 允许覆盖：s 模型显存翻倍，8GB 卡上 640 尺寸 batch=8 会 OOM
    batch = cfg.pop("batch", 8)
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
        batch=batch,
        device=0,
        workers=4,
        project=RUNS_DIR,
        name="exp_%s" % tag,
        exist_ok=True,
        pretrained=True,
        seed=seed,
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
    r = m.val(data=TMP_YAML, imgsz=imgsz, batch=batch, verbose=False)
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
