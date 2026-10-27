# -*- coding: utf-8 -*-
"""
生成 records/对比结果.csv —— 把实测数据填进任务书要求的对比表。

为什么不手工填：
  表格里的每个数字都来自 tools/eval_draft.py 和 tools/compare_rounds.py 的
  实测输出，手抄容易抄错，而且答辩前重跑训练后数字会变。
  这里直接读 JSON 产物生成，数字和实验永远对得上。

数据来源：
  data/round_compare.json  -> R1/R2 修正率、精确率、召回率
  runs/seg/results.csv     -> 自训模型的 mAP@0.5

用法：
  python tools/make_compare_csv.py
"""
import csv
import json
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMPARE_JSON = os.path.join(BASE_DIR, "data", "round_compare.json")
RESULTS_CSV = os.path.join(BASE_DIR, "runs", "seg", "results.csv")
RECORD_DIR = os.path.join(BASE_DIR, "records")
OUT_CSV = os.path.join(RECORD_DIR, "对比结果.csv")


def read_best_map():
    """取mAP@0.5（检测 / 掩码）。

    【口径问题，必须说清楚】
    results.csv 里每个 epoch 都记了当时的验证 mAP。
    - 用 max() 取全表最大值 = 训练过程中的「最好看的那一版」
    - 用最后一个epoch = early stopping 停下来的那一版
    两者都不是「实际回标时用的那个模型」。

    真正对外该报的是best.pt 在验证集上的最终表现，
    也就是 train.py 训练结束时那段 "Validating best.pt" 打印的 all 行。
    训练日志落在 data/train.log，取不到时退回用 results.csv 最后一轮，
    并在返回值里带一个来源标记，由调用方写进 CSV 说明列。
    """
    box_map = mask_map = None
    source = "未获取"

    # 【最高优先级：data/best_metrics.json】
    # 这份文件由 tools/eval_best.py 生成，是「拿当前最新标签测 best.pt」的结果，
    # 并附带测量时的数据状态（修正条数、验证集张数）。
    # 为什么它比 train.log 更可信：train.log 里那句是训练刚结束那一刻打的，
    # 之后人工修正还会改标签（本项目实测从 1 条改到 60 条），
    # 再测同一份权重的 mAP 就会变（0.742/0.415 -> 0.718/0.442）。
    # 文档里报的数字必须是「评委照现在的仓库复现能得到的」，所以以这份为准。
    metrics_path = os.path.join(BASE_DIR, "data", "best_metrics.json")
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, encoding="utf-8") as f:
                m = json.load(f)
            box_map = float(m["box_map50"])
            mask_map = float(m["mask_map50"])
            source = ("best.pt 在最终修正标签上实测"
                      "（%s，%d 条修正）"
                      % (m.get("measured_at", ""), m.get("dataset_edited", 0)))
        except Exception:  # noqa: BLE001
            pass

    # 其次从训练日志里解析 best.pt 的验证结果
    log_path = os.path.join(BASE_DIR, "data", "train.log")
    if box_map is None and os.path.exists(log_path):
        try:
            import re
            with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                txt = f.read()
            # 匹配 "all  30  200  0.808  0.639  0.742  0.545  0.549 ..."
            m = re.findall(
                r"^\s*all\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+"
                r"([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)",
                txt, re.M)
            if m:
                last = m[-1]
                box_map = float(last[4])
                mask_map = float(last[8])
                source = "best.pt 验证集实测（回标实际使用的权重）"
        except Exception:  # noqa: BLE001
            pass

    if box_map is None and os.path.exists(RESULTS_CSV):
        import csv as _csv
        with open(RESULTS_CSV, "r", encoding="utf-8") as f:
            rows = list(_csv.DictReader(f))
        if rows:
            # 退回用最后一个 epoch，而不是 max()——
            # max() 是训练过程的峰值，不代表最终权重的真实水平，报出去会偏高。
            last = rows[-1]
            box_map = float(last["metrics/mAP50(B)"])
            mask_map = float(last["metrics/mAP50(M)"])
            source = "results.csv 最后一个 epoch（未找到验证日志）"

    return box_map, mask_map, source


def main():
    if not os.path.exists(COMPARE_JSON):
        print("找不到", COMPARE_JSON)
        print("请先运行：python tools/compare_rounds.py")
        sys.exit(1)

    with open(COMPARE_JSON, "r", encoding="utf-8") as f:
        d = json.load(f)
    r1, r2 = d["r1"], d["r2"]

    box_map, mask_map, map_source = read_best_map()

    # R2 用的权重已经被 R3 替换掉了，best_metrics.json 现在记的是 R3 的数。
    # R2 这一行的 mAP 必须用当时留档的实测值，不能现读——
    # 否则换一次权重，历史行会跟着变，对比表就失去意义了。
    R2_BOX_MAP, R2_MASK_MAP = 0.718, 0.442

    # R3（换用冻结主干继续微调的权重）的回标实测数据
    r3 = None
    r3_path = os.path.join(BASE_DIR, "data", "round_eval.json")
    if os.path.exists(r3_path):
        try:
            with open(r3_path, encoding="utf-8") as f:
                cand = json.load(f)
            if cand.get("round") == "R3":
                r3 = cand
        except Exception:  # noqa: BLE001
            r3 = None

    os.makedirs(RECORD_DIR, exist_ok=True)
    with open(OUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["轮次", "模型", "总对象数", "修正对象数", "修正率(%)",
                    "精确率", "召回率", "检测mAP50", "掩码mAP50", "同类重复对"])
        w.writerow([
            "R1", "官方预训练权重 (YOLO-World + YOLO26n-seg)",
            r1["total_objects"], r1["need_fix"], round(r1["fix_rate"] * 100, 1),
            round(r1["precision"], 3), round(r1["recall"], 3),
            "—", "—", r1["dup_pairs"],
        ])
        w.writerow([
            "R2", "自训权重 (YOLO26n-seg 微调)",
            r2["total_objects"], r2["need_fix"], round(r2["fix_rate"] * 100, 1),
            round(r2["precision"], 3), round(r2["recall"], 3),
            R2_BOX_MAP, R2_MASK_MAP,
            r2["dup_pairs"],
        ])

        if r3:
            w.writerow([
                "R3", "自训权重 (冻结主干继续微调，当前 model/best.pt)",
                r3["objects"], r3["fix_doc"], r3["fix_rate_doc"],
                "%.3f" % r3["precision"], "%.3f" % r3["recall"],
                "%.3f" % box_map if box_map else "—",
                "%.3f" % mask_map if mask_map else "—",
                r3["dup_pairs"],
            ])

        # 变化量行，方便评委一眼看出趋势
        dr = (r2["fix_rate"] - r1["fix_rate"]) * 100
        w.writerow([
            "变化", "R2 - R1",
            r2["total_objects"] - r1["total_objects"],
            r2["need_fix"] - r1["need_fix"],
            round(dr, 1),
            round(r2["precision"] - r1["precision"], 3),
            round(r2["recall"] - r1["recall"], 3),
            "—", "—",
            r2["dup_pairs"] - r1["dup_pairs"],
        ])
        if r3:
            w.writerow([
                "变化", "R3 - R2",
                r3["objects"] - r2["total_objects"],
                r3["fix_doc"] - r2["need_fix"],
                round(r3["fix_rate_doc"] - r2["fix_rate"] * 100, 1),
                "%.3f" % (r3["precision"] - r2["precision"]),
                "%.3f" % (r3["recall"] - r2["recall"]),
                "%.3f" % (box_map - R2_BOX_MAP) if box_map else "—",
                "%.3f" % (mask_map - R2_MASK_MAP) if mask_map else "—",
                r3["dup_pairs"] - r2["dup_pairs"],
            ])

    print("已生成", os.path.relpath(OUT_CSV, BASE_DIR))
    with open(OUT_CSV, "r", encoding="utf-8-sig") as f:
        for line in f:
            print("  " + line.rstrip())


if __name__ == "__main__":
    main()
