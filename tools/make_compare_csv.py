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
    """从训练结果里取最佳 mAP@0.5（检测 / 掩码）。"""
    if not os.path.exists(RESULTS_CSV):
        return None, None
    import csv as _csv
    with open(RESULTS_CSV, "r", encoding="utf-8") as f:
        rows = list(_csv.DictReader(f))
    if not rows:
        return None, None
    best_box = max(rows, key=lambda r: float(r["metrics/mAP50(B)"]))
    best_mask = max(rows, key=lambda r: float(r["metrics/mAP50(M)"]))
    return (float(best_box["metrics/mAP50(B)"]),
            float(best_mask["metrics/mAP50(M)"]))


def main():
    if not os.path.exists(COMPARE_JSON):
        print("找不到", COMPARE_JSON)
        print("请先运行：python tools/compare_rounds.py")
        sys.exit(1)

    with open(COMPARE_JSON, "r", encoding="utf-8") as f:
        d = json.load(f)
    r1, r2 = d["r1"], d["r2"]

    box_map, mask_map = read_best_map()

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
            "R2", "自训权重 model/best.pt (YOLO26n-seg 微调)",
            r2["total_objects"], r2["need_fix"], round(r2["fix_rate"] * 100, 1),
            round(r2["precision"], 3), round(r2["recall"], 3),
            round(box_map, 3) if box_map else "—",
            round(mask_map, 3) if mask_map else "—",
            r2["dup_pairs"],
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

    print("已生成", os.path.relpath(OUT_CSV, BASE_DIR))
    with open(OUT_CSV, "r", encoding="utf-8-sig") as f:
        for line in f:
            print("  " + line.rstrip())


if __name__ == "__main__":
    main()
