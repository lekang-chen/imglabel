# -*- coding: utf-8 -*-
"""
跑一轮「回标」，算出这一轮的真实修正率——但不碰任何现有标注。

【为什么要单独做这件事】
换权重之后，文档里 R2 那行「627 对象 / 319 修正 / 50.9%」是旧权重测出来的，
新权重到底让工作量变成多少，得用**完全相同的口径**再算一遍才有可比性。
但直接跑 app 里的回标会把 data/annotations 覆盖掉——那里有 60 条人工修正，
是任务书的硬性要求，绝对不能丢。

所以这里绕开落盘：
  1. 直接调 prelabel.run_relabel_job()，它只在内存里返回结果，本身不写文件；
  2. 把结果写到 %TEMP%/imglabel_r3_anno/ 这个临时目录；
  3. 复用 tools/make_fix_tasks.py 和 tools/eval_draft.py 的现成口径去统计；
  4. 算完删掉临时目录。

【口径一致性】
  修正数 = 错类 + 漏标 + 多标 + 边界差（框 IoU<0.75），
  与 make_fix_tasks.py 完全一致；
  精确率/召回率来自 eval_draft.py，IoU=0.5 判定匹配。

用法：
  python tools/eval_relabel_round.py
  python tools/eval_relabel_round.py --conf 0.25 --tag R3
"""
import argparse
import json
import os
import shutil
import sys
import tempfile

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "tools"))

IMAGE_DIR = os.path.join(BASE_DIR, "data", "images")
CURRENT_ANNO = os.path.join(BASE_DIR, "data", "annotations")
WEIGHTS = os.path.join(BASE_DIR, "model", "best.pt")
OUT_JSON = os.path.join(BASE_DIR, "data", "round_eval.json")


class _DummyJob:
    """run_relabel_job 只需要 total / done / advance / results 这几个字段"""

    def __init__(self):
        self.total = 0
        self.done = 0
        self.results = None

    def advance(self, name, msg):
        self.done += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", type=float, default=0.25,
                    help="回标置信度，R2 当时用的是 0.25，保持一致才能对比")
    ap.add_argument("--tag", default="R3")
    ap.add_argument("--weights", default=WEIGHTS)
    args = ap.parse_args()

    from PIL import Image

    import prelabel

    # 先确认现有标注完好，跑完再核对一次——防止意外被覆盖
    before = len(os.listdir(CURRENT_ANNO))
    before_mtime = max(os.path.getmtime(os.path.join(CURRENT_ANNO, f))
                       for f in os.listdir(CURRENT_ANNO))

    print("=" * 66)
    print("%s 回标评估（conf=%.2f，权重 %s）"
          % (args.tag, args.conf,
             os.path.relpath(args.weights, BASE_DIR).replace("\\", "/")))
    print("=" * 66)
    print("现有标注 %d 份，跑完会核对是否被改动" % before)

    job = _DummyJob()
    results = prelabel.run_relabel_job(job, IMAGE_DIR, args.weights,
                                       conf=args.conf, round_tag=args.tag)

    # 落到临时目录，让后面两个统计工具直接复用
    tmp = tempfile.mkdtemp(prefix="imglabel_%s_" % args.tag.lower())
    try:
        total_obj = 0
        for img_name, data in results.items():
            stem = os.path.splitext(img_name)[0]
            w, h = Image.open(os.path.join(IMAGE_DIR, img_name)).size
            total_obj += len(data.get("objects") or [])
            payload = {
                "image": img_name,
                "width": w,
                "height": h,
                "objects": data.get("objects") or [],
                "source": data.get("source", "model"),
                "round": args.tag,
            }
            with open(os.path.join(tmp, stem + ".json"), "w",
                      encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)

        print("回标产出对象数: %d（%d 张图）" % (total_obj, len(results)))

        # --- 用 eval_draft 的现成口径算精确率/召回率 ---
        from eval_draft import COCO_JSON, SOURCE_MAP, evaluate
        q = evaluate(tmp, COCO_JSON, SOURCE_MAP)

        # --- 用 make_fix_tasks 的现成口径算"要人工改多少条" ---
        from make_fix_tasks import build_tasks
        tasks = build_tasks(tmp)
        cnt = {"wrong_class": 0, "missing": 0, "extra": 0, "overlap": 0}
        for t in tasks:
            for k in cnt:
                cnt[k] += len(t[k])
        fixes = sum(cnt.values())
        rate = fixes / q["total_draft"] * 100 if q["total_draft"] else 0.0

        print("")
        print("-" * 66)
        print("  错类（改类别）: %4d" % cnt["wrong_class"])
        print("  漏标（补对象）: %4d" % cnt["missing"])
        print("  多标（删对象）: %4d" % cnt["extra"])
        print("  边界（改轮廓）: %4d" % cnt["overlap"])
        print("  ------------------------------------")
        print("  需人工修正    : %4d / %d 个对象 = %.1f%%"
              % (fixes, q["total_draft"], rate))
        print("  精确率 P      : %.3f" % q["precision"])
        print("  召回率 R      : %.3f" % q["recall"])
        print("  F1            : %.3f" % q["f1"])
        print("  同类重复对    : %d" % q["dup_pairs"])
        print("  对照真值对象  : %d" % q["total_truth"])

        # 提交材料里用的「修正数」口径是：草稿对象数 - 正确对象数
        # （也就是 1 - 精确率）。这里把它单独算出来，避免和上面
        # make_fix_tasks 的四类问题和混。
        fix_doc = q["total_draft"] - q["tp"]
        rate_doc = fix_doc / q["total_draft"] * 100 if q["total_draft"] else 0.0
        print("  ------------------------------------")
        print("  按提交口径    : %d / %d = %.1f%%（正确 %d 个）"
              % (fix_doc, q["total_draft"], rate_doc, q["tp"]))

        # 把 mAP 一起并进来，凑成一整行
        best_file = os.path.join(BASE_DIR, "data", "best_metrics.json")
        best = {}
        if os.path.exists(best_file):
            with open(best_file, encoding="utf-8") as f:
                best = json.load(f)

        row = {
            "round": args.tag,
            "weights": os.path.relpath(args.weights, BASE_DIR).replace("\\", "/"),
            "conf": args.conf,
            "images": q["files"],
            "objects": q["total_draft"],
            "fixes": fixes,
            "fix_breakdown": cnt,
            "fix_rate": round(rate, 1),
            "tp": q["tp"],
            "fp": q["fp"],
            "fn": q["fn"],
            "fix_doc": fix_doc,
            "fix_rate_doc": round(rate_doc, 1),
            "precision": q["precision"],
            "recall": q["recall"],
            "f1": q["f1"],
            "dup_pairs": q["dup_pairs"],
            "box_map50": best.get("box_map50"),
            "mask_map50": best.get("mask_map50"),
        }
        with open(OUT_JSON, "w", encoding="utf-8") as f:
            json.dump(row, f, ensure_ascii=False, indent=2)
        print("")
        print("  已写入 %s" % OUT_JSON)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # 最后核对现有标注有没有被动过
    after = len(os.listdir(CURRENT_ANNO))
    after_mtime = max(os.path.getmtime(os.path.join(CURRENT_ANNO, f))
                      for f in os.listdir(CURRENT_ANNO))
    print("")
    if before == after and before_mtime == after_mtime:
        print("✓ 现有标注未被改动（%d 份，时间戳一致）" % after)
    else:
        print("✗ 警告：现有标注发生变化！%d -> %d 份" % (before, after))
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
