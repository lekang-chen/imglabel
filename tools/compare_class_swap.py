#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""对比「换类前后」两个权重在同一个验证集上的分指标。

【为什么不能直接比 all 那一行】
把 book 换成 bed 之后，all 是所有类的平均。旧模型里 book 学得还行，
新模型里 bed 只有 14 个训练样本、必然很差——all 一降，看起来像「训崩了」，
其实是**换了考卷上的题**。所以要看的是：
  1. 9 个没动过的类，指标有没有掉（这才是「续训有没有冲掉旧能力」）
  2. 换掉的第 9 位，新旧各自多少（bed 能从 0 学到多少）

【为什么旧权重也跑同一个数据集】
同一个 val 名单、同一批标签，两边才是同一张考卷。旧权重的位置 9 是 book，
在这里会被当成 bed 打分——它当然很低，这恰恰说明「不重训就直接换词表」
是不可行的，正好当反证。

用法：
  python tools/compare_class_swap.py --old model/_backup/best_xxx.pt --new model/best.pt
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_YAML = os.path.join(ROOT, "dataset", "data.yaml")


def make_abs_yaml():
    """data.yaml 里的 path 是相对值（提交包要可移植），ultralytics 会按当前
    工作目录解析，直接喂给它就找不到图。这里生成一份 path 为绝对路径的临时
    yaml——和 train.py 训练时做的事一样——跑完删掉。"""
    with open(DATA_YAML, encoding="utf-8") as f:
        lines = f.read().splitlines()
    out = []
    for ln in lines:
        if ln.strip().startswith("path:"):
            out.append("path: %s" % os.path.join(ROOT, "dataset").replace("\\", "/"))
        else:
            out.append(ln)
    tmp = os.path.join(ROOT, "dataset", "_cmp.yaml")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    return tmp


def run(weights, yaml_path):
    from ultralytics import YOLO
    m = YOLO(weights)
    r = m.val(data=yaml_path, split="val", imgsz=640, verbose=False)
    names = [m.names[i] for i in range(len(m.names))] if isinstance(m.names, dict) else list(m.names)
    box = r.box.maps          # 每类 mAP50-95
    seg = r.seg.maps if r.seg is not None else [0] * len(names)
    b50 = r.box.map50
    m50 = r.seg.map50 if r.seg is not None else 0
    per = []
    for i, n in enumerate(names):
        per.append({"id": i, "name": n,
                    "box_map50_95": round(float(box[i]), 4) if i < len(box) else 0,
                    "mask_map50_95": round(float(seg[i]), 4) if i < len(seg) else 0})
    return {"box_map50": round(float(b50), 4),
            "mask_map50": round(float(m50), 4),
            "per_class": per,
            "val_instances": int(sum(r.box.nc and [len(r.box.cls)] or [0])) if False else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "swap_metrics.json"))
    args = ap.parse_args()

    yml = make_abs_yaml()
    try:
        old = run(args.old, yml)
        new = run(args.new, yml)
    finally:
        try:
            os.remove(yml)
        except OSError:
            pass

    print("%-16s %10s %10s %10s %10s" % ("类别", "旧 box", "新 box", "旧 mask", "新 mask"))
    old_map = {p["id"]: p for p in old["per_class"]}
    new_map = {p["id"]: p for p in new["per_class"]}
    steady = []
    for i in sorted(set(old_map) | set(new_map)):
        o = old_map.get(i, {})
        n = new_map.get(i, {})
        name = n.get("name") or o.get("name")
        ob, nb = o.get("box_map50_95", 0), n.get("box_map50_95", 0)
        om, nm = o.get("mask_map50_95", 0), n.get("mask_map50_95", 0)
        mark = ""
        if i < 9:
            steady.append((ob, nb, om, nm))
        else:
            mark = "  <- 换掉的这一位"
        print("%-16s %10.4f %10.4f %10.4f %10.4f%s" % (name, ob, nb, om, nm, mark))

    print("\n总体: box %.4f -> %.4f    mask %.4f -> %.4f"
          % (old["box_map50"], new["box_map50"], old["mask_map50"], new["mask_map50"]))
    if steady:
        ob = sum(s[0] for s in steady) / len(steady)
        nb = sum(s[1] for s in steady) / len(steady)
        om = sum(s[2] for s in steady) / len(steady)
        nm = sum(s[3] for s in steady) / len(steady)
        print("未动的 9 个类平均: box %.4f -> %.4f (%+.4f)    mask %.4f -> %.4f (%+.4f)"
              % (ob, nb, nb - ob, om, nm, nm - om))

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"old": args.old, "new": args.new, "old_metrics": old,
                   "new_metrics": new}, f, ensure_ascii=False, indent=2)
    print("明细已存 %s" % args.out)


if __name__ == "__main__":
    main()
