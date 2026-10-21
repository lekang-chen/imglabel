# -*- coding: utf-8 -*-
"""R1 全量重跑：修复矩形标注后重新生成第一轮预标注。

为什么要重跑
------------
排查「界面上标注全是矩形」时定位到两层根因：
  1. core.py 的 simplify_polygon() 对闭合环退化（已修）
  2. prelabel.py 的 IoU 配对成功率 0%，配不上的框被兜底成矩形（已修）
修复后150 张图的只读验证结果：
  对象总数 576 -> 576（不变）矩形占比 100% -> 45.7%、平均顶点数 4.0 -> 69.1

代码改了但数据没变，所以必须重跑才能看到效果。
注意重跑会作废现有 mAP 数字与修正率，需要重新测量。

流程
----
  1. 预标注（conf=0.15，与原界面默认一致，只让「形状」这一个变量变化）
  2. 导出数据集
  3. 训练
  4. 用 best.pt 回标得到 R2
  5. 两轮对比

用法: .venv/Scripts/python.exe tools/rerun_r1.py [阶段]
      阶段可选 all / prelabel / export / train / relabel / compare
"""
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

CLASSES = ["person", "chair", "dining table", "couch", "tv",
           "laptop", "cup", "bottle", "potted plant", "book"]
IMAGE_DIR = os.path.join(ROOT, "data", "images")
CONF = 0.15   # 与界面默认值保持一致，避免引入额外变量


def step(n, msg):
    print("\n" + "=" * 66)
    print("[步骤 %s] %s" % (n, msg))
    print("=" * 66, flush=True)


def do_prelabel():
    step("1", "R1 全量预标注（conf=%.2f）" % CONF)
    t0 = time.time()
    from prelabel import PrelabelJob, run_prelabel_job
    job = PrelabelJob("rerun_r1", 0)
    objs = run_prelabel_job(job, IMAGE_DIR, CLASSES, conf=CONF)
    print("耗时 %.1f 分钟" % ((time.time() - t0) / 60))

    # 统计形状质量
    # results 结构是 {图片名: {"objects": [对象...], ...}}，不是 {图名: [对象]}
    n = 0
    rect = 0
    pts = 0
    for img_name, item in objs.items():
        lst = item.get("objects", []) if isinstance(item, dict) else item
        for o in lst:
            poly = o.get("polygon") if isinstance(o, dict) else getattr(o, "polygon", None)
            if not poly:
                continue
            n += 1
            pts += len(poly)
            if len(poly) == 4:
                xs = sorted(set(round(p[0], 6) for p in poly))
                ys = sorted(set(round(p[1], 6) for p in poly))
                if len(xs) == 2 and len(ys) == 2:
                    rect += 1
    print()
    print("本次预标注结果: %d 个对象（覆盖 %d 张图）" % (n, len(objs)))
    if n:
        print("  矩形占比: %.1f%%  (修复前 100%%)" % (100 * rect / n))
        print("  平均顶点数: %.1f  (修复前 4.0)" % (pts / n))

    # run_prelabel_job 只返回结果、不写盘（写盘在 app.py 接口层），
    # 这里手动落盘，并把每张图的错误列出来。
    #
    # 【注意】不能直接抄 results 的结构：它缺image / width / height，
    # 而这三个字段 Annotation.from_dict() 必须要（导出时要靠它们算分辨率）。
    # 所以要自己读图片补齐，否则后续 load_dir / export 会直接 KeyError。
    import json
    from PIL import Image

    out_dir = os.path.join(ROOT, "data", "annotations")
    errs = []
    #img_name 已经是 "000109.jpg" 这种带扩展名的名字，
    # 而落盘约定是 "000109.json"（去掉扩展名再加 .json），
    # 直接 img_name + ".json" 会多出一层变成 "000109.jpg.json"。
    stem = img_name.rsplit(".", 1)[0]
    for img_name, item in objs.items():
        with Image.open(os.path.join(IMAGE_DIR, img_name)) as im:
            w, h = im.size
        payload = {
            "image": img_name,
            "width": w,
            "height": h,
            "objects": item.get("objects", []),
        }
        with open(os.path.join(out_dir, stem + ".json"), "w",
                  encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        if item.get("error"):
            errs.append("%s: %s" % (img_name, item["error"]))
    print()
    print("已写入 %s" % out_dir)
    if errs:
        print("!! %d 张图报错:" % len(errs))
        for e in errs[:10]:
            print("  -", e)
    else:
        print("无报错图片")
    return objs


def do_export():
    step("2", "导出数据集")
    import glob
    import json
    from core import Annotation
    from exporter import export_dataset, validate_dataset

    anno_dir = os.path.join(ROOT, "data", "annotations")
    paths = sorted(glob.glob(os.path.join(anno_dir, "*.json")))
    # 【防呆】落盘约定是 000109.json，不该出现 000109.jpg.json 这种双后缀。
    # 一旦出现说明写入端和读取端对名字的理解不一致，必须当场报出来，
    # 否则后面导出会把同一张图当成两个样本，训练数据直接翻倍。
    bad = [p for p in paths if ".jpg." in os.path.basename(p)
           or ".png." in os.path.basename(p)]
    if bad:
        print("!! 发现 %d 个双后缀文件（写入端 bug）：" % len(bad))
        for p in bad[:5]:
            print("  -", os.path.basename(p))
        print("请修正 rerun_r1.py 的落盘文件名后再继续。")
        return False

    anns = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            anns.append(Annotation.from_dict(json.load(f)))
    print("读入 %d 张标注" % len(anns))

    out = os.path.join(ROOT, "dataset")
    # 【坑】export_dataset 的第4 个位置参数是 val_ratio，不是 classes。
    # 按位置传会把 CLASSES 列表当成 val_ratio，导致 total*list 报
    # "type list doesn't define __round__"。一律用关键字调用。
    stats = export_dataset(anns, out, IMAGE_DIR,
                           has_seg=True, copy_images=True)
    print("导出完成: %s" % stats)

    ok, problems, counts = validate_dataset(out)
    print("自检: %s (%d 个问题)" % ("通过" if ok else "不通过", len(problems)))
    for pr in problems[:8]:
        print("  -", pr)
    return ok


def do_train():
    step("3", "训练（沿用现有超参）")
    exe = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
    cmd = [exe, os.path.join(ROOT, "train.py")]
    log_path = os.path.join(ROOT, "data", "train.log")

    # 【为什么要落盘】train.py 不写日志文件，训练输出只在终端。
    # 而 tools/make_compare_csv.py 要从日志里解析 best.pt 的验证结果
    # （那才是回标实际使用的权重），所以这里必须留档。
    print("执行: %s\n日志: %s" % (" ".join(cmd), log_path))
    with open(log_path, "w", encoding="utf-8", errors="ignore") as lf:
        proc = subprocess.run(cmd, cwd=ROOT, stdout=lf,
                              stderr=subprocess.STDOUT)
    print("训练结束，退出码 %d（完整输出见 %s）" % (proc.returncode, log_path))
    return proc.returncode == 0


def do_relabel():
    step("4", "用 best.pt 回标得到 R2")
    cmd = [os.path.join(ROOT, ".venv", "Scripts", "python.exe"),
           os.path.join(ROOT, "tools", "relabel_r2.py")]
    print("执行: %s\n" % " ".join(cmd))
    r = subprocess.run(cmd, cwd=ROOT)
    return r.returncode == 0


def do_compare():
    step("5", "R1/R2 对比")
    for name in ("compare_rounds.py", "make_compare_csv.py"):
        cmd = [os.path.join(ROOT, ".venv", "Scripts", "python.exe"),
               os.path.join(ROOT, "tools", name)]
        print("执行: %s" % name)
        subprocess.run(cmd, cwd=ROOT)


ALL = {
    "prelabel": do_prelabel,
    "export": do_export,
    "train": do_train,
    "relabel": do_relabel,
    "compare": do_compare,
}

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    todo = ["prelabel", "export", "train", "relabel", "compare"] \
        if which == "all" else [which]
    t0 = time.time()
    for k in todo:
        if k not in ALL:
            print("未知阶段: %s，可选 %s / all" % (k, " | ".join(ALL)))
            sys.exit(2)
        ALL[k]()
    print("\n总耗时 %.1f 分钟" % ((time.time() - t0) / 60))