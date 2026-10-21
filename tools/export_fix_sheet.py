# -*- coding: utf-8 -*-
"""把修正清单导成更好照着改的表格（Markdown + CSV）。

为什么不用现成的 fix_tasks.txt
--------------------------
txt 里每条只写「第N 个对象」，而 N 是对象在 JSON 数组里的下标，
跟界面右栏的显示顺序对不上，很容易数错。
这里把 obj_id、类别、改成什么、画布上的归一化坐标全都列出来，
对照界面能直接找到对象，点上去就改。

用法:
    .venv/Scripts/python.exe tools/export_fix_sheet.py            # 前 3 张图
    .venv/Scripts/python.exe tools/export_fix_sheet.py --images 8 # 前 8 张图
"""
import csv
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "data", "fix_tasks.json")
MD = os.path.join(ROOT, "人工修正清单.md")
CSV = os.path.join(ROOT, "data", "fix_tasks.csv")

CLS_CN = {
    "person": "人", "chair": "椅子", "dining table": "餐桌", "couch": "沙发",
    "tv": "电视", "laptop": "笔记本电脑", "cup": "杯子",
    "bottle": "瓶子", "potted plant": "盆栽", "book": "书",
}


def cn(c):
    return CLS_CN.get(c, c)


def main():
    n_img = 3
    if "--images" in sys.argv:
        n_img = int(sys.argv[sys.argv.index("--images") + 1])

    with open(SRC, encoding="utf-8") as f:
        tasks = json.load(f)

    used = tasks[:n_img]
    rows = []
    for t in used:
        img = t["image"]

        def base(idx):
            # 【序号口径】清单里的序号是 JSON 数组下标（从 0 开始），
            # 而界面右栏显示的是 idx+1（从 1 开始）。两个都给出，
            # 以免照着清单找过去时数错一位。
            return dict(图片=img, 清单序号=idx,
                        界面序号=(idx + 1 if idx != "" else ""),
                        对象id="", 类别="", 应改为="", IoU="",
                        中心x="", 中心y="", 宽="", 高="")

        for w in t.get("wrong_class", []):
            r = base(w["obj_index"])
            r.update(类型="错类", 对象id=w["obj_id"], 类别=w["from"],
                     应改为=w["to"])
            rows.append(r)
        for e in t.get("extra", []):
            r = base(e["obj_index"])
            r.update(类型="多标(删)", 对象id=e["obj_id"], 类别=e["class"],
                     应改为="删除")
            rows.append(r)
        for o in t.get("overlap", []):
            r = base(o["obj_index"])
            r.update(类型="边界", 对象id=o["obj_id"], 类别=o["class"],
                     应改为="修轮廓", IoU=o.get("iou", ""))
            rows.append(r)
        for m in t.get("missing", []):
            r = base("")
            cx, cy = m["center"]
            w, h = m["size"]
            r.update(类型="漏标(补)", 类别=m["class"], 应改为="新建",
                     中心x=round(cx, 3), 中心y=round(cy, 3),
                     宽=round(w, 3), 高=round(h, 3))
            rows.append(r)

    # --- Markdown ---
    L = []
    L.append("# 人工修正清单（照着改就行）")
    L.append("")
    L.append("> 任务书要求：**人工修正不少于 60 条**，且记录可核验。")
    L.append("> 本清单由 `tools/make_fix_tasks.py` 对照 COCO 官方真值反查得出，")
    L.append("> 已按「省力程度」排序：**错类 > 多标 > 漏标 > 边界**。")
    L.append(">")
    L.append("> 当前覆盖 **%d 张图**，共 **%d 条**可改。做完这 %d 条即达标。"
             % (len(used), len(rows), len(rows)))
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 怎么用")
    L.append("")
    L.append("1. 打开软件：双击桌面「人机协同图像标注系统」，或浏览器访问 "
             "`http://127.0.0.1:5000`")
    L.append("2. 左边图片列表点选下表对应的图片")
    L.append("3. 找到「界面序号」对应的对象：**点画布上的对象**，"
             "或 **点右栏对象列表那一行**")
    L.append("   - 右栏每行左边有个序号，**界面序号**就是它")
    L.append("   - ⚠ 「清单序号」从 0 数，「界面序号」从 1 数，"
             "两者差 1。对不上时以「界面序号」为准，"
             "或者直接用对象ID 核对（最保险）")
    L.append("   - 右栏列表带复选框，勾上才显示；点行会进入「仅看选中」模式，"
             "画面只剩这一个对象，最容易对准")
    L.append("4. 按类型操作：")
    L.append("   | 类型 | 界面操作 |")
    L.append("   |---|---|")
    L.append("   | 错类 | 右栏「选中对象」区，把类别下拉改成目标类别 |")
    L.append("   | 多标(删) | 按 `Delete` 键，或点右栏该行的删除按钮 |")
    L.append("   | 漏标(补) | 底部工具栏选「画笔」直接涂，或选「多边形」画轮廓后指定类别 |")
    L.append("   | 边界 | 选「多边形」工具拖顶点，或用「画笔」「橡皮」局部增减 |")
    L.append("5. 看进度：界面右上角「修正」计数，或运行")
    L.append("   `curl http://127.0.0.1:5000/api/records`")
    L.append("")
    L.append("---")
    L.append("")
    total = 0
    for t in used:
        img = t["image"]
        L.append("## %s（%d 处）" % (img, t["total"]))
        L.append("")
        L.append("| 类型 | 界面序号 | 对象ID | 当前 | 操作 | 参考 |")
        L.append("|---|---|---|---|---|---|")
        sub = [r for r in rows if r["图片"] == img]
        # 排序：错类 -> 多标 -> 漏标 -> 边界
        order = {"错类": 0, "多标(删)": 1, "漏标(补)": 2, "边界": 3}
        sub.sort(key=lambda r: (order[r["类型"]],
                                r["清单序号"] if isinstance(r["清单序号"], int)
                                else 999))
        for r in sub:
            total += 1
            if r["类型"] == "漏标(补)":
                ref = ("中心约 (%.3f, %.3f)，大小 %.3f x %.3f"
                       % (r["中心x"], r["中心y"], r["宽"], r["高"]))
            elif r["类型"] == "边界":
                ref = "IoU=%s（越低越偏）" % r["IoU"]
            else:
                ref = ""
            L.append("| %s | %s | `%s` | %s | → %s | %s |"
                     % (r["类型"], r["界面序号"], r["对象id"],
                        cn(r["类别"]), cn(str(r["应改为"])), ref))
        L.append("")

    L.append("---")
    L.append("")
    L.append("## 满 60 条后：导出修正记录")
    L.append("")
    L.append("```bash")
    L.append("curl -X POST http://127.0.0.1:5000/api/records/export")
    L.append("```")
    L.append("")
    L.append("会生成 `records/修正记录.csv`，打包脚本会检查条数。")
    L.append("")
    L.append("## 改完之后")
    L.append("")
    L.append("告诉我，我帮你跑：导出数据集 → 训练 → 回标 R2 → 两轮对比，")
    L.append("然后把新数字同步到答辩材料。")
    L.append("")

    with open(MD, "w", encoding="utf-8") as f:
        f.write("\n".join(L))

    with open(CSV, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("已生成:")
    print("  %s (%d 条 / %d 张图)" % (MD, len(rows), len(used)))
    print("  %s" % CSV)


if __name__ == "__main__":
    main()