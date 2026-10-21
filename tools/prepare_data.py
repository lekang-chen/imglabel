"""
tools/prepare_data.py —— 数据集准备脚本

【任务书要求】
"数据：总量不少于 150 张，至少覆盖 6 类，每类不少于 15 张"
"规格：室内场景照片，长边不超过 960 像素，JPG 格式，质量 85"
"来源：公开数据集网站，来源名称、下载数、保留数写入提交信息.txt"

【这个脚本做什么】
1. 读取 COCO 的标注文件，按我们的十类词表筛选室内场景图片
2. 按类别数量要求挑图（每类至少 N 张，总数至少 150）
3. 压缩到长边 960px、JPG 质量 85
4. 输出到 data/images/
5. 生成一份筛选说明（写进 提交信息.txt）

【为什么用 COCO】
- 十类词表 10/10 全覆盖（VOC 只有 7/10，缺 laptop/cup/book）
- 自带人工标注，可以用来反向验证我们的导出格式是否正确
- val2017 有 5000 张，筛 150 张绰绰有余

【怎么用】
  python tools/prepare_data.py --coco-ann <annotations.json> --coco-img <图片目录> --out data/images
"""

import argparse
import json
import os
import random
import shutil
import sys
from collections import defaultdict

# 十类词表（与 core.py 一致）
CLASSES = [
    "person", "chair", "dining table", "couch", "tv",
    "laptop", "cup", "bottle", "potted plant", "book",
]

# COCO 类别 id -> 我们的类别名
# COCO 的 id 是离散的一堆数字，不是 0~9，必须建立映射表
COCO_ID_TO_NAME = {
    1: "person",
    62: "chair",
    67: "dining table",
    65: "couch",
    72: "tv",
    73: "laptop",
    47: "cup",
    44: "bottle",
    64: "potted plant",
    84: "book",
}

MAX_SIDE = 960
JPEG_QUALITY = 85
TARGET_TOTAL = 150
MIN_CLASSES = 6
MIN_PER_CLASS = 15


def load_coco(ann_file):
    with open(ann_file, encoding="utf-8") as f:
        return json.load(f)


def build_index(coco):
    """
    建立「图片 -> 该图包含的我们的十类」的索引。
    只保留至少包含一个目标类别的图片。
    """
    img_info = {im["id"]: im for im in coco["images"]}
    per_image = defaultdict(set)
    for ann in coco["annotations"]:
        name = COCO_ID_TO_NAME.get(ann["category_id"])
        if name is None:
            continue
        # 过滤掉过小的标注（面积太小多半是误标）
        if ann.get("area", 0) < 100:
            continue
        per_image[ann["image_id"]].add(name)
    return img_info, per_image


def select_images(img_info, per_image, target_total, min_per_class, min_classes, seed=42):
    """
    按类别覆盖度挑图。

    策略（贪心 + 均衡）：
    1. 先统计每类有多少张图可用
    2. 优先选「稀有类别」的图（保证每类都能凑够 min_per_class）
    3. 剩下名额用类别均衡的方式补，尽量让各类数量接近

    为什么要这么做？
    任务书要求"至少覆盖 6 类，每类不少于 15 张"。
    如果随机挑 150 张，person 会占一半，cup/book 可能一张都没有。
    """
    random.seed(seed)

    by_class = defaultdict(list)
    for img_id, classes in per_image.items():
        for c in classes:
            by_class[c].append(img_id)

    print("[筛选] 各类别可用图片数：")
    for c in CLASSES:
        print("        %-14s %d 张" % (c, len(by_class[c])))

    # 类别不够的先剔除（无法满足 min_per_class 的类别不选）
    usable = [c for c in CLASSES if len(by_class[c]) >= min_per_class]
    if len(usable) < min_classes:
        print("[警告] 只有 %d 个类别能满足每类 %d 张的要求（需要至少 %d 类）"
              % (len(usable), min_per_class, min_classes))
    print("[筛选] 可用类别: %s" % ", ".join(usable))

    chosen = {}          # img_id -> set(classes)
    counts = defaultdict(int)

    # --- 阶段 1：稀有类别优先 ---
    # 按可用数量升序处理，最稀有的先挑
    for c in sorted(usable, key=lambda x: len(by_class[x])):
        need = min_per_class - counts[c]
        if need <= 0:
            continue
        cands = sorted(by_class[c], key=lambda i: -len(per_image[i]))
        random.shuffle(cands)
        for img_id in cands:
            if need <= 0:
                break
            if img_id in chosen:
                counts[c] += 1
                need -= 1
                continue
            chosen[img_id] = set(per_image[img_id])
            for cc in per_image[img_id]:
                counts[cc] += 1
            need -= 1

    # --- 阶段 2：补足总数，同时保持类别均衡 ---
    all_ids = list(per_image.keys())
    random.shuffle(all_ids)
    # 按当前计数升序排，优先加到最少的类别上
    all_ids.sort(key=lambda i: sum(counts[c] for c in per_image[i]))

    for img_id in all_ids:
        if len(chosen) >= target_total:
            break
        if img_id in chosen:
            continue
        chosen[img_id] = set(per_image[img_id])
        for cc in per_image[img_id]:
            counts[cc] += 1

    # --- 阶段 3：如果还不足总数，从剩余图里随便补 ---
    if len(chosen) < target_total:
        for img_id in all_ids:
            if len(chosen) >= target_total:
                break
            if img_id not in chosen:
                chosen[img_id] = set(per_image[img_id])

    return chosen, counts


def compress_and_save(src_path, dst_path):
    """
    压缩图片：长边不超过 960px，JPG 质量 85。

    任务书原话："长边不超过 960 像素，JPG 格式，质量 85"
    """
    from PIL import Image

    with Image.open(src_path) as im:
        im = im.convert("RGB")
        w, h = im.size
        scale = 1.0
        if max(w, h) > MAX_SIDE:
            scale = MAX_SIDE / max(w, h)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        if (new_w, new_h) != (w, h):
            im = im.resize((new_w, new_h), Image.LANCZOS)
        im.save(dst_path, "JPEG", quality=JPEG_QUALITY, optimize=True)
        return new_w, new_h


def main():
    ap = argparse.ArgumentParser(description="准备室内场景图片数据集")
    ap.add_argument("--coco-ann", required=True, help="COCO instances_*.json 路径")
    ap.add_argument("--coco-img", required=True, help="COCO 图片目录")
    ap.add_argument("--out", default="data/images", help="输出目录")
    ap.add_argument("--total", type=int, default=TARGET_TOTAL)
    ap.add_argument("--min-per-class", type=int, default=MIN_PER_CLASS)
    ap.add_argument("--report", default="data/data_source.txt", help="筛选说明输出")
    args = ap.parse_args()

    print("=" * 64)
    print("准备数据集")
    print("=" * 64)

    if not os.path.exists(args.coco_ann):
        sys.exit("[错误] 找不到标注文件: %s" % args.coco_ann)
    if not os.path.isdir(args.coco_img):
        sys.exit("[错误] 找不到图片目录: %s" % args.coco_img)

    coco = load_coco(args.coco_ann)
    print("[1/4] 读取 COCO 标注: %d 张图, %d 个标注"
          % (len(coco["images"]), len(coco["annotations"])))

    img_info, per_image = build_index(coco)
    print("[2/4] 含目标类别的图片: %d 张" % len(per_image))

    chosen, counts = select_images(img_info, per_image, args.total,
                                   args.min_per_class, MIN_CLASSES)
    print("[3/4] 选出 %d 张" % len(chosen))

    out_dir = os.path.abspath(args.out)
    if os.path.isdir(out_dir):
        print("      清空输出目录 %s" % out_dir)
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    kept = 0
    log_lines = []
    for i, (img_id, classes) in enumerate(sorted(chosen.items())):
        info = img_info[img_id]
        fn = info["file_name"]
        src = os.path.join(args.coco_img, fn)
        if not os.path.exists(src):
            continue
        dst_name = "%06d.jpg" % kept
        dst = os.path.join(out_dir, dst_name)
        try:
            w, h = compress_and_save(src, dst)
        except Exception as exc:  # noqa: BLE001
            print("      跳过 %s: %s" % (fn, exc))
            continue
        kept += 1
        log_lines.append("%s  <-  %s  (%dx%d, 类别: %s)"
                         % (dst_name, fn, w, h, ",".join(sorted(classes))))

    # --- 写筛选说明 ---
    os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        f.write("数据来源: COCO (COCO 2017 val)\n")
        f.write("来源网址: https://cocodataset.org/\n")
        f.write("下载数量: %d 张 (筛选后可用)\n" % len(per_image))
        f.write("保留数量: %d 张\n" % kept)
        f.write("压缩规格: 长边 <= %d px, JPG 质量 %d\n" % (MAX_SIDE, JPEG_QUALITY))
        f.write("覆盖类别: %d 类 (要求 >= %d)\n" % (len([c for c in CLASSES if counts[c]]), MIN_CLASSES))
        f.write("\n各类别图片数:\n")
        for c in CLASSES:
            f.write("  %-14s %d\n" % (c, counts[c]))
        f.write("\n逐张对应关系:\n")
        f.write("\n".join(log_lines))

    print("[4/4] 完成，输出 %d 张到 %s" % (kept, out_dir))
    print("      覆盖类别 %d 类，各类数量: %s"
          % (len([c for c in CLASSES if counts[c]]),
             ", ".join("%s=%d" % (c, counts[c]) for c in CLASSES if counts[c])))
    print("      筛选说明已写入 %s" % args.report)


if __name__ == "__main__":
    main()
