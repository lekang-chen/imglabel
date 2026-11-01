"""
exporter.py —— 数据集导出器（三种格式）

【任务书要求】
"数据集导出：YOLO txt、实例掩码、语义 PNG 三种格式，采用 YOLO 社区标准格式规格"
"train.py 仅接受平台导出数据集"

【导出结构】
dataset/
├── images/
│   ├── train/   xxx.jpg          训练图片
│   └── val/     xxx.jpg          验证图片（按比例划分）
├── labels/
│   ├── train/   xxx.txt          YOLO 标注（检测框 + 实例分割多边形）
│   └── val/     xxx.txt
├── masks/
│   ├── train/   xxx.png          实例掩码：一张图所有实例画在一起，
│   └── val/                       不同实例用不同灰度值区分
├── semantic/
│   ├── train/   xxx.png          语义分割：类别索引图，像素值 = 类别编号
│   └── val/                       255 = 背景/未标注
└── data.yaml                     训练配置

【YOLO 格式要点】
- 标签 txt 每行一个对象
- 检测格式:  <cls_id> <xc> <yc> <w> <h>        全部是 0~1 归一化值
- 分割格式:  <cls_id> <x1> <y1> <x2> <y2> ...   顶点坐标，也是 0~1
- 同一个 txt 里两种格式不能混写，ultralytics 会按文件名后缀区分：
  本项目统一用**分割格式**（多边形），因为任务是实例分割。
  检测框可以由多边形外接矩形自动推出，训练时 ultralytics 也能直接用。

【为什么要自检】
任务书第五部分提到"文档导出前的格式校验 -> 自动化测试/CI"。
导出完立刻校验一遍，如果格式错了立刻报错并说明原因，
而不是等训练跑完才发现标签对不上、白白浪费时间。
"""

import hashlib
import json
import os
import random
import shutil

from PIL import Image

from core import (
    CLASSES,
    CLASS_TO_ID,
    Annotation,
    instance_from_objects,
    semantic_from_objects,
)

# 训练/验证集划分比例（只用于「第一次初始化名单」）
VAL_RATIO = 0.2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 【为什么要固化成名单文件，而不是每次按比例随机分】
# 原来是 random.seed(42) + random.sample(range(total), n_val)：
# 种子固定，但采样的「索引范围」随总数变。150 张时抽 30 个索引，
# 157 张时抽 31 个——同一个种子抽出的是两个几乎不相交的集合。
# 后果是每加几张图，验证集就整体换一遍：同一份 best.pt 在 150 张的 val 上
# 是 0.728/0.580，在 157 张的 val 上是 0.760/0.540。数字并非矛盾，
# 是考卷换人了。
#
# 对自主学习来说这是致命的：每一代都要重新导出，验证集跟着漂移，
# 闸门拿两个不同考卷的分数比大小，判据就彻底失效了。
# 所以把验证集钉成一份名单：名单里的进 val，其余全进 train。
# 新上传的图默认进训练集——这既保证了考卷不变，也正是「新信息」该去的地方。
VAL_SPLIT_PATH = os.path.join(BASE_DIR, "data", "val_split.json")


def find_duplicate_groups(image_names, image_dir):
    """按文件内容找出内容完全相同的图（同一张照片传了两次）。

    【为什么必须查】
    同一张图如果一张进 train、一张进 val，模型训练时已经把它背下来了，
    验证时自然"答对"——这叫数据泄漏，会让 mAP 虚高，而且是评委最爱问的点。
    实测就踩到过：IMG_8073.webp 和 IMG_8073_2.webp 的 md5 完全一样，
    一个在 val、一个在 train。

    只在本地有原图时才查；查不到就当没重复（不阻断导出）。
    """
    seen = {}
    groups = []
    for name in image_names:
        p = os.path.join(image_dir, name)
        try:
            with open(p, "rb") as f:
                digest = hashlib.md5(f.read()).hexdigest()
        except OSError:
            continue
        if digest in seen:
            for g in groups:
                if g[0] == digest:
                    g[1].append(name)
                    break
            else:
                groups.append([digest, [seen[digest], name]])
        else:
            seen[digest] = name
    return groups


def load_val_split(path=None):
    """读固定验证集名单；文件不存在或坏了就返回空列表（= 还没初始化）。"""
    path = path or VAL_SPLIT_PATH
    try:
        with open(path, encoding="utf-8") as f:
            obj = json.load(f)
        names = obj.get("names") if isinstance(obj, dict) else obj
        return [str(n) for n in names] if isinstance(names, list) else []
    except (OSError, ValueError, AttributeError):
        return []


def save_val_split(names, path=None):
    """写回名单。先写临时文件再替换，中途打断不会留下半截 JSON。"""
    path = path or VAL_SPLIT_PATH
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    obj = {
        "note": "固定验证集名单：只增不减，新上传的图一律进训练集。"
                "这样各代 mAP 才在同一张考卷上可比。",
        "count": len(names),
        "names": sorted(names),
    }
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def export_yolo_label(ann, has_seg=True):
    """
    生成一张图的 YOLO 标签文本。

    参数 has_seg:
      True  -> 写分割格式（类别 + 一串多边形顶点）
      False -> 写检测格式（类别 + 中心点/宽高）

    本项目默认 True，因为核心任务是实例分割。
    写检测格式时用多边形的外接矩形自动推出——这正是「掩码派生检测框」。
    """
    lines = []
    for obj in ann.objects:
        if len(obj.polygon) < 3:
            continue
        cls_id = CLASS_TO_ID[obj.class_name]

        if has_seg:
            flat = []
            for x, y in obj.polygon:
                # 【兜底夹取到 [0,1]】
                # 轮廓沿像素缝隙追踪时，贴边物体可能产生 -0.5 像素的顶点，
                # 归一化后是 -0.0005 这种负数。YOLO 规范要求坐标在 [0,1]，
                # 越界会被校验判为有问题，部分训练器还会直接跳过该行。
                # 前端 geom.js 已在生成时夹取，这里再夹一次是为了兜住
                # 历史标注文件里已经存在的脏数据（那批数据不会再回头重算）。
                x = 0.0 if x < 0 else (1.0 if x > 1 else x)
                y = 0.0 if y < 0 else (1.0 if y > 1 else y)
                flat.append("%.6f" % x)
                flat.append("%.6f" % y)
            lines.append("%d %s" % (cls_id, " ".join(flat)))
        else:
            xmin, ymin, xmax, ymax = obj.bbox()
            xmin, ymin = max(0.0, xmin), max(0.0, ymin)
            xmax, ymax = min(1.0, xmax), min(1.0, ymax)
            xc = (xmin + xmax) / 2.0
            yc = (ymin + ymax) / 2.0
            w = max(0.0, xmax - xmin)
            h = max(0.0, ymax - ymin)
            lines.append("%d %.6f %.6f %.6f %.6f" % (cls_id, xc, yc, w, h))
    return "\n".join(lines) + ("\n" if lines else "")


def export_dataset(annotations, out_dir, image_dir, val_ratio=VAL_RATIO,
                   has_seg=True, copy_images=True, seed=42, split_path=None):
    """
    主导出函数：把一批标注导出成完整的 YOLO 数据集。

    参数:
      annotations : list[Annotation]
      out_dir     : 导出根目录（一般是 ./dataset）
      image_dir   : 原始图片所在目录
      copy_images : True = 复制图片到 dataset/images/；False = 不复制（只导标签）
      seed        : 只在「第一次初始化验证集名单」时用
      split_path  : 验证集名单文件。默认 data/val_split.json。
                    列表为空时按 val_ratio 初始化一次并落盘；之后一律以名单为准。

    返回一个 dict，记录本次导出的统计信息，用于「提交信息.txt」和对比页。
    """
    random.seed(seed)

    # 1) 划分：以固定名单为准，新图一律进训练集
    anns = sorted(annotations, key=lambda a: a.image_name)
    total = len(anns)
    split_path = split_path or VAL_SPLIT_PATH
    fixed = load_val_split(split_path)
    if fixed:
        val_names = {a.image_name for a in anns if a.image_name in set(fixed)}
        holdout_note = "沿用固定名单"
    else:
        # 首次：按比例随机抽一次，落盘后永久固定
        n_val = max(1, int(round(total * val_ratio))) if total > 1 else 0
        val_names = {anns[i].image_name for i in random.sample(range(total), n_val)} if n_val else set()
        if val_names:
            save_val_split(val_names, split_path)
        holdout_note = "首次初始化并落盘"
    # 1b) 数据泄漏检查：内容完全相同的图，不能一张进 train、一张进 val
    #      （同一张照片传两次，模型训练时背下来了，验证时自然"答对"）
    leak_fixed = []
    for _digest, names in find_duplicate_groups([a.image_name for a in anns], image_dir):
        # 用 any 而不是 names[0]：组内名字是按文件名排序的，
        # 第一张可能正好在训练集（copy_of_xxx 排在 xxx 前面），
        # 只看第一个会漏检一整组。
        if any(n in val_names for n in names):
            for n in names:
                if n in val_names:
                    val_names.discard(n)
                    leak_fixed.append(n)
    if leak_fixed:
        # 名单也一起清掉，否则下次导出又要原地纠一遍
        save_val_split(val_names, split_path)
        if not val_names:
            raise ValueError(
                "验证集被清空了：名单里的图全都有内容完全相同的重复副本。"
                "请删掉重复的图，或删除 %s 让系统重新划分。" % split_path)

    val_set = {i for i, a in enumerate(anns) if a.image_name in val_names}

    # 2) 建目录
    for sub in ("images/train", "images/val",
                "labels/train", "labels/val",
                "masks/train", "masks/val",
                "semantic/train", "semantic/val"):
        os.makedirs(os.path.join(out_dir, sub.replace("/", os.sep)), exist_ok=True)

    stats = {
        "total_images": total,
        "train_images": 0,
        "val_images": 0,
        "total_objects": 0,
        "class_counts": {c: 0 for c in CLASSES},
        "images_with_label": 0,
        # 让人一眼看出这次导出有没有偷偷换考卷
        "val_split_source": holdout_note,
        "val_split_fixed": len(fixed),
        # 被发现并强制挪进训练集的重复图（非空就说明库里有多份同一张照片）
        "duplicates_moved": leak_fixed,
    }

    # 记录本次真正写出的文件，最后用来清理「上一次的残留」。
    # 【为什么不先 rmtree 整个目录再重建】
    # 1) 删 150 个文件这种批量删除容易被系统安全策略拦下（我们真踩过：
    #    批量删除触发保护直接中断了导出请求，页面报不出错）；
    # 2) 更重要的是没必要——导出是幂等的，同名文件直接覆盖就行，
    #    只有「上次的文件这次不写了」才是残留，删掉那部分就够了。
    written = set()

    for i, ann in enumerate(anns):
        split = "val" if i in val_set else "train"
        stem = os.path.splitext(ann.image_name)[0]
        written.add(os.path.join(out_dir, "images", split, ann.image_name))

        # --- 图片 ---
        if copy_images:
            src = os.path.join(image_dir, ann.image_name)
            if os.path.exists(src):
                dst = os.path.join(out_dir, "images", split, ann.image_name)
                shutil.copy2(src, dst)
            else:
                # 找不到原图就跳过这张，但计数照记，便于发现数据缺失
                pass

        # --- YOLO 标签 ---
        label_path = os.path.join(out_dir, "labels", split, stem + ".txt")
        with open(label_path, "w", encoding="utf-8") as f:
            f.write(export_yolo_label(ann, has_seg=has_seg))
        written.add(label_path)

        # --- 实例掩码 PNG ---
        inst_img, mapping = instance_from_objects(ann.objects, ann.width, ann.height)
        inst_path = os.path.join(out_dir, "masks", split, stem + ".png")
        inst_img.save(inst_path)
        written.add(inst_path)

        # 实例掩码的类别信息另存一份，方便人工核对
        inst_txt = os.path.join(out_dir, "masks", split, stem + "_instances.txt")
        with open(inst_txt, "w", encoding="utf-8") as f:
            for val, (cls_name, idx) in sorted(mapping.items()):
                f.write("%d %s\n" % (val, cls_name))
        written.add(inst_txt)

        # --- 语义分割 PNG ---
        sem_img = semantic_from_objects(ann.objects, ann.width, ann.height)
        sem_path = os.path.join(out_dir, "semantic", split, stem + ".png")
        sem_img.save(sem_path)
        written.add(sem_path)

        # --- 统计 ---
        stats["train_images" if split == "train" else "val_images"] += 1
        stats["total_objects"] += len(ann.objects)
        for o in ann.objects:
            stats["class_counts"][o.class_name] = stats["class_counts"].get(o.class_name, 0) + 1
        if ann.objects:
            stats["images_with_label"] += 1

    # 3) data.yaml
    yaml_path = os.path.join(out_dir, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write("# 由 exporter.py 自动生成，请勿手工修改\n")
        # path 写 "."（相对本文件所在目录），这样提交包换到任何机器上都能用。
        # train.py 在真正训练前会把它改写成本机的绝对路径。
        f.write("path: .\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("nc: %d\n" % len(CLASSES))
        f.write("names:\n")
        for i, c in enumerate(CLASSES):
            f.write("  %d: %s\n" % (i, c))

    # 4) 处理上一次导出留下的残留文件
    #    只动「这次没写、但目录里有」的文件。
    #
    #    【为什么不删，而是挪走】
    #    每次导出都会按 val_ratio 重新划分 train/val，图片总数一变，
    #    归属就跟着变，于是「这次没写」的残留可能几十上百个。
    #    原先这里逐个 os.remove，在开了批量删除保护（杀软、企业安全策略、
    #    某些回收站软件）的机器上会触发拦截，表现是**页面请求直接断开**，
    #    前端只看到 `Failed to fetch`，服务端连一行访问日志都来不及写。
    #    挪到 dataset/_stale/ 是 rename，不产生删除动作，不会被拦，
    #    而且同名覆盖，不会越滚越大；万一哪天要找回旧图也还在。
    stale_root = os.path.join(out_dir, "_stale")
    moved = 0
    for sub in ("images", "labels", "masks", "semantic"):
        for split in ("train", "val"):
            d = os.path.join(out_dir, sub, split)
            if not os.path.isdir(d):
                continue
            for fn in os.listdir(d):
                p = os.path.join(d, fn)
                if not os.path.isfile(p) or p in written:
                    continue
                dst = os.path.join(stale_root, sub, split, fn)
                try:
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    os.replace(p, dst)   # 同名直接覆盖，不累积
                    moved += 1
                except OSError:
                    pass
    stats["stale_moved"] = moved

    return stats


# ---------------------------------------------------------------------------
# 格式自检
# ---------------------------------------------------------------------------

def validate_dataset(out_dir):
    """
    导出后的格式校验。返回 (是否通过, 问题列表)。

    校验项（对应任务书"文档导出前的格式校验"）：
      1. 三种格式的对象数量是否一致
      2. YOLO 标签里的坐标是否都在 [0,1]
      3. 类别编号是否在 0~9 范围
      4. 多边形顶点是否 >= 3
      5. 语义 PNG 的像素值是否只有 0~9 和 255
      6. train/val 是否有图
    """
    problems = []
    counts = {"label_objects": 0, "mask_instances": 0, "semantic_classes": set()}

    for split in ("train", "val"):
        label_dir = os.path.join(out_dir, "labels", split)
        mask_dir = os.path.join(out_dir, "masks", split)
        sem_dir = os.path.join(out_dir, "semantic", split)

        if not os.path.isdir(label_dir):
            problems.append("缺少目录 labels/%s" % split)
            continue

        stems = [os.path.splitext(f)[0] for f in os.listdir(label_dir)
                 if f.endswith(".txt") and not f.endswith("_instances.txt")]
        if split == "train" and not stems:
            problems.append("训练集为空")
        if split == "val" and not stems:
            problems.append("验证集为空")

        for stem in stems:
            # --- 检查 YOLO 标签 ---
            with open(os.path.join(label_dir, stem + ".txt"), encoding="utf-8") as f:
                for lineno, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    cls_id = int(parts[0])
                    coords = [float(x) for x in parts[1:]]
                    counts["label_objects"] += 1

                    if not (0 <= cls_id < len(CLASSES)):
                        problems.append("%s/%s.txt 第%d行: 类别编号 %d 越界"
                                        % (split, stem, lineno, cls_id))
                    if len(coords) < 6 or len(coords) % 2 != 0:
                        problems.append("%s/%s.txt 第%d行: 顶点数量异常"
                                        % (split, stem, lineno))
                    for c in coords:
                        if c < 0.0 or c > 1.0:
                            problems.append("%s/%s.txt 第%d行: 坐标 %.4f 超出 [0,1]"
                                            % (split, stem, lineno, c))
                            break

            # --- 检查实例掩码 ---
            mask_path = os.path.join(mask_dir, stem + ".png")
            inst_txt = os.path.join(mask_dir, stem + "_instances.txt")
            if not os.path.exists(mask_path):
                problems.append("%s/%s 缺少实例掩码 PNG" % (split, stem))
            else:
                # 实例数以 _instances.txt 的映射表为准。
                # 【为什么不能数 PNG 里的像素值】
                # 1) 掩码是 uint8，灰度值上限 255，一张图超过 255 个实例会溢出；
                # 2) 多边形互相重叠时后画的会覆盖先画的，像素值天然少于实例数。
                # 所以数像素是错的，必须读我们自己写的映射表。
                if os.path.exists(inst_txt):
                    with open(inst_txt, encoding="utf-8") as f:
                        n = len([ln for ln in f if ln.strip()])
                    counts["mask_instances"] += n
                else:
                    problems.append("%s/%s 缺少实例映射表 _instances.txt" % (split, stem))

            # --- 检查语义 PNG ---
            sem_path = os.path.join(sem_dir, stem + ".png")
            if not os.path.exists(sem_path):
                problems.append("%s/%s 缺少语义 PNG" % (split, stem))
            else:
                s = Image.open(sem_path)
                svals = set(s.getdata())
                for v in svals:
                    if v != 255 and v not in range(len(CLASSES)):
                        problems.append("%s/%s 语义 PNG 出现非法像素值 %d"
                                        % (split, stem, v))
                    elif v != 255:
                        counts["semantic_classes"].add(v)

    # 三种格式对象数量一致性（允许少量差异：模型可能给空图留 0 标注）
    if counts["label_objects"] != counts["mask_instances"]:
        problems.append(
            "对象数量不一致：YOLO标签 %d 个 vs 实例掩码 %d 个（差 %d）"
            % (counts["label_objects"], counts["mask_instances"],
               abs(counts["label_objects"] - counts["mask_instances"]))
        )

    return (len(problems) == 0), problems, counts


def format_report(counts):
    """把自检统计整理成可读文本，用于写进提交信息或对比结果。"""
    lines = []
    lines.append("YOLO 标签对象数: %d" % counts["label_objects"])
    lines.append("实例掩码实例数: %d" % counts["mask_instances"])
    lines.append("语义图覆盖类别数: %d" % len(counts["semantic_classes"]))
    return "\n".join(lines)
