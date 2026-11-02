"""
train.py —— 模型微调脚本

【任务书硬性要求】
1. "train.py 仅接受平台导出数据集"
   -> 本脚本只读 ./dataset（由 exporter.py 导出的那份），
      路径写死在 DATASET_DIR，任何其他输入都拒绝。
      这样就堵死了「直接拿 COCO 官方标注去训练」这条捷径，
      保证训练数据一定是本项目自己标注、导出的结果。

2. "train.py 自动检测配置并选择训练参数"
   -> 下面 probe_config() 会探测显存、CPU 核数、数据量，
      自动决定 batch / workers / epochs / imgsz。

3. "最低配置：Windows 10 x64、8GB 内存、20GB 可用磁盘"
   -> 纯 CPU 也能跑（只是慢）。任务书说"训练耗时一律不计分"。

【怎么用】
  python train.py                 # 自动配置
  python train.py --epochs 50     # 指定轮数
  python train.py --device cpu    # 强制 CPU

【产出】
  model/best.pt        最佳权重（这是要提交的那份）
  runs/seg/train/      完整训练过程记录
"""

import argparse
import json
import os
import shutil
import sys
import time

# ---------------------------------------------------------------------------
# 固定路径：只接受平台导出的数据集
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")          # ← 唯一允许的数据源
DATA_YAML = os.path.join(DATASET_DIR, "data.yaml")
MODEL_DIR = os.path.join(BASE_DIR, "model")
RUNS_DIR = os.path.join(BASE_DIR, "runs")
BACKUP_DIR = os.path.join(MODEL_DIR, "_backup")   # 被覆盖前的旧权重
BACKUP_KEEP = 3                                    # 只留最近 3 份（每份 6~7 MB）
# 【续训配方的来历】
# 这套超参不是拍脑袋定的，是第七~十一轮实验里唯一一条「检测和掩码同时涨」的
# （见 ai_log.md 的 iter3_long：0.733 / 0.583，R4 的 0.580 就出自它）。
# 关键几点，缺一个就退化：
#   · optimizer=SGD + 小 lr0=8e-4：续训不能给大学习率
#     （auto 会自己挑 AdamW 7e-4，实测一个 epoch 就把掩码从 0.580 打到 0.530）
#   · freeze=10：冻结主干，只动检测头与分割分支，抗遗忘
#   · copy_paste=0.5 + scale=0.7 + degrees=8：实例分割专用增强，靠它抗遗忘
#   · cls=1.2：分类损失提权，压「椅子判成沙发」这类跨类混淆
#   · cos_lr + close_mosaic=20 + patience=40：长训练要余弦退火，最后 20 轮关掉 mosaic
FT_LR = 8e-4

# 类别词表不再写在这里：改由 core.get_classes() 读 data/classes.json。
# 【为什么删掉这份副本】抄一份常量最省事，但词表一改（界面能改了）就会
# 训练用的 nc 和导出的标签编号对不上——这种错不报异常，只是指标莫名其妙地掉。
# 需要类别名的地方一律从 core 取。


def refuse_if_not_exported():
    """
    校验训练输入确实是「平台导出的数据集」。

    判据（必须全部满足）：
      1. dataset/data.yaml 存在
      2. dataset/images/train 和 dataset/labels/train 都存在且非空
      3. 三个格式的目录都存在（images / labels / masks / semantic）

    只要有一条不满足就退出并说明原因。
    这个函数是任务书「train.py 仅接受平台导出数据集」这条规定的落地。
    """
    if not os.path.exists(DATA_YAML):
        sys.exit(
            "[错误] 找不到 %s\n"
            "        train.py 只接受由平台导出的数据集。\n"
            "        请先在界面上点击「导出数据集」，或运行 exporter.py。" % DATA_YAML
        )

    required = [
        os.path.join(DATASET_DIR, "images", "train"),
        os.path.join(DATASET_DIR, "labels", "train"),
        os.path.join(DATASET_DIR, "masks", "train"),
        os.path.join(DATASET_DIR, "semantic", "train"),
    ]
    for d in required:
        if not os.path.isdir(d):
            sys.exit(
                "[错误] 导出数据集不完整，缺少目录: %s\n"
                "        请重新执行导出，确保三种格式都生成。" % d
            )
        if not os.listdir(d):
            sys.exit("[错误] 目录为空: %s\n        请重新导出数据集。" % d)

    n_images = len([f for f in os.listdir(required[0])
                    if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))])
    if n_images == 0:
        sys.exit("[错误] 训练集里没有图片，请重新导出。")

    # --- 生成一份「带本机绝对路径」的训练用 yaml ---
    # 【为什么不再直接改写 dataset/data.yaml】
    # 原来是把 data.yaml 的 path 原地改成绝对路径，训练完再还原。
    # 但这有个隐患：训练过程中如果用户在界面上点了「导出数据集」，
    # exporter 会重写 data.yaml（path 变回 "."），而 ultralytics 在
    # 训练收尾做最终验证时会重新读这份 yaml，于是 path 按当前工作目录
    # 解析成 D:/imglabel/images/val（不存在）——
    # 实测就报过：Dataset images not found, missing path 'D:\imglabel\images\val'，
    # 51 轮白训，best.pt 也不会被拷到 model/。
    #
    # 改成训练专用的一份临时 yaml：导出怎么改 data.yaml 都影响不到训练。
    # 数据集配置必须跟着机器走，同时又不该被别的操作干扰。
    _write_train_yaml()

    return n_images


TRAIN_YAML = os.path.join(DATASET_DIR, "_train.yaml")


def _write_train_yaml():
    """生成训练专用 yaml（path 指向本机绝对路径），返回文件路径。

    与 dataset/data.yaml 的区别：那份的 path 刻意保持 "."（可移植，进提交包），
    这一份只在本机训练时用，跑完即删，不进版本库。
    """
    try:
        with open(DATA_YAML, encoding="utf-8") as f:
            lines = f.read().splitlines()
    except OSError:
        # dataset/data.yaml 不存在时用当前词表兜底（正常情况下走不到这里，
        # refuse_if_not_exported 已经先拦了）
        from core import get_classes
        names = get_classes()
        lines = ["nc: %d" % len(names), "names:"]
        lines += ["  %d: %s" % (i, c) for i, c in enumerate(names)]

    new_path = DATASET_DIR.replace("\\", "/")
    out, replaced = [], False
    for ln in lines:
        if ln.strip().startswith("path:"):
            out.append("path: %s" % new_path)
            replaced = True
        else:
            out.append(ln)
    if not replaced:
        out.insert(0, "path: %s" % new_path)

    with open(TRAIN_YAML, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    return TRAIN_YAML


def _cleanup_train_yaml():
    """删掉训练用的临时 yaml，避免混进提交包。"""
    try:
        if os.path.exists(TRAIN_YAML):
            os.remove(TRAIN_YAML)
    except OSError:
        pass


def _rewrite_data_yaml_path():
    """
    把 data.yaml 的 path 字段改成本机 DATASET_DIR 的绝对路径。

    为什么需要：
    - data.yaml 是导出时生成的，里面记的是当时的绝对路径；
    - 提交包被评委解压到别的目录后，这个路径就不存在了；
    - 训练前重写一次，等于让数据集配置"跟着机器走"。

    只改 path 这一行，nc / names / train / val 保持导出时的样子不动。

    注意：改写后必须调用 restore_data_yaml_path() 还原成相对路径，
    否则这份被污染的 data.yaml 会跟着进提交包，评委换机器就训不了。
    """
    try:
        with open(DATA_YAML, encoding="utf-8") as f:
            lines = f.read().splitlines()
    except OSError:
        return

    new_path = DATASET_DIR.replace("\\", "/")
    out, replaced = [], False
    for ln in lines:
        if ln.strip().startswith("path:"):
            out.append("path: %s" % new_path)
            replaced = True
        else:
            out.append(ln)
    if not replaced:
        out.insert(0, "path: %s" % new_path)

    with open(DATA_YAML, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")


def restore_data_yaml_path():
    """
    把 data.yaml 的 path 还原成相对路径 "."。

    为什么需要：
    训练时为了让 ultralytics 找到数据，会把 path 改写成绝对路径。
    但提交包里的 data.yaml 必须是可移植的——评委解压到任何目录都能用。
    所以训练结束后立刻还原，避免污染提交物。

    相对路径 "." 的含义是"以 data.yaml 所在目录为根"，
    配合 train: images/train 这种相对写法，整个数据集目录可以整体搬走。
    """
    try:
        with open(DATA_YAML, encoding="utf-8") as f:
            lines = f.read().splitlines()
    except OSError:
        return

    out, replaced = [], False
    for ln in lines:
        if ln.strip().startswith("path:"):
            out.append("path: .")
            replaced = True
        else:
            out.append(ln)
    if not replaced:
        out.insert(0, "path: .")

    with open(DATA_YAML, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")


def probe_config(n_images):
    """
    自动探测硬件配置，选择训练参数。

    为什么要自动化？
    任务书要求"train.py 自动检测配置并选择训练参数"，
    而且评委复测时可能用一台完全不同的机器（甚至 8GB 内存无独显）。
    写死参数会导致在别人机器上跑不起来。

    探测项：
      - 有没有 CUDA、显存多大  -> 决定 batch 和 imgsz
      - CPU 核心数            -> 决定 dataloader workers
      - 数据量                -> 决定 epochs（数据少就多训几轮）
    """
    import torch

    cfg = {
        "device": "cpu",
        "batch": 4,
        "imgsz": 640,
        "workers": 2,
        "epochs": 100,
        "gpu_name": "无独显（纯 CPU 训练）",
        "vram_gb": 0,
    }

    # --- GPU / 显存 ---
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        vram_gb = props.total_memory / (1024 ** 3)
        cfg["device"] = "0"
        cfg["gpu_name"] = props.name
        cfg["vram_gb"] = round(vram_gb, 1)

        if vram_gb >= 10:
            cfg["batch"] = 16
            cfg["imgsz"] = 640
        elif vram_gb >= 6:
            cfg["batch"] = 8
            cfg["imgsz"] = 640
        elif vram_gb >= 4:
            cfg["batch"] = 4
            cfg["imgsz"] = 512
        else:
            cfg["batch"] = 2
            cfg["imgsz"] = 448
    else:
        # 纯 CPU：核多就多开 workers
        cores = os.cpu_count() or 4
        cfg["batch"] = 4
        cfg["imgsz"] = 640
        cfg["workers"] = max(0, min(4, cores // 2))

    # --- 数据量少 -> 轮数可以多一些 ---
    if n_images < 100:
        cfg["epochs"] = 150
    elif n_images < 200:
        cfg["epochs"] = 100
    else:
        cfg["epochs"] = 60

    # --- CPU 训练时轮数减半，否则太慢 ---
    if cfg["device"] == "cpu":
        cfg["epochs"] = max(30, cfg["epochs"] // 2)

    return cfg


def pick_base_weights():
    """
    选择用于微调的基线权重。

    任务书说"预训练权重：官方在大规模数据集上已学好的模型参数，
    本项目以其为起点训练而非从零训练"—— 也就是微调（fine-tuning）。

    优先用本项目第一轮用的实例分割模型，保持迭代的一致性：
      R1 预标注用 YOLO26n-seg -> R2 微调也应该从 YOLO26n-seg 出发
    这样 R2 相对 R1 的提升才是"在同一模型上微调"的效果，归因清楚。
    """
    candidates = ["yolo26n-seg.pt", "yolov8n-seg.pt", "yolo11n-seg.pt"]
    for name in candidates:
        p = os.path.join(BASE_DIR, "weights", name)
        if os.path.exists(p):
            return p
    # 没放到 weights/ 就用 ultralytics 自动下载的
    for name in candidates:
        try:
            from ultralytics import YOLO
            YOLO(name)   # 触发下载
            return name
        except Exception:  # noqa: BLE001
            continue
    return "yolo26n-seg.pt"


def backup_existing_best(dst):
    """
    覆盖 model/best.pt 之前，先把旧的那份存进 model/_backup/。

    【为什么必须做】best.pt 是「要提交的那份」，覆盖是不可逆的。
    只要新训练比原来差（换种子、数据集没更新、早停早、只跑了几轮），
    验收用的权重就被换掉了，而本地没有第二份。
    实测同配置只换随机种子，掩码 mAP 能差 ±0.035——
    在这种波动下「再训一次」经常不是变好而是变差，所以旧的那份必须留得住。

    返回备份路径；原来没有 best.pt 时返回 None。只保留最近 BACKUP_KEEP 份。
    """
    if not os.path.exists(dst):
        return None
    os.makedirs(BACKUP_DIR, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = os.path.join(BACKUP_DIR, "best_%s.pt" % stamp)
    n = 1
    while os.path.exists(out):            # 同一秒内连训两次也不能互相覆盖
        out = os.path.join(BACKUP_DIR, "best_%s_%d.pt" % (stamp, n))
        n += 1
    shutil.copy2(dst, out)

    # 名字里带 baseline 的是人工钉住的基准权重，不参与「只留最近 N 份」的清理。
    # 否则连训几次之后，验收用的那份反而被自动删掉了——那正是备份要防的事。
    olds = [os.path.join(BACKUP_DIR, f) for f in os.listdir(BACKUP_DIR)
            if f.startswith("best_") and f.endswith(".pt")
            and "baseline" not in f]
    olds.sort(key=os.path.getmtime)          # 按时间留，不按文件名（同秒会加后缀，字母序会骗人）
    for p in olds[:-BACKUP_KEEP]:
        try:
            os.remove(p)
        except OSError:
            pass
    return out


def main():
    parser = argparse.ArgumentParser(description="人机协同图像标注系统 - 模型微调")
    parser.add_argument("--epochs", type=int, default=None, help="训练轮数")
    parser.add_argument("--batch", type=int, default=None, help="批大小")
    parser.add_argument("--imgsz", type=int, default=None, help="输入尺寸")
    parser.add_argument("--device", type=str, default=None,
                        help="设备，如 0 / cpu")
    parser.add_argument("--weights", type=str, default=None,
                        help="基线权重路径")
    parser.add_argument("--out", type=str, default=None,
                        help="把最佳权重写到指定路径（默认写 model/best.pt）。"
                             "自主学习用它产出「候选权重」，好让决策层先评估再决定要不要采纳")
    parser.add_argument("--lr0", type=float, default=None,
                        help="初始学习率。从已有权重续训时要调小（如 0.002），"
                             "否则一步就把学到的走样了")
    parser.add_argument("--patience", type=int, default=None,
                        help="早停耐心轮数（默认 50）")
    parser.add_argument("--ft", action="store_true",
                        help="快速微调模式：小学习率、关掉 mosaic / copy_paste、"
                             "不做 warmup。从已经调好的权重再学时用这条路，"
                             "默认那套强增强会把好不容易学到的掩码打坏")
    parser.add_argument("--unfreeze", action="store_true",
                        help="配合 --ft：不冻结主干。换一个新类别（形态和旧类差很远）时用")
    parser.add_argument("--allow-class-rename", action="store_true",
                        help="允许「类别数不变、只是某个位置换了名字」时续训，"
                             "例如把 book 换成 bed。检测头维度没变，其余类别原样保留，"
                             "只有换掉的那个位置会重新学。不加这个开关仍然硬拦")
    args = parser.parse_args()

    print("=" * 64)
    print("人机协同图像标注系统 - 模型微调")
    print("=" * 64)

    # --- 第一步：确认输入是平台导出的数据集 ---
    n_images = refuse_if_not_exported()
    print("[1/4] 数据集校验通过：%d 张训练图片（来自 %s）"
          % (n_images, os.path.relpath(DATASET_DIR, BASE_DIR)))

    # --- 第二步：探测配置 ---
    cfg = probe_config(n_images)
    if args.epochs:
        cfg["epochs"] = args.epochs
    if args.batch:
        cfg["batch"] = args.batch
    if args.imgsz:
        cfg["imgsz"] = args.imgsz
    if args.device:
        cfg["device"] = args.device
    if args.lr0:
        cfg["lr0"] = args.lr0
    if args.patience:
        cfg["patience"] = args.patience
    if args.ft:
        # 续训配方（iter3_long）：参数写在 cfg 里，下面统一塞给 model.train
        cfg.setdefault("optimizer", "SGD")     # 钉死优化器，否则 lr0 会被 auto 覆盖
        cfg.setdefault("lr0", FT_LR)
        cfg.setdefault("lrf", 0.01)
        cfg["cos_lr"] = True
        # 冻结前 10 层（backbone）：换场景时防止把已有特征打散。
        # 但「换一个新类别」不同——新类的形态可能和旧类差很远（book 小、bed 大），
        # 冻着主干只能调检测头，容易学不进去，所以留了 --unfreeze 让人自己决定。
        cfg["freeze"] = 0 if args.unfreeze else 10
        cfg["close_mosaic"] = 20
        cfg["copy_paste"] = 0.5
        cfg["scale"] = 0.7
        cfg["degrees"] = 8.0
        cfg["cls"] = 1.2
        cfg.setdefault("patience", 40)

    print("[2/4] 自动选择的训练配置：")
    print("        设备      : %s (%s, 显存 %.1f GB)"
          % (cfg["device"], cfg["gpu_name"], cfg["vram_gb"]))
    print("        批大小    : %d" % cfg["batch"])
    print("        输入尺寸  : %d" % cfg["imgsz"])
    print("        轮数      : %d" % cfg["epochs"])
    print("        workers   : %d" % cfg["workers"])

    # --- 第三步：训练 ---
    from ultralytics import YOLO

    base = args.weights or pick_base_weights()
    print("[3/4] 从预训练权重开始微调: %s" % base)
    model = YOLO(base)

    # 【词表改了就不能续训】类别词表现在可以在界面上增删。一旦类别数变了，
    # 权重的检测头维度和新的 nc 对不上：ultralytics 会按新 nc 重建 head
    # 并随机初始化，--ft 那几十轮等于从头学分类头，原来会的类一起被冲掉。
    # 这种错不抛异常、只是指标悄悄崩，所以必须在这里硬拦。
    if args.ft:
        from core import get_classes
        want = list(get_classes())
        have_map = getattr(model, "names", {}) or {}
        if isinstance(have_map, dict):
            have = [have_map.get(i, "") for i in range(max(len(want), len(have_map)))]
        else:
            have = list(have_map)
        if set(want) != set(have):
            # 【同位置换名是唯一安全的例外】book -> bed 这种：类别数没变，
            # 检测头张量形状完全一样，ultralytics 不会重建 head，前 9 个类的
            # 权重原样保留，只有换掉的那个位置从旧类的参数起步重新学。
            # 这比基座重训划算得多——重训要把其余九类从 COCO 重新学一遍。
            #
            # 但它必须由人显式声明：光看「长度相同」判断不了这是有意的换名，
            # 还是词表顺序被误改（那种情况下位置会错位，第 3 个类学的东西
            # 被当成第 5 个类，指标全线崩且不报错）。所以给开关，不给自动放行。
            same_len = len(want) == len(have)
            if args.allow_class_rename and same_len:
                swaps = [(i, have[i], want[i]) for i in range(len(want))
                         if have[i] != want[i]]
                print("[警告] 类别数相同但名字不同，按「同位置换名」处理：")
                for i, old, new in swaps:
                    print("         位置 %d：%s -> %s（这个类会从旧权重起步重新学）" % (i, old, new))
                print("         其余 %d 个类原样保留。" % (len(want) - len(swaps)))
            else:
                only_new = sorted(set(want) - set(have))
                only_old = sorted(set(have) - set(want))
                hint = ""
                if same_len:
                    hint = ("\n        类别数没变（都是 %d），如果这是「把某个类换成另一个类」"
                            "（例如 book -> bed），\n        加 --allow-class-rename 可以续训，"
                            "其余类别不会被冲掉。" % len(want))
                sys.exit(
                    "[错误] 词表和权重对不上，不能续训。\n"
                    "        当前词表：%s\n"
                    "        权重里的类：%s\n"
                    "        词表多出来：%s\n"
                    "        权重多出来：%s\n"
                    "        类别数变了，检测头就得重建，续训只会把学过的类冲掉。%s\n"
                    "        否则请改用基座重训（去掉 --ft，或界面上点「训练」）。"
                    % (", ".join(sorted(want)), ", ".join(sorted(h for h in have if h)),
                       ", ".join(only_new) or "无", ", ".join(only_old) or "无", hint))

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RUNS_DIR, exist_ok=True)

    t0 = time.time()
    try:
        model.train(
            data=TRAIN_YAML,
            epochs=cfg["epochs"],
            imgsz=cfg["imgsz"],
            batch=cfg["batch"],
            device=cfg["device"],
            workers=cfg["workers"],
            project=RUNS_DIR,
            name="seg",
            exist_ok=True,
            pretrained=True,
            optimizer=cfg.get("optimizer", "auto"),
            seed=42,
            lr0=cfg.get("lr0", 0.01),
            lrf=cfg.get("lrf", 0.01),
            cos_lr=cfg.get("cos_lr", False),
            freeze=cfg.get("freeze", None),
            close_mosaic=cfg.get("close_mosaic", 10),
            copy_paste=cfg.get("copy_paste", 0.0),
            scale=cfg.get("scale", 0.5),
            degrees=cfg.get("degrees", 0.0),
            cls=cfg.get("cls", 0.5),
            # 小数据集容易过拟合，加一点正则
            weight_decay=0.0005,
            patience=cfg.get("patience", 50),
            verbose=True,
            plots=True,
        )
    except Exception as exc:  # noqa: BLE001
        print("[错误] 训练失败: %s" % exc)
        # 失败也要清掉临时 yaml，别留垃圾
        _cleanup_train_yaml()
        sys.exit(1)

    # --- 第四步：把最佳权重拷到 model/ ---
    src = os.path.join(RUNS_DIR, "seg", "weights", "best.pt")
    if not os.path.exists(src):
        # 某些 ultralytics 版本路径不同，兜底找一下
        cands = []
        for root, _dirs, files in os.walk(RUNS_DIR):
            if "best.pt" in files:
                cands.append(os.path.join(root, "best.pt"))
        if not cands:
            print("[错误] 训练结束但没找到 best.pt")
            _cleanup_train_yaml()
            sys.exit(1)
        src = max(cands, key=os.path.getmtime)

    # --out 是「自主学习」用的候选权重出口：不覆盖 model/best.pt，
    # 先让决策层在验证集上评估，指标过线才由 autolearn 换上去。
    if args.out:
        dst = os.path.abspath(args.out)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        print("[4/4] 训练完成，耗时 %.1f 分钟" % ((time.time() - t0) / 60))
        print("        候选权重已输出: %s（model/best.pt 未被改动）" % dst)
        _cleanup_train_yaml()
        restore_data_yaml_path()
        return 0

    dst = os.path.join(MODEL_DIR, "best.pt")
    old = backup_existing_best(dst)
    if old:
        print("        旧权重已备份: %s" % old)
    shutil.copy2(src, dst)

    elapsed = time.time() - t0
    print("[4/4] 训练完成，耗时 %.1f 分钟" % (elapsed / 60))
    print("        最佳权重已保存: %s" % dst)

    _cleanup_train_yaml()

    # 还原 data.yaml 的 path 为相对路径（历史流程留下的兜底，正常已不会被改）。
    # 训练时改成绝对路径是为了让 ultralytics 找到数据，
    # 但提交包里的 data.yaml 必须可移植——评委解压到任何目录都能直接训练。
    restore_data_yaml_path()
    print("        data.yaml 已还原为相对路径（保证提交包可移植）")

    # 训练摘要写进 state，供对比页读取
    state_path = os.path.join(BASE_DIR, "data", "state.json")
    state = {}
    if os.path.exists(state_path):
        with open(state_path, encoding="utf-8") as f:
            state = json.load(f)
    state.setdefault("train_history", []).append({
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "epochs": cfg["epochs"],
        "batch": cfg["batch"],
        "imgsz": cfg["imgsz"],
        "device": cfg["gpu_name"],
        "minutes": round(elapsed / 60, 1),
        "n_images": n_images,
    })
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

    print("\n下一步：在界面上点击「预标注」并选择 model/best.pt 进行第二轮回标。")


if __name__ == "__main__":
    main()
