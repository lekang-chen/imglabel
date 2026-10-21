"""
tools/package.py —— 提交包打包脚本

【任务书要求】
"项目根目录须自行编写 .gitignore（排除权重文件、原始图片压缩包等大文件）；
 dataset/ 与 model/best.pt 必须纳入压缩包。"

【提交目录结构】
├─ （项目源码）    app.py、canvas/、prelabel.py 等 + 启动.bat + requirements.txt
├─ dataset/        平台导出的三种格式数据集（YOLO txt / 实例掩码 / 语义 PNG）
├─ model/          自训权重 best.pt（必须为自训结果，评委复测）
├─ records/        修正记录 + 对比结果.csv（均由平台自动导出，无需手写）
├─ demo/           演示录屏不超过 3 分钟（启动→三模式标注→导出→对比页），或 3 张关键截图
├─ ai_log.md       AI 使用记录：时间 | 工具（Trae/Qoder/WorkBuddy）| 用途说明
├─ 提交信息.txt     姓名、学号、联系方式、数据来源与筛选说明、迭代轮次、各轮 mAP 与修正率
└─ README.md       一页以内：三条命令或双击即可运行

【怎么用】
  python tools/package.py --name 张三-学号20260101 --out D:/submit
"""

import argparse
import os
import shutil
import sys
import zipfile

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 要打进提交包的源码文件
CODE_FILES = [
    "app.py", "core.py", "prelabel.py", "exporter.py", "train.py",
    "requirements.txt", "README.md", "启动.bat", "ai_log.md", "提交信息.txt",
    ".gitignore",
]
CODE_DIRS = ["canvas", "tools"]

# 不需要进包的大目录
EXCLUDE_DIRS = {".venv", "venv", "__pycache__", "runs", ".git", "weights",
                 "data", ".idea", ".vscode"}
EXCLUDE_SUFFIX = (".pyc", ".log", ".tmp", ".zip")


def copy_code(dst_root):
    """拷贝源码部分。"""
    for f in CODE_FILES:
        src = os.path.join(BASE_DIR, f)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(dst_root, f))
        else:
            print("  跳过（不存在）: %s" % f)

    for d in CODE_DIRS:
        src_dir = os.path.join(BASE_DIR, d)
        if not os.path.isdir(src_dir):
            continue
        dst_dir = os.path.join(dst_root, d)
        shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns(*EXCLUDE_SUFFIX,
                                                     "__pycache__"))
        print("  拷贝目录: %s/" % d)


def copy_tree(src, dst):
    """拷贝一个目录（用于 dataset/ model/ records/ demo/）。"""
    if not os.path.isdir(src):
        print("  跳过（目录不存在）: %s" % os.path.basename(src))
        return 0
    n = 0
    os.makedirs(dst, exist_ok=True)
    for root, dirs, files in os.walk(src):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        rel = os.path.relpath(root, src)
        target = os.path.join(dst, rel) if rel != "." else dst
        os.makedirs(target, exist_ok=True)
        for f in files:
            if f.endswith(EXCLUDE_SUFFIX):
                continue
            shutil.copy2(os.path.join(root, f), os.path.join(target, f))
            n += 1
    print("  拷贝 %s/ 共 %d 个文件" % (os.path.basename(src.rstrip(os.sep)), n))
    return n


def check_required(dst_root):
    """检查提交包里的关键内容是否齐全。"""
    problems = []
    required = [
        ("README.md", "README"),
        ("提交信息.txt", "提交信息"),
        ("ai_log.md", "AI 使用记录"),
        ("model/best.pt", "自训权重"),
        ("dataset/data.yaml", "导出数据集"),
    ]
    for path, label in required:
        p = os.path.join(dst_root, path.replace("/", os.sep))
        if not os.path.exists(p):
            problems.append("缺少 %s (%s)" % (path, label))

    # 数据集三个格式
    for sub in ("images/train", "labels/train", "masks/train", "semantic/train"):
        p = os.path.join(dst_root, "dataset", sub.replace("/", os.sep))
        if not os.path.isdir(p) or not os.listdir(p):
            problems.append("dataset/%s 为空" % sub)

    # 修正记录
    rec = os.path.join(dst_root, "records", "修正记录.csv")
    if os.path.exists(rec):
        with open(rec, encoding="utf-8-sig") as f:
            n = sum(1 for _ in f) - 1
        if n < 60:
            problems.append("修正记录只有 %d 条，任务书要求不少于 60 条" % n)
        else:
            print("  修正记录: %d 条" % n)
    else:
        problems.append("缺少 records/修正记录.csv")

    # demo
    demo = os.path.join(dst_root, "demo")
    if not os.path.isdir(demo) or not os.listdir(demo):
        problems.append("demo/ 为空（需要录屏 ≤3 分钟或 3 张关键截图）")

    return problems


def main():
    ap = argparse.ArgumentParser(description="打包提交物")
    ap.add_argument("--name", default="提交包", help="压缩包名（建议 姓名-学号）")
    ap.add_argument("--out", default="D:/submit", help="输出目录")
    args = ap.parse_args()

    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)
    stage = os.path.join(out_dir, "stage")
    if os.path.isdir(stage):
        shutil.rmtree(stage)
    os.makedirs(stage)

    print("=" * 60)
    print("打包提交物")
    print("=" * 60)

    print("[1/4] 拷贝源码")
    copy_code(stage)

    print("[2/4] 拷贝成果目录")
    for d in ("dataset", "model", "records", "demo"):
        copy_tree(os.path.join(BASE_DIR, d), os.path.join(stage, d))

    print("[3/4] 检查完整性")
    problems = check_required(stage)
    if problems:
        print("\n  ⚠ 以下问题需要在提交前解决：")
        for p in problems:
            print("    - %s" % p)
    else:
        print("  ✓ 提交包内容完整")

    print("[4/4] 压缩")
    zip_path = os.path.join(out_dir, "%s.zip" % args.name)
    if os.path.exists(zip_path):
        os.remove(zip_path)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for root, dirs, files in os.walk(stage):
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
            for f in files:
                full = os.path.join(root, f)
                rel = os.path.relpath(full, stage)
                z.write(full, rel)

    size = os.path.getsize(zip_path) / 1048576
    print("\n  压缩包: %s  (%.1f MB)" % (zip_path, size))
    print("  上传地址: ftp://10.4.16.13  账号 ruiyi")
    if problems:
        print("\n  注意：上面列出的问题解决后请重新运行本脚本。")
    print("=" * 60)


if __name__ == "__main__":
    main()
