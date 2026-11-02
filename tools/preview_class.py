#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把某个类别的标注拼成一张总览图，一次看完所有样本，判断质量。

【为什么要这个】
零样本补标注（tools/add_class_zeroshot.py）拿到的是模型原始输出，质量参差。
20 个样本逐个打开看要 3 分钟且容易看漏；拼成网格一眼就能挑出「明显不对」的
那些——比如整张桌子被判成床。训练前花这 30 秒，比训完发现指标崩了再回头查
便宜得多。

红框 = 该类别；蓝框 = 同一张图上的其他类别（用来判断「这块区域是不是已经
被标成别的东西了」）。

用法：
  python tools/preview_class.py --cls bed
"""
import argparse
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image, ImageDraw  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE_DIR = os.path.join(ROOT, "data", "images")
ANNO_DIR = os.path.join(ROOT, "data", "annotations")
CELL_W, CELL_H = 360, 270


def collect(cls):
    """返回 [(图片名, 该类的多边形列表, 其他类多边形列表)]。"""
    out = []
    for fn in sorted(os.listdir(ANNO_DIR)):
        if not fn.endswith(".json"):
            continue
        with io.open(os.path.join(ANNO_DIR, fn), encoding="utf-8") as f:
            data = json.load(f)
        objs = data.get("objects") or []
        mine = [o["polygon"] for o in objs if o.get("class") == cls]
        if not mine:
            continue
        others = [o["polygon"] for o in objs if o.get("class") != cls]
        out.append((data["image"], mine, others))
    return out


def draw_cell(img, mine, others, label):
    W, H = img.size
    scale = min(CELL_W / W, (CELL_H - 16) / H)
    thumb = img.resize((max(1, int(W * scale)), max(1, int(H * scale))))
    cell = Image.new("RGB", (CELL_W, CELL_H), (245, 247, 250))
    ox, oy = (CELL_W - thumb.size[0]) // 2, 16 + (CELL_H - 16 - thumb.size[1]) // 2
    cell.paste(thumb, (ox, oy))
    d = ImageDraw.Draw(cell)

    def poly(pts, color, width):
        xy = [(ox + p[0] * thumb.size[0], oy + p[1] * thumb.size[1]) for p in pts]
        if len(xy) >= 3:
            d.line(xy + [xy[0]], fill=color, width=width, joint="curve")

    for pts in others:
        poly(pts, (120, 160, 220), 1)
    for pts in mine:
        poly(pts, (220, 30, 40), 2)
    d.rectangle([0, 0, CELL_W, 15], fill=(30, 41, 59))
    d.text((4, 3), label, fill=(240, 245, 250))
    return cell


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cls", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--cols", type=int, default=4)
    args = ap.parse_args()

    items = collect(args.cls)
    if not items:
        print("没有任何「%s」对象" % args.cls)
        return 1

    cells = []
    for name, mine, others in items:
        path = os.path.join(IMAGE_DIR, name)
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:                     # noqa: BLE001
            print("  读不到 %s (%s)" % (name, e))
            continue
        W, H = img.size
        # 面积最大的那个框占比，用来快速找「整张图都框进去」的样本
        big = 0.0
        for pts in mine:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            big = max(big, (max(xs) - min(xs)) * (max(ys) - min(ys)))
        cells.append(draw_cell(img, mine, others,
                               "%s  %d  max %.0f%%" % (name, len(mine), big * 100)))

    if not cells:
        print("没有可画的图")
        return 1

    cols = min(args.cols, len(cells))
    rows = (len(cells) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * CELL_W, rows * CELL_H), (255, 255, 255))
    for i, c in enumerate(cells):
        sheet.paste(c, ((i % cols) * CELL_W, (i // cols) * CELL_H))

    out = args.out or os.path.join(ROOT, "demo", "preview_%s.png" % args.cls)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    sheet.save(out)
    print("共 %d 张含「%s」的图，总览图已存 %s" % (len(cells), args.cls, out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
