"""手机竖屏效果图（demo/08、09）。

【为什么不用 --window-size=390,844 直接截】
无头模式下 position:fixed 的元素会被定位到"截图视口"的底部，
而截图视口的高度和真实屏幕不一致，结果就是右下角那两个抽屉按钮
在图上根本画不出来——看上去像"手机上没入口"，其实是截图工具的问题。
用固定尺寸的 iframe（390×844）渲染，fixed 相对 iframe，位置稳定可复现。

用法：python tools/shoot_mobile.py
产出：demo/08_手机竖屏.png、demo/09_手机_图片抽屉.png
"""
import os
import subprocess
import sys

import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEMO_DIR = os.path.join(BASE_DIR, "demo")
HARNESS = os.path.join(BASE_DIR, "canvas", "_mob_harness.html")
PROFILE = os.path.join(os.environ.get("TEMP", "/tmp"), "mob_shot_profile")

EDGE_CANDIDATES = [
    r"C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",
    r"C:/Program Files/Microsoft/Edge/Application/msedge.exe",
    r"C:/Program Files/Google/Chrome/Application/chrome.exe",
]

W, H = 390, 844          # 典型手机竖屏 CSS 尺寸
SHOT_W, SHOT_H = 1400, 1000   # 外层浏览器窗口（要装得下 iframe）


def find_browser():
    for p in EDGE_CANDIDATES:
        if os.path.exists(p):
            return p
    raise SystemExit("找不到 Edge/Chrome")


def make_harness(script=""):
    html = """<!DOCTYPE html><html><head><meta charset="utf-8">
<style>html,body{margin:0;background:#ddd}iframe{border:0;width:%dpx;height:%dpx}</style>
</head><body>
<iframe id="fr" src="/?img=%s&t=%d"></iframe>
<script>
const fr = document.getElementById('fr');
fr.onload = () => { setTimeout(() => { %s }, 3500); };
</script></body></html>""" % (W, H, IMG, int(__import__("time").time()), script)
    with open(HARNESS, "w", encoding="utf-8") as f:
        f.write(html)


def shoot(browser, out_path):
    if os.path.exists(out_path):
        os.remove(out_path)
    cmd = [
        browser, "--headless=new", "--disable-gpu", "--hide-scrollbars",
        "--window-size=%d,%d" % (SHOT_W, SHOT_H),
        "--virtual-time-budget=11000",
        "--user-data-dir=" + PROFILE,
        "--incognito",
        "--screenshot=" + out_path,
        "http://127.0.0.1:5000/static/_mob_harness.html",
    ]
    subprocess.run(cmd, capture_output=True, timeout=120)
    if not os.path.exists(out_path):
        print("  FAIL 未生成", out_path)
        return False
    # 裁掉外层窗口的空白，只留 iframe 区域
    # 【为什么不用 cv2.imread】OpenCV 在 Windows 下读不了含中文的路径，
    # 会静默返回 None（文件名是「08_手机竖屏.png」，正好踩中）
    im = cv2.imdecode(np.fromfile(out_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if im is None:
        print("  FAIL 读不出截图", out_path)
        return False
    crop = im[0:H, 0:W]
    # 放大 2 倍：和手机 DPR 2 的观感一致，文字不发虚
    crop = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    ok, enc = cv2.imencode(".png", crop)
    enc.tofile(out_path)
    print("  OK  %s  %dx%d" % (os.path.basename(out_path), crop.shape[1], crop.shape[0]))
    return True


IMG = "000001.jpg"


def main():
    browser = find_browser()
    os.makedirs(DEMO_DIR, exist_ok=True)
    try:
        print("手机竖屏截图：")
        make_harness("")
        shoot(browser, os.path.join(DEMO_DIR, "08_手机竖屏.png"))

        # 第二张：点一下左上抽屉按钮，展示图片列表抽屉
        make_harness(
            "fr.contentWindow.__APP__.toggleDrawer('left');"
            "fr.contentWindow.__APP__.renderImageList();")
        shoot(browser, os.path.join(DEMO_DIR, "09_手机_图片抽屉.png"))
    finally:
        if os.path.exists(HARNESS):
            os.remove(HARNESS)


if __name__ == "__main__":
    main()
