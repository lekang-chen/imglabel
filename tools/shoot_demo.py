# -*- coding: utf-8 -*-
"""
demo 截图脚本：抓取关键界面截图放进 demo/。

为什么不用 Playwright：
  pip 源里装不上 playwright（离线环境），但 Windows 11 自带 Edge，
  Edge 的 headless 模式一样能截图，效果等价、零额外依赖。

为什么要有这个脚本：
  任务书要求 demo/ 里放"录屏或关键截图"。评委可能不跑你的环境，
  截图是他们了解系统长什么样的最低成本入口。
  脚本化之后，答辩前改了界面只要重跑一次，
  不会出现"截图和实际界面不一致"的尴尬。

用法：
  1) 先启动服务：python app.py
  2) 再运行：python tools/shoot_demo.py

输出到 demo/ 目录。
"""
import os
import subprocess
import sys
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEMO_DIR = os.path.join(BASE_DIR, "demo")
BASE_URL = "http://127.0.0.1:5000"
PROFILE = os.path.join(BASE_DIR, ".edge_tmp")

# Edge 路径按可能性排序
EDGE_CANDIDATES = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
]


def find_browser():
    for p in EDGE_CANDIDATES:
        if os.path.exists(p):
            return p
    return None


def shoot(browser, out_path, url=BASE_URL, wait=6000, size="1600,950"):
    """调 headless 浏览器截图。"""
    cmd = [
        browser, "--headless=new", "--disable-gpu", "--hide-scrollbars",
        "--window-size=" + size,
        "--virtual-time-budget=%d" % wait,
        "--user-data-dir=" + PROFILE,
        "--screenshot=" + out_path,
        url,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    ok = os.path.exists(out_path) and os.path.getsize(out_path) > 5000
    print("  %s  %s" % ("OK " if ok else "FAIL", os.path.basename(out_path)))
    return ok


def main():
    browser = find_browser()
    if not browser:
        print("找不到 Edge 或 Chrome，请检查安装路径")
        sys.exit(1)
    print("使用浏览器:", browser)

    os.makedirs(DEMO_DIR, exist_ok=True)
    # 清掉上次的结果，避免残留旧截图混淆
    for f in os.listdir(DEMO_DIR):
        if f.endswith(".png"):
            os.remove(os.path.join(DEMO_DIR, f))

    # 选一张对象最多的图做演示，这样截图里能看到多种类别
    demo_img = "000109.jpg"

    shots = [
        # (说明, 输出文件名, URL 后缀, 额外等待毫秒)
        ("主界面_三栏布局与图片列表", "00_主界面_三栏布局.png", "", 6000),
        ("实例分割模式", "01_实例分割_掩码编辑.png",
         "?mode=seg&img=%s" % demo_img, 9000),
        ("目标检测模式", "02_目标检测_自动生成矩形框.png",
         "?mode=det&img=%s" % demo_img, 9000),
        ("语义分割模式", "03_语义分割_按类合并.png",
         "?mode=sem&img=%s" % demo_img, 9000),
    ]

    for desc, fname, query, wait in shots:
        print("截图:", desc)
        shoot(browser, os.path.join(DEMO_DIR, fname),
              url=BASE_URL + query, wait=wait)

    print("\n完成。demo/ 目录内容：")
    for f in sorted(os.listdir(DEMO_DIR)):
        if f.endswith(".png"):
            kb = os.path.getsize(os.path.join(DEMO_DIR, f)) // 1024
            print("  %-40s %5d KB" % (f, kb))


if __name__ == "__main__":
    main()
