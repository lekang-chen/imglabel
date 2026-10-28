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


def shoot_scripted(browser, out_path, script, wait=9000, size="1600,950"):
    """先执行一段页面脚本，再截图。

    用于拍「需要交互才能出现的状态」——比如选中某个对象后的聚焦视图、
    勾选框筛选后的效果。这些状态没法用 URL 参数直接进入。
    """
    # 把脚本挂到页面上，用 ?uitest=1 触发加载
    tmp_script = os.path.join(BASE_DIR, "canvas", "demo_script.js")
    with open(tmp_script, "w", encoding="utf-8") as f:
        f.write(script)
    cmd = [
        browser, "--headless=new", "--disable-gpu", "--hide-scrollbars",
        "--window-size=" + size,
        "--virtual-time-budget=%d" % wait,
        "--user-data-dir=" + PROFILE,
        "--screenshot=" + out_path,
        BASE_URL + "?uitest=1",
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        ok = os.path.exists(out_path) and os.path.getsize(out_path) > 5000
        print("  %s  %s" % ("OK " if ok else "FAIL", os.path.basename(out_path)))
        return ok
    finally:
        # 【必须在 finally 里删，不能靠调用方收尾】
        # 任何带 ?uitest=1 的页面都会自动加载这个文件。一旦残留，
        # 下次跑 UI 测试时它会在后台抢着 selectImage、切换工具，
        # 症状是「测试里的当前图片莫名变成别的图、S.img 为空后崩在 toCanvas」——
        # 排查起来极难，因为报错位置和真实原因（一个没删的文件）完全无关。
        if os.path.exists(tmp_script):
            try:
                os.remove(tmp_script)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# 交互态截图用的脚本
# ---------------------------------------------------------------------------
SCRIPT_FOCUS = """
(async () => {
  // 等页面初始化完成
  for (let i = 0; i < 60; i++) {
    if (window.__APP__ && window.__APP__.S && window.__APP__.S.images.length) break;
    await new Promise(r => setTimeout(r, 250));
  }
  {
    const A = window.__APP__;
    if (!A) return;
    // 选一张对象多的图
    const rich = A.S.images.filter(i => i.total > 6).sort((a,b)=>b.total-a.total)[0];
    if (rich) { await A.selectImage(rich.name); await new Promise(r=>setTimeout(r,1500)); }
    // 选中第 3 个对象 -> 进入聚焦
    const objs = A.S.anno.objects;
    if (objs[2]) {
      A.S.selectedId = objs[2].id;
      A.S.focusMode = true;
      A.renderObjects();
      A.draw();
    }
  }
})();
"""

SCRIPT_CHECKBOX = """
(async () => {
  for (let i = 0; i < 60; i++) {
    if (window.__APP__ && window.__APP__.S && window.__APP__.S.images.length) break;
    await new Promise(r => setTimeout(r, 250));
  }
  {
    const A = window.__APP__;
    if (!A) return;
    const rich = A.S.images.filter(i => i.total > 6).sort((a,b)=>b.total-a.total)[0];
    if (rich) { await A.selectImage(rich.name); await new Promise(r=>setTimeout(r,1500)); }
    // 取消勾选前几个对象，演示筛选
    A.S.focusMode = false;
    A.S.anno.objects.slice(0, 4).forEach(o => A.S.unchecked.add(o.id));
    A.renderObjects();
    A.draw();
  }
})();
"""


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

    # 交互态截图（需要脚本驱动才能出现的状态）
    print("截图: 聚焦模式（选中单个对象）")
    shoot_scripted(browser, os.path.join(DEMO_DIR, "04_聚焦模式_只显示选中对象.png"),
                   SCRIPT_FOCUS, wait=11000)

    print("截图: 复选框筛选（隐藏部分对象）")
    shoot_scripted(browser, os.path.join(DEMO_DIR, "05_复选框筛选_隐藏部分对象.png"),
                   SCRIPT_CHECKBOX, wait=11000)

    # 清理临时脚本
    tmp_script = os.path.join(BASE_DIR, "canvas", "demo_script.js")
    if os.path.exists(tmp_script):
        os.remove(tmp_script)

    print("\n完成。demo/ 目录内容：")
    for f in sorted(os.listdir(DEMO_DIR)):
        if f.endswith(".png"):
            kb = os.path.getsize(os.path.join(DEMO_DIR, f)) // 1024
            print("  %-40s %5d KB" % (f, kb))


if __name__ == "__main__":
    main()
