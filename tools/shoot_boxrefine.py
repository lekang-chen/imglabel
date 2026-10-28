# -*- coding: utf-8 -*-
"""为「框选修形」拍前后对比图（demo/07_框选修形_前.png / _后.png）。

走 Edge headless + ?uitest=1（测试模式不落盘），
所以截图过程不会改到 data/annotations 里的真实标注。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shoot_demo import find_browser, shoot_scripted, DEMO_DIR  # noqa: E402

IMG = "2.jpg"                          # 沙发被模型只圈了右半边的那张
BOX = [0.290, 0.550, 0.868, 0.752]     # 人拉的框：盖住整条沙发

BEFORE = """
(async () => {
  for (let i = 0; i < 80; i++) {
    if (window.__APP__ && window.__APP__.S && window.__APP__.S.images.length) break;
    await new Promise(r => setTimeout(r, 250));
  }
  const A = window.__APP__;
  await A.selectImage(IMG_NAME);
  await new Promise(r => setTimeout(r, 1200));
  A.S.focusMode = false;
  const couch = A.S.anno.objects.find(o => o.class === 'couch');
  window.__COUCH__ = couch || null;
  if (couch) A.S.selectedId = couch.id;
  document.querySelector('.tool-btn[data-tool="box"]').click();
  A.S.boxMode = 'add';
  A.syncBoxHint();
  A.renderObjects(); A.renderProps(); A.draw();
})();
"""

AFTER = """
(async () => {
  await new Promise(r => setTimeout(r, 1600));
  const A = window.__APP__;
  if (!window.__COUCH__) return;
  A.S.selectedId = window.__COUCH__.id;
  A.S.boxDrag = { x0: BOX[0], y0: BOX[1], x1: BOX[2], y1: BOX[3] };
  A.draw();
  await new Promise(r => setTimeout(r, 400));
  A.applyBoxShape();          // 等价于松手那一刻
  // 截图用：把选中态再贴回去，右侧面板才会显示顶点数（测试模式不落盘）
  A.S.selectedId = window.__COUCH__.id;
  A.S.focusMode = false;
  A.renderObjects(); A.renderProps(); A.draw();
})();
"""


def main():
    browser = find_browser()
    if not browser:
        print("找不到 Edge / Chrome")
        return
    os.makedirs(DEMO_DIR, exist_ok=True)

    head = "const IMG_NAME = %r;\nconst BOX = %r;\n" % (IMG, BOX)
    shoot_scripted(browser, os.path.join(DEMO_DIR, "07_框选修形_前.png"),
                   head + BEFORE, wait=12000)
    shoot_scripted(browser, os.path.join(DEMO_DIR, "07_框选修形_后.png"),
                   head + BEFORE + AFTER, wait=16000)
    # 再拍一张「正在拖框」的中间态：虚线框 + 并入模式
    mid = BEFORE.replace("A.renderObjects(); A.renderProps(); A.draw();",
                         "A.S.boxDrag = { x0: BOX[0], y0: BOX[1], x1: BOX[2], y1: BOX[3] };\n  A.renderObjects(); A.renderProps(); A.draw();")
    shoot_scripted(browser, os.path.join(DEMO_DIR, "07_框选修形_拖框中.png"),
                   head + mid, wait=13000)


if __name__ == "__main__":
    main()
