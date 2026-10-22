/* 拍一张撤销/重做按钮的「可用态」截图。
   ui_test.png 里测试跑完后栈被清空了，按钮是灰的，看不出启用长什么样。
   这里造几步操作再拍，让用户直观看到按钮亮起来 + title 里的剩余步数。

   用法：先启动 app.py，再 node tools/shoot_undo.js
*/
'use strict';
const { execFileSync } = require('child_process');
const fs = require('fs');
const path = require('path');

const EDGE = 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe';
const OUT = path.join(__dirname, '..', 'demo');
const HARNESS = path.join(__dirname, '..', 'canvas', '_undoharness.html');
const URL = 'http://127.0.0.1:5000/static/_undoharness.html';
const stamp = Date.now().toString(36);
const PROFILE = path.join(process.env.TEMP || '.', 'edge_undo_' + stamp);
if (!fs.existsSync(OUT)) fs.mkdirSync(OUT, { recursive: true });

// 页面内脚本：造几步撤销 + 1 步重做，把按钮点亮
const PAGE_SCRIPT = `
<script>
const st = document.getElementById('state');
function setState(s) { st.textContent = s; }
// 【为什么要 try/catch + 反复重试】
// iframe 的 onload 只表示文档加载完，但主页面 init() 是async 的，
// 还要拉图片列表、初始化画布。onload 触发时 __APP__.S.images 可能还是空的。
// 之前没包try/catch，一旦中途抛错 state 永远停在 pending，排查起来毫无线索。
(async function run() {
  const fr = document.getElementById('fr');
  await new Promise(res => {
    if (fr.contentDocument && fr.contentDocument.readyState === 'complete') res();
    else fr.onload = res;
  });
  const w = fr.contentWindow;
  for (let i = 0; i < 80; i++) {
    try {
      if (w.__APP__ && w.__APP__.S && w.__APP__.S.images && w.__APP__.S.images.length) break;
    } catch (e) { /* 跨域或尚未就绪 */ }
    await new Promise(r => setTimeout(r, 250));
  }
  if (!w.__APP__ || !w.__APP__.S || !w.__APP__.S.images || !w.__APP__.S.images.length) {
    setState('FATAL:__APP__ 未就绪'); return;
  }
  const A = w.__APP__, S = A.S;
  // 必须切到一张真有标注对象的图，否则画布是空的，
  // 造出来的对象挂在null anno 上，addObject 直接报错。
  // 【两个坑】selectImage 要传 .name（不是对象），而且是 async 必须 await。
  if (!S.anno || !S.anno.objects || !S.anno.objects.length) {
    const withObj = S.images.find(im => im.total > 0);
    if (withObj) {
      try { await A.selectImage(withObj.name); }
      catch (e) { /* 加载失败就退出了 */ }
      await new Promise(r => setTimeout(r, 1200));
    }
  }
  if (!S.anno || !S.anno.objects.length) {
    setState('FATAL:noanno current=' + S.current); return;
  }
  // 造几步：新增两个、改一次类别
  A.addObject('cup',  [[0.55,0.45],[0.65,0.45],[0.65,0.55],[0.55,0.55]], 'human');
  A.addObject('book', [[0.70,0.60],[0.85,0.60],[0.85,0.72],[0.70,0.72]], 'human');
  A.changeClass('tv');
  // 撤销一步 -> 留下「可撤销 2 步 / 可重做 1 步」
  A.undo();
  A.draw();
  // 把按钮的真实状态写出来，便于排查「按钮没显示」这类问题
  const ub = w.document.getElementById('btnUndo');
  const rb = w.document.getElementById('btnRedo');
  setState('ready|undo='
    + (ub ? ub.offsetWidth + 'x' + ub.offsetHeight + ' disabled=' + ub.disabled
              + ' title=' + ub.title : 'NOT FOUND')
    + ' | redo=' + (rb ? rb.disabled + ' title=' + rb.title : 'NOT FOUND')
    + ' | objects=' + S.anno.objects.length);
})().catch(e => setState('FATAL:' + (e && e.stack || e)));
</script>`;

fs.writeFileSync(HARNESS, `<!DOCTYPE html><html><head><meta charset="utf-8">
<style>html,body{margin:0;height:100%;background:#fff}
iframe{border:0;width:100%;height:100%}</style></head><body>
<iframe id="fr" src="/?uitest=1&t=${Date.now()}"></iframe>
<span id="state" style="display:none">pending</span>
${PAGE_SCRIPT}
</body></html>`, 'utf8');

const shot = path.join(OUT, 'undo_buttons.png');
execFileSync(EDGE, [
  '--headless=new', '--disable-gpu', '--no-sandbox',
  '--window-size=1760,980',
  '--virtual-time-budget=30000',
  '--user-data-dir=' + PROFILE,
  '--screenshot=' + shot,
  URL,
], { stdio: 'pipe', timeout: 90000 });

const kb = fs.statSync(shot).size;
console.log('✓ 截图已生成:', shot, '(' + (kb / 1024).toFixed(0) + ' KB)');
if (kb < 50 * 1024) {
  console.error('✗ 截图过小，可能截到了空白页');
  process.exit(1);
}