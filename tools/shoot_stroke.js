/* 描边效果专项对比截图。
   目的：把「选中」和「未选中」两种状态的描边放大截出来，
   肉眼直接对比哪里不对劲，而不是靠猜。

   用法：node tools/shoot_stroke.js
*/
'use strict';

const { execFileSync } = require('child_process');
const fs = require('fs');
const path = require('path');

const EDGE = 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe';
const OUT = path.join(__dirname, '..', 'demo');
const CANVAS = path.join(__dirname, '..', 'canvas');
if (!fs.existsSync(OUT)) fs.mkdirSync(OUT, { recursive: true });

// 生成一个「同源」页面：直接把主页面 iframe 放大，
// 并在里面注入脚本设置选中状态、放大画布。
const script = `
(async () => {
  for (let i = 0; i < 60; i++) {
    if (window.__APP__ && window.__APP__.S && window.__APP__.S.images.length) break;
    await new Promise(r => setTimeout(r, 250));
  }
  const A = window.__APP__;
  if (!A) return;
  // 选一张对象多、目标大的图
  const rich = A.S.images.filter(i => i.total > 8).sort((a,b)=>b.total-a.total)[0];
  if (rich) { await A.selectImage(rich.name); await new Promise(r=>setTimeout(r,1200)); }
  // 选面积最大的对象
  let best = null, ba = 0;
  for (const o of A.S.anno.objects) {
    let t = 0;
    const p = o.polygon;
    for (let i = 0; i < p.length; i++) {
      const [x1,y1] = p[i], [x2,y2] = p[(i+1)%p.length];
      t += x1*y2 - x2*y1;
    }
    const a = Math.abs(t/2);
    if (a > ba) { ba = a; best = o; }
  }
  A.S.selectedId = best.id;
  A.S.focusMode = false;
  A.S.showLabel = false;
  A.renderObjects();
  A.draw();
  document.title = 'READY';
})();
`;

// 注入到主页面
fs.writeFileSync(path.join(CANVAS, 'stroke_script.js'), script, 'utf8');

// 页面 URL 参数
const url = 'http://127.0.0.1:5000/?uitest=1&img=000109.jpg';

function shoot(out, wait, size) {
  const profile = path.join(process.env.TEMP || '.', 'edge_stroke_' + Math.random().toString(36).slice(2, 8));
  try {
    execFileSync(EDGE, [
      '--headless=new', '--disable-gpu', '--hide-scrollbars',
      '--window-size=' + size,
      '--virtual-time-budget=' + wait,
      '--user-data-dir=' + profile,
      '--screenshot=' + out,
      url,
    ], { stdio: 'pipe', timeout: 90000 });
    const ok = fs.existsSync(out) && fs.statSync(out).size > 5000;
    console.log((ok ? '  OK   ' : '  FAIL ') + path.basename(out));
    return ok;
  } catch (e) {
    console.log('  FAIL ' + path.basename(out) + '  ' + e.message);
    return false;
  }
}

console.log('描边对比截图:');
shoot(path.join(OUT, 'stroke_选中_轮廓描边.png'), 12000, '1600,950');

fs.unlinkSync(path.join(CANVAS, 'stroke_script.js'));
