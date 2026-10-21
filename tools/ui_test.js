/* 用 Edge 无头模式实际驱动页面，验证三项交互改动。
   手段：注入一段测试脚本到页面里，模拟真实鼠标操作，把结果写进 DOM，
        然后截图 + 读取页面上的测试报告。

   用法：node tools/ui_test.js
*/
'use strict';

const { execFileSync } = require('child_process');
const fs = require('fs');
const path = require('path');
const http = require('http');

const EDGE = 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe';
const OUT = path.join(__dirname, '..', 'demo');
if (!fs.existsSync(OUT)) fs.mkdirSync(OUT, { recursive: true });

// 1) 生成同源测试页。
//    必须和应用同源（放在 canvas/ 目录下由 Flask 提供），
//    否则 iframe 会被跨域策略拦住，父页拿不到 iframe 的 window。
const HARNESS = path.join(__dirname, '..', 'canvas', '_ui_harness.html');
const harness = `<!DOCTYPE html><html><head><meta charset="utf-8">
<style>html,body{margin:0;height:100%}iframe{border:0;width:100%;height:100%}
#report{position:fixed;right:0;bottom:0;width:460px;max-height:70vh;overflow:auto;
background:#fff;font:11px monospace;padding:6px;z-index:9;border:1px solid #000;white-space:pre}</style>
</head><body>
<iframe id="fr" src="/?uitest=1"></iframe>
<pre id="report">running...</pre>
<script>
const fr = document.getElementById('fr');
fr.onload = async () => {
  const w = fr.contentWindow, d = fr.contentDocument;
  // 等主页面初始化完
  for (let i = 0; i < 40; i++) {
    if (w.__APP__ && w.__APP__.S && w.__APP__.S.images.length) break;
    await new Promise(r => setTimeout(r, 250));
  }
  try {
    const out = await w.__runUiTests(d, w);
    document.getElementById('report').textContent = out;
  } catch (e) {
    document.getElementById('report').textContent = 'FATAL: ' + (e && e.stack || e);
  }
};
</script>
</body></html>`;
fs.writeFileSync(HARNESS, harness, 'utf8');

const URL = 'http://127.0.0.1:5000/static/_ui_harness.html';

// 2) 无头浏览器跑一遍
const shot = path.join(OUT, 'ui_test.png');
const profile = path.join(process.env.TEMP || '.', 'edge_ui_test');
try {
  execFileSync(EDGE, [
    '--headless=new',
    '--disable-gpu',
    '--no-sandbox',
    '--window-size=1600,1000',
    '--virtual-time-budget=20000',
    '--user-data-dir=' + profile,
    '--screenshot=' + shot,
    URL,
  ], { stdio: 'pipe', timeout: 90000 });
  console.log('✓ 截图已生成:', shot);
} catch (e) {
  console.error('✗ 无头浏览器失败:', e.message);
  process.exit(1);
}

// 3) 报告：再跑一次拿文本（headless 截图不返回 DOM，
//    用 --dump-dom 提取页面上的报告块）
try {
  const dom = execFileSync(EDGE, [
    '--headless=new',
    '--disable-gpu',
    '--no-sandbox',
    '--virtual-time-budget=20000',
    '--user-data-dir=' + profile + '2',
    '--dump-dom',
    URL,
  ], { encoding: 'utf8', timeout: 90000, maxBuffer: 1 << 24 });

  const m = dom.match(/<pre id="report">([\s\S]*?)<\/pre>/);
  const report = m ? m[1]
    .replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&amp;/g, '&').replace(/&quot;/g, '"')
    : '(未取到报告)';
  console.log('\n===== UI 测试报告 =====');
  console.log(report.trim());
  fs.writeFileSync(path.join(OUT, 'ui_test_report.txt'), report.trim(), 'utf8');
  if (/FATAL|FAIL/.test(report)) process.exitCode = 1;
} catch (e) {
  console.error('✗ 提取报告失败:', e.message);
} finally {
  // 清理临时 harness
  if (fs.existsSync(HARNESS)) fs.unlinkSync(HARNESS);
}
