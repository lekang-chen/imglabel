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
<iframe id="fr" src="/?uitest=1&t=${Date.now()}"></iframe>
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

// 1.5) 防呆：确认页面加载的测试脚本就是canvas/ 下这一份。
// 【踩过的坑】早期把测试脚本同时放在 tools/ 和 canvas/ 两处，
// 实际被 Flask 提供的是 canvas/ 那份，而改的是 tools/ 那份——
// 于是「代码改了半天，测试跑出来还是旧的23 条」，
// 且旧结果全绿，看起来一切正常，极难察觉。
// 这里直接比对服务端返回的内容与本地文件，不一致就立刻报错。
(function assertServerHasLatestScript() {
  const LOCAL = path.join(__dirname, '..', 'canvas', 'ui_test_script.js');
  const local = fs.readFileSync(LOCAL, 'utf8');
  const http = require('http');
  http.get({ host: '127.0.0.1', port: 5000, path: '/static/ui_test_script.js?nocache=' + Date.now() },
    res => {
      let b = '';
      res.on('data', c => b += c);
      res.on('end', () => {
        if (b.trim() !== local.trim()) {
          console.error('✗ 服务端返回的 ui_test_script.js 与本地文件不一致。');
          console.error('  常见原因：另有一份同名文件在别处，或 Flask 未重启。');
          process.exit(2);
        }
      });
    }).on('error', () => {
      console.error('✗ 无法连接http://127.0.0.1:5000，请先启动 app.py');
      process.exit(2);
    });
})();

// 2) 无头浏览器跑一遍
const shot = path.join(OUT, 'ui_test.png');
// 【为什么 profile 每次都换新目录】
// ui_test_script.js 会被浏览器缓存。若复用同一个 user-data-dir，
// 改了测试脚本再跑，拿到的仍是旧脚本的 23 条结果，
// 看起来"测试通过"其实新加的用例根本没执行——极具欺骗性。
// 所以每次跑用带时间戳的全新 profile，强制重新拉脚本。
const stamp = Date.now().toString(36);
const profile = path.join(process.env.TEMP || '.', 'edge_ui_test_' + stamp);
const NO_CACHE = [
  '--disable-application-cache',
  '--disk-cache-size=1',
  '--media-cache-size=1',
  '--incognito',
];
try {
  execFileSync(EDGE, [
    '--headless=new',
    '--disable-gpu',
    '--no-sandbox',
    '--window-size=1600,1000',
    '--virtual-time-budget=20000',
    '--user-data-dir=' + profile,
    ...NO_CACHE,
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
    '--user-data-dir=' + profile,
    ...NO_CACHE,
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
