/* 用 Edge 无头模式实际驱动页面，验证交互行为。

   手段：生成一个同源 harness 页，里面套 iframe 加载真实界面，
        由 canvas/ui_test_script.js 在 iframe 里操作真实 DOM，
        结果既画在页面上（截图能看见），也 POST 回 Node 这边打印。

   用法：node tools/ui_test.js

   【为什么报告不用 --dump-dom 取】
   那是原本的做法，但新版 Edge（headless=new）的 --dump-dom 已经不出内容了：
   命令正常退出、stdout 恒为 0 字节。结果测试全过了却打印「未取到报告」，
   看起来像挂了——这种"工具坏了伪装成被测对象坏了"的假阴性最难查。
   改成页面主动 POST 回来（fetch / sendBeacon），和浏览器的 DOM 导出解耦。
*/
'use strict';

const { execFile } = require('child_process');
const fs = require('fs');
const path = require('path');
const http = require('http');

const EDGE = 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe';
const OUT = path.join(__dirname, '..', 'demo');
if (!fs.existsSync(OUT)) fs.mkdirSync(OUT, { recursive: true });

const HARNESS = path.join(__dirname, '..', 'canvas', '_ui_harness.html');
const APP = 'http://127.0.0.1:5000';
const shot = path.join(OUT, 'ui_test.png');

// ---------------------------------------------------------------- 报告回传
// 端口用 0 让系统分配，避免 8899 被别的进程占着时整个测试跑不起来。
let REPORT = '';
let BEACON_HITS = 0;
let srv = null;

function startBeacon() {
  return new Promise((resolve, reject) => {
    srv = http.createServer((req, res) => {
      res.setHeader('Access-Control-Allow-Origin', '*');
      res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
      res.setHeader('Access-Control-Allow-Headers', '*');
      if (req.method === 'OPTIONS') { res.writeHead(204); res.end(); return; }
      if (req.method !== 'POST') {
        res.writeHead(200, { 'Content-Type': 'text/plain' });
        res.end('ui-test beacon');
        return;
      }
      let body = '';
      req.on('data', c => { body += c; });
      req.on('end', () => {
        REPORT = body;
        BEACON_HITS += 1;
        res.writeHead(204);
        res.end();
      });
    });
    srv.on('error', reject);
    srv.listen(0, '127.0.0.1', () => resolve(srv.address().port));
  });
}

function waitReport(ms) {
  const t0 = Date.now();
  return new Promise(resolve => {
    const tick = () => {
      if (REPORT) return resolve(REPORT);
      if (Date.now() - t0 > ms) return resolve('');
      setTimeout(tick, 150);
    };
    tick();
  });
}

// ---------------------------------------------------------------- harness
function writeHarness(port) {
  const harness = `<!DOCTYPE html><html><head><meta charset="utf-8">
<style>html,body{margin:0;height:100%}iframe{border:0;width:100%;height:100%}
#report{position:fixed;right:0;bottom:0;width:460px;max-height:70vh;overflow:auto;
background:#fff;font:11px monospace;padding:6px;z-index:9;border:1px solid #000;white-space:pre}</style>
</head><body>
<iframe id="fr" src="/?uitest=1&t=${Date.now()}"></iframe>
<pre id="report">running...</pre>
<script>
const BEACON = 'http://127.0.0.1:${port}/report';
function send(text) {
  try { navigator.sendBeacon(BEACON, text); return; } catch (e) {}
  try { fetch(BEACON, { method: 'POST', mode: 'no-cors', body: text }); } catch (e) {}
}
const fr = document.getElementById('fr');
fr.onload = async () => {
  const w = fr.contentWindow, d = fr.contentDocument;
  const report = document.getElementById('report');
  for (let i = 0; i < 40; i++) {
    if (w.__APP__ && w.__APP__.S && w.__APP__.S.images.length) break;
    await new Promise(r => setTimeout(r, 250));
  }
  let out;
  try {
    out = await w.__runUiTests(d, w);
  } catch (e) {
    const done = (w.__UITEST_LOG__ || []).join('\\n');
    out = (done ? done + '\\n\\n' : '') + 'FATAL: ' + (e && e.stack || e);
  }
  report.textContent = out;
  send(out);
  window.__DONE__ = true;
};
</script>
</body></html>`;
  fs.writeFileSync(HARNESS, harness, 'utf8');
}

// ---------------------------------------------------------------- 前置检查
// 页面加载的测试脚本必须是 canvas/ 下这一份。
// 【踩过的坑】早期把测试脚本同时放在 tools/ 和 canvas/ 两处，
// 实际被 Flask 提供的是 canvas/ 那份，改的却是 tools/ 那份——
// 「代码改了半天，测试跑出来还是旧的 23 条」，且旧结果全绿，极难察觉。
function assertServerHasLatestScript() {
  const LOCAL = path.join(__dirname, '..', 'canvas', 'ui_test_script.js');
  const local = fs.readFileSync(LOCAL, 'utf8');
  return new Promise((resolve, reject) => {
    http.get({ host: '127.0.0.1', port: 5000,
               path: '/static/ui_test_script.js?nocache=' + Date.now() }, res => {
      let b = '';
      res.on('data', c => b += c);
      res.on('end', () => {
        if (b.trim() !== local.trim()) {
          return reject(new Error(
            '服务端返回的 ui_test_script.js 与本地文件不一致。'
            + '常见原因：另有一份同名文件在别处，或 Flask 未重启。'));
        }
        resolve();
      });
    }).on('error', () =>
      reject(new Error('无法连接 http://127.0.0.1:5000，请先启动 app.py')));
  });
}

// ---------------------------------------------------------------- 跑浏览器
function runBrowser(profile) {
  // 【为什么 profile 每次都换新目录】
  // ui_test_script.js 会被浏览器缓存。若复用同一个 user-data-dir，
  // 改了测试脚本再跑，拿到的仍是旧脚本的结果，看起来"测试通过"
  // 其实新用例根本没执行——极具欺骗性。
  const NO_CACHE = [
    '--disable-application-cache',
    '--disk-cache-size=1',
    '--media-cache-size=1',
    '--incognito',
  ];
  return new Promise((resolve, reject) => {
    execFile(EDGE, [
      '--headless=new',
      '--disable-gpu',
      '--no-sandbox',
      '--window-size=1600,1000',
      '--virtual-time-budget=25000',
      '--user-data-dir=' + profile,
      ...NO_CACHE,
      '--screenshot=' + shot,
      APP + '/static/_ui_harness.html',
    ], { timeout: 120000 }, err =>
      err ? reject(err) : resolve());
  });
}

// ---------------------------------------------------------------- 主流程
(async () => {
  let port;
  try {
    port = await startBeacon();
  } catch (e) {
    console.error('✗ 无法启动报告接收端口:', e.message);
    process.exit(1);
  }

  writeHarness(port);

  try {
    await assertServerHasLatestScript();
  } catch (e) {
    console.error('✗ ' + e.message);
    if (fs.existsSync(HARNESS)) fs.unlinkSync(HARNESS);
    process.exit(2);
  }

  const profile = path.join(process.env.TEMP || '.', 'edge_ui_test_' + Date.now().toString(36));
  try {
    await runBrowser(profile);
    console.log('✓ 截图已生成:', shot);
  } catch (e) {
    console.error('✗ 无头浏览器失败:', e.message);
  }

  const report = await waitReport(10000);
  console.log('\n===== UI 测试报告 =====');
  if (report) {
    console.log(report.trim());
    fs.writeFileSync(path.join(OUT, 'ui_test_report.txt'), report.trim(), 'utf8');
    if (/FATAL|FAIL/.test(report)) process.exitCode = 1;
  } else {
    // 报告没回来不等于测试没跑——截图里可能已经有了。
    console.log('(报告未回传，共收到 ' + BEACON_HITS + ' 次)');
    console.log('看截图：' + shot);
    process.exitCode = 3;
  }

  if (srv) srv.close();
  if (fs.existsSync(HARNESS)) fs.unlinkSync(HARNESS);
})();
