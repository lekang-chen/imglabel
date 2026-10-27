/* ==========================================================================
   e2e_upload.js —— 「上传照片 → 自动预标注」端到端验证
   --------------------------------------------------------------------------
   为什么单独一个工具，而不是并进 ui_test.js：
   这套断言必须真的把一张图 POST 到 /api/upload，也就是会往 data/images 里
   写文件。ui_test.js 的原则是「测试不能污染数据集」，所以它只做 DOM 级检查。
   这里单独跑，跑完自动删掉自己造的图片和标注，两边互不干扰。

   验证的链路（就是页面上那一下点击做的事）：
     选文件 → 逐张上传 → 刷新图片列表 → 自动触发预标注
            → 预标注只带新上传的文件名 → 完成后自动打开新图

   其中「预标注」这一步用桩替换掉了：塞进去一个立刻返回 done 的假任务，
   这样不用等模型推理，几秒钟就能跑完，也不会覆盖任何一个真实标注。

   用法：先启动 app.py，然后
         node tools/e2e_upload.js
   ========================================================================== */
'use strict';

const { execFileSync } = require('child_process');
const fs = require('fs');
const path = require('path');

const ROOT = path.join(__dirname, '..');
const EDGE = 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe';
const BASE = process.env.IMGLABEL_BASE || 'http://127.0.0.1:5000';
const HARNESS = path.join(ROOT, 'canvas', '_e2e_harness.html');
const SRC = path.join(ROOT, 'canvas', '_e2e_src.jpg');
const TEST_NAME = '端到端测试';

const cleanup = () => {
  for (const p of [HARNESS, SRC]) {
    try { fs.unlinkSync(p); } catch (e) { /* 不存在就算了 */ }
  }
  const imgDir = path.join(ROOT, 'data', 'images');
  const annoDir = path.join(ROOT, 'data', 'annotations');
  for (const dir of [imgDir, annoDir]) {
    if (!fs.existsSync(dir)) continue;
    for (const f of fs.readdirSync(dir)) {
      if (!f.startsWith(TEST_NAME)) continue;
      try { fs.unlinkSync(path.join(dir, f)); } catch (e) { /* ignore */ }
    }
  }
  // 上传接口会往 data/state.json 追加一条「上传图片」运行日志。
  // 那是给答辩展示"人真的操作过"用的，不该混进自动化测试的记录，
  // 所以整个文件原样还原。
  if (STATE_BACKUP !== null) {
    try { fs.writeFileSync(STATE_FILE, STATE_BACKUP); } catch (e) { /* ignore */ }
  }
};

const STATE_FILE = path.join(ROOT, 'data', 'state.json');
const STATE_BACKUP = fs.existsSync(STATE_FILE)
  ? fs.readFileSync(STATE_FILE) : null;

process.on('exit', cleanup);
process.on('SIGINT', () => process.exit(130));

// ---------------------------------------------------------------- 准备素材
function pickSourceImage() {
  const dir = path.join(ROOT, 'data', 'images');
  const list = fs.readdirSync(dir).filter(f => /\.jpe?g$/i.test(f)).sort();
  if (!list.length) {
    console.error('✗ data/images 里没有 jpg 可以做测试素材');
    process.exit(1);
  }
  return path.join(dir, list[0]);
}

const srcImage = pickSourceImage();
fs.copyFileSync(srcImage, SRC);

// ---------------------------------------------------------------- 生成测试页
// 必须和应用同源：iframe 里要读子页的 window.__APP__，跨域会被拦
const harness = `<!DOCTYPE html><html><head><meta charset="utf-8">
<style>html,body{margin:0;height:100%}iframe{border:0;width:100%;height:100%}
#report{position:fixed;right:0;bottom:0;width:560px;max-height:70vh;overflow:auto;
background:#fff;font:11px monospace;padding:6px;z-index:9;border:1px solid #000;white-space:pre}</style>
</head><body>
<iframe id="fr" src="/?t=${Date.now()}"></iframe>
<pre id="report">running...</pre>
<script>
const fr = document.getElementById('fr');
const rep = document.getElementById('report');
const lines = [];
function say(s) { lines.push(s); rep.textContent = lines.join('\\n'); }
function ok(name, cond, extra) {
  say((cond ? 'PASS  ' : 'FAIL  ') + name + (extra ? '  (' + extra + ')' : ''));
}
fr.onload = async () => {
  const w = fr.contentWindow;
  const d = fr.contentDocument;
  const wait = ms => new Promise(r => setTimeout(r, ms));
  try {
    for (let i = 0; i < 80 && !w.__APP__; i++) await wait(100);
    if (!w.__APP__) { say('FATAL 没有 __APP__'); return; }
    const A = w.__APP__;

    // 无头浏览器里原生 confirm 会把虚拟时钟卡住，直接放行
    w.confirm = () => true;

    // 把预标注换成桩：只检查前端发了什么，不让真模型跑
    const realFetch = w.fetch;
    let prelabelBody = null;
    w.fetch = function (url, opt) {
      if (String(url).indexOf('/api/prelabel/progress') >= 0) {
        return Promise.resolve(new w.Response(
          JSON.stringify({ status: 'done', done: 1, total: 1, percent: 100,
                           current: '', message: '' }),
          { status: 200, headers: { 'Content-Type': 'application/json' } }));
      }
      if (String(url).indexOf('/api/prelabel/start') >= 0) {
        prelabelBody = JSON.parse(opt.body);
        return Promise.resolve(new w.Response(
          JSON.stringify({ ok: true, job_id: 'stub', total: 1 }),
          { status: 200, headers: { 'Content-Type': 'application/json' } }));
      }
      return realFetch.apply(this, arguments);
    };

    // 等首页把图片列表拉回来，否则 before 会读到 0
    for (let i = 0; i < 100 && A.S.images.length === 0; i++) await wait(100);

    const blob = await (await realFetch.call(w, '/static/_e2e_src.jpg')).blob();
    const file = new w.File([blob], '${TEST_NAME}.jpg', { type: 'image/jpeg' });

    d.getElementById('autoPrelabel').checked = true;
    const before = A.S.images.length;
    await A.uploadFiles([file]);
    await wait(300);

    ok('上传后图片列表 +1', A.S.images.length === before + 1,
       before + ' -> ' + A.S.images.length);
    ok('新图出现在列表里', A.S.images.some(i => i.name.indexOf('${TEST_NAME}') === 0));
    ok('自动触发了预标注', !!prelabelBody);
    if (prelabelBody) {
      ok('预标注只带新上传的那一张',
         Array.isArray(prelabelBody.images) && prelabelBody.images.length === 1
         && prelabelBody.images[0].indexOf('${TEST_NAME}') === 0,
         JSON.stringify(prelabelBody));
      ok('首轮用官方权重', prelabelBody.model === 'official');
    }
    for (let i = 0; i < 50 && !A.S.current; i++) await wait(100);
    ok('完成后自动打开新图',
       !!A.S.current && A.S.current.indexOf('${TEST_NAME}') === 0,
       '当前=' + A.S.current);
    ok('页面统计已刷新', d.getElementById('imgCount').textContent === String(before + 1),
       '计数=' + d.getElementById('imgCount').textContent);
    say('DONE');
  } catch (e) { say('EXCEPTION ' + (e && e.stack || e)); }
};
</script></body></html>`;
fs.writeFileSync(HARNESS, harness, 'utf8');

// ---------------------------------------------------------------- 跑
const profile = path.join(process.env.TEMP || '.',
                          'edge_e2e_' + Date.now().toString(36));
let dom = '';
try {
  dom = execFileSync(EDGE, [
    '--headless=new',
    '--disable-gpu',
    '--no-sandbox',
    '--window-size=1400,900',
    '--virtual-time-budget=30000',
    '--user-data-dir=' + profile,
    '--disable-application-cache',
    '--disk-cache-size=1',
    '--media-cache-size=1',
    '--incognito',
    '--dump-dom',
    BASE + '/static/_e2e_harness.html',
  ], { stdio: ['ignore', 'pipe', 'pipe'], timeout: 90000, encoding: 'utf8' });
} catch (e) {
  console.error('✗ 无头浏览器执行失败:', e.message);
  if (e.stdout) dom = String(e.stdout);
}

const m = dom.match(/<pre id="report">([\s\S]*?)<\/pre>/);
if (!m) {
  console.error('✗ 没拿到测试报告。请确认 app.py 已在 ' + BASE + ' 启动。');
  process.exit(1);
}

const report = m[1]
  .replace(/&gt;/g, '>').replace(/&lt;/g, '<').replace(/&amp;/g, '&');
console.log(report);

const pass = (report.match(/^PASS/gm) || []).length;
const fail = (report.match(/^FAIL/gm) || []).length;
console.log('');
console.log('结果：' + pass + ' 通过 / ' + fail + ' 失败');
if (!report.includes('DONE')) {
  console.error('✗ 脚本没跑完（见上面的 EXCEPTION）');
  process.exit(2);
}
process.exit(fail ? 1 : 0);
