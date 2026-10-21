/* 多边形工具的可用性回归测试。

   为什么专门测这个
   --------------
   用户实测反馈：「我怎么框不了？」
   排查发现根因不是功能缺失，而是**零反馈**：
   多边形只能靠双击或 Enter 闭合，但界面上没有任何提示，
   新手点了一串点发现没反应，以为工具坏了。

   这里锁住三件事：
     1. 选中多边形工具后，提示条必须出现
     2. 描边过程中，画布指引层必须显示并随点数更新
     3. 点数不足 3 个就点「完成描边」，必须报错而不是静默失败
   防止以后改代码时又把这个提示删掉。
*/
(async function () {
  const A = window.__APP__;
  const $ = (id) => document.getElementById(id);
  const results = [];
  const ok = (name, cond, extra) =>
    results.push({ ok: !!cond, name: name, extra: extra || '' });
  const sleep = (ms) => new Promise(r => setTimeout(r, ms));

  if (!A) return;

  // 【必须等 init() 的异步绑定完成】
  // app.js 里init() 是 async 函数，按钮 onclick（btnFinishPoly 等）挂在
  // 它靠后的位置。而 window.__APP__ 是在文件末尾同步赋值的，
  // 于是「__APP__ 已就绪」比「按钮已绑定」早得多。
  // 之前只等 300ms 就去点按钮，onclick 还是 null，点击静默无效——
  // 外部看到的就是 tool 一直停在 select、整份测试结果为 null。
  // 这里轮询等 onclick 真正挂上，最长 20 秒。
  for (let i = 0; i < 80; i++) {
    const b = $('btnFinishPoly');
    if (b && typeof b.onclick === 'function') break;
    await sleep(250);
  }

  // --- 1. 选中多边形工具后提示条出现 ---
  const btn = document.querySelector('.tool-btn[data-tool="polygon"]');
  btn.click();
  const bar = $('polyHint');
  ok('选中多边形后提示条可见', bar && !bar.hidden);

  // --- 2. 按钮有 title 悬浮说明（鼠标悬停能看到）---
  ok('多边形按钮有 title 说明',
    btn.getAttribute('title') && btn.getAttribute('title').length > 10,
    btn.getAttribute('title') || '');
  ok('「完成描边」按钮存在', !!$('btnFinishPoly'));

  // --- 3. 点数不足时点完成，必须 toast 报错而不是静默 ---
  // 模拟点了 2 个点
  A.S.drawing = true;
  A.S.draftPts = [[0.1, 0.1], [0.2, 0.2]];
  $('btnFinishPoly').click();
  const t = $('toast');
  ok('点数不足时给出明确报错',
    t && !t.classList.contains('hidden') && (t.textContent || '').indexOf('3') >= 0,
    (t && t.textContent) || '');
  ok('报错后自动清空草稿', A.S.drawing === false && A.S.draftPts.length === 0);

  // --- 4. 描边过程中指引层显示，且随点数更新 ---
  A.S.tool = 'polygon';
  A.S.drawing = true;
  A.S.draftPts = [[0.1, 0.1]];
  // 直接调draw 不会触发 syncPolyHint，这里模拟真实点击路径
  const c = document.getElementById('canvas');
  function clickAt(nx, ny) {
    const r = c.getBoundingClientRect();
    const ev = new MouseEvent('mousedown', {
      bubbles: true, clientX: r.left + r.width * nx,
      clientY: r.top + r.height * ny, button: 0,
    });
    c.dispatchEvent(ev);
  }
  // 走真实路径：重新激活工具并点击 3 次
  btn.click();
  A.S.drawing = false; A.S.draftPts = [];
  clickAt(0.2, 0.2);
  ok('点击后进入描边状态', A.S.drawing === true);
  ok('点击后记录了顶点', A.S.draftPts.length === 1);
  ok('描边中提示层可见', !$('drawingTip').classList.contains('hidden'));
  ok('提示层显示当前点数',
    ($('drawingTip').textContent || '').indexOf('1') >= 0,
    $('drawingTip').textContent || '');

  clickAt(0.4, 0.2);
  clickAt(0.4, 0.4);
  ok('顶点累积到 3 个', A.S.draftPts.length === 3, String(A.S.draftPts.length));
  ok('点数>=3 时提示改为闭合指引',
    ($('drawingTip').textContent || '').indexOf('双击') >= 0,
    $('drawingTip').textContent || '');

  // --- 5. 点「完成描边」应该弹出类别选择框 ---
  $('btnFinishPoly').click();
  const picker = document.querySelector('#clsPick');
  ok('点数足够时弹出类别选择', !!picker);
  ok('类别数与任务书十类一致',
    picker && picker.querySelectorAll('button').length === 10,
    picker ? String(picker.querySelectorAll('button').length) : '');
  // 取消掉，不要真落盘
  const cancel = document.getElementById('btnCancelPoly');
  if (cancel) cancel.click();

  // --- 6. 放弃按钮能清掉草稿 ---
  btn.click();
  A.S.drawing = true; A.S.draftPts = [[0.1, 0.1]];
  $('btnCancelPoly2').click();
  ok('「放弃」清掉描边草稿',
    A.S.drawing === false && A.S.draftPts.length === 0);
  ok('放弃后提示层隐藏',
    $('drawingTip').classList.contains('hidden'));

  // --- 7. 空对象图的对象列表说明文案 ---
  // 同样不切图（切图在 headless 下不可靠）。
  // 直接把当前标注的对象数组临时清空再渲染，看是否输出说明文案，验完恢复。
  const anno = A.S.anno;
  if (anno) {
    const saved = anno.objects;      // 持有原数组引用，不是拷贝
    anno.objects = [];
    A.renderObjects();
    const box = $('objList');
    const txt = (box && box.textContent) || '';
    ok('空对象图给出原因说明', txt.indexOf('十类') >= 0, txt.slice(0, 30));
    anno.objects = saved;            // 直接还回原引用
    A.renderObjects();
    ok('恢复后对象列表回到原状', A.S.anno.objects.length === saved.length,
       String(A.S.anno.objects.length));
  }

  window.__TEST_RESULTS__ = results;
})().catch(function (e) {
  // 【关键】无论如何都要把结果写出去。
  // 之前没有 catch，脚本里任何一处抛异常都会导致
  // window.__TEST_RESULTS__ 永远不被赋值，外部只能看到
  // 「未取到结果」，连错在哪都看不到。
  window.__TEST_RESULTS__ = [{
    ok: false,
    name: '脚本执行异常中断：' + (e && e.message ? e.message : String(e)),
    extra: '',
  }];
});