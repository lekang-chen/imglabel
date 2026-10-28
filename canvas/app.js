/* ==========================================================================
   app.js —— 前端交互逻辑
   --------------------------------------------------------------------------
   【设计原则】三种模式共用同一份数据
   底层始终是「实例多边形 + 类别」，切换模式只是换渲染方式：
     seg 模式 -> 画多边形填充
     det 模式 -> 画多边形的外接矩形
     sem 模式 -> 同类别的多边形合并成一片色块（互斥填充）

   这就是后端 core.py 里「检测框 = 掩码外接矩形、语义 = 掩码按类合并」
   在前端的对应实现。改一次，三种模式同时变。
   ========================================================================== */

'use strict';

// ---------------------------------------------------------------- 状态
const S = {
  classes: [],            // [{name, id, color}]
  colorMap: {},           // 类别名 -> hex
  images: [],             // 图片列表
  lastTrash: null,        // 最近一次删除的批次 {stamp, count}，用于「恢复」提示条
  current: null,          // 当前图片名
  anno: null,             // 当前标注 {image,width,height,objects:[]}
  mode: 'seg',            // seg | det | sem
  tool: 'select',         // select | polygon | brush | erase | box
  boxDrag: null,          // 框选修形的橡皮筋 {x0,y0,x1,y1}（归一化），null = 没在拉
  boxMode: 'add',         // add 并入 | cut 挖掉 | replace 换成这个框
  selectedId: null,       // 选中的对象 id
  scale: 1,               // 缩放比例
  offsetX: 0, offsetY: 0, // 画布偏移
  showMask: true, showBox: false, showLabel: true,
  unchecked: new Set(),    // 复选框取消勾选的对象 id（勾选 = 显示）
  focusMode: true,         // 聚焦：画布只显示当前选中的对象
  hoverId: null,           // 鼠标悬停的对象 id
  brushSize: 18,
  brushMask: null,         // 画笔位图掩码 {id, cv, cx, w, h, k}
  scratch: null,           // 画笔着色暂存画布
  mouse: { x: 0, y: 0, in: false },
  pointerId: null,        // 正在按下的指针 id（触摸时用来忽略第二根手指）
  drawing: false,         // 多边形绘制中
  draftPts: [],           // 多边形草稿点
  painting: false,        // 画笔涂抹中
  paintTarget: null,      // 画笔作用的对象
  img: null,              // Image 对象
  dirty: false,           // 有未保存修改
  edits: [],              // 本图的修正记录
  origById: {},           // 初始状态快照，用于判断哪里被改了
  undoStack: [],          // 撤销栈：每次改标注前的标注快照
  redoStack: [],          // 重做栈
  undoLimit: 50,          // 栈上限，防止长时间标注吃掉内存
};

/**
 * 撤销 / 重做。
 *
 * 【为什么用「整图快照」而不是逐操作反向函数】
 * 标注操作有好几种（新增、删除、改类别、画笔改形状、多边形重描），
 * 每种写一个反向函数要处理「对象是否还存在」「顺序怎么恢复」等边界，
 * 很容易漏。而且新增对象时对象 id 是随机的，反向函数没法凭空造回来。
 * 直接快照整份 anno（深拷贝）最稳：撤销就是把整份状态换回去。
 *
 * 一张图的标注通常几十个对象、多边形顶点上百，JSON 快照约几十 KB，
 * 50 步上限也就一两 MB，可以接受。
 */

/** 深拷贝一份标注状态（要连数组一起复制，否则改回��会互相影响） */
function snapshotAnno() {
  return JSON.parse(JSON.stringify({
    objects: S.anno ? S.anno.objects : [],
    selectedId: S.selectedId,
    edits: S.edits,
  }));
}

/** 记录一次可撤销的操作（在真正改状态之前调用） */
function pushUndo() {
  S.undoStack.push(snapshotAnno());
  if (S.undoStack.length > S.undoLimit) S.undoStack.shift();
  // 新操作后清空重做栈——这是所有编辑器的通用行为：
  // 撤销回去之后再做新操作，原来那条「重做」分支就作废了
  S.redoStack.length = 0;
  updateUndoBtns();
}

/** 撤销一步 */
function undo() {
  if (!S.undoStack.length) { toast('没有可撤销的操作'); return; }
  const snap = S.undoStack.pop();
  S.redoStack.push(snapshotAnno());
  applySnapshot(snap);
  toast('已撤销');
}

/** 重做一步 */
function redo() {
  if (!S.redoStack.length) { toast('没有可重做的操作'); return; }
  const snap = S.redoStack.pop();
  S.undoStack.push(snapshotAnno());
  applySnapshot(snap);
  toast('已重做');
}

/** 把快照恢复成当前状态 */
function applySnapshot(snap) {
  S.anno.objects = snap.objects;
  S.selectedId = snap.selectedId;
  S.edits = snap.edits;
  // 画笔/多边形的临时状态必须一起清掉，否则撤销后会残留半截草稿
  S.brushMask = null;
  S.drawing = false;
  S.draftPts = [];
  syncPolyHint();
  // 焦点可能落在已删除的对象上，选中列表里要能重新找到
  if (S.selectedId && !S.anno.objects.some(o => o.id === S.selectedId)) {
    S.selectedId = null;
  }
  S.dirty = true;
  renderObjects(); renderProps(); renderEditLog(); draw();
  updateUndoBtns();
  saveSoon();
}

/** 撤销/重做按钮的可用状态 */
function updateUndoBtns() {
  const u = $('btnUndo'), r = $('btnRedo');
  if (u) u.disabled = !S.undoStack.length;
  if (r) r.disabled = !S.redoStack.length;
  if (u) u.title = S.undoStack.length
    ? `撤销上一步（Ctrl+Z），还可撤销 ${S.undoStack.length} 步`
    : '没有可撤销的操作（Ctrl+Z）';
  if (r) r.title = S.redoStack.length
    ? `重做（Ctrl+Shift+Z），还可重做 ${S.redoStack.length} 步`
    : '没有可重做的操作（Ctrl+Shift+Z）';
}

// ---------------------------------------------------------------- 工具
const $ = (id) => document.getElementById(id);
const canvas = $('canvas');
const ctx = canvas.getContext('2d');

function toast(msg, isErr) {
  const t = $('toast');
  t.textContent = msg;
  t.classList.remove('hidden');
  t.classList.toggle('err', !!isErr);
  clearTimeout(t._timer);
  t._timer = setTimeout(() => t.classList.add('hidden'), 3200);
}

async function api(url, opts) {
  const r = await fetch(url, opts);
  if (!r.ok) {
    let msg = 'HTTP ' + r.status;
    try { const j = await r.json(); msg = j.error || msg; } catch (e) {}
    throw new Error(msg);
  }
  return r.json();
}

// ---------------------------------------------------------------- 初始化
async function init() {
  const cls = await api('/api/classes');
  S.classes = cls.classes;
  S.colorMap = {};
  S.classes.forEach(c => { S.colorMap[c.name] = c.color; });

  // 支持用 URL 参数直接进入指定模式和指定图片：
  //   ?mode=det&img=000003.jpg
  // 用途有两个：
  //   1) 演示/截图脚本可以直达某个状态，不用模拟点击
  //   2) 答辩时想让评委直接看某张图，可以把链接发过去
  const params = new URLSearchParams(location.search);
  const qMode = params.get('mode');
  if (qMode && ['seg', 'det', 'sem'].includes(qMode)) {
    S.mode = qMode;
    document.querySelectorAll('.mode-btn').forEach(x => {
      x.classList.toggle('active', x.dataset.mode === qMode);
    });
  }
  const qImg = params.get('img');
  if (qImg) S.pendingImage = qImg;

  await loadImages();
  bindEvents();
  if (S.pendingImage) {
    const it = S.images.find(i => i.name === S.pendingImage);
    if (it) await selectImage(it.name);
  }
  draw();
}

async function loadImages() {
  const d = await api('/api/images');
  S.images = d.images;
  S.lastTrash = d.last_trash || null;
  $('imgCount').textContent = d.total;
  $('statReviewed').textContent = `${d.reviewed} / ${d.total}`;
  $('statEdited').textContent = `${d.edited_total} / 60`;
  renderImageList();
  renderTrashBar();
  updateEmptyHint();
  syncPrelabelBtn();
  await refreshRate();
}

/** 预标注按钮上标出待标张数——点之前就能看出这次会处理几张。 */
function syncPrelabelBtn() {
  const btn = $('btnPrelabel');
  if (!btn) return;
  const n = S.images.filter(i => !i.reviewed).length;
  btn.textContent = n ? `预标注 (${n})` : '预标注';
  btn.title = n
    ? `只给还没有标注的 ${n} 张图生成草稿，其余图不动`
    : '全部图片都已有标注；点了会先问是否重做全部';
}

/** 有删除记录时，在列表上方显示一条可恢复的提示。 */
function renderTrashBar() {
  const bar = $('trashBar');
  if (!bar) return;
  if (!S.lastTrash || !S.lastTrash.count) { bar.classList.add('hidden'); return; }
  bar.classList.remove('hidden');
  $('trashText').textContent = `已删除 ${S.lastTrash.count} 张`;
}

/**
 * 画布中央的空白提示。
 * 分三种情况：没图 / 有图但没选 / 选中了（隐藏）。
 * 【为什么要分】原来是写死一句「还没有图片」，用户上传了 150 张
 * 但还没点某张图时，页面正中央仍然写着「还没有图片」，明显不对。
 */
function updateEmptyHint() {
  const hint = $('emptyHint');
  if (!hint) return;
  if (S.current) { hint.classList.add('hidden'); return; }
  hint.classList.remove('hidden');
  const hasImg = S.images.length > 0;
  $('emptyTitle').textContent = hasImg
    ? `已加载 ${S.images.length} 张图片`
    : '还没有图片';
  $('emptySub').textContent = hasImg
    ? '从左侧列表选一张开始，或上传新照片'
    : '把照片拖进这个窗口，或点下面的按钮';
  $('btnUploadEmpty').textContent = hasImg ? '＋ 再传几张' : '＋ 上传照片';
}

async function refreshRate() {
  try {
    const r = await api('/api/records');
    $('statRate').textContent = r.correction_rate.toFixed(1) + '%';
  } catch (e) {}
}

function renderImageList() {
  const box = $('imgList');
  box.innerHTML = '';
  S.images.forEach(it => {
    const div = document.createElement('div');
    div.className = 'img-item';
    if (it.reviewed) div.classList.add('done');
    else if (it.total > 0) div.classList.add('pending');
    if (it.name === S.current) div.classList.add('active');

    // 文件名可能是用户上传的任意文字，统一用 textContent，不拼 innerHTML
    const nm = document.createElement('span');
    nm.className = 'img-name';
    nm.textContent = it.name;
    nm.title = it.name;

    const badge = document.createElement('span');
    badge.className = 'img-badge';
    badge.textContent = it.reviewed
      ? (it.edited ? '改' + it.edited : '已审')
      : (it.total ? '待审' : '—');

    const del = document.createElement('button');
    del.type = 'button';
    del.className = 'img-del';
    del.textContent = '×';
    del.title = '删除这张照片（图片与标注移到 data/_trash，可恢复）';
    // stopPropagation：否则点 × 会顺带触发整行的 selectImage
    del.onclick = ev => { ev.stopPropagation(); deleteImage(it); };

    div.append(nm, badge, del);
    div.onclick = () => selectImage(it.name);
    box.appendChild(div);
  });
}

// ---------------------------------------------------------------- 删除照片
/**
 * 删除一张照片。
 *
 * 【为什么要反复确认】
 * 后端是软删除（搬到 data/_trash/，可恢复），但用户不会去翻那个目录，
 * 所以提示里必须写清后果：这张图有几个对象、其中几条是人工改过的。
 * 尤其是数据集自带的那 150 张——删掉之后提交材料里
 * 「150 张 / 60 条修正」的口径就对不上了，得单独加重警告。
 */
async function deleteImage(it) {
  const isPreset = /^\d{6}\.\w+$/i.test(it.name);

  let msg = `确定删除「${it.name}」？\n\n`;
  if (it.total) {
    msg += `这张图有 ${it.total} 个标注对象`;
    msg += it.edited ? `，其中 ${it.edited} 条是人工修正过的。\n\n` : '。\n\n';
  }
  if (isPreset && it.edited) {
    msg += '⚠ 这是数据集自带的图片，删除后「150 张 / 60 条修正」\n'
         + '   的口径就对不上，提交材料里的数字需要重算。\n\n';
  }
  if (S.current === it.name && S.dirty) {
    msg += '当前这张图有未保存的修改，删除会一并丢弃。\n\n';
  }
  msg += '图片和标注会一起移到 data/_trash/，可以点「恢复」找回。';
  if (!confirm(msg)) return;

  const wasCurrent = S.current === it.name;
  const idx = S.images.findIndex(x => x.name === it.name);

  // 保存是 700ms 防抖的（saveSoon）。删的就是当前这张图时，
  // 必须先把待执行的保存取消——否则删除请求先到、标注文件被搬走，
  // 延迟的 saveNow 又把整份标注写回来，留下一个没有图片的孤儿 json。
  // 顺手把 saveTimer 置空：clearTimeout 只取消定时器、不清变量，
  // 留着旧 id 会让「还有没有待执行的保存」变得无法判断（也不便测试）。
  if (wasCurrent) { clearTimeout(saveTimer); saveTimer = null; }

  try {
    const r = await api('/api/images/delete', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ names: [it.name] }),
    });
    if (!r.deleted.length) {
      toast('删除失败：' + (r.missing.length ? '图片已不存在' : '未知原因'), true);
      if (S.dirty) saveSoon();   // 没删成，把刚才取消的保存还回去
      return;
    }
    await loadImages();
    if (wasCurrent) {
      // 删掉的正是打开着的那张：清空画布并顺位选下一张。
      // 不自动跳转的话，画布上还挂着已删图片的残影，看着像没删成功。
      S.dirty = false;
      clearStage();
      const next = S.images[Math.min(idx, S.images.length - 1)];
      if (next) await selectImage(next.name);
    }
    toast(`已删除 ${it.name}（可在左侧「恢复」）`);
  } catch (e) {
    toast('删除失败: ' + e.message, true);
    if (S.dirty) saveSoon();
  }
}

/** 把画布清空到「没选图」的状态（删掉当前图后用）。 */
function clearStage() {
  S.current = null;
  S.img = null;
  S.anno = null;
  S.selectedId = null;
  S.edits = [];
  S.draftPts = [];
  S.drawing = false;
  S.painting = false;
  S.brushMask = null;
  S.hoverId = null;
  S.undoStack = [];
  S.redoStack = [];
  updateUndoBtns();
  draw();
  updateEmptyHint();
}

/** 恢复最近一次删除（含标注与人工修正标记）。 */
async function restoreImages() {
  if (!S.lastTrash) return;
  try {
    const r = await api('/api/images/restore', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ stamp: S.lastTrash.stamp }),
    });
    await loadImages();
    if (r.restored.length) {
      const head = r.restored.slice(0, 3).join('、');
      toast(`已恢复 ${r.restored.length} 张：${head}`
            + (r.restored.length > 3 ? ' 等' : ''));
    } else {
      toast('没有恢复任何文件（可能同名文件已存在）', true);
    }
  } catch (e) {
    toast('恢复失败: ' + e.message, true);
  }
}

// ---------------------------------------------------------------- 选图
/* ---------------------------------------------------------------------------
 * 窄屏抽屉（手机 / 平板竖屏）
 *
 * 【为什么交给 JS 而不是纯 CSS】
 * 需要「打开一边自动收起另一边」和「点遮罩关闭」两个状态判断，
 * 用 JS 十几行就说清楚了；纯 CSS 得靠 :target 或 checkbox hack，反而绕。
 * 桌面端左右栏是常驻的，这些函数调用不会产生任何视觉变化。
 * ------------------------------------------------------------------------- */

function closeDrawers() {
  const open = document.querySelectorAll('.panel-left.open, .panel-right.open');
  if (!open.length) return;
  open.forEach(p => p.classList.remove('open'));
  const mask = $('drawerMask');
  mask.classList.remove('show');
  // 等淡出动画走完再真正隐藏，否则遮罩会瞬间消失、抽屉还在滑
  setTimeout(() => {
    if (!document.querySelector('.panel-left.open, .panel-right.open')) mask.hidden = true;
  }, 200);
}

function toggleDrawer(side) {
  const target = document.querySelector(side === 'left' ? '.panel-left' : '.panel-right');
  if (!target) return;
  const wasOpen = target.classList.contains('open');
  closeDrawers();
  if (wasOpen) return;              // 再点一次同一个按钮 = 收起
  target.classList.add('open');
  const mask = $('drawerMask');
  mask.hidden = false;
  // 先显示、下一帧再加 class：同一帧改这两个属性，transition 会被合并掉不播放
  requestAnimationFrame(() => mask.classList.add('show'));
}

async function selectImage(name) {
  if (S.dirty && !confirm('当前图片有未保存的修改，确定切换？')) return;
  S.current = name;
  // 窄屏：选完图就把左抽屉收起来，否则画布一直被盖着看不到东西。
  // 桌面端这两个类本来就不存在，调用是无副作用的。
  closeDrawers();
  S.edits = [];
  S.selectedId = null;
  S.draftPts = [];
  S.drawing = false;
  // 换图时重置可见性状态：新图默认全部显示
  S.unchecked = new Set();
  S.focusMode = true;
  S.hoverId = null;
  S.brushMask = null;
  // 撤销栈是「当前这张图」的历史，换图必须清空，
  // 否则 Ctrl+Z 会把上一张图的对象恢复出来
  S.undoStack = [];
  S.redoStack = [];
  updateUndoBtns();
  syncFocusBtn();

  $('emptyHint').classList.add('hidden');
  syncPolyHint();

  const img = new Image();
  img.onload = () => {
    S.img = img;
    fitCanvas();
    draw();
  };
  img.src = '/images/' + encodeURIComponent(name);

  try {
    const a = await api('/api/annotation/' + encodeURIComponent(name));
    S.anno = a;
    S.origById = {};
    a.objects.forEach(o => { S.origById[o.id] = JSON.stringify(o); });
  } catch (e) {
    // 还没有标注数据 -> 建一个空的
    S.anno = { image: name, width: 960, height: 720, objects: [] };
    S.origById = {};
  }
  S.dirty = false;
  renderImageList();
  renderObjects();
  renderProps();
  renderEditLog();
  draw();
}

// ---------------------------------------------------------------- 画布
function fitCanvas() {
  const wrap = $('canvasWrap');
  const cw = wrap.clientWidth, ch = wrap.clientHeight;
  if (!S.img) return;
  const pad = 24;
  const s = Math.min((cw - pad) / S.img.width, (ch - pad) / S.img.height);
  S.scale = s;
  S.offsetX = (cw - S.img.width * s) / 2;
  S.offsetY = (ch - S.img.height * s) / 2;
  canvas.width = cw; canvas.height = ch;
  canvas.style.width = cw + 'px'; canvas.style.height = ch + 'px';
  $('zoomLabel').textContent = Math.round(s * 100) + '%';
}

function toCanvas(x, y) {
  // 归一化坐标 -> 画布像素
  return [x * S.img.width * S.scale + S.offsetX,
          y * S.img.height * S.scale + S.offsetY];
}

function toNorm(cx, cy) {
  // 画布像素 -> 归一化坐标
  return [(cx - S.offsetX) / (S.img.width * S.scale),
          (cy - S.offsetY) / (S.img.height * S.scale)];
}

function draw() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!S.img || !S.anno) return;

  ctx.drawImage(S.img, S.offsetX, S.offsetY,
                S.img.width * S.scale, S.img.height * S.scale);

  const objs = visibleObjects();

  if (S.mode === 'sem') {
    drawSemantic(objs);
  } else {
    objs.forEach(o => drawObject(o));
  }

  // 多边形草稿
  if (S.drawing && S.draftPts.length) {
    ctx.save();
    ctx.strokeStyle = '#22d3ee';
    ctx.fillStyle = 'rgba(34,211,238,.16)';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([4, 3]);
    ctx.beginPath();
    S.draftPts.forEach((p, i) => {
      const [cx, cy] = toCanvas(p[0], p[1]);
      i === 0 ? ctx.moveTo(cx, cy) : ctx.lineTo(cx, cy);
    });
    ctx.closePath(); ctx.fill(); ctx.stroke();
    ctx.restore();
  }

  // 框选修形的橡皮筋
  drawBoxDrag();
}

function drawObject(o) {
  const col = S.colorMap[o.class] || '#94a3b8';
  const sel = o.id === S.selectedId;
  const hov = o.id === S.hoverId && !sel;
  const pts = o.polygon.map(p => toCanvas(p[0], p[1]));

  ctx.save();
  if (S.mode === 'det') {
    // 检测模式：画外接矩形（这是检测任务本身的语义）
    const xs = pts.map(p => p[0]), ys = pts.map(p => p[1]);
    const x = Math.min(...xs), y = Math.min(...ys);
    const w = Math.max(...xs) - x, h = Math.max(...ys) - y;
    ctx.strokeStyle = col;
    ctx.lineWidth = sel ? 2.5 : 1.5;
    ctx.strokeRect(x, y, w, h);
    if (S.showLabel && (sel || hov)) drawLabel(o, x, y, col);
  } else {
    // 掩码模式：填充多边形
    ctx.beginPath();
    pts.forEach((p, i) => i === 0 ? ctx.moveTo(p[0], p[1]) : ctx.lineTo(p[0], p[1]));
    ctx.closePath();
    ctx.fillStyle = hexA(col, sel ? 0.30 : (hov ? 0.26 : 0.18));
    ctx.fill();

    // 描边：沿轮廓走，不套矩形框
    //
    // 选中的对象要「在任何背景上都看得清」，做法是描三层：
    //   外层白边（粗）-> 暗色分隔线（中）-> 类别色细线（细）
    // 白边打底，暗线隔开白边和填充，最后细彩线给出类别识别。
    // 三层叠加后轮廓是"亮边 + 彩线"，不会和填充色糊在一起。
    if (sel) {
      // 第 1 层：外侧白色底衬
      ctx.strokeStyle = 'rgba(255,255,255,.92)';
      ctx.lineWidth = 5;
      ctx.lineJoin = 'round';
      ctx.stroke();
      // 第 2 层：暗色分隔，让白边和填充不粘连
      ctx.strokeStyle = 'rgba(0,0,0,.45)';
      ctx.lineWidth = 3;
      ctx.stroke();
      // 第 3 层：类别色细线
      ctx.strokeStyle = col;
      ctx.lineWidth = 1.8;
      ctx.stroke();
    } else {
      // 未选中：细一点、淡一点，避免满屏线条看不清图
      ctx.strokeStyle = hexA(col, hov ? 0.9 : 0.6);
      ctx.lineWidth = hov ? 1.8 : 1.1;
      ctx.lineJoin = 'round';
      ctx.stroke();
    }

    if (S.showBox && sel) {
      // 「框」开关仍然可用：额外叠加外接矩形（调试用）
      const xs = pts.map(p => p[0]), ys = pts.map(p => p[1]);
      ctx.setLineDash([3, 3]);
      ctx.strokeStyle = 'rgba(255,255,255,.45)';
      ctx.lineWidth = 1;
      ctx.strokeRect(Math.min(...xs), Math.min(...ys),
                     Math.max(...xs) - Math.min(...xs),
                     Math.max(...ys) - Math.min(...ys));
      ctx.setLineDash([]);
    }
    if (S.showLabel && (sel || hov)) {
      const xs = pts.map(p => p[0]), ys = pts.map(p => p[1]);
      drawLabel(o, Math.min(...xs), Math.min(...ys), col);
    }
  }
  ctx.restore();
}

function drawSemantic(objs) {
  // 语义模式：同类别的多边形合并成一片色块
  // 用「最大面积优先 + 后面的不覆盖前面的」模拟类别互斥
  const sorted = objs.slice().sort((a, b) => polyArea(b.polygon) - polyArea(a.polygon));
  sorted.forEach(o => {
    const col = S.colorMap[o.class] || '#94a3b8';
    const pts = o.polygon.map(p => toCanvas(p[0], p[1]));
    ctx.beginPath();
    pts.forEach((p, i) => i === 0 ? ctx.moveTo(p[0], p[1]) : ctx.lineTo(p[0], p[1]));
    ctx.closePath();
    ctx.fillStyle = hexA(col, o.id === S.selectedId ? 0.6 : 0.45);
    ctx.fill();
    ctx.strokeStyle = hexA(col, 0.9);
    ctx.lineWidth = 0.8;
    ctx.stroke();
  });
}
function drawLabel(o, x, y, col) {
  const txt = o.class;
  ctx.save();
  ctx.font = '11px -apple-system, "Microsoft YaHei", sans-serif';
  const w = ctx.measureText(txt).width + 8;
  ctx.fillStyle = col;
  ctx.fillRect(x, y - 14, w, 14);
  ctx.fillStyle = '#fff';
  ctx.fillText(txt, x + 4, y - 3.5);
  ctx.restore();
}

function polyArea(poly) {
  let t = 0;
  for (let i = 0; i < poly.length; i++) {
    const [x1, y1] = poly[i], [x2, y2] = poly[(i + 1) % poly.length];
    t += x1 * y2 - x2 * y1;
  }
  return Math.abs(t / 2);
}

function hexA(hex, a) {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
}

// ---------------------------------------------------------------- 可见性
/**
 * 判断对象是否应该在画布上显示。
 *
 * 两层控制，优先级从高到低：
 *   1. 复选框没勾选  -> 不显示（用户主动隐藏）
 *   2. 聚焦模式      -> 只显示选中的那一个
 * 这样"点开一个对象想仔细看轮廓"时，别的对象不会糊在脸上。
 */
function isVisible(o) {
  if (S.unchecked.has(o.id)) return false;
  if (S.focusMode && S.selectedId) return o.id === S.selectedId;
  return true;
}

function visibleObjects() {
  if (!S.anno) return [];
  return S.anno.objects.filter(isVisible);
}

// ---------------------------------------------------------------- 命中检测
function hitTest(nx, ny) {
  // 从后往前找（后画的在上面）；只命中可见的对象
  const objs = visibleObjects();
  for (let i = objs.length - 1; i >= 0; i--) {
    const o = objs[i];
    if (pointInPoly(nx, ny, o.polygon)) return o;
  }
  return null;
}

function pointInPoly(x, y, poly) {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const xi = poly[i][0], yi = poly[i][1];
    const xj = poly[j][0], yj = poly[j][1];
    if ((yi > y) !== (yj > y)) {
      const xc = (xj - xi) * (y - yi) / (yj - yi) + xi;
      if (x < xc) inside = !inside;
    }
  }
  return inside;
}

// ---------------------------------------------------------------- 编辑操作
function markEdited(objId) {
  const o = S.anno.objects.find(x => x.id === objId);
  if (!o) return;
  const before = S.origById[objId];
  const now = JSON.stringify({ id: o.id, class: o.class,
                                polygon: o.polygon, source: o.source,
                                edited: o.edited });
  if (before === now) return;
  o.edited = true;
  S.origById[objId] = JSON.stringify({ id: o.id, class: o.class,
                                       polygon: o.polygon, source: o.source,
                                       edited: true });
  S.dirty = true;
  const existed = S.edits.find(e => e.object_id === objId);
  if (!existed) {
    S.edits.push({ object_id: objId, class: o.class, type: '修正' });
    renderEditLog();
    renderObjects();
  }
}

function addObject(cls, poly, source) {
  pushUndo();// 撤销点：新增之前的整图状态
  const obj = {
    id: 'o' + Math.random().toString(36).slice(2, 10),
    class: cls, polygon: poly, source: source || 'human', edited: true,
  };
  S.anno.objects.push(obj);
  S.origById[obj.id] = JSON.stringify(obj);
  S.edits.push({ object_id: obj.id, class: cls, type: '新增' });
  S.dirty = true;
  S.selectedId = obj.id;
  renderObjects(); renderProps(); renderEditLog();
  return obj;
}

function deleteSelected() {
  if (!S.selectedId) return;
  const idx = S.anno.objects.findIndex(o => o.id === S.selectedId);
  if (idx < 0) return;
  const o = S.anno.objects[idx];
  pushUndo();              // 撤销点：删除之前的整图状态（对象还能拿回来）
  S.anno.objects.splice(idx, 1);
  S.edits.push({ object_id: o.id, class: o.class, type: '删除' });
  S.selectedId = null;
  S.dirty = true;
  renderObjects(); renderProps(); renderEditLog();
  saveSoon();
}

function changeClass(newCls) {
  if (!S.selectedId) return;
  const o = S.anno.objects.find(x => x.id === S.selectedId);
  if (!o || o.class === newCls) return;
  pushUndo();              // 撤销点：改类别之前
  o.class = newCls;
  markEdited(o.id);
  S.edits[S.edits.length - 1].type = '类别';
  renderObjects(); renderProps(); renderEditLog();
  saveSoon();
}

// ---------------------------------------------------------------- 渲染面板
function renderObjects() {
  const box = $('objList');
  if (!S.anno) { $('objCount').textContent = '0'; box.innerHTML = ''; return; }
  const total = S.anno.objects.length;
  $('objCount').textContent = total ? `${visibleObjects().length}/${total}` : '0';
  box.innerHTML = '';
  if (!total) {
    // 【实测踩到的困惑】用户打开 000000.jpg（一张小熊照片），
    // 右栏只写「没有标注对象」，他不知道是工具坏了还是自己操作错了，
    // 于是去点「多边形」试图框小熊——但小熊不在任务书十类词表里。
    // 这里明确说清原因：不是没标，而是这张图没有属于十类的目标。
    const canAdd = S.classes.length > 0;
    box.innerHTML = canAdd
      ? `<p class="muted small" style="padding:8px;line-height:1.7">
           没有标注对象。<br>
           如果图里有属于任务书十类的目标（人、椅子、餐桌、沙发、电视、
           笔记本、杯子、瓶子、盆栽、书），可以用底部
           <b>「多边形」</b> 工具描出轮廓。</p>`
      : '<p class="muted small" style="padding:8px">没有标注对象</p>';
    return;
  }

  S.anno.objects.forEach((o, idx) => {
    const div = document.createElement('div');
    div.className = 'obj-item';
    if (o.id === S.selectedId) div.classList.add('active');
    if (!isVisible(o)) div.classList.add('dimmed');

    // 复选框：勾选 = 在画布上显示
    const chk = document.createElement('input');
    chk.type = 'checkbox';
    chk.className = 'obj-chk';
    chk.checked = !S.unchecked.has(o.id);
    chk.title = '显示 / 隐藏该对象';
    chk.onclick = (e) => e.stopPropagation();
    chk.onchange = () => {
      if (chk.checked) S.unchecked.delete(o.id);
      else S.unchecked.add(o.id);
      renderObjects();
      draw();
    };

    const idxSpan = document.createElement('span');
    idxSpan.className = 'obj-idx';
    idxSpan.textContent = idx + 1;

    const sw = document.createElement('span');
    sw.className = 'obj-swatch';
    sw.style.background = S.colorMap[o.class] || '#94a3b8';

    const nm = document.createElement('span');
    nm.className = 'obj-name';
    nm.textContent = o.class;

    const tg = document.createElement('span');
    tg.className = 'obj-tag' + (o.edited ? ' edited' : '');
    tg.textContent = o.edited ? '已改' : (o.source === 'model' ? '草稿' : '人工');

    div.appendChild(chk);
    div.appendChild(idxSpan);
    div.appendChild(sw);
    div.appendChild(nm);
    div.appendChild(tg);

    // 点一行 = 选中它，并自动进入聚焦（只显示它）
    div.onclick = () => {
      S.selectedId = o.id;
      S.focusMode = true;
      syncFocusBtn();
      renderObjects(); renderProps(); draw();
    };
    box.appendChild(div);
  });
}

/** 聚焦按钮的启用态要和状态一致 */
function syncFocusBtn() {
  const b = $('btnFocusOnly');
  if (b) b.classList.toggle('active', S.focusMode);
}

/**
 * 多边形工具的状态提示。
 *
 * 【为什么要专门做这个】
 * 用实测踩出来的坑：用户点了「多边形」按钮，在轮廓上点了一串点，
 * 然后发现「框不了」—— 因为闭合要双击或按 Enter，
 * 而界面上从头到尾没有任何地方告诉过他这件事。
 * 新手会以为工具坏了，反复点击，最后放弃。
 *
 *所以做两件事：
 *   1. 工具栏下方常驻一条提示（选中多边形工具就出现）
 *   2. 画布上跟随一个指引层，随点数变化告诉用户下一步该干什么
 */
function syncPolyHint() {
  const bar = $('polyHint');
  const tip = $('drawingTip');
  const on = S.tool === 'polygon';
  if (bar) bar.hidden = !on;
  syncBoxHint();
  if (!tip) return;
  if (!on || !S.drawing) {
    tip.classList.add('hidden');
    return;
  }
  const n = S.draftPts.length;
  tip.classList.remove('hidden');
  tip.textContent = n < 3
    ? `已点 ${n} 个点 · 继续沿物体轮廓点击（至少 3 点），然后双击或按 Enter 闭合`
    : `已点 ${n} 个点 · 双击或按「完成描边」闭合；点歪了用「撤回一点」或画布右键`;
}

/** 框选修形的提示条：只有切到这个工具才显示，并把当前模式高亮出来。 */
function syncBoxHint() {
  const bar = $('boxHint');
  if (!bar) return;
  bar.hidden = S.tool !== 'box';
  document.querySelectorAll('.box-mode').forEach(b => {
    b.classList.toggle('active', b.dataset.boxmode === S.boxMode);
  });
}

function renderProps() {
  const box = $('props');
  const o = S.anno && S.anno.objects.find(x => x.id === S.selectedId);
  if (!o) {
    box.innerHTML = '<p class="muted small">在画布或上方列表中选择一个对象</p>';
    return;
  }
  const opts = S.classes.map(c =>
    `<option value="${c.name}" ${c.name === o.class ? 'selected' : ''}>${c.name}</option>`).join('');
  const xs = o.polygon.map(p => p[0]), ys = o.polygon.map(p => p[1]);
  box.innerHTML = `
    <div class="prop-row">
      <span class="prop-label">类别</span>
      <select class="class-select" id="clsSelect">${opts}</select>
    </div>
    <div class="prop-row">
      <span class="prop-label">掩码顶点数</span>
      <span class="prop-value">${o.polygon.length}</span>
    </div>
    <div class="prop-row">
      <span class="prop-label">外接框</span>
      <span class="prop-value">${(Math.min(...xs)).toFixed(3)}, ${(Math.min(...ys)).toFixed(3)} → ${(Math.max(...xs)).toFixed(3)}, ${(Math.max(...ys)).toFixed(3)}</span>
    </div>
    <div class="prop-row">
      <span class="prop-label">来源</span>
      <span class="prop-value">${o.source === 'model' ? '模型草稿' : '人工新建'}${o.edited ? ' · 已修正' : ''}</span>
    </div>
    <button class="btn-danger" id="btnDelete">删除该对象</button>`;
  $('clsSelect').onchange = (e) => changeClass(e.target.value);
  $('btnDelete').onclick = deleteSelected;
}

function renderEditLog() {
  const box = $('editLog');
  $('editCount').textContent = S.edits.length;
  if (!S.edits.length) {
    box.innerHTML = '<p class="muted small">还没有修正</p>';
    return;
  }
  box.innerHTML = S.edits.map((e, i) => `
    <div class="edit-item">
      <span class="edit-idx">${String(i + 1).padStart(2, '0')}</span>
      <span class="edit-text">${e.class}</span>
      <span class="edit-kind">${e.type}</span>
    </div>`).join('');
}

// ---------------------------------------------------------------- 保存
let saveTimer = null;
function saveSoon() {
  clearTimeout(saveTimer);
  saveTimer = setTimeout(saveNow, 700);
}

async function saveNow() {
  saveTimer = null;          // 定时器已经触发，"还有待执行的保存"这一状态结束
  if (!S.current || !S.anno) return;
  // 测试模式下不落盘，避免自动化测试污染真实标注数据
  if (window.__UITEST__) { S.dirty = false; return; }
  try {
    const d = await api('/api/annotation/' + encodeURIComponent(S.current), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(S.anno),
    });
    S.dirty = false;
    $('statEdited').textContent = `${d.edited} / 60`;
    await loadImages();
    await refreshRate();
  } catch (e) {
    toast('保存失败: ' + e.message, true);
  }
}

// ---------------------------------------------------------------- 事件
function bindEvents() {
  // 模式切换
  document.querySelectorAll('.mode-btn').forEach(b => {
    b.onclick = () => {
      document.querySelectorAll('.mode-btn').forEach(x => x.classList.remove('active'));
      b.classList.add('active');
      S.mode = b.dataset.mode;
      draw();
    };
  });

  // 工具切换
  document.querySelectorAll('.tool-btn[data-tool]').forEach(b => {
    b.onclick = () => {
      document.querySelectorAll('.tool-btn[data-tool]').forEach(x => x.classList.remove('active'));
      b.classList.add('active');
      const prev = S.tool;
      S.tool = b.dataset.tool;
      canvas.className = 'tool-' + S.tool;
      if (S.tool !== 'polygon') { S.drawing = false; S.draftPts = []; }
      if (S.tool !== 'box') S.boxDrag = null;
      syncPolyHint();
      if (S.tool === 'box' && !S.selectedId) {
        // 不拦着不让进（有人习惯先切工具再点对象），但要明确说清楚顺序
        toast('框选修形：先用「选择」点一个对象，再拉框');
      }
      if (S.tool === 'brush' || S.tool === 'erase') {
        if (!S.selectedId) {
          toast('先用「选择」工具点一个对象，再来涂抹');
          S.tool = prev;
          canvas.className = 'tool-' + S.tool;
          b.classList.remove('active');
          const pb = document.querySelector(`.tool-btn[data-tool="${prev}"]`);
          if (pb) pb.classList.add('active');
          return;
        }
        // 画笔需要一张位图掩码，把当前多边形栅格化进去
        const o = S.anno.objects.find(x => x.id === S.selectedId);
        if (o) buildBrushMask(o);
      } else {
        S.brushMask = null;
      }
      draw();
    };
  });

  // 对象可见性：全选 / 全不选 / 仅看选中
  $('btnShowAll').onclick = () => {
    S.unchecked.clear();
    S.focusMode = false;
    syncFocusBtn();
    renderObjects(); draw();
  };
  $('btnHideAll').onclick = () => {
    S.unchecked = new Set(S.anno ? S.anno.objects.map(o => o.id) : []);
    S.focusMode = false;
    syncFocusBtn();
    renderObjects(); draw();
  };
  $('btnFocusOnly').onclick = () => {
    S.focusMode = !S.focusMode;
    syncFocusBtn();
    renderObjects(); draw();
  };

  // 多边形描边的显式操作按钮。
  // 【为什么加这个】原来只能靠双击或 Enter 闭合，界面上一个提示都没有，
  // 用户点了半天发现「框不了」，以为是工具坏了。
  // 给出可见按钮 + 常驻提示条，把「怎么结束」明说出来。
  $('btnFinishPoly').onclick = () => {
    if (!S.drawing || S.draftPts.length < 3) {
      toast('先在物体轮廓上点至少 3 个点', true);
      return;
    }
    finishPolygon();
  };
  $('btnCancelPoly2').onclick = () => {
    S.drawing = false; S.draftPts = [];
    syncPolyHint(); draw();
  };

  // 框选修形：模式切换 + 退出
  document.querySelectorAll('.box-mode').forEach(b => {
    b.onclick = () => {
      S.boxMode = b.dataset.boxmode;
      syncBoxHint();
      draw();
    };
  });
  const cancelBox = $('btnCancelBox');
  if (cancelBox) {
    cancelBox.onclick = () => {
      S.boxDrag = null;
      S.tool = 'select';
      canvas.className = 'tool-select';
      document.querySelectorAll('.tool-btn[data-tool]').forEach(x => x.classList.remove('active'));
      const sb = document.querySelector('.tool-btn[data-tool="select"]');
      if (sb) sb.classList.add('active');
      syncBoxHint();
      draw();
    };
  }

  // 撤销 / 重做
  $('btnUndo').onclick = undo;
  $('btnRedo').onclick = redo;
  updateUndoBtns();

  // 描边过程中的「撤回一点」——等价于在画布上右键，
  // 但按钮更显眼，小目标点歪了不用每次都去画布上找右键。
  $('btnUndoPt').onclick = () => {
    if (!S.drawing || !S.draftPts.length) { toast('还没有点过任何点', true); return; }
    S.draftPts.pop();
    syncPolyHint(); draw();
  };

  // 画布交互
  // 【为什么用 Pointer Events 而不是 mouse + touch 两套监听】
  // 一套代码同时覆盖鼠标、手指、触控笔，不用维护两份分支逻辑；
  // 也不会出现「浏览器对一次触摸同时派发 mouse 和 touch」导致的重复处理。
  // 代价是触摸没有 e.detail（双击计数）、没有 button===2（右键），
  // 所以下面用手势自己补：双击自己数，撤回上一点用提示条上的按钮。
  canvas.addEventListener('pointerdown', onDown);
  // move / up 挂在 window 上：拖出画布再松手也能收到。
  // 挂 canvas 上的话手指滑出边界就丢 up，画笔会一直停在「涂抹中」。
  window.addEventListener('pointermove', onMove);
  window.addEventListener('pointerup', onUp);
  window.addEventListener('pointercancel', onUp);
  canvas.oncontextmenu = e => e.preventDefault();

  // 显示开关
  $('showMask').onchange = e => { S.showMask = e.target.checked; draw(); };
  $('showBox').onchange = e => { S.showBox = e.target.checked; draw(); };
  $('showLabel').onchange = e => { S.showLabel = e.target.checked; draw(); };

  // 笔刷大小
  $('brushSize').oninput = e => {
    S.brushSize = +e.target.value;
    $('brushSizeVal').textContent = e.target.value;
  };

  // 缩放
  $('btnZoomIn').onclick = () => { S.scale *= 1.2; $('zoomLabel').textContent = Math.round(S.scale * 100) + '%'; draw(); };
  $('btnZoomOut').onclick = () => { S.scale /= 1.2; $('zoomLabel').textContent = Math.round(S.scale * 100) + '%'; draw(); };
  $('btnFit').onclick = () => { fitCanvas(); draw(); };

  // 顶栏按钮
  // 用箭头函数包一层：直接挂函数的话，点击事件对象会被当成 opts 传进去
  $('btnPrelabel').onclick = () => startPrelabel();
  $('btnRelabel').onclick = startRelabel;
  $('btnRestore').onclick = restoreImages;
  $('btnExport').onclick = doExport;
  $('btnTrain').onclick = doTrain;
  $('btnCompare').onclick = showCompare;
  $('btnRecords').onclick = showRecords;

  $('modalClose').onclick = () => $('modal').classList.add('hidden');
  window.onresize = () => { fitCanvas(); draw(); };

  // 窄屏抽屉（桌面端这两个按钮被 CSS 隐藏，点不到）
  $('btnLeftDrawer').onclick = () => toggleDrawer('left');
  $('btnRightDrawer').onclick = () => toggleDrawer('right');
  $('drawerMask').onclick = closeDrawers;
  window.onbeforeunload = e => { if (S.dirty) { e.preventDefault(); e.returnValue = ''; } };

  // 上传：按钮 → 文件选择框 → 选完立刻上传
  $('btnUpload').onclick = pickFiles;
  $('btnUploadEmpty').onclick = pickFiles;
  $('fileInput').onchange = e => uploadFiles(e.target.files);
  bindDropUpload();
}

// ---------------------------------------------------------------- 画布事件
function canvasPos(e) {
  const r = canvas.getBoundingClientRect();
  return [e.clientX - r.left, e.clientY - r.top];
}

function onDown(e) {
  if (!S.anno) return;
  // 多指同时按住时只认第一根手指：否则两指会各拖一个框/各画一条线，
  // 坐标互相打架。第一根抬起后 pointerId 归 null，下一根才能接管。
  if (S.pointerId !== null && S.pointerId !== e.pointerId) return;
  S.pointerId = e.pointerId;

  const [cx, cy] = canvasPos(e);
  const [nx, ny] = toNorm(cx, cy);

  if (S.tool === 'polygon') {
    if (e.button === 2) {           // 右键：退回上一个点
      S.draftPts.pop(); syncPolyHint(); draw(); return;
    }
    if (!S.drawing) { S.drawing = true; S.draftPts = []; }
    S.draftPts.push([nx, ny]);
    syncPolyHint();
    draw();
    return;
  }

  if (S.tool === 'box') {
    if (e.button === 2) { S.boxDrag = null; draw(); return; }
    S.boxDrag = { x0: nx, y0: ny, x1: nx, y1: ny };
    draw();
    return;
  }

  if (S.tool === 'select') {
    const hit = hitTest(nx, ny);
    S.selectedId = hit ? hit.id : null;
    if (!hit) {
      // 点空白：退出聚焦，恢复显示全部（未勾选的除外）
      S.focusMode = false;
      syncFocusBtn();
    } else {
      S.focusMode = true;
    }
    renderObjects(); renderProps(); draw();
    return;
  }

  if (S.tool === 'brush' || S.tool === 'erase') {
    if (!S.selectedId) { toast('先选中一个对象再涂抹'); return; }
    // 掩码不存在或属于别的对象 -> 重建
    if (!S.brushMask || S.brushMask.id !== S.selectedId) {
      const o = S.anno.objects.find(x => x.id === S.selectedId);
      if (!o) return;
      buildBrushMask(o);
    }
    S.painting = true;
    S.paintTarget = S.selectedId;
    paintAt(nx, ny);
  }
}

function onMove(e) {
  const [cx, cy] = canvasPos(e);
  // move 挂在 window 上（为了拖出画布也能收到），所以指针可能在画布外。
  // 画笔/拖框这两种"已按下"的状态要照常处理——用户就是想在画布外继续拖；
  // 只有悬停高亮需要知道指针是不是真的在画布上。
  const inside = cx >= 0 && cy >= 0 && cx <= canvas.clientWidth && cy <= canvas.clientHeight;
  S.mouse.x = cx; S.mouse.y = cy; S.mouse.in = inside;

  if (S.painting) {
    const [nx, ny] = toNorm(cx, cy);
    paintAt(nx, ny);
    return;
  }

  if (S.boxDrag) {
    const [nx, ny] = toNorm(cx, cy);
    S.boxDrag.x1 = nx; S.boxDrag.y1 = ny;
    draw();
    return;
  }

  // 悬停高亮：让用户知道指针底下是哪个对象（触摸时不会触发，无害）
  if (S.tool === 'select' && S.anno) {
    if (!inside) {
      if (S.hoverId !== null) { S.hoverId = null; canvas.style.cursor = 'default'; draw(); }
      return;
    }
    const [nx, ny] = toNorm(cx, cy);
    const hit = hitTest(nx, ny);
    const hid = hit ? hit.id : null;
    if (hid !== S.hoverId) {
      S.hoverId = hid;
      canvas.style.cursor = hid ? 'pointer' : 'default';
      draw();
    }
  }
}

// 触摸事件没有 e.detail（双击计数），双击得自己数：
// 两次抬起间隔 < 320ms 且几乎落在同一位置。
// 【为什么位置阈值只有 12px】描边时用户会快速点下几个顶点，
// 阈值放宽（比如 30px）会把「快速点两个相邻顶点」误判成双击，多边形提前闭合。
const lastTap = { t: 0, x: 0, y: 0 };
function isDoubleTap(e) {
  const now = (typeof performance !== 'undefined' && performance.now)
    ? performance.now() : Date.now();
  const near = Math.abs(e.clientX - lastTap.x) < 12 && Math.abs(e.clientY - lastTap.y) < 12;
  const hit = near && (now - lastTap.t) < 320;
  lastTap.t = hit ? 0 : now;   // 命中后清零：连点三下只闭合一次
  lastTap.x = e.clientX;
  lastTap.y = e.clientY;
  return hit;
}

function onUp(e) {
  if (S.pointerId !== null && S.pointerId !== e.pointerId) return;
  S.pointerId = null;

  if (S.painting) {
    S.painting = false;
    commitBrush();
    draw();
    return;
  }
  if (S.boxDrag) {
    applyBoxShape();
    return;
  }
  // 多边形：双击闭合。鼠标有 detail 可用；触摸只能靠上面的手势判断。
  if (S.drawing && e.target === canvas) {
    if (e.detail === 2 || (e.pointerType === 'touch' && isDoubleTap(e))) finishPolygon();
  }
}

/**
 * 框选修形：把「拉出来的框」并进 / 挖掉 / 替换当前对象的轮廓。
 *
 * 【为什么不是让模型重新抠一次】
 * 这条路的起因是「沙发只圈了一半」。查下来的结论是：模型给的掩码本身就是
 * 右半边——官方 yolo26n-seg 与自训 best.pt 都只认到 x≈0.47，掩码是完整单块
 * （连通块 1 个，占 100%），不存在「转多边形丢了像素」。也试过 SAM 的框提示：
 * 框拉大一点它把小狗雕塑/毯子当成目标，点提示又只给单个坐垫。
 * 也就是说这是模型能力边界，再套一层模型只会引入新的不确定性。
 *
 * 所以这里做成**确定性**的操作：人给一个框（「这一块也属于它」），
 * 系统把框内的像素并进对象的位图掩码，再把掩码提回多边形。
 * 复用的是画笔已经验证过的栅格→多边形管线（commitBrush），
 * 撤销、修正记录、自动保存全都白拿，不用另写一套状态管理。
 */
function applyBoxShape() {
  const r = S.boxDrag;
  S.boxDrag = null;
  if (!r || !S.anno) { draw(); return; }

  const o = S.anno.objects.find(x => x.id === S.selectedId);
  if (!o) {
    toast('先选中一个对象（「选择」工具点它一下），再拉框', true);
    draw();
    return;
  }

  const x1 = Math.min(r.x0, r.x1), x2 = Math.max(r.x0, r.x1);
  const y1 = Math.min(r.y0, r.y1), y2 = Math.max(r.y0, r.y1);
  if (x2 - x1 < 0.004 || y2 - y1 < 0.004) {   // 手抖点了一下，不算操作
    toast('框太小了，按住拖一个框出来', true);
    draw();
    return;
  }

  if (!S.brushMask || S.brushMask.id !== o.id) buildBrushMask(o);
  const m = S.brushMask;
  const ctx = m.ctx;
  ctx.save();
  if (S.boxMode === 'replace') {
    // 整个轮廓换成这个框：先清空位图，再填满框
    ctx.globalCompositeOperation = 'source-over';
    ctx.clearRect(0, 0, m.w, m.h);
    ctx.fillStyle = '#fff';
  } else if (S.boxMode === 'cut') {
    ctx.globalCompositeOperation = 'destination-out';
    ctx.fillStyle = '#000';
  } else {
    ctx.globalCompositeOperation = 'source-over';
    ctx.fillStyle = '#fff';
  }
  ctx.fillRect(x1 * m.w, y1 * m.h, (x2 - x1) * m.w, (y2 - y1) * m.h);
  ctx.restore();

  // commitBrush 认的是 paintTarget，不是 selectedId
  S.paintTarget = o.id;
  commitBrush();
}

/** 框选过程中画橡皮筋 */
function drawBoxDrag() {
  const r = S.boxDrag;
  if (!r) return;
  const [x1, y1] = toCanvas(Math.min(r.x0, r.x1), Math.min(r.y0, r.y1));
  const [x2, y2] = toCanvas(Math.max(r.x0, r.x1), Math.max(r.y0, r.y1));
  ctx.save();
  ctx.setLineDash([6, 4]);
  ctx.lineWidth = 1.5;
  ctx.strokeStyle = S.boxMode === 'cut' ? '#f87171'
                  : S.boxMode === 'replace' ? '#fbbf24' : '#34d399';
  ctx.fillStyle = S.boxMode === 'cut' ? 'rgba(248,113,113,.12)'
                : S.boxMode === 'replace' ? 'rgba(251,191,36,.12)'
                : 'rgba(52,211,153,.12)';
  ctx.beginPath();
  ctx.rect(x1, y1, x2 - x1, y2 - y1);
  ctx.fill();
  ctx.stroke();
  ctx.restore();
}

/**
 * 画笔：位图掩码方案。
 *
 * 旧实现是「取凸包」—— 一涂就把物体的凹角填平、轮廓变方，越涂越不像原形。
 * 现在的做法是正经的位图编辑：
 *   1. 把目标物体的多边形栅格化成一张位图掩码
 *   2. 画笔在掩码上涂抹（加=涂白，擦=擦掉）
 *   3. 松手后把掩码重新提取成多边形
 * 这样凹角能保留，也能真的把多边形"擦掉一块"。
 */
function paintAt(nx, ny) {
  const o = S.anno.objects.find(x => x.id === S.paintTarget);
  if (!o) return;
  const m = S.brushMask;
  if (!m) return;

  // 归一化坐标 -> 掩码像素坐标
  const px = nx * m.w, py = ny * m.h;
  // 笔刷半径：brushSize 是「屏幕像素」，换算到掩码像素
  const r = S.brushSize / S.scale * (m.w / (S.anno.width || m.w));

  const ctx = m.ctx;
  ctx.save();
  if (S.tool === 'erase') {
    // 擦：把命中的像素变透明
    ctx.globalCompositeOperation = 'destination-out';
    ctx.fillStyle = '#000';
  } else {
    ctx.globalCompositeOperation = 'source-over';
    ctx.fillStyle = '#fff';
  }
  ctx.beginPath();
  ctx.arc(px, py, Math.max(1, r), 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();

  draw();
}

/** 把掩码的当前状态提取成多边形并写回对象 */
function commitBrush() {
  const o = S.anno.objects.find(x => x.id === S.paintTarget);
  const m = S.brushMask;
  if (!o || !m) return;

  const img = m.ctx.getImageData(0, 0, m.w, m.h);
  const poly = Geo.maskToPolygon(img.data, m.w, m.h, {
    eps: 1.0,
    maxPts: 180,
    scaleX: m.w,          // 掩码像素 -> 归一化（x/y 分别缩放，图未必是正方形）
    scaleY: m.h,
  });

  if (!poly || poly.length < 3) {
    toast('涂抹后形状为空，已放弃这次修改', true);
    S.brushMask = null;
    draw();
    return;
  }

  // 撤销点：涂抹之前的整图状态。
  // 必须放在写回 polygon 之前，快照里存的才是涂抹前的形状。
  pushUndo();
  o.polygon = poly;
  o.edited = true;
  S.dirty = true;
  // 测试模式下不写修正流水、不落盘
  if (window.__UITEST__) {
    S.brushMask = null;
    renderObjects();
    draw();
    return;
  }
  const existed = S.edits.find(e => e.object_id === o.id);
  if (!existed) {
    S.edits.push({ object_id: o.id, class: o.class, type: '修正' });
    renderEditLog();
  }
  S.origById[o.id] = JSON.stringify({ id: o.id, class: o.class,
                                     polygon: o.polygon, source: o.source,
                                     edited: true });
  S.brushMask = null;
  renderObjects();
  saveSoon();
  toast('已更新轮廓（' + poly.length + ' 个顶点）');
}

/** 为目标对象建立一张位图掩码（把现有多边形栅格化进去） */
function buildBrushMask(o) {
  const w = Math.max(16, Math.round(S.anno.width || 960));
  const h = Math.max(16, Math.round(S.anno.height || 720));
  // 掩码不需要原图那么精细，取一个能保证形状的下采样倍率即可
  const k = Math.min(1, 700 / Math.max(w, h));
  const mw = Math.max(8, Math.round(w * k));
  const mh = Math.max(8, Math.round(h * k));

  const cv = document.createElement('canvas');
  cv.width = mw; cv.height = mh;
  const cx = cv.getContext('2d');
  cx.fillStyle = '#fff';
  cx.beginPath();
  o.polygon.forEach((p, i) => {
    const x = p[0] * mw, y = p[1] * mh;
    i === 0 ? cx.moveTo(x, y) : cx.lineTo(x, y);
  });
  cx.closePath();
  cx.fill();

  S.brushMask = { id: o.id, cv, ctx: cx, w: mw, h: mh, k };
}

/** 闭合多边形：问用户类别，然后加入对象列表。 */
function finishPolygon() {
  if (S.draftPts.length < 3) {
    // 【原来这里静默return，用户看不出为什么没反应】
    // 少于 3 个点无法构成多边形，必须明确告知，否则像是工具坏了。
    toast('至少需要 3 个点才能构成多边形，当前只有 '
          + S.draftPts.length + ' 个', true);
    S.drawing = false; S.draftPts = []; syncPolyHint(); draw();
    return;
  }
  const poly = S.draftPts.slice();
  S.drawing = false; S.draftPts = [];
  syncPolyHint();
  showModal('新建标注对象', `
    <p class="muted small" style="margin-bottom:10px">选择这个对象的类别（共 ${poly.length} 个顶点）：</p>
    <div id="clsPick" style="display:grid;grid-template-columns:repeat(2,1fr);gap:6px">
      ${S.classes.map(c => `
        <button class="btn" data-cls="${c.name}" style="text-align:left;display:flex;align-items:center;gap:7px;height:32px">
          <span class="obj-swatch" style="background:${c.color}"></span>${c.name}
        </button>`).join('')}
    </div>
    <div style="margin-top:14px"><button class="btn btn-ghost" id="btnCancelPoly">取消</button></div>
  `);
  document.querySelectorAll('#clsPick button').forEach(b => {
    b.onclick = () => {
      addObject(b.dataset.cls, poly, 'human');
      $('modal').classList.add('hidden');
      saveSoon();
      toast('已新增 ' + b.dataset.cls);
    };
  });
  $('btnCancelPoly').onclick = () => $('modal').classList.add('hidden');
}

// ---------------------------------------------------------------- 上传
/* ==========================================================================
   上传照片 → 自动预标注 → 直接开始标注
   --------------------------------------------------------------------------
   【为什么一次只传一张】
   fetch 没有上传进度事件。一次多张打成一个包，进度条只能一直转，
   用户不知道卡在哪。逐张串行上传，每传完一张进度条走一格，
   而且某张坏了不影响其余张（单张失败只记进 skipped）。
   ========================================================================== */

const UPLOAD_EXT = /\.(jpe?g|png|bmp|webp)$/i;

function pickFiles() {
  const inp = $('fileInput');
  // 先清空：否则连续两次选同一个文件时不会触发 change
  inp.value = '';
  inp.click();
}

function isImageFile(f) {
  return (f.type && f.type.startsWith('image/')) || UPLOAD_EXT.test(f.name || '');
}

async function uploadFiles(fileList) {
  const files = Array.from(fileList || []);
  if (!files.length) return;

  const imgs = files.filter(isImageFile);
  const notImage = files.length - imgs.length;
  if (!imgs.length) {
    toast('选中的文件不是图片（支持 jpg / png / bmp / webp）', true);
    return;
  }

  const btn = $('btnUpload');
  const btnHTML = btn.innerHTML;
  btn.disabled = true;
  btn.classList.add('busy');
  btn.textContent = '上传中…';

  $('progressWrap').classList.remove('hidden');
  $('progressFill').style.width = '0%';
  $('progressText').textContent = '准备上传…';

  const saved = [];
  const skipped = [];
  try {
    for (let i = 0; i < imgs.length; i++) {
      const f = imgs[i];
      $('progressText').textContent = `上传 ${i + 1}/${imgs.length}  ${f.name}`;
      const fd = new FormData();
      fd.append('file', f);
      try {
        const r = await api('/api/upload', { method: 'POST', body: fd });
        saved.push(...(r.saved || []));
        skipped.push(...(r.skipped || []));
      } catch (e) {
        skipped.push({ name: f.name, reason: e.message });
      }
      $('progressFill').style.width =
        Math.round((i + 1) / imgs.length * 100) + '%';
    }
  } finally {
    btn.disabled = false;
    btn.classList.remove('busy');
    btn.innerHTML = btnHTML;
  }

  await loadImages();

  if (!saved.length) {
    $('progressText').textContent = '上传失败';
    toast('没有图片上传成功：'
          + (skipped[0] ? skipped[0].reason : '未知原因'), true);
    return;
  }

  const names = saved.map(s => s.name);
  let msg = `已上传 ${saved.length} 张`;
  if (notImage) msg += `，忽略 ${notImage} 个非图片文件`;
  $('progressText').textContent = msg;

  if (skipped.length) {
    toast(`${saved.length} 张成功，${skipped.length} 张跳过：`
          + skipped[0].reason, true);
  }

  // 一键流程：上传完直接预标注，然后自动打开第一张新图开始改
  if ($('autoPrelabel').checked) {
    await startPrelabel({ images: names, pickAfter: names[0] });
  } else {
    toast(msg + '，点「预标注」生成草稿');
    await selectImage(names[0]);
  }
}

/** 拖拽：整个窗口都能接图片 */
function bindDropUpload() {
  const mask = $('dropMask');
  let depth = 0;   // dragenter/dragleave 会在子元素间反复触发，用计数抵消

  const hasFiles = e =>
    e.dataTransfer && Array.from(e.dataTransfer.types || []).includes('Files');

  window.addEventListener('dragenter', e => {
    if (!hasFiles(e)) return;
    e.preventDefault();
    depth++;
    mask.classList.remove('hidden');
  });

  window.addEventListener('dragover', e => {
    if (!hasFiles(e)) return;
    e.preventDefault();          // 不阻止的话浏览器会直接打开这张图
    e.dataTransfer.dropEffect = 'copy';
  });

  window.addEventListener('dragleave', e => {
    if (!hasFiles(e)) return;
    depth = Math.max(0, depth - 1);
    if (!depth) mask.classList.add('hidden');
  });

  window.addEventListener('drop', e => {
    if (!hasFiles(e)) return;
    e.preventDefault();
    depth = 0;
    mask.classList.add('hidden');
    uploadFiles(e.dataTransfer.files);
  });
}

// ---------------------------------------------------------------- 预标注
/** 还没有标注文件的图（列表里状态显示「—」的那些）。 */
function pendingDrafts() {
  return S.images.filter(i => !i.reviewed).map(i => i.name);
}

async function startPrelabel(opts) {
  const o = opts || {};
  if (!S.images.length) { toast('还没有图片，先上传照片', true); return; }

  // 显式指定（上传后自动预标注）→ 只处理这几张；
  // 否则 → 只处理「还没有标注文件」的图。
  //
  // 【为什么默认不是全部】
  // 用户点「预标注」的本意是「给我刚传的图配上草稿」，不是「把 150 张
  // （里面有 60 条人工修正）全部重做一遍」。旧版一律全量，一次误点就把修正
  // 冲掉了（见 ai_log 第十六轮）。按钮上会显示待标张数，点之前就知道范围。
  const given = Array.isArray(o.images) && o.images.length ? o.images : null;
  const pending = pendingDrafts();
  const only = given || (pending.length ? pending : null);
  const editedTotal = S.images.reduce((s, i) => s + (i.edited || 0), 0);

  if (!o.silent) {
    let msg;
    if (only) {
      const head = only.slice(0, 3).join('、')
                 + (only.length > 3 ? ` 等 ${only.length} 张` : '');
      msg = `将用官方预训练权重对 ${only.length} 张图生成草稿：${head}\n\n`
          + `其余 ${S.images.length - only.length} 张已有标注的图`
          + (editedTotal ? `（含 ${editedTotal} 条人工修正）` : '')
          + '完全不动。\n\n';
    } else if (editedTotal > 0) {
      msg = `全部 ${S.images.length} 张图都已有标注，其中 ${editedTotal} 条人工修正。\n\n`
          + '⚠ 重做全部会重写每一个标注文件，这些修正会被清除，且无法撤销。\n'
          + '（records/修正记录.csv 的历史记录还在，'
          + '但图上的修改会回到模型原始输出）\n\n';
    } else {
      msg = `全部 ${S.images.length} 张图都已有标注。\n\n`
          + '重做全部会覆盖现有标注文件（当前没有人工修正，只是回到模型草稿）。\n\n';
    }
    const ask = only ? '确定继续？' : '确定对全部图片重新生成草稿吗？';
    if (!confirm(msg + ask)) return;
  }

  $('progressWrap').classList.remove('hidden');
  try {
    const r = await api('/api/prelabel/start', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      // force 只在「全量重做、且用户已在弹窗确认」时才带；服务端靠它放行覆盖闸门
      body: JSON.stringify({ model: 'official', images: only, force: !only }),
    });
    pollProgress(r.job_id, 'official', o.pickAfter || null);
  } catch (e) { toast('启动失败: ' + e.message, true); }
}

async function startRelabel() {
  if (!S.images.length) { toast('没有图片', true); return; }
  // 回标会整体覆盖标注文件，人工修正会被清掉。
  // 已经有修正记录时，提示要说清楚"记录已导出但标注会被重置"，
  // 避免用户辛苦改了 60 条之后点错一下全没了。
  const edited = S.images.reduce((s, i) => s + (i.edited || 0), 0);
  let msg = '将用自训权重 model/best.pt 对全部图片回标。\n已有的人工修正会被覆盖，确定继续？';
  if (edited > 0) {
    msg = '⚠ 当前已有 ' + edited + ' 条人工修正。\n\n'
        + '回标会重置全部标注文件，这些修正将被清除。\n'
        + '（如果已导出 records/修正记录.csv，记录本身不会丢，'
        + '但标注文件里的 edited 标记会没了）\n\n'
        + '确定要继续吗？';
  }
  if (!confirm(msg)) return;
  $('progressWrap').classList.remove('hidden');
  try {
    const r = await api('/api/prelabel/start', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      // 上面的 confirm 已经明确告知会覆盖修正，这里带 force 让服务端放行
      body: JSON.stringify({ model: 'custom', force: true }),
    });
    pollProgress(r.job_id, 'custom');
  } catch (e) { toast('启动失败: ' + e.message, true); }
}

function pollProgress(jobId, mode, pickAfter) {
  const timer = setInterval(async () => {
    try {
      const p = await api('/api/prelabel/progress/' + jobId);
      $('progressFill').style.width = p.percent + '%';
      $('progressText').textContent =
        `${p.done}/${p.total} ${p.percent}%  ${p.current}`;
      if (p.status === 'done' || p.status === 'failed') {
        clearInterval(timer);
        $('progressText').textContent = p.status === 'done'
          ? `完成，共 ${p.total} 张` : p.message;
        if (p.status === 'done') {
          toast(mode === 'custom' ? '回标完成' : '预标注完成，可以开始改了');
          await loadImages();
          // 上传流程会指定 pickAfter：直接跳到刚标好的新图，省一次点击
          const target = pickAfter || S.current;
          if (target) {
            S.dirty = false;   // 草稿刚生成，没有待保存的修改
            await selectImage(target);
          }
        }
      }
    } catch (e) { clearInterval(timer); }
  }, 800);
}

// ---------------------------------------------------------------- 导出
async function doExport() {
  if (!S.images.length) { toast('没有图片', true); return; }
  toast('正在导出，请稍候…');
  try {
    const r = await api('/api/export', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ val_ratio: 0.2, has_seg: true, copy_images: true }),
    });
    const s = r.stats, v = r.validation;
    showModal('导出结果', `
      <div class="kv">
        <dt>训练图片</dt><dd>${s.train_images}</dd>
        <dt>验证图片</dt><dd>${s.val_images}</dd>
        <dt>标注对象</dt><dd>${s.total_objects}</dd>
        <dt>含标注的图片</dt><dd>${s.images_with_label}</dd>
        <dt>格式校验</dt><dd>${v.passed
          ? '<span class="ok-badge">通过</span>'
          : '<span class="err-badge">有问题</span>'}</dd>
      </div>
      <div class="mono" style="font-size:11px;color:var(--text-2);margin-bottom:12px">
        ${v.report.replace(/\n/g, '<br>')}
      </div>
      ${v.problems.length ? `<div style="background:#fef2f2;padding:10px;border-radius:6px;font-size:12px">
        ${v.problems.map(p => '· ' + p).join('<br>')}</div>` : ''}
      <h3 style="font-size:13px;margin:14px 0 8px">各类别对象数</h3>
      <div class="mono" style="font-size:12px">
        ${Object.entries(s.class_counts).filter(([k,v]) => v > 0)
          .map(([k,v]) => `${k}: ${v}`).join('　') || '（无）'}
      </div>
      <div style="margin-top:14px"><button class="btn btn-ghost" onclick="document.getElementById('modal').classList.add('hidden')">关闭</button></div>
    `);
  } catch (e) { toast('导出失败: ' + e.message, true); }
}

// ---------------------------------------------------------------- 训练
async function doTrain() {
  if (!confirm('开始训练？这会覆盖 model/best.pt。')) return;
  try {
    await api('/api/train', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({}),
    });
    toast('训练已在后台启动，可在终端查看日志');
    setTimeout(showTrainLog, 4000);
  } catch (e) { toast('启动失败: ' + e.message, true); }
}

async function showTrainLog() {
  try {
    const r = await api('/api/train/log');
    showModal('训练日志', `<pre class="mono" style="font-size:11px;white-space:pre-wrap;line-height:1.6">${r.log || '（暂无日志）'}</pre>
      <div style="margin-top:12px"><button class="btn btn-ghost" onclick="document.getElementById('modal').classList.add('hidden')">关闭</button></div>`);
  } catch (e) {}
}

// ---------------------------------------------------------------- 对比/记录
async function showCompare() {
  const r = await api('/api/comparison');
  const rounds = r.rounds || {};
  const keys = Object.keys(rounds).sort();
  showModal('各轮对比结果', `
    <table class="cmp">
      <tr><th>轮次</th><th>模型</th><th>总对象</th><th>修正</th><th>修正率</th><th>mAP50</th></tr>
      ${keys.map(k => {
        const v = rounds[k];
        return `<tr><td>${k}</td><td>${v.model || ''}</td>
          <td class="num">${v.total_objects}</td><td class="num">${v.edited_objects}</td>
          <td class="num">${v.correction_rate}%</td><td class="num">${v.map50 || '—'}</td></tr>`;
      }).join('') || '<tr><td colspan="6" class="muted">还没有轮次数据</td></tr>'}
    </table>
    <h3 style="font-size:13px;margin:16px 0 8px">操作流水</h3>
    <div style="max-height:200px;overflow:auto">
      ${(r.logs || []).slice().reverse().map(l =>
        `<div class="log-line">${l.time}　${l.action}　${l.detail}</div>`).join('')
        || '<div class="muted small">暂无</div>'}
    </div>
    <div style="margin-top:14px"><button class="btn btn-ghost" onclick="document.getElementById('modal').classList.add('hidden')">关闭</button></div>
  `);
}

async function showRecords() {
  const r = await api('/api/records');
  showModal('修正记录', `
    <div class="kv">
      <dt>总对象数</dt><dd>${r.total_objects}</dd>
      <dt>已修正</dt><dd>${r.edited_objects} / 60</dd>
      <dt>修正率</dt><dd>${r.correction_rate}%</dd>
    </div>
    <h3 style="font-size:13px;margin:10px 0 8px">各类别修正情况</h3>
    <table class="cmp">
      <tr><th>类别</th><th>总对象</th><th>已修正</th><th>修正率</th></tr>
      ${Object.entries(r.per_class).map(([k, v]) => `<tr>
        <td>${k}</td><td class="num">${v.total}</td><td class="num">${v.edited}</td>
        <td class="num">${v.total ? (v.edited / v.total * 100).toFixed(0) : 0}%</td></tr>`).join('')
        || '<tr><td colspan="4" class="muted">暂无</td></tr>'}
    </table>
    <div style="margin-top:14px;display:flex;gap:8px">
      <button class="btn" id="btnExportRec">导出 CSV</button>
      <button class="btn btn-ghost" onclick="document.getElementById('modal').classList.add('hidden')">关闭</button>
    </div>
  `);
  $('btnExportRec').onclick = async () => {
    try {
      const d = await api('/api/records/export', { method: 'POST' });
      toast('已导出 ' + d.records + ' 条修正记录到 records/');
    } catch (e) { toast('导出失败', true); }
  };
}

// ---------------------------------------------------------------- 弹层
function showModal(title, html) {
  $('modalTitle').textContent = title;
  $('modalBody').innerHTML = html;
  $('modal').classList.remove('hidden');
}

// 键盘快捷键
document.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
  if (e.key === 'Delete' || e.key === 'Backspace') { deleteSelected(); }
  // 撤销 / 重做。
  // 用e.key.toLowerCase 兼容 Caps Lock；但 Ctrl+Y 也是重做的通用键。
  if ((e.ctrlKey || e.metaKey) && !e.shiftKey) {
    const k = e.key.toLowerCase();
    if (k === 'z') { e.preventDefault(); undo(); return; }
    if (k === 'y') { e.preventDefault(); redo(); return; }
  }
  if ((e.ctrlKey || e.metaKey) && e.shiftKey &&
      e.key.toLowerCase() === 'z') { e.preventDefault(); redo(); return; }
  if (e.key === 'Escape') { S.drawing = false; S.draftPts = []; syncPolyHint(); draw(); }
  if (e.key === 'Enter' && S.drawing) { finishPolygon(); }
});

init();

// 暴露内部状态给自动化测试和调试用。
// （顶层 const 不会挂到 window 上，这里显式导出）
window.__APP__ = {
  get S() { return S; },
  get saveTimer() { return saveTimer; },
  draw, buildBrushMask, visibleObjects, isVisible, toCanvas, toNorm,
  renderObjects, renderProps, selectImage, hitTest, commitBrush,
  // 撤销栈：自动化测试要直接触发并检查栈深，所以一并导出。
  undo, redo, pushUndo, addObject, deleteSelected, changeClass,
  saveSoon, saveNow,
  snapshot: snapshotAnno, applySnapshot,
  // 上传：测试要验证「选完文件后状态机怎么走」，所以把入口导出去
  uploadFiles, pickFiles,
  // 删除：测试要验证删除后画布是否清空、列表是否同步
  deleteImage, restoreImages, clearStage, renderImageList, renderTrashBar,
  // 框选修形：测试要验证三种模式对轮廓的影响
  applyBoxShape, syncBoxHint,
  // 预标注范围：测试要验证「只标没有标注的图」这条规则与按钮上的数字
  pendingDrafts, syncPrelabelBtn,
  // 窄屏抽屉：测试要验证「开一边自动收另一边」和遮罩点击关闭
  toggleDrawer, closeDrawers,
};

// 自动化测试脚本（仅在带 ?uitest=1 时按需加载）
if (location.search.includes('uitest=1')) {
  // 测试模式下禁止落盘：UI 测试会伪造对象来验证交互，
  // 如果让它写进真实标注文件，就会污染项目数据。
  window.__UITEST__ = true;
  const s = document.createElement('script');
  s.src = '/static/ui_test_script.js?t=' + Date.now();
  document.body.appendChild(s);
  // 演示截图脚本：拍交互态（聚焦模式、筛选效果）时用。
  // 这个文件由 tools/shoot_demo.py 在截图前临时生成，不进版本库，
  // 所以用 onerror 静默忽略（404 不影响主功能）。
  const d = document.createElement('script');
  d.src = '/static/demo_script.js';
  d.onerror = () => {};
  document.body.appendChild(d);

  // 描边专项截图脚本（tools/shoot_stroke.js 生成，同样是临时的）
  const st = document.createElement('script');
  st.src = '/static/stroke_script.js';
  st.onerror = () => {};
  document.body.appendChild(st);
}
