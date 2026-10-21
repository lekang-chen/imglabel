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
  current: null,          // 当前图片名
  anno: null,             // 当前标注 {image,width,height,objects:[]}
  mode: 'seg',            // seg | det | sem
  tool: 'select',         // select | polygon | brush | erase
  selectedId: null,       // 选中的对象 id
  scale: 1,               // 缩放比例
  offsetX: 0, offsetY: 0, // 画布偏移
  showMask: true, showBox: false, showLabel: true,
  brushSize: 18,
  drawing: false,         // 多边形绘制中
  draftPts: [],           // 多边形草稿点
  painting: false,        // 画笔涂抹中
  paintTarget: null,      // 画笔作用的对象
  img: null,              // Image 对象
  dirty: false,           // 有未保存修改
  edits: [],              // 本图的修正记录
  origById: {},           // 初始状态快照，用于判断哪里被改了
};

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
  $('imgCount').textContent = d.total;
  $('statReviewed').textContent = `${d.reviewed} / ${d.total}`;
  $('statEdited').textContent = `${d.edited_total} / 60`;
  renderImageList();
  await refreshRate();
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
    div.innerHTML = `<span class="img-name">${it.name}</span>` +
      `<span class="img-badge">${it.reviewed ? (it.edited ? '改' + it.edited : '已审') : (it.total ? '待审' : '—')}</span>`;
    div.onclick = () => selectImage(it.name);
    box.appendChild(div);
  });
}

// ---------------------------------------------------------------- 选图
async function selectImage(name) {
  if (S.dirty && !confirm('当前图片有未保存的修改，确定切换？')) return;
  S.current = name;
  S.edits = [];
  S.selectedId = null;
  S.draftPts = [];
  S.drawing = false;

  $('emptyHint').classList.add('hidden');

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

  const objs = S.anno.objects;

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
}

function drawObject(o) {
  const col = S.colorMap[o.class] || '#94a3b8';
  const sel = o.id === S.selectedId;
  const pts = o.polygon.map(p => toCanvas(p[0], p[1]));

  ctx.save();
  if (S.mode === 'det') {
    // 检测模式：只画外接矩形
    const xs = pts.map(p => p[0]), ys = pts.map(p => p[1]);
    const x = Math.min(...xs), y = Math.min(...ys);
    const w = Math.max(...xs) - x, h = Math.max(...ys) - y;
    ctx.strokeStyle = col;
    ctx.lineWidth = sel ? 2.5 : 1.5;
    ctx.strokeRect(x, y, w, h);
    if (S.showLabel) drawLabel(o, x, y, col);
  } else {
    // 掩码模式：填充多边形
    ctx.beginPath();
    pts.forEach((p, i) => i === 0 ? ctx.moveTo(p[0], p[1]) : ctx.lineTo(p[0], p[1]));
    ctx.closePath();
    ctx.fillStyle = hexA(col, sel ? 0.38 : 0.26);
    ctx.fill();
    ctx.strokeStyle = col;
    ctx.lineWidth = sel ? 2.5 : 1.2;
    ctx.stroke();

    if (S.showBox && sel) {
      const xs = pts.map(p => p[0]), ys = pts.map(p => p[1]);
      ctx.setLineDash([3, 3]);
      ctx.strokeStyle = 'rgba(255,255,255,.55)';
      ctx.lineWidth = 1;
      ctx.strokeRect(Math.min(...xs), Math.min(...ys),
                     Math.max(...xs) - Math.min(...xs),
                     Math.max(...ys) - Math.min(...ys));
      ctx.setLineDash([]);
    }
    if (S.showLabel) {
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

// ---------------------------------------------------------------- 命中检测
function hitTest(nx, ny) {
  // 从后往前找（后画的在上面）
  for (let i = S.anno.objects.length - 1; i >= 0; i--) {
    const o = S.anno.objects[i];
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
  o.class = newCls;
  markEdited(o.id);
  S.edits[S.edits.length - 1].type = '类别';
  renderObjects(); renderProps(); renderEditLog();
  saveSoon();
}

// ---------------------------------------------------------------- 渲染面板
function renderObjects() {
  const box = $('objList');
  $('objCount').textContent = S.anno ? S.anno.objects.length : 0;
  box.innerHTML = '';
  if (!S.anno || !S.anno.objects.length) {
    box.innerHTML = '<p class="muted small" style="padding:8px">没有标注对象</p>';
    return;
  }
  S.anno.objects.forEach(o => {
    const div = document.createElement('div');
    div.className = 'obj-item';
    if (o.id === S.selectedId) div.classList.add('active');
    div.innerHTML =
      `<span class="obj-swatch" style="background:${S.colorMap[o.class]||'#94a3b8'}"></span>` +
      `<span class="obj-name">${o.class}</span>` +
      `<span class="obj-tag ${o.edited ? 'edited' : ''}">${o.edited ? '已改' : (o.source === 'model' ? '草稿' : '人工')}</span>`;
    div.onclick = () => { S.selectedId = o.id; renderObjects(); renderProps(); draw(); };
    box.appendChild(div);
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
  if (!S.current || !S.anno) return;
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
      S.tool = b.dataset.tool;
      canvas.className = S.tool === 'select' ? 'tool-select' : '';
      if (S.tool !== 'polygon') { S.drawing = false; S.draftPts = []; }
      if (S.tool === 'brush' || S.tool === 'erase') {
        if (!S.selectedId) toast('先用「选择」工具点一个对象，再来涂抹');
      }
      draw();
    };
  });

  // 画布交互
  canvas.onmousedown = onDown;
  canvas.onmousemove = onMove;
  canvas.onmouseup = onUp;
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
  $('btnPrelabel').onclick = startPrelabel;
  $('btnRelabel').onclick = startRelabel;
  $('btnExport').onclick = doExport;
  $('btnTrain').onclick = doTrain;
  $('btnCompare').onclick = showCompare;
  $('btnRecords').onclick = showRecords;

  $('modalClose').onclick = () => $('modal').classList.add('hidden');
  window.onresize = () => { fitCanvas(); draw(); };
  window.onbeforeunload = e => { if (S.dirty) { e.preventDefault(); e.returnValue = ''; } };
}

// ---------------------------------------------------------------- 画布事件
function canvasPos(e) {
  const r = canvas.getBoundingClientRect();
  return [e.clientX - r.left, e.clientY - r.top];
}

function onDown(e) {
  if (!S.anno) return;
  const [cx, cy] = canvasPos(e);
  const [nx, ny] = toNorm(cx, cy);

  if (S.tool === 'polygon') {
    if (e.button === 2) {           // 右键：退回上一个点
      S.draftPts.pop(); draw(); return;
    }
    if (!S.drawing) { S.drawing = true; S.draftPts = []; }
    S.draftPts.push([nx, ny]);
    draw();
    return;
  }

  if (S.tool === 'select') {
    const hit = hitTest(nx, ny);
    S.selectedId = hit ? hit.id : null;
    renderObjects(); renderProps(); draw();
    return;
  }

  if (S.tool === 'brush' || S.tool === 'erase') {
    if (!S.selectedId) { toast('先选中一个对象再涂抹'); return; }
    S.painting = true;
    S.paintTarget = S.selectedId;
    paintAt(nx, ny);
  }
}

function onMove(e) {
  if (S.painting) {
    const [cx, cy] = canvasPos(e);
    const [nx, ny] = toNorm(cx, cy);
    paintAt(nx, ny);
  }
}

function onUp(e) {
  if (S.painting) {
    S.painting = false;
    const o = S.anno.objects.find(x => x.id === S.paintTarget);
    if (o) { markEdited(o.id); saveSoon(); renderObjects(); }
    draw();
  }
  // 多边形：双击闭合
  if (S.drawing && e.detail === 2) finishPolygon();
}

/**
 * 画笔涂抹。
 * 做法：在目标对象的多边形上，把笔刷覆盖到的像素加进多边形。
 * 简化实现：以笔刷为圆心，取圆周上的点做「凸包合并」——
 * 实际项目里通常用位图 mask 栅格化后转轮廓，这里为了前端轻量，
 * 采用「圆周采样 + 轮廓合并」的近似方式。
 */
function paintAt(nx, ny) {
  const o = S.anno.objects.find(x => x.id === S.paintTarget);
  if (!o) return;
  const r = S.brushSize / S.img.width / S.scale;

  if (S.tool === 'erase') {
    // 橡皮：把落在笔刷内的多边形顶点去掉
    o.polygon = o.polygon.filter(p =>
      Math.hypot(p[0] - nx, p[1] - ny) > r);
    if (o.polygon.length < 3) o.polygon = o.polygon.slice(0, 3);
  } else {
    // 画笔：把笔刷圆周上的点并入多边形
    const pts = [];
    const N = 12;
    for (let i = 0; i < N; i++) {
      const a = (i / N) * Math.PI * 2;
      pts.push([nx + r * Math.cos(a), ny + r * Math.sin(a)]);
    }
    o.polygon = mergePolygon(o.polygon, pts);
  }
  draw();
}

/** 把 addPts 合并进 polygon（凸包 + 简化），实现「涂抹即扩张」。 */
function mergePolygon(poly, addPts) {
  const all = poly.concat(addPts);
  // 凸包
  const pts = all.slice().sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const cross = (o, a, b) => (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0]);
  const lower = [];
  for (const p of pts) {
    while (lower.length >= 2 && cross(lower[lower.length-2], lower[lower.length-1], p) <= 0) lower.pop();
    lower.push(p);
  }
  const upper = [];
  for (let i = pts.length - 1; i >= 0; i--) {
    const p = pts[i];
    while (upper.length >= 2 && cross(upper[upper.length-2], upper[upper.length-1], p) <= 0) upper.pop();
    upper.push(p);
  }
  const hull = lower.slice(0, -1).concat(upper.slice(0, -1));
  // 抽稀
  const simple = [];
  for (let i = 0; i < hull.length; i++) {
    const prev = hull[(i - 1 + hull.length) % hull.length];
    const cur = hull[i];
    const next = hull[(i + 1) % hull.length];
    const c = Math.abs((cur[0]-prev[0])*(next[1]-prev[1]) - (cur[1]-prev[1])*(next[0]-prev[0]));
    if (c > 1e-5) simple.push(cur);
  }
  return simple.length >= 3 ? simple : hull;
}

/** 闭合多边形：问用户类别，然后加入对象列表。 */
function finishPolygon() {
  if (S.draftPts.length < 3) {
    S.drawing = false; S.draftPts = []; draw();
    return;
  }
  const poly = S.draftPts.slice();
  S.drawing = false; S.draftPts = [];
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

// ---------------------------------------------------------------- 预标注
async function startPrelabel() {
  if (!S.images.length) { toast('data/images/ 里还没有图片', true); return; }
  if (!confirm('将用官方预训练权重对全部图片生成草稿。\n已有的人工修正会被覆盖，确定继续？')) return;
  $('progressWrap').classList.remove('hidden');
  try {
    const r = await api('/api/prelabel/start', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: 'official' }),
    });
    pollProgress(r.job_id, 'official');
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
      body: JSON.stringify({ model: 'custom' }),
    });
    pollProgress(r.job_id, 'custom');
  } catch (e) { toast('启动失败: ' + e.message, true); }
}

function pollProgress(jobId, mode) {
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
          toast(mode === 'custom' ? '回标完成' : '预标注完成');
          await loadImages();
          if (S.current) await selectImage(S.current);
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
  if (e.key === 'Escape') { S.drawing = false; S.draftPts = []; draw(); }
  if (e.key === 'Enter' && S.drawing) { finishPolygon(); }
});

init();
