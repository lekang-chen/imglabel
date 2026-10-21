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
  unchecked: new Set(),    // 复选框取消勾选的对象 id（勾选 = 显示）
  focusMode: true,         // 聚焦：画布只显示当前选中的对象
  hoverId: null,           // 鼠标悬停的对象 id
  brushSize: 18,
  brushMask: null,         // 画笔位图掩码 {id, cv, cx, w, h, k}
  scratch: null,           // 画笔着色暂存画布
  mouse: { x: 0, y: 0, in: false },
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
  // 换图时重置可见性状态：新图默认全部显示
  S.unchecked = new Set();
  S.focusMode = true;
  S.hoverId = null;
  S.brushMask = null;
  syncFocusBtn();

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
  if (!S.anno) { $('objCount').textContent = '0'; box.innerHTML = ''; return; }
  const total = S.anno.objects.length;
  $('objCount').textContent = total ? `${visibleObjects().length}/${total}` : '0';
  box.innerHTML = '';
  if (!total) {
    box.innerHTML = '<p class="muted small" style="padding:8px">没有标注对象</p>';
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
  S.mouse.x = cx; S.mouse.y = cy; S.mouse.in = true;

  if (S.painting) {
    const [nx, ny] = toNorm(cx, cy);
    paintAt(nx, ny);
    return;
  }

  // 悬停高亮：让用户知道鼠标底下是哪个对象
  if (S.tool === 'select' && S.anno) {
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

function onUp(e) {
  if (S.painting) {
    S.painting = false;
    commitBrush();
    draw();
    return;
  }
  // 多边形：双击闭合
  if (S.drawing && e.detail === 2) finishPolygon();
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

// 暴露内部状态给自动化测试和调试用。
// （顶层 const 不会挂到 window 上，这里显式导出）
window.__APP__ = {
  get S() { return S; },
  draw, buildBrushMask, visibleObjects, isVisible, toCanvas, toNorm,
  renderObjects, selectImage, hitTest, commitBrush,
};

// 自动化测试脚本（仅在带 ?uitest=1 时按需加载）
if (location.search.includes('uitest=1')) {
  // 测试模式下禁止落盘：UI 测试会伪造对象来验证交互，
  // 如果让它写进真实标注文件，就会污染项目数据。
  window.__UITEST__ = true;
  const s = document.createElement('script');
  s.src = '/static/ui_test_script.js';
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
