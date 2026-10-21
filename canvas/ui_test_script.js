/* 在页面内执行的 UI 测试脚本（被 ui_test.js 注入到 iframe 父页）
   需要主页面已加载完，并通过 __runUiTests(d, w) 调用。
   模拟真实鼠标事件，验证：
     T1  复选框能隐藏/显示对象
     T2  点对象自动进入聚焦（其它对象不显示）
     T3  全选 / 全不选 / 仅看选中 三个按钮
     T4  选中对象用轮廓描边，不画矩形外框
     T5  画笔：位图掩码涂抹后，多边形真的变了且凹角没被填平
     T6  橡皮：能擦掉一块
*/
(function () {
  'use strict';

  function fire(w, d, el, type, opts) {
    const r = el.getBoundingClientRect();
    const o = Object.assign({ bubbles: true, cancelable: true, view: w,
      clientX: r.left + r.width / 2, clientY: r.top + r.height / 2 }, opts || {});
    el.dispatchEvent(new w.MouseEvent(type, o));
  }

  function polyArea(poly) {
    let t = 0;
    for (let i = 0; i < poly.length; i++) {
      const [x1, y1] = poly[i], [x2, y2] = poly[(i + 1) % poly.length];
      t += x1 * y2 - x2 * y1;
    }
    return Math.abs(t / 2);
  }

  window.__runUiTests = async function (d, w) {
    const log = [];
    let pass = 0, fail = 0;
    function ok(name, cond, extra) {
      if (cond) { pass++; log.push('  PASS  ' + name); }
      else { fail++; log.push('  FAIL  ' + name + (extra ? '  -> ' + extra : '')); }
    }

    // 等主页面初始化完
    for (let i = 0; i < 80; i++) {
      if (w.__APP__ && w.__APP__.S && w.__APP__.S.images &&
          w.__APP__.S.images.length) break;
      await new Promise(r => setTimeout(r, 250));
    }
    const A = w.__APP__;
    if (!A) return 'FATAL: __APP__ 未暴露';
    if (!A.S) return 'FATAL: A.S 未定义';
    if (!A.S.images || !A.S.images.length)
      return 'FATAL: 图片列表为空';

    // 挑一张有标注对象的图（第一张可能是空的），并等它加载完
    if (!A.S.anno || !A.S.anno.objects || !A.S.anno.objects.length) {
      const withObj = A.S.images.find(im => im.total > 0);
      if (withObj) {
        try { await A.selectImage(withObj.name); }
        catch (e) { log.push('selectImage 抛错: ' + (e && e.stack || e)); }
        await new Promise(r => setTimeout(r, 1000));
      }
    }
    if (!A.S.anno) return 'FATAL: 选图后 A.S.anno 仍为空';
    if (!A.S.anno.objects.length)
      return 'FATAL: 选图后仍无对象; 当前=' + A.S.current;
    const S = A.S;
    log.push('当前图片: ' + S.current + '  对象数: ' + S.anno.objects.length);
    log.push('');

    // ---------- T1 复选框隐藏 / 显示 ----------
    const list = d.getElementById('objList');
    const items0 = list.querySelectorAll('.obj-item');
    const chks = list.querySelectorAll('.obj-chk');
    ok('对象列表每一行都有复选框', chks.length === items0.length,
       chks.length + ' vs ' + items0.length);

    // 聚焦模式下画布只画选中的 -> 先关掉聚焦，测复选框
    S.focusMode = false;
    const before = A.visibleObjects().length;
    chks[0].checked = false;
    fire(w, d, chks[0], 'change');
    const after = A.visibleObjects().length;
    ok('取消勾选后该对象不再显示', after === before - 1, before + ' -> ' + after);
    chks[0].checked = true;
    fire(w, d, chks[0], 'change');
    ok('重新勾选后恢复显示', A.visibleObjects().length === before);

    // ---------- T2 点对象自动聚焦 ----------
    S.focusMode = false;
    S.selectedId = null;
    // 点击列表里的第 2 行（对象不足 2 个就点第 1 行）
    const items = d.getElementById('objList').querySelectorAll('.obj-item');
    const rowIdx = items.length > 1 ? 1 : 0;
    const expSel = S.anno.objects[rowIdx].id;
    items[rowIdx].dispatchEvent(new w.MouseEvent('click', { bubbles: true }));
    ok('点列表会选中该对象', S.selectedId === expSel, S.selectedId + ' vs ' + expSel);
    ok('点列表自动进入聚焦模式', S.focusMode === true);
    const vis = A.visibleObjects();
    ok('聚焦后画布只剩 1 个对象', vis.length === 1, vis.length + ' 个');
    ok('聚焦的就是点中的那个', vis[0] && vis[0].id === expSel);

    // ---------- T3 三个筛选按钮 ----------
    d.getElementById('btnShowAll').click();
    ok('「全选」后聚焦关闭', S.focusMode === false);
    ok('「全选」后全部可见', A.visibleObjects().length === S.anno.objects.length,
       A.visibleObjects().length + '/' + S.anno.objects.length);

    d.getElementById('btnHideAll').click();
    ok('「全不选」后一个都不显示', A.visibleObjects().length === 0,
       A.visibleObjects().length + ' 个');
    ok('「全不选」后复选框全部未勾选',
       [...d.getElementById('objList').querySelectorAll('.obj-chk')].every(c => !c.checked));

    d.getElementById('btnShowAll').click();
    // 选一个再点「仅看选中」
    S.selectedId = S.anno.objects[0].id;
    d.getElementById('btnFocusOnly').click();
    ok('「仅看选中」后只剩 1 个', A.visibleObjects().length === 1);
    d.getElementById('btnFocusOnly').click();
    ok('再点一次恢复全部', A.visibleObjects().length === S.anno.objects.length);

    // ---------- T4 选中用轮廓描边，不画矩形 ----------
    S.focusMode = false;
    S.selectedId = null;
    const cv = d.getElementById('canvas');
    const cctx = cv.getContext('2d');
    function snap() {
      return cctx.getImageData(0, 0, cv.width, cv.height).data;
    }
    // 选中一个对象
    S.selectedId = S.anno.objects[0].id;
    A.draw();
    // 关键：默认（showBox=false）不应该出现外接矩形
    const boxBefore = S.showBox;
    S.showBox = false;
    A.draw();
    const pxNoBox = snap();
    S.showBox = true;
    A.draw();
    const pxWithBox = snap();
    S.showBox = boxBefore;
    A.draw();
    let diff = 0;
    for (let i = 0; i < pxNoBox.length; i += 4) if (pxNoBox[i] !== pxWithBox[i]) diff++;
    ok('勾选「框」后才会额外画外接矩形', diff > 0, diff + ' 像素不同');
    ok('默认不画外接矩形（只用轮廓描边）', boxBefore === false);

    // ---------- T5 画笔：位图掩码涂抹 ----------
    // 造一个带凹角的对象（凹形矩形），验证涂完凹角没被填平
    const cv2 = d.createElement('canvas');
    const deep = {
      id: 'test_deep', class: 'book', edited: true, source: 'human',
      polygon: [[0.3,0.3],[0.5,0.3],[0.5,0.45],[0.7,0.45],[0.7,0.7],[0.3,0.7]],
    };
    S.anno.objects = [deep];
    S.selectedId = deep.id;
    S.unchecked = new Set();
    S.focusMode = false;
    S.origById[deep.id] = JSON.stringify(deep);
    S.edits = [];
    const areaBefore = polyArea(deep.polygon);

    // 切到画笔工具
    const brushBtn = d.querySelector('.tool-btn[data-tool="brush"]');
    brushBtn.click();
    ok('画笔工具已激活', S.tool === 'brush');
    ok('画笔建立了位图掩码', !!S.brushMask, S.brushMask ? S.brushMask.w + 'x' + S.brushMask.h : 'null');
    ok('画笔光标是笔状', cv.className === 'tool-brush', 'class=' + cv.className);

    // 在对象内部涂一笔（不该改变面积 —— 在已有形状内涂）
    const cwr = cv.getBoundingClientRect();
    const toClient = (nx, ny) => {
      const p = A.toCanvas(nx, ny);
      return { x: cwr.left + p[0], y: cwr.top + p[1] };
    };
    let pt = toClient(0.4, 0.6);
    fire(w, d, cv, 'mousedown', { clientX: pt.x, clientY: pt.y, button: 0 });
    for (let i = 0; i <= 10; i++) {
      pt = toClient(0.4 + i * 0.02, 0.6);
      fire(w, d, cv, 'mousemove', { clientX: pt.x, clientY: pt.y });
    }
    fire(w, d, cv, 'mouseup', { clientX: pt.x, clientY: pt.y, button: 0 });
    await new Promise(r => setTimeout(r, 300));
    ok('涂抹后多边形仍有效', Array.isArray(deep.polygon) && deep.polygon.length >= 3,
       deep.polygon ? deep.polygon.length + ' 点' : 'null');
    const areaAfter = polyArea(deep.polygon);
    ok('在形状内部涂抹，面积基本不变（±15%）',
       Math.abs(areaAfter - areaBefore) / areaBefore < 0.15,
       areaBefore.toFixed(4) + ' -> ' + areaAfter.toFixed(4));

    // 凹角是否被填平：数一下多边形的"凹"顶点
    function concaveCount(poly) {
      let n = 0;
      const m = poly.length;
      for (let i = 0; i < m; i++) {
        const a = poly[(i - 1 + m) % m], b = poly[i], c = poly[(i + 1) % m];
        const cr = (b[0]-a[0])*(c[1]-b[1]) - (b[1]-a[1])*(c[0]-b[0]);
        if (cr > 1e-9) n++;
      }
      return n;
    }
    ok('凹角被保留（至少 1 个凹顶点）', concaveCount(deep.polygon) >= 1,
       concaveCount(deep.polygon) + ' 个凹顶点');

    // ---------- T6 橡皮能擦掉一块 ----------
    const areaPre = polyArea(deep.polygon);
    S.brushMask = null;
    A.buildBrushMask(deep);
    const eraseBtn = d.querySelector('.tool-btn[data-tool="erase"]');
    eraseBtn.click();
    ok('橡皮工具已激活', S.tool === 'erase');
    pt = toClient(0.3, 0.5);   // 左边缘
    fire(w, d, cv, 'mousedown', { clientX: pt.x, clientY: pt.y, button: 0 });
    for (let i = 0; i <= 8; i++) {
      pt = toClient(0.3, 0.5 + i * 0.02);
      fire(w, d, cv, 'mousemove', { clientX: pt.x, clientY: pt.y });
    }
    fire(w, d, cv, 'mouseup', { clientX: pt.x, clientY: pt.y, button: 0 });
    await new Promise(r => setTimeout(r, 300));
    const areaErased = polyArea(deep.polygon);
    ok('橡皮擦掉了面积', areaErased < areaPre * 0.95,
       areaPre.toFixed(4) + ' -> ' + areaErased.toFixed(4));

    // ---------- 汇总 ----------
    log.unshift('结果：' + pass + ' 通过 / ' + fail + ' 失败\n');
    return log.join('\n');
  };
})();
