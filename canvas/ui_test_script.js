/* 在页面内执行的 UI 测试脚本（被 ui_test.js 注入到 iframe 父页）
   需要主页面已加载完，并通过 __runUiTests(d, w) 调用。
   模拟真实鼠标事件，验证：
     T1  复选框能隐藏/显示对象
     T2  点对象自动进入聚焦（其它对象不显示）
     T3  全选 / 全不选 / 仅看选中 三个按钮
     T4  选中对象用轮廓描边，不画矩形外框
     T5  画笔：位图掩码涂抹后，多边形真的变了且凹角没被填平
     T6  橡皮：能擦掉一块
     T7  撤销「新增对象」，且重做能恢复（按 id 判定）
     T8  撤销「删除对象」，对象连同原id 和位置一起回来
     T9  撤销「改类别」
     T10 撤销「画笔改形状」，多边形逐点还原
     T11 撤销栈上限 50；新操作清空重做分支
     T12 上传入口：按钮/多选/默认勾选、非图片被拒、拖拽遮罩显隐

   【这个文件必须放在 canvas/ 下】
   Flask 的static_folder 指向 canvas/，app.js 在 ?uitest=1 时
   从 /static/ui_test_script.js 加载本页。放在tools/ 下不会被执行。
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
    // 【为什么这里要取顶点中点而不是写死 0.3, 0.5】
    // 橡皮擦的是掩码，而掩码按deep 的多边形填充。
    // 写死一个坐标，一旦那个点恰好落在形状外，擦的就是空气，
    // 面积不变 -> 误判成"橡皮坏了"。
    // 取最靠左顶点的邻边中点，保证落在形状内部。
    let minX = 1, li = 0;
    deep.polygon.forEach((p, i) => { if (p[0] < minX) { minX = p[0]; li = i; } });
    const nb = deep.polygon[(li + 1) % deep.polygon.length];
    const ex = (minX + nb[0]) / 2, ey = (deep.polygon[li][1] + nb[1]) / 2;
    pt = toClient(ex, ey);   // 靠左的边中点
    fire(w, d, cv, 'mousedown', { clientX: pt.x, clientY: pt.y, button: 0 });
    for (let i = 0; i <= 8; i++) {
      pt = toClient(0.3, 0.5 + i * 0.02);
      fire(w, d, cv, 'mousemove', { clientX: pt.x, clientY: pt.y });
    }
    fire(w, d, cv, 'mouseup', { clientX: pt.x, clientY: pt.y, button: 0 });
    await new Promise(r => setTimeout(r, 300));
    const areaErased = polyArea(deep.polygon);
    // 阈值取2% 而不是 5%：擦的是一条窄边，
    // 窄边占整体面积本来就小，5% 会把「擦掉了」误判成「没擦掉」。
    // 这里的目的是确认橡皮真的改动了形状，不是量擦了多少。
    ok('橡皮擦掉了面积', areaErased < areaPre * 0.98,
       areaPre.toFixed(4) + ' -> ' + areaErased.toFixed(4)
       + ' (' + ((1 - areaErased / areaPre) * 100).toFixed(1) + '%)');

    // ---------- 撤销栈基线 ----------
    // 前面 T5/T6 已经改过 deep 的形状，先把栈清干净，
    // 让下面每项断言的「几步」数目确定。
    S.undoStack.length = 0; S.redoStack.length = 0;

    // ---------- T7 撤销「新增对象」 ----------
    const nPre = S.anno.objects.length;
    A.addObject('chair', [[0.1, 0.1], [0.3, 0.1], [0.3, 0.3], [0.1, 0.3]], 'human');
    const nAdd = S.anno.objects.length;
    ok('新增了一个对象', nAdd === nPre + 1, nPre + ' -> ' + nAdd);
    ok('新增后撤销栈有 1 步', S.undoStack.length === 1, S.undoStack.length + ' 步');
    const addedId = S.anno.objects[S.anno.objects.length - 1].id;
    A.undo();
    ok('撤销后对象数还原', S.anno.objects.length === nPre,
       S.anno.objects.length + ' vs ' + nPre);
    ok('撤销后新增的对象消失了（按 id 判定）',
       !S.anno.objects.some(o => o.id === addedId));
    ok('撤销后栈空、重做栈有 1 步',
       S.undoStack.length === 0 && S.redoStack.length === 1,
       S.undoStack.length + '/' + S.redoStack.length);
    A.redo();
    ok('重做后对象又回来了（id 一致）',
       S.anno.objects.some(o => o.id === addedId));
    A.undo();   // 回到基线

    // ---------- T8 撤销「删除对象」 ----------
    const victim = S.anno.objects[0];
    S.selectedId = victim.id;
    A.deleteSelected();
    ok('删除后少了一个对象', !S.anno.objects.some(o => o.id === victim.id));
    A.undo();
    const back = S.anno.objects.find(o => o.id === victim.id);
    ok('撤销删除后对象回来了', !!back);
    // 关键：删除时对象 id 从列表消失，撤销必须原样恢复同一个 id，
    // 否则「选中它改类别」这类后续操作会对不上号。
    // 【注意不能用引用比较 back === victim】
    // 快照是 JSON 深拷贝，恢复出来的必然是新对象，
    // 引用比较恒为 false，那是测试写错而不是功能坏了。
    ok('恢复的是同一个 id，且回到原位置',
       !!back && back.id === victim.id
       && S.anno.objects[0].id === victim.id
       && JSON.stringify(back.polygon) === JSON.stringify(victim.polygon),
       back ? back.id + ' vs ' + victim.id : '未恢复');

    // ---------- T9 撤销「改类别」 ----------
    const t9obj = S.anno.objects[0];
    S.selectedId = t9obj.id;
    const clsPre = t9obj.class;
    const clsNext = S.classes.find(c => c.name !== clsPre).name;
    A.changeClass(clsNext);
    ok('类别已改', S.anno.objects[0].class === clsNext, clsNext);
    A.undo();
    ok('撤销后类别还原', S.anno.objects[0].class === clsPre, S.anno.objects[0].class);

    // ---------- T10 撤销「修改多边形形状」 ----------
    // 走真实路径：buildBrushMask -> commitBrush，改完后撤销应回到原形状。
    const t10 = S.anno.objects[0];
    S.selectedId = t10.id;
    const polyPre = JSON.stringify(t10.polygon);
    const nPtsPre = t10.polygon.length;
    A.buildBrushMask(t10);
    // 用一次「全选填充」代替手工涂抹：直接铺满掩码，
    // commitBrush 会把它转成一个新的多边形，形状必然改变。
    const m = S.brushMask;
    ok('画笔掩码已建立', !!m);
    if (m) {
      const im = m.ctx.createImageData(m.w, m.h);
      for (let i = 0; i < im.data.length; i += 4) {
        im.data[i] = 255; im.data[i + 3] = 255;
      }
      m.ctx.putImageData(im, 0, 0);
      S.paintTarget = t10.id;
      A.commitBrush();
      ok('commitBrush 改变了形状',
         JSON.stringify(S.anno.objects[0].polygon) !== polyPre,
         nPtsPre + ' -> ' + S.anno.objects[0].polygon.length + ' 个点');
      A.undo();
      ok('撤销画笔后形状逐点还原',
         JSON.stringify(S.anno.objects[0].polygon) === polyPre,
         S.anno.objects[0].polygon.length + ' 个点');
      ok('撤销后选中对象仍然存在',
         S.anno.objects.some(o => o.id === t10.id));
    }

    // ---------- T11 栈上限 / 新操作清空重做分支 ----------
    S.undoStack.length = 0; S.redoStack.length = 0;
    for (let i = 0; i < S.undoLimit + 12; i++) A.pushUndo();
    ok('撤销栈不超过上限 ' + S.undoLimit,
       S.undoStack.length === S.undoLimit, S.undoStack.length + ' 步');
    A.redo();                 // 先造一个重做分支
    const redoBefore = S.redoStack.length;
    A.pushUndo();
    ok('新操作清空重做分支', S.redoStack.length === 0,
       redoBefore + ' -> ' + S.redoStack.length);
    S.undoStack.length = 0; S.redoStack.length = 0;

    // ---------- T12 上传入口（上传 → 一键预标注） ----------
    // 这一组只验证「界面接线」和「防御分支」：
    // 真上传会往 data/images 写文件，自动化测试不该污染数据集，
    // 所以端到端上传由 curl 手工验证，这里只测不会产生副作用的部分。
    const upBtn = d.getElementById('btnUpload');
    const fileInp = d.getElementById('fileInput');
    const autoChk = d.getElementById('autoPrelabel');
    ok('上传按钮存在', !!upBtn);
    ok('文件输入支持多选且限定图片', !!fileInp && fileInp.multiple
       && /image/.test(fileInp.accept || ''),
       fileInp ? `multiple=${fileInp.multiple} accept=${fileInp.accept}` : '缺失');
    ok('默认勾选「上传后自动预标注」', !!autoChk && autoChk.checked);

    // 非图片文件必须被挡在发请求之前，且不能改动图片列表
    const imgBefore = S.images.length;
    let netCalled = false;
    const realFetch = w.fetch;
    w.fetch = function () { netCalled = true; return realFetch.apply(this, arguments); };
    try {
      await A.uploadFiles([new w.File(['x'], 'a.txt', { type: 'text/plain' })]);
    } finally { w.fetch = realFetch; }
    ok('非图片文件直接拒绝，不发上传请求', !netCalled);
    ok('被拒绝时不改动图片列表', S.images.length === imgBefore,
       imgBefore + ' -> ' + S.images.length);

    // 空选择不能报错
    await A.uploadFiles([]);
    ok('空文件列表安全退出', true);

    // 拖拽提示遮罩：拖入显示、拖出隐藏
    const mask = d.getElementById('dropMask');
    ok('拖拽提示遮罩存在', !!mask);
    if (mask) {
      const dt = new w.DataTransfer();
      dt.items.add(new w.File(['x'], 'a.png', { type: 'image/png' }));
      const ev = t => new w.DragEvent(t, {
        bubbles: true, cancelable: true, dataTransfer: dt,
      });
      w.dispatchEvent(ev('dragenter'));
      ok('拖入文件时显示遮罩', !mask.classList.contains('hidden'));
      w.dispatchEvent(ev('dragleave'));
      ok('拖出后遮罩自动隐藏', mask.classList.contains('hidden'));
    }

    // ---------- 汇总 ----------
    log.unshift('结果：' + pass + ' 通过 / ' + fail + ' 失败\n');
    return log.join('\n');
  };
})();
