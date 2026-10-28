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
     T13 删除照片：每项都有删除按钮、取消不发请求、请求只带一个名字、
         删除按钮不误触选中、恢复提示条显隐
     T14 框选修形：并入/挖掉/替换三种模式对轮廓的作用、无选中与
         框过小时的边界行为、退出后回到选择工具

   【这个文件必须放在 canvas/ 下】
   Flask 的static_folder 指向 canvas/，app.js 在 ?uitest=1 时
   从 /static/ui_test_script.js 加载本页。放在tools/ 下不会被执行。
*/
(function () {
  'use strict';

  // 【为什么要把 mouse* 映射成 pointer*】
  // 画布改成监听 Pointer Events（鼠标和触摸共用一套逻辑）之后，
  // MouseEvent 不会再触发 pointerdown。这里在派发时顺手发一次同义的 pointer 事件，
  // 测试里"按下-移动-抬起"的写法就不用逐个改。
  const POINTER_OF = { mousedown: 'pointerdown', mousemove: 'pointermove', mouseup: 'pointerup' };

  function fire(w, d, el, type, opts) {
    const r = el.getBoundingClientRect();
    const o = Object.assign({ bubbles: true, cancelable: true, view: w,
      clientX: r.left + r.width / 2, clientY: r.top + r.height / 2 }, opts || {});
    const ptype = POINTER_OF[type];
    if (ptype && w.PointerEvent) {
      el.dispatchEvent(new w.PointerEvent(ptype, Object.assign({}, o, {
        pointerId: 1, pointerType: 'mouse', isPrimary: true,
        button: o.button || 0, buttons: type === 'mouseup' ? 0 : 1,
      })));
      return;
    }
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
    // 把 log 挂到 window 上：万一中途 FATAL，报告里也能看到跑到哪一步了
    w.__UITEST_LOG__ = log;
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
        // 【为什么不能固定 sleep】图片列表涨到 150+ 张后，缩略图会占满
        // 浏览器对同一域名的并发连接（HTTP/1.1 只有 6 条），主图要排队，
        // 加载时间忽快忽慢：等 1 秒有时够、有时不够，就会出现
        // 「anno 有了但 S.img 还没 onload」，后面 toCanvas 读 S.img.width 直接崩。
        // 改成轮询到主图真正解码完成，判断依据是 naturalWidth。
        for (let i = 0; i < 80; i++) {
          if (A.S.img && A.S.img.naturalWidth > 0) break;
          await new Promise(r => setTimeout(r, 100));
        }
        if (!A.S.img) return 'FATAL: 主图未加载完（S.img 为空）: ' + withObj.name;
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

    // ---------- T13 删除照片 ----------
    // 同样只测「接线 + 防御分支」：真删会往 data/_trash 搬文件，
    // 真删流程由 curl 端到端验证，这里用 confirm 桩和 fetch 桩，
    // 保证跑完 data/images 一张不少。
    const imgItems = d.querySelectorAll('.img-item');
    const dels = d.querySelectorAll('.img-del');
    ok('每个列表项都有删除按钮', imgItems.length > 0 && dels.length === imgItems.length,
       imgItems.length + ' 项 / ' + dels.length + ' 按钮');
    ok('删除按钮提示写明可恢复',
       dels.length > 0 && /_trash/.test(dels[0].title || ''),
       dels.length ? dels[0].title : '缺失');

    const realConfirm = w.confirm;
    const realFetchDel = w.fetch;
    let confirmAnswer = false;
    let delCalled = 0;
    let delBody = null;
    w.confirm = () => confirmAnswer;
    w.fetch = function (url, opt) {
      if (String(url).indexOf('/api/images/delete') >= 0) {
        delCalled++;
        delBody = JSON.parse(opt.body);
        return Promise.resolve(new w.Response(JSON.stringify({
          ok: true, deleted: delBody.names, missing: [], failed: [],
          trash: 'data/_trash/stub', total_images: S.images.length,
        }), { status: 200, headers: { 'Content-Type': 'application/json' } }));
      }
      return realFetchDel.apply(this, arguments);
    };

    try {
      // ① 取消确认：一个请求都不能发
      const target = S.images.find(x => x.name !== S.current);
      const curBefore = S.current;
      confirmAnswer = false;
      A.deleteImage(target);
      await new Promise(r => setTimeout(r, 120));
      ok('取消确认时不发删除请求', delCalled === 0, 'delCalled=' + delCalled);
      // 点 × 不能顺带把整行选中（stopPropagation 是否生效）
      ok('点删除按钮不会切换当前图片', S.current === curBefore,
         curBefore + ' -> ' + S.current);

      // ② 确认删除：请求只带这一个文件名
      confirmAnswer = true;
      A.deleteImage(target);
      await new Promise(r => setTimeout(r, 600));
      ok('确认后发出删除请求', delCalled === 1, 'delCalled=' + delCalled);
      ok('请求体只含被删的那一个文件名',
         !!delBody && Array.isArray(delBody.names)
         && delBody.names.length === 1 && delBody.names[0] === target.name,
         delBody ? JSON.stringify(delBody) : 'null');

      // ③ 恢复提示条：没有删除记录时隐藏，有记录时显示张数
      const bar = d.getElementById('trashBar');
      S.lastTrash = null;
      A.renderTrashBar();
      ok('无删除记录时不显示恢复条', !!bar && bar.classList.contains('hidden'));

      S.lastTrash = { stamp: '20260101_000000', count: 3 };
      A.renderTrashBar();
      ok('有删除记录时显示恢复条并标出张数',
         !!bar && !bar.classList.contains('hidden')
         && /3/.test(d.getElementById('trashText').textContent),
         bar ? d.getElementById('trashText').textContent : '缺失');

      // ④ 防抖保存不能把标注「写回来」
      // saveSoon 是 700ms 防抖。若在窗口内删掉当前图，删除请求会先到
      // （标注文件被搬走），延迟的 saveNow 再把整份标注写回，
      // 就留下一个没有图片的孤儿 json。删当前图必须先取消待执行的保存。
      //
      // 【怎么观测】测试模式下 saveNow 不落盘（__UITEST__ 直接 return），
      // 所以数不到请求。改看 saveTimer：saveSoon 会留下 timer id，
      // 取消后必须为 null。移除修复后这条会 FAIL，验证过有效。
      const curName = S.current;
      confirmAnswer = true;
      S.dirty = true;
      A.saveSoon();
      const queued = A.saveTimer !== null;
      ok('saveSoon 排上了待执行的保存', queued);
      A.deleteImage(S.images.find(x => x.name === curName));
      await new Promise(r => setTimeout(r, 400));
      ok('删除当前图后取消了待执行的保存（不留孤儿标注）',
         A.saveTimer === null, 'saveTimer=' + A.saveTimer);
      await new Promise(r => setTimeout(r, 1200));   // 越过防抖窗口再看一次
      ok('防抖窗口过去也没有补写', A.saveTimer === null,
         'saveTimer=' + A.saveTimer);

      // 收尾：把状态还原，别影响后面的断言
      S.lastTrash = null;
      A.renderTrashBar();
    } finally {
      w.confirm = realConfirm;
      w.fetch = realFetchDel;
    }

    // ---------- T14 框选修形：并入 / 挖掉 / 替换 ----------
    // 起因：模型只把沙发圈了右半边。人的诉求是「这一块也属于它」，
    // 所以工具要能把一个框并进已有轮廓，而不是逼人重画一百个点。
    {
      S.tool = 'select';
      cv.className = 'tool-select';
      const base = {
        id: 'test_box', class: 'couch', edited: true, source: 'human',
        polygon: [[0.45,0.40],[0.60,0.40],[0.60,0.60],[0.45,0.60]],
      };
      S.anno.objects = [base];
      S.selectedId = base.id;
      S.unchecked = new Set();
      S.focusMode = false;
      S.undoStack.length = 0;
      S.origById[base.id] = JSON.stringify(base);
      S.edits = [];
      S.brushMask = null;

      const boxBtn = d.querySelector('.tool-btn[data-tool="box"]');
      boxBtn.click();
      ok('框选工具已激活', S.tool === 'box');
      ok('框选光标是十字', cv.className === 'tool-box', 'class=' + cv.className);
      const hint = d.getElementById('boxHint');
      ok('框选提示条已显示', !!hint && !hint.hidden);
      ok('默认模式是「并入」', S.boxMode === 'add', 'mode=' + S.boxMode);

      // 用真实鼠标拖一个框：x 0.25~0.50，与对象左半重叠 -> 应连成一块
      const cwr = cv.getBoundingClientRect();
      const toClient = (nx, ny) => {
        const p = A.toCanvas(nx, ny);
        return { x: cwr.left + p[0], y: cwr.top + p[1] };
      };
      async function dragBox(x0, y0, x1, y1) {
        let p = toClient(x0, y0);
        fire(w, d, cv, 'mousedown', { clientX: p.x, clientY: p.y, button: 0 });
        p = toClient(x1, y1);
        fire(w, d, cv, 'mousemove', { clientX: p.x, clientY: p.y });
        fire(w, d, cv, 'mouseup', { clientX: p.x, clientY: p.y, button: 0 });
        await new Promise(r => setTimeout(r, 250));
      }

      const areaBase = polyArea(base.polygon);
      await dragBox(0.25, 0.40, 0.50, 0.60);
      const areaAdd = polyArea(base.polygon);
      const minX = Math.min(...base.polygon.map(q => q[0]));
      ok('并入后轮廓变大', areaAdd > areaBase * 1.5,
         areaBase.toFixed(4) + ' -> ' + areaAdd.toFixed(4));
      ok('并入后轮廓延伸到框的左边缘（真的补全了）', minX < 0.28,
         'minX=' + minX.toFixed(3));
      ok('并入后仍是一个连续多边形', base.polygon.length >= 3,
         base.polygon.length + ' 点');
      ok('并入算一次可撤销的修改', S.undoStack.length >= 1,
         'undo=' + S.undoStack.length);

      // 挖掉：框落在对象身体里，面积应下降
      S.boxMode = 'cut';
      A.syncBoxHint();
      ok('切到挖掉模式', S.boxMode === 'cut');
      const areaPreCut = polyArea(base.polygon);
      S.brushMask = null;
      await dragBox(0.52, 0.45, 0.60, 0.55);
      const areaCut = polyArea(base.polygon);
      ok('挖掉后轮廓变小', areaCut < areaPreCut,
         areaPreCut.toFixed(4) + ' -> ' + areaCut.toFixed(4));

      // 替换：轮廓直接变成这个框
      S.boxMode = 'replace';
      A.syncBoxHint();
      S.brushMask = null;
      await dragBox(0.20, 0.20, 0.40, 0.32);
      const xs = base.polygon.map(q => q[0]), ys = base.polygon.map(q => q[1]);
      const bw = Math.max(...xs) - Math.min(...xs);
      const bh = Math.max(...ys) - Math.min(...ys);
      ok('替换后轮廓范围等于框',
         Math.abs(bw - 0.20) < 0.03 && Math.abs(bh - 0.12) < 0.03,
         bw.toFixed(3) + ' x ' + bh.toFixed(3));
      ok('替换后轮廓确实挪到了新位置',
         Math.abs(Math.min(...xs) - 0.20) < 0.03,
         'minX=' + Math.min(...xs).toFixed(3));

      // 没选中对象时拉框：不许崩，也不许改坏数据
      S.boxMode = 'add';
      const keep = JSON.stringify(base.polygon);
      S.selectedId = null;
      S.brushMask = null;
      await dragBox(0.60, 0.60, 0.80, 0.80);
      ok('没选中对象时拉框不改任何轮廓',
         JSON.stringify(base.polygon) === keep);

      // 手抖点一下（框极小）：不算操作
      S.selectedId = base.id;
      S.brushMask = null;
      await dragBox(0.30, 0.30, 0.301, 0.301);
      ok('框太小时忽略，不产生空轮廓',
         base.polygon.length >= 3 && JSON.stringify(base.polygon) === keep);

      // 收尾：回到选择工具，清掉测试对象
      d.getElementById('btnCancelBox').click();
      ok('退出框选后回到选择工具', S.tool === 'select', 'tool=' + S.tool);
      S.anno.objects = [];
      S.selectedId = null;
      S.undoStack.length = 0;
    }

    // ---------- T14 手机 / 平板：抽屉 + 触摸标注 ----------
    {
      // 抽屉按钮在桌面视口被 CSS 隐藏，但元素和状态逻辑必须在
      const bL = d.getElementById('btnLeftDrawer');
      const bR = d.getElementById('btnRightDrawer');
      const mask = d.getElementById('drawerMask');
      const pL = d.querySelector('.panel-left');
      const pR = d.querySelector('.panel-right');
      ok('窄屏入口存在（图片 / 属性两个按钮）', !!bL && !!bR);
      ok('初始状态两栏都没展开',
         !pL.classList.contains('open') && !pR.classList.contains('open'));

      bL.click();
      await new Promise(r => setTimeout(r, 60));
      ok('点「图片」展开左栏', pL.classList.contains('open'));
      ok('抽屉展开时遮罩同步显示', mask.hidden === false);

      bR.click();
      await new Promise(r => setTimeout(r, 60));
      ok('点「属性」展开右栏', pR.classList.contains('open'));
      ok('打开一边自动收起另一边（不会同时压住画布）',
         !pL.classList.contains('open'));

      bR.click();                      // 再点一次 = 收起
      await new Promise(r => setTimeout(r, 260));
      ok('再点同一个按钮收起抽屉',
         !pR.classList.contains('open') && mask.hidden === true);

      bL.click();
      await new Promise(r => setTimeout(r, 60));
      mask.click();
      await new Promise(r => setTimeout(r, 260));
      ok('点遮罩能关闭抽屉',
         !pL.classList.contains('open') && mask.hidden === true);
    }

    {
      const cvT = d.getElementById('canvas');
      // 不关掉浏览器默认手势，手指一涂整页会跟着滚，pointermove 还会被系统夺走
      ok('画布禁用了浏览器默认触摸手势',
         w.getComputedStyle(cvT).touchAction === 'none',
         w.getComputedStyle(cvT).touchAction);

      S.anno.objects = [];
      S.selectedId = null;
      S.undoStack.length = 0;
      S.edits = [];
      S.focusMode = false;
      S.draftPts = [];
      S.drawing = false;
      const cwr = cvT.getBoundingClientRect();

      // 用 PointerEvent + pointerType:'touch' 模拟手指
      function touch(type, nx, ny, pid) {
        const p = A.toCanvas(nx, ny);
        cvT.dispatchEvent(new w.PointerEvent(type, {
          bubbles: true, cancelable: true, view: w,
          clientX: cwr.left + p[0], clientY: cwr.top + p[1],
          pointerId: pid || 11, pointerType: 'touch', isPrimary: true,
          button: 0, buttons: type === 'pointerup' ? 0 : 1,
        }));
      }
      function tap(nx, ny) { touch('pointerdown', nx, ny); touch('pointerup', nx, ny); }

      // 1) 触摸描边：三个点落下
      d.querySelector('.tool-btn[data-tool="polygon"]').click();
      tap(0.30, 0.30); tap(0.55, 0.30); tap(0.55, 0.55);
      ok('触摸能画多边形（三个点已落下）',
         S.drawing && S.draftPts.length === 3, 'draftPts=' + S.draftPts.length);

      // 2) 同一位置快速点两下 = 双击闭合（触摸没有 e.detail，靠手势判断）
      tap(0.40, 0.60); tap(0.40, 0.60);
      await new Promise(r => setTimeout(r, 60));
      ok('触摸双击能闭合多边形', !S.drawing, 'drawing=' + S.drawing);
      // 闭合后和鼠标路径一样弹类别框，点一个类别才算真的落地
      const pick = d.querySelector('#clsPick button');
      ok('触摸闭合后弹出类别选择框', !!pick);
      if (pick) pick.click();
      await new Promise(r => setTimeout(r, 60));
      ok('选完类别生成了对象', S.anno.objects.length === 1,
         'objects=' + S.anno.objects.length);

      // 3) 触摸选中：点在对象质心上
      const obj = S.anno.objects[0];
      if (obj) {
        const cx = obj.polygon.reduce((s, p) => s + p[0], 0) / obj.polygon.length;
        const cy = obj.polygon.reduce((s, p) => s + p[1], 0) / obj.polygon.length;
        d.querySelector('.tool-btn[data-tool="select"]').click();
        S.selectedId = null;
        tap(cx, cy);
        ok('触摸能选中对象', S.selectedId === obj.id,
           'selected=' + S.selectedId);

        // 4) 多指：第一根按在空白处，第二根按在对象上 —— 不许抢走选中
        S.selectedId = null;
        touch('pointerdown', 0.05, 0.05);           // 第一根（空白）
        touch('pointerdown', cx, cy, 12);           // 第二根（对象上）
        ok('第二根手指被忽略，不会抢走操作',
           S.selectedId === null, 'selected=' + S.selectedId);
        touch('pointerup', 0.05, 0.05);
      }

      // 收尾：清干净，避免影响后面的汇总统计
      S.anno.objects = [];
      S.selectedId = null;
      S.undoStack.length = 0;
      S.edits = [];
      S.draftPts = [];
      S.drawing = false;
      A.draw();
    }

    // ---------- T15 预标注的范围：只标「还没有标注文件」的图 ----------
    // 【为什么值得单测】这个按钮原先一律全量，一次误点把 60 条人工修正冲掉了
    // （ai_log 第十六轮）。规则很容易被改回去，所以钉在测试里：
    // 没有标注文件的才算待标；文件存在但 0 对象的（用户确认过这张图没有目标物）
    // 不重做——否则每次点预标注都会把那 10 张空图又灌上模型误检。
    {
      const keep = S.images;
      S.images = [
        { name: 'a.jpg', reviewed: true, total: 3, edited: 1 },
        { name: 'b.jpg', reviewed: true, total: 0, edited: 0 },
        { name: 'c.jpg', reviewed: false, total: 0, edited: 0 },
        { name: 'd.jpg', reviewed: false, total: 0, edited: 0 },
      ];
      const pend = A.pendingDrafts();
      ok('只有缺标注文件的图才算待标', pend.join(',') === 'c.jpg,d.jpg',
         pend.join(','));

      A.syncPrelabelBtn();
      const btn = d.getElementById('btnPrelabel');
      ok('按钮上标出待标张数', btn.textContent === '预标注 (2)', btn.textContent);

      S.images = [{ name: 'a.jpg', reviewed: true, total: 3, edited: 0 }];
      A.syncPrelabelBtn();
      ok('全部已有标注时按钮不带数字', btn.textContent === '预标注',
         btn.textContent);

      S.images = keep;
      A.syncPrelabelBtn();
    }

    // ---------- 汇总 ----------
    log.unshift('结果：' + pass + ' 通过 / ' + fail + ' 失败\n');
    return log.join('\n');
  };
})();
