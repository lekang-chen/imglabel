/* ==========================================================================
   geom.js —— 前端几何工具（掩码轮廓提取 + 多边形抽稀）
   --------------------------------------------------------------------------
   为什么单独成文件：
   画笔要能把「位图掩码」还原成「多边形」，需要两件核心算法：
     1. traceContour()  —— Moore 邻域轮廓追踪（从二值图里抠出闭合边界）
     2. simplifyDP()    —— Douglas-Peucker 抽稀（把锯齿边界压成少量顶点）
   这两段代码逻辑不平凡，单独成文件后可以用 node 直接跑单元测试，
   不用每次开浏览器才验证（见 tools/test_geom.js）。

   同一套算法在 Python 侧也有对应实现（core.py 的 simplify_polygon），
   两边保持一致的语义：抽稀后形状误差可控，顶点数可控。
   ========================================================================== */

(function (root, factory) {
  const api = factory();
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.Geo = api;
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  // 8 邻域方向，按「顺时针」排列，从正左方开始。
  // 屏幕坐标 y 轴向下，所以 W→NW→N→NE→E 是视觉上的顺时针。
  const DIRS = [
    [-1, 0],   // 0 W
    [-1, -1],  // 1 NW
    [0, -1],   // 2 N
    [1, -1],   // 3 NE
    [1, 0],    // 4 E
    [1, 1],    // 5 SE
    [0, 1],    // 6 S
    [-1, 1],   // 7 SW
  ];

  // 屏幕坐标系下的方向：+x 右、+y 下
  // 顺时针顺序：右 -> 下 -> 左 -> 上
  const CDIR = [[1, 0], [0, 1], [-1, 0], [0, -1]];

  /**
   * 沿「像素缝隙」追踪边界。
   *
   * 相比 Moore 邻域（走像素中心），缝隙追踪走的是像素与像素之间的分界线，
   * 好处是面积能精确还原：填出来的方形就是 N×N 像素，一分不多一分不少。
   * 如果走像素中心，边界会向内缩半个像素，反复涂抹编辑会让物体越缩越小。
   *
   * @param {Uint8ClampedArray} alpha 像素 alpha 通道
   * @param {number} w @param {number} h
   * @returns {Array<Array<[number,number]>>} 若干条闭合轮廓（晶格坐标）
   */
  function traceBoundaries(alpha, w, h) {
    const filled = (x, y) =>
      x >= 0 && y >= 0 && x < w && y < h && alpha[(y * w + x) * 4 + 3] > 128;

    const CW = w + 1;                       // 晶格宽度
    const key = (x, y) => y * CW + x;       // 晶格点编号
    const out = new Map();                 // 起点 -> 出边列表

    const addEdge = (x0, y0, x1, y1) => {
      const k = key(x0, y0);
      if (!out.has(k)) out.set(k, []);
      out.get(k).push({ x0, y0, x1, y1, used: false });
    };

    // 逐像素检查四条边，邻居为空则这条边是边界
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        if (!filled(x, y)) continue;
        if (!filled(x, y - 1)) addEdge(x, y, x + 1, y);         // 上边 向右
        if (!filled(x + 1, y)) addEdge(x + 1, y, x + 1, y + 1); // 右边 向下
        if (!filled(x, y + 1)) addEdge(x + 1, y + 1, x, y + 1); // 下边 向左
        if (!filled(x - 1, y)) addEdge(x, y + 1, x, y);         // 左边 向上
      }
    }

    const loops = [];
    for (const list of out.values()) {
      for (const e0 of list) {
        if (e0.used) continue;
        const loop = [];
        let cur = e0;
        let guard = 0;
        const maxSteps = w * h * 4 + 32;
        while (cur && !cur.used && guard++ < maxSteps) {
          cur.used = true;
          loop.push([cur.x0, cur.y0]);

          // 从 cur.x1,cur.y1 继续找下一条边
          const cands = out.get(key(cur.x1, cur.y1)) || [];
          const unused = cands.filter(c => !c.used);
          if (!unused.length) break;
          if (unused.length === 1) { cur = unused[0]; continue; }

          // 鞍点（对角相接的两个前景块）：按「右转优先」配对，
          // 等价于前景 4 连通、背景 8 连通，避免把两块黏成一坨。
          const d = dirIndex(cur);
          const pref = [CDIR[(d + 1) % 4], CDIR[d], CDIR[(d + 3) % 4], CDIR[(d + 2) % 4]];
          let chosen = null;
          for (const p of pref) {
            const c = unused.find(u => u.x1 - u.x0 === p[0] && u.y1 - u.y0 === p[1]);
            if (c) { chosen = c; break; }
          }
          cur = chosen || unused[0];
        }
        if (loop.length >= 4) {
          // 补上闭合点：走完一圈后回到起点
          const a = loop[0], b = loop[loop.length - 1];
          if (a[0] !== b[0] || a[1] !== b[1]) loop.push([a[0], a[1]]);
          loops.push(loop);
        }
      }
    }
    return loops;
  }

  /** 边方向在 CDIR 里的下标 */
  function dirIndex(e) {
    for (let i = 0; i < 4; i++) {
      if (CDIR[i][0] === e.x1 - e.x0 && CDIR[i][1] === e.y1 - e.y0) return i;
    }
    return 0;
  }

  /** 多边形有向面积（晶格坐标，y 向下） */
  function signedArea(pts) {
    let s = 0;
    for (let i = 0; i < pts.length; i++) {
      const p1 = pts[i], p2 = pts[(i + 1) % pts.length];
      s += p1[0] * p2[1] - p2[0] * p1[1];
    }
    return s / 2;
  }

  /**
   * Moore 邻域轮廓追踪：从二值位图里抠出一条闭合边界。
   *
   * @param {Uint8ClampedArray|number[]} alpha 像素 alpha 通道（长度 w*h）
   * @param {number} w 宽
   * @param {number} h 高
   * @returns {Array<[number,number]>|null} 像素坐标闭合轮廓（首尾相同），空图返回 null
   *
   * 说明：只追踪「外轮廓」。多边形格式无法表达孔洞，
   * 所以掩码内部的洞在转回多边形时会被填实——这和主流标注工具一致。
   */
  function traceContour(alpha, w, h) {
    const at = (x, y) =>
      x >= 0 && y >= 0 && x < w && y < h && alpha[(y * w + x) * 4 + 3] > 128;

    // 起点：最上面一行里最左边的前景像素。它的左邻一定是背景，
    // 这样从「正左方」开始顺时针扫，方向定义才唯一。
    let sx = -1, sy = -1;
    for (let y = 0; y < h && sx < 0; y++) {
      for (let x = 0; x < w; x++) {
        if (at(x, y)) { sx = x; sy = y; break; }
      }
    }
    if (sx < 0) return null;

    const contour = [];
    let cx = sx, cy = sy;
    // back：回溯方向（相对于当前像素），初始为正左方
    let back = 0;
    let guard = 0;
    const maxSteps = w * h * 4 + 16;

    while (guard++ < maxSteps) {
      contour.push([cx, cy]);

      let nf = -1, np = -1;
      for (let k = 0; k < 8; k++) {
        const d = (back + k) % 8;
        const nx = cx + DIRS[d][0], ny = cy + DIRS[d][1];
        if (at(nx, ny)) {
          nf = d;
          np = (d + 7) % 8;   // 找到前景之前的那个邻点必为背景，作为新的回溯点
          break;
        }
      }
      if (nf < 0) break;     // 孤立像素，无轮廓

      const ncx = cx + DIRS[nf][0], ncy = cy + DIRS[nf][1];
      // 新回溯像素相对新当前位置的方向
      const dx = (cx + DIRS[np][0]) - ncx;
      const dy = (cy + DIRS[np][1]) - ncy;
      let nd = 0;
      for (let i = 0; i < 8; i++) {
        if (DIRS[i][0] === dx && DIRS[i][1] === dy) { nd = i; break; }
      }

      cx = ncx; cy = ncy; back = nd;

      if (cx === sx && cy === sy) { contour.push([cx, cy]); break; }
    }

    return contour.length >= 4 ? contour : null;
  }

  /** 点到线段的距离（Douglas-Peucker 用） */
  function distToSeg(p, a, b) {
    const vx = b[0] - a[0], vy = b[1] - a[1];
    const wx = p[0] - a[0], wy = p[1] - a[1];
    const len2 = vx * vx + vy * vy;
    if (len2 === 0) return Math.hypot(wx, wy);
    let t = (wx * vx + wy * vy) / len2;
    t = Math.max(0, Math.min(1, t));
    return Math.hypot(p[0] - (a[0] + t * vx), p[1] - (a[1] + t * vy));
  }

  /**
   * Douglas-Peucker 抽稀（开放链）。
   * @param {Array<[number,number]>} pts
   * @param {number} eps 容差（与 pts 同单位）
   * @returns {Array<[number,number]>}
   */
  function simplifyDP(pts, eps) {
    if (!pts || pts.length < 3) return (pts || []).slice();
    const keep = new Uint8Array(pts.length);
    keep[0] = 1;
    keep[pts.length - 1] = 1;

    const stack = [[0, pts.length - 1]];
    while (stack.length) {
      const [i, j] = stack.pop();
      let maxD = -1, idx = -1;
      for (let k = i + 1; k < j; k++) {
        const d = distToSeg(pts[k], pts[i], pts[j]);
        if (d > maxD) { maxD = d; idx = k; }
      }
      if (maxD > eps && idx > 0) {
        keep[idx] = 1;
        stack.push([i, idx], [idx, j]);
      }
    }
    return pts.filter((_, i) => keep[i]);
  }

  /**
   * Douglas-Peucker 抽稀（闭合环）。
   *
   * 直接把闭合环丢给开放链版 D-P 会退化：首尾是同一个点，
   * 整条链被当成一条零长线段，每个点到它的距离都很大，
   * 结果一个点都删不掉。所以要先在「最远对角点」处把环切成两条开放链，
   * 分别抽稀后再拼回去。
   *
   * @param {Array<[number,number]>} ring 闭合环（首尾同点）
   * @param {number} eps 容差
   */
  function simplifyClosed(ring, eps) {
    if (!ring || ring.length < 4) return (ring || []).slice();
    // 去掉重复的收尾点
    const pts = ring.slice();
    const last = pts[pts.length - 1], first = pts[0];
    if (last[0] === first[0] && last[1] === first[1]) pts.pop();
    if (pts.length < 4) return ring.slice();

    // 找离 pts[0] 最远的点，在那儿对半切
    let m = 0, bestD = -1;
    for (let i = 1; i < pts.length; i++) {
      const d = (pts[i][0] - first[0]) ** 2 + (pts[i][1] - first[1]) ** 2;
      if (d > bestD) { bestD = d; m = i; }
    }

    const chainA = pts.slice(0, m + 1);
    const chainB = pts.slice(m).concat([first]);

    const sa = simplifyDP(chainA, eps);
    const sb = simplifyDP(chainB, eps);

    // 拼回：sa 已含两端点，sb 去掉头部(=m)和尾部(=first)
    const out = sa.concat(sb.slice(1, -1));
    out.push([out[0][0], out[0][1]]);
    return out;
  }

  /**
   * 从二值掩码还原成多边形：边界追踪 + 抽稀。
   * @param {Uint8ClampedArray} alpha
   * @param {number} w @param {number} h
   * @param {object} opt
   *   eps      抽稀容差（像素）
   *   maxPts   顶点上限（超出则加大容差重试）
   *   scaleX   像素 x -> 归一化 的除数
   *   scaleY   像素 y -> 归一化 的除数
   *
   * 注意 x/y 必须分别给：图片往往不是正方形（例如 640x419），
   * 用同一个 scale 会把图形压扁，面积直接算错。
   */
  function maskToPolygon(alpha, w, h, opt) {
    const o = Object.assign({ eps: 1.2, maxPts: 180, scaleX: w, scaleY: h }, opt || {});

    // 取面积最大的那条外轮廓（顺时针方向的即外边界）
    const loops = traceBoundaries(alpha, w, h);
    if (!loops.length) return null;
    let best = null, bestA = 0;
    for (const l of loops) {
      const a = signedArea(l);
      if (a > bestA) { bestA = a; best = l; }
    }
    if (!best) return null;

    let eps = o.eps;
    let simple = simplifyClosed(best, eps);
    // 二分找「最小容差」，保证顶点数不超过上限
    let lo = 0.05, hi = Math.max(2, Math.min(w, h) / 4);
    while (simple.length > o.maxPts && lo < hi) {
      const mid = (lo + hi) / 2;
      const s2 = simplifyClosed(best, mid);
      if (s2.length > o.maxPts) lo = mid; else { hi = mid; simple = s2; }
    }

    const sx = o.scaleX || w, sy = o.scaleY || h;
    // 【为什么要把坐标夹到 [0,1]】
    // traceBoundaries 是沿「像素缝隙」走的，轮廓贴着图像边缘时会产出 -0.5
    // 或 w-0.5 这样的坐标（半个像素在画布外）。除以 w 之后就是 -0.0005 之类
    // 的负数，写进 YOLO txt 就是非法坐标——导出校验会报
    // 「坐标 -0.0009 超出 [0,1]」，训练器读到还可能直接跳过该行。
    // 半个像素的夹取对形状没有任何可见影响，但能让导出永远合法。
    const clamp01 = v => (v < 0 ? 0 : (v > 1 ? 1 : v));
    return simple.map(p => [clamp01(p[0] / sx), clamp01(p[1] / sy)]);
  }

  return { traceContour, traceBoundaries, simplifyDP, simplifyClosed,
           maskToPolygon, distToSeg, signedArea };
});
