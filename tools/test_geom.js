/* geom.js 的单元测试（node 直接跑，不用开浏览器）
   用法：node tools/test_geom.js
   覆盖：轮廓追踪、D-P 抽稀、掩码转多边形 */
'use strict';

const path = require('path');
const Geo = require(path.join(__dirname, '..', 'canvas', 'geom.js'));

let pass = 0, fail = 0;
function ok(name, cond, extra) {
  if (cond) { pass++; console.log('  ✓ ' + name); }
  else { fail++; console.log('  ✗ ' + name + (extra ? '  -> ' + extra : '')); }
}

/* 造一个 w*h 的 alpha 缓冲，圆形区域填 255 */
function makeBuf(w, h) {
  return { alpha: new Uint8ClampedArray(w * h * 4), w, h };
}
function fillCircle(buf, cx, cy, r) {
  for (let y = 0; y < buf.h; y++) {
    for (let x = 0; x < buf.w; x++) {
      if (Math.hypot(x - cx, y - cy) <= r) {
        buf.alpha[(y * buf.w + x) * 4 + 3] = 255;
      }
    }
  }
}
function fillRect(buf, x0, y0, x1, y1) {
  for (let y = y0; y < y1; y++) {
    for (let x = x0; x < x1; x++) {
      if (x >= 0 && y >= 0 && x < buf.w && y < buf.h) {
        buf.alpha[(y * buf.w + x) * 4 + 3] = 255;
      }
    }
  }
}

console.log('\n== traceContour ==');
{
  const b = makeBuf(40, 40);
  fillRect(b, 10, 10, 30, 30);
  const c = Geo.traceContour(b.alpha, b.w, b.h);
  ok('方形掩码能追踪到闭合轮廓', c && c.length > 4, c ? c.length + ' 点' : 'null');
  ok('轮廓首尾相同（闭合）', c && c[0][0] === c[c.length - 1][0] && c[0][1] === c[c.length - 1][1]);
  let minX = 1e9, maxX = -1e9, minY = 1e9, maxY = -1e9;
  c.forEach(p => {
    minX = Math.min(minX, p[0]); maxX = Math.max(maxX, p[0]);
    minY = Math.min(minY, p[1]); maxY = Math.max(maxY, p[1]);
  });
  ok('包围盒约等于 10..30', minX >= 9 && maxX <= 30 && minY >= 9 && maxY <= 30,
     `${minX},${minY} .. ${maxX},${maxY}`);
}
{
  const b = makeBuf(60, 60);
  fillCircle(b, 30, 30, 20);
  const c = Geo.traceContour(b.alpha, b.w, b.h);
  ok('圆形掩码能追踪到轮廓', c && c.length > 4, c ? c.length + ' 点' : 'null');
  let minX = 1e9, maxX = -1e9;
  c.forEach(p => { minX = Math.min(minX, p[0]); maxX = Math.max(maxX, p[0]); });
  ok('圆左右端点约 10 / 50', minX >= 9 && minX <= 12 && maxX >= 48 && maxX <= 51,
     `${minX}..${maxX}`);
}
{
  const b = makeBuf(20, 20);
  const c = Geo.traceContour(b.alpha, b.w, b.h);
  ok('空掩码返回 null', c === null, String(c));
}
{
  // L 形：凹角必须被正确绕过
  const b = makeBuf(40, 40);
  fillRect(b, 10, 10, 20, 30);
  fillRect(b, 10, 10, 30, 20);
  const c = Geo.traceContour(b.alpha, b.w, b.h);
  ok('L 形凹角被绕过（轮廓点数 > 8）', c && c.length > 8, c ? c.length + ' 点' : 'null');
}

console.log('\n== traceBoundaries（缝隙追踪）==');
{
  const b = makeBuf(40, 40);
  fillRect(b, 10, 10, 30, 30);
  const loops = Geo.traceBoundaries(b.alpha, b.w, b.h);
  ok('方形产生 1 条闭合轮廓', loops.length === 1, loops.length + ' 条');
  const l = loops[0];
  ok('轮廓首尾相同', l[0][0] === l[l.length - 1][0] && l[0][1] === l[l.length - 1][1]);
  // 缝隙追踪的面积必须精确等于像素数
  const area = Math.abs(Geo.signedArea(l));
  ok('面积精确 = 20*20 = 400', Math.abs(area - 400) < 1e-6, area + '');
  let minX = 1e9, maxX = -1e9;
  l.forEach(p => { minX = Math.min(minX, p[0]); maxX = Math.max(maxX, p[0]); });
  ok('左右边界在 10 / 30', minX === 10 && maxX === 30, `${minX}..${maxX}`);
}
{
  // L 形：凹角必须保留
  const b = makeBuf(40, 40);
  fillRect(b, 10, 10, 20, 30);
  fillRect(b, 10, 10, 30, 20);
  const loops = Geo.traceBoundaries(b.alpha, b.w, b.h);
  const area = Math.abs(Geo.signedArea(loops[0]));
  // L 形 = 10*20 + 20*10 - 10*10 = 300
  ok('L 形面积 = 300', Math.abs(area - 300) < 1e-6, area + '');
}
{
  // 两个分离的块 -> 两条轮廓
  const b = makeBuf(60, 40);
  fillRect(b, 5, 5, 15, 15);
  fillRect(b, 30, 20, 45, 30);
  const loops = Geo.traceBoundaries(b.alpha, b.w, b.h);
  ok('分离的两块产生 2 条轮廓', loops.length === 2, loops.length + ' 条');
}
{
  // 对角相接的两个块 -> 不应黏成一坨（鞍点正确配对）
  const b = makeBuf(40, 40);
  fillRect(b, 10, 10, 15, 15);
  fillRect(b, 15, 15, 20, 20);
  const loops = Geo.traceBoundaries(b.alpha, b.w, b.h);
  ok('对角相接产生 2 条独立轮廓', loops.length === 2, loops.length + ' 条');
  loops.forEach((l, i) => {
    ok('  第' + (i + 1) + '条面积 = 25', Math.abs(Math.abs(Geo.signedArea(l)) - 25) < 1e-6,
       Math.abs(Geo.signedArea(l)) + '');
  });
}
{
  const b = makeBuf(20, 20);
  ok('空掩码无轮廓', Geo.traceBoundaries(b.alpha, b.w, b.h).length === 0);
}

console.log('\n== simplifyDP ==');
{
  const line = [];
  for (let i = 0; i <= 100; i++) line.push([i, 0]);
  const s = Geo.simplifyDP(line, 0.5);
  ok('共线 101 点抽成 2 点', s.length === 2, s.length + ' 点');
}
{
  const pts = [[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]];
  const s = Geo.simplifyDP(pts, 0.01);
  ok('方形 5 点保持 5 点', s.length === 5, s.length + ' 点');
}
{
  // 真实输入是「像素阶梯」边界：偏差约 0.5px，D-P 的强项
  const stair = [];
  for (let i = 0; i < 360; i++) {
    const a = i / 360 * Math.PI * 2;
    const r = 50 + (i % 2 ? 0.5 : -0.5);
    stair.push([50 + r * Math.cos(a), 50 + r * Math.sin(a)]);
  }
  stair.push([stair[0][0], stair[0][1]]);
  const s = Geo.simplifyClosed(stair, 1.0);
  ok('阶梯噪声圆抽稀到 < 40 点', s.length < 40, s.length + ' 点');
  ok('闭合版首尾相同', s[0][0] === s[s.length - 1][0] && s[0][1] === s[s.length - 1][1]);
  let maxErr = 0;
  for (const p of stair) {
    let d = 1e9;
    for (let i = 0; i < s.length - 1; i++) {
      d = Math.min(d, Geo.distToSeg(p, s[i], s[i + 1]));
    }
    maxErr = Math.max(maxErr, d);
  }
  ok('最大误差 <= 1.5px（eps=1）', maxErr <= 1.5, maxErr.toFixed(3) + ' px');
}
{
  // 交替噪声是 D-P 的已知弱点：eps 必须大于噪声幅度才能压下去
  // 这里验证「误差不受影响」—— 放大 eps 后误差仍在容差内
  const noisy = [];
  for (let i = 0; i < 360; i++) {
    const a = i / 360 * Math.PI * 2;
    const r = 50 + (i % 2 ? 1.5 : -1.5);
    noisy.push([50 + r * Math.cos(a), 50 + r * Math.sin(a)]);
  }
  noisy.push([noisy[0][0], noisy[0][1]]);
  const s = Geo.simplifyClosed(noisy, 3.0);
  ok('交替噪声在 eps=3 时可压到 < 40 点', s.length < 40, s.length + ' 点');
  let maxErr = 0;
  for (const p of noisy) {
    let d = 1e9;
    for (let i = 0; i < s.length - 1; i++) {
      d = Math.min(d, Geo.distToSeg(p, s[i], s[i + 1]));
    }
    maxErr = Math.max(maxErr, d);
  }
  ok('最大误差 <= 3.5px（eps=3）', maxErr <= 3.5, maxErr.toFixed(3) + ' px');
}

console.log('\n== maskToPolygon ==');
{
  const b = makeBuf(80, 80);
  fillCircle(b, 40, 40, 30);
  const poly = Geo.maskToPolygon(b.alpha, b.w, b.h, { eps: 1.0, maxPts: 180, scale: 80 });
  ok('能转出多边形', poly && poly.length >= 3, poly ? poly.length + ' 点' : 'null');
  ok('坐标归一化在 0~1', poly.every(p => p[0] >= 0 && p[0] <= 1 && p[1] >= 0 && p[1] <= 1));
  ok('顶点数 <= 上限', poly.length <= 180, poly.length + ' 点');
  // 面积应接近 π r² / (80²)
  let a = 0;
  for (let i = 0; i < poly.length; i++) {
    const p1 = poly[i], p2 = poly[(i + 1) % poly.length];
    a += p1[0] * p2[1] - p2[0] * p1[1];
  }
  a = Math.abs(a / 2);
  const expect = Math.PI * (30 / 80) ** 2;
  ok('面积误差 < 3%', Math.abs(a - expect) / expect < 0.03,
     `${a.toFixed(4)} vs ${expect.toFixed(4)}`);
}
{
  // 方形：面积必须精确
  const b = makeBuf(80, 80);
  fillRect(b, 20, 20, 60, 60);
  const poly = Geo.maskToPolygon(b.alpha, b.w, b.h, { eps: 0.5, maxPts: 180, scale: 80 });
  let a = 0;
  for (let i = 0; i < poly.length; i++) {
    const p1 = poly[i], p2 = poly[(i + 1) % poly.length];
    a += p1[0] * p2[1] - p2[0] * p1[1];
  }
  a = Math.abs(a / 2);
  const expect = (40 / 80) ** 2;
  ok('方形面积精确 = 0.25', Math.abs(a - expect) < 0.001, a.toFixed(6) + '');
}
{
  // 顶点数超限时应自动加大容差
  const b = makeBuf(200, 200);
  fillCircle(b, 100, 100, 90);
  const poly = Geo.maskToPolygon(b.alpha, b.w, b.h, { eps: 0.5, maxPts: 60, scale: 200 });
  ok('超限时自动压缩到 <= 60 点', poly.length <= 60, poly.length + ' 点');
}

console.log(`\n结果：${pass} 通过 / ${fail} 失败\n`);
process.exit(fail ? 1 : 0);
