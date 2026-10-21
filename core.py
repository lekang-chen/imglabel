"""
core.py —— 数据模型与几何工具（本项目最核心的模块）

【这个文件为什么重要】
答辩时你最核心的一句话是：
    "以实例掩码为唯一核心标注对象，检测框由掩码外接矩形自动生成，
     语义掩码由实例掩码按类别合并自动派生。"
本文件就是把这句话变成可执行代码的地方。

【设计要点】
1. 唯一真值：一张图的标注状态就是一个 Annotation 对象，里面装若干 Object。
2. 每个 Object 只存 polygon（一串归一化坐标点）和 class（类别名），
   它的"检测框"和"语义掩码"都不存，都是算出来的。
   -> 好处：人改一次，三种格式自动同步，绝不会对不上。
3. 所有坐标用「归一化浮点数」(0~1)，与图片实际像素尺寸解耦。
   -> 好处：换分辨率、导出时不需要重新缩放。

【坐标约定】
- polygon: [[x0,y0, x1,y1, ...]]，每个点都是 0~1 的浮点数
- 外接框 bbox: [xmin, ymin, xmax, ymax]，同样是 0~1 浮点数
- 掩码位图: PIL Image，尺寸 = 原图像素尺寸，模式 'L'
"""

import json
import os
import uuid
from collections import Counter

# ---------------------------------------------------------------------------
# 十类词表（任务书明确规定"不得更改"）
# 英文名是模型调用用名，顺序即 YOLO 训练时的类别编号 0~9
# ---------------------------------------------------------------------------
CLASSES = [
    "person",        # 人
    "chair",         # 椅子
    "dining table",  # 桌子
    "couch",         # 沙发
    "tv",            # 电视
    "laptop",        # 笔记本电脑
    "cup",           # 杯子
    "bottle",        # 瓶子
    "potted plant",  # 盆栽
    "book",          # 书
]

# 类别名 -> 编号（训练时用）
CLASS_TO_ID = {name: i for i, name in enumerate(CLASSES)}

# 界面上给每个类别分配的固定颜色（十六进制 RGB）
# 同一类别在任何图、任何模式下都是同一个颜色，方便肉眼核对
CLASS_COLORS = {
    "person":       (59, 130, 246),   # 蓝
    "chair":        (16, 185, 129),   # 绿
    "dining table": (245, 158, 11),   # 橙
    "couch":        (139, 92, 246),   # 紫
    "tv":           (236, 72, 153),   # 粉
    "laptop":       (14, 165, 233),   # 天蓝
    "cup":          (234, 179, 8),    # 黄
    "bottle":       (20, 184, 166),   # 青
    "potted plant": (132, 204, 22),   # 黄绿
    "book":         (217, 70, 239),   # 品红
}


def class_color(class_name):
    """取类别颜色；遇到词表外的类别给个灰色兜底。"""
    return CLASS_COLORS.get(class_name, (148, 163, 184))


# ---------------------------------------------------------------------------
# 几何工具
# ---------------------------------------------------------------------------

def polygon_bbox(polygon):
    """
    从多边形算出外接矩形（检测框）。

    为什么要单独一个函数？
    因为「检测框 = 掩码外接矩形」是本项目的核心设计。
    如果你把框单独存一份，人改了掩码忘了改框，两边就对不上了。
    算出来的东西永远一致。

    返回 [xmin, ymin, xmax, ymax]，坐标已归一化到 0~1。
    """
    if not polygon:
        return [0.0, 0.0, 0.0, 0.0]
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return [min(xs), min(ys), max(xs), max(ys)]


def polygon_area(polygon):
    """
    多边形面积（鞋带公式）。
    用途：过滤掉模型吐出来的极小噪声块（面积 < 阈值直接丢弃）。
    """
    if len(polygon) < 3:
        return 0.0
    total = 0.0
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        total += x1 * y2 - x2 * y1
    return abs(total) / 2.0


def point_in_polygon(x, y, polygon):
    """
    射线法判断点是否在多边形内。
    用途：画笔模式下，用户点一下某个位置，问「这里属于哪个对象」，
          从而选中它（而不是每次都靠点边界线，容易点不中）。
    """
    inside = False
    n = len(polygon)
    if n < 3:
        return False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if (yi > y) != (yj > y):
            # 边的 y 跨越了 y 这一水平线，交点横坐标
            x_cross = (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi
            if x < x_cross:
                inside = not inside
        j = i
    return inside


def simplify_polygon(polygon, max_points=180):
    """
    简化多边形，限制顶点数上限。

    为什么需要？
    预标注模型的掩码转成多边形后可能有几百上千个点，
    1) 画布渲染慢  2) 导出的 txt 文件巨大  3) 人工拖点改不动

    做法：Douglas-Peucker 算法（递归找最远的点，保留它，把线段两侧分开处理）。

    【为什么不用"相邻三点共线就删中间点"】
    之前用的是共线抽稀，问题是它只能删掉严格共线的点。
    掩码轮廓是像素级的锯齿，几乎没有三点严格共线，
    所以循环删不掉任何点，函数在 180 个点以上就"卡住"了——
    实测导出里出现了 519 顶点的多边形，直接超标。
    Douglas-Peucker 是"在误差范围内简化曲线"，保证一定能收敛到指定点数。

    【踩过的坑：闭合环不能直接喂给开放折线的 D-P】
    掩码轮廓是闭合环（首尾同点）。而 _dp 是按"开放折线"设计的：
    它把首尾连成一条线段，然后看各点到这条线的距离。
    闭合环首尾同点 -> 这条"线段"长度为零 -> 所有点到它的距离都很大
    -> 每轮都判定"必须保留"并继续递归，行为完全退化。
    实测 1000 点的圆被抽成9 点（近似正八边形），面积误差 9.9%。
    正确做法（见JS 版 canvas/geom.js 的 simplifyClosed，思路一致）：
    在"离起点最远的对角点"处把环切成两条开放链，分别 D-P，再拼回去。
    对角点保证两段都不是退化的零长链。

    误差 eps 不再写死 0.002，而是二分搜索「能让点数落到 max_points 以内的
    最小 eps」—— 这样形状精度优先，点数只作为上限约束。
    """
    if len(polygon) <= max_points:
        return polygon

    pts = [(float(p[0]), float(p[1])) for p in polygon]

    # 闭合环在 D-P 之前先去掉重复的尾点，否则会引入零长链。
    if abs(pts[0][0] - pts[-1][0]) < 1e-12 and abs(pts[0][1] - pts[-1][1]) < 1e-12:
        pts = pts[:-1]
    if len(pts) <= max_points:
        return [[p[0], p[1]] for p in pts]

    def _dp(points, eps):
        """对开放折线做 Douglas-Peucker 简化，返回保留点的索引列表。"""
        n = len(points)
        if n < 3:
            return list(range(n))
        # 找距离首尾连线最远的点
        x1, y1 = points[0]
        x2, y2 = points[-1]
        dx, dy = x2 - x1, y2 - y1
        norm = (dx * dx + dy * dy) ** 0.5
        max_dist, idx = -1.0, 0
        for i in range(1, n - 1):
            px, py = points[i]
            if norm < 1e-12:
                dist = ((px - x1) ** 2 + (py - y1) ** 2) ** 0.5
            else:
                # 点到直线的距离
                dist = abs(dy * px - dx * py + x2 * y1 - y2 * x1) / norm
            if dist > max_dist:
                max_dist, idx = dist, i
        if max_dist > eps:
            left = _dp(points[:idx + 1], eps)
            right = _dp(points[idx:], eps)
            return left[:-1] + [idx] + [idx + i for i in right[1:]]
        return [0, n - 1]

    def _dp_closed(points, eps):
        """闭合环的 D-P：在离首点最远的对角点处切成两半，分别抽稀再拼回。"""
        n = len(points)
        if n < 4:
            return list(range(n))
        x0, y0 = points[0]
        # 找离首点最远的点作为切分点（对角点，保证两段都是非退化链）
        cut, far = 0, -1.0
        for i in range(1, n):
            d = (points[i][0] - x0) ** 2 + (points[i][1] - y0) ** 2
            if d > far:
                far, cut = d, i
        if cut == 0:
            return list(range(n))
        # A: points[0..cut]，B: points[cut..n-1]再绕回 points[0]
        ka = _dp(points[:cut + 1], eps)
        kb = _dp(points[cut:] + [points[0]], eps)
        # 拼回：A 的全部 + B 的内部点（去掉两端，B 的两端已由 A 覆盖）
        keep = list(ka) + [cut + i for i in kb[1:-1]]
        # 去重并保持顺序
        out, seen = [], set()
        for i in keep:
            if i not in seen and 0 <= i < n:
                seen.add(i)
                out.append(i)
        out.sort()
        return out

    # 二分搜索最小的 eps，使简化后点数 <= max_points。
    # 【注意】这里绝不能因为「点数已经够少」就 break——
    #   那样二分只试了 mid=0.025 就退出，误差容差被放大到约 24 像素，
    #   形状会被压成近似矩形。必须继续收敛去逼近更小的 eps。
    lo, hi = 0.0, 0.05
    best = pts
    for _ in range(18):
        mid = (lo + hi) / 2
        keep = _dp_closed(pts, mid)
        cand = [pts[i] for i in keep]
        if len(cand) <= max_points:
            best = cand
            hi = mid
        else:
            lo = mid

    if len(best) < 3:
        return polygon[:max_points]
    return [[p[0], p[1]] for p in best]


# ---------------------------------------------------------------------------
# Object：一个标注对象
# ---------------------------------------------------------------------------

class Object:
    """
    一个标注对象 = 一个物体的一次实例分割标注。

    字段说明：
      id        : 唯一短 id，界面上用来标识选中项
      class_name: 类别名（必须是 CLASSES 里的十类之一）
      polygon   : 归一化多边形点集 [[x,y], ...]  ← 唯一的几何真值
      source    : 'model' 表示模型预标注生成的，'human' 表示人工新建的
      edited    : True 表示这个对象被人改过（改类别/改形状/移动过）
                  修正率统计就靠这个字段
    """

    def __init__(self, class_name, polygon, source="model", edited=False, obj_id=None):
        if class_name not in CLASS_TO_ID:
            raise ValueError(
                "类别 '%s' 不在任务书规定的十类词表内。可选：%s"
                % (class_name, ", ".join(CLASSES))
            )
        self.id = obj_id or uuid.uuid4().hex[:8]
        self.class_name = class_name
        self.polygon = [[float(p[0]), float(p[1])] for p in polygon]
        self.source = source
        self.edited = edited

    def to_dict(self):
        return {
            "id": self.id,
            "class": self.class_name,
            "polygon": self.polygon,
            "source": self.source,
            "edited": self.edited,
        }

    @staticmethod
    def from_dict(d):
        return Object(
            class_name=d["class"],
            polygon=d["polygon"],
            source=d.get("source", "model"),
            edited=d.get("edited", False),
            obj_id=d.get("id"),
        )

    def bbox(self):
        return polygon_bbox(self.polygon)


# ---------------------------------------------------------------------------
# Annotation：一张图的完整标注状态
# ---------------------------------------------------------------------------

class Annotation:
    """
    一张图片的标注状态 = 若干 Object 的集合。

    这就是「唯一真值」。三种导出格式全部从它派生：
      - 检测框    : [obj.bbox() for obj in objects]
      - 实例掩码  : 每个 obj 一个多边形栅格化成独立 PNG
      - 语义 PNG  : 把同类 obj 的掩码合并到同一张图的类别索引图
    """

    def __init__(self, image_name, width, height, objects=None, image_path=None):
        self.image_name = image_name
        self.width = width
        self.height = height
        self.objects = objects or []
        self.image_path = image_path

    # ---- 序列化：存成 JSON 落盘 ----
    def to_dict(self):
        return {
            "image": self.image_name,
            "width": self.width,
            "height": self.height,
            "objects": [o.to_dict() for o in self.objects],
        }

    @staticmethod
    def from_dict(d):
        return Annotation(
            image_name=d["image"],
            width=d["width"],
            height=d["height"],
            objects=[Object.from_dict(x) for x in d.get("objects", [])],
        )

    # ---- 常用查询 ----
    def find(self, obj_id):
        for o in self.objects:
            if o.id == obj_id:
                return o
        return None

    def class_counts(self):
        """各类别对象数量统计，用于界面右侧面板和提交信息。"""
        return dict(Counter(o.class_name for o in self.objects))

    def edited_count(self):
        """本图中被人实际修改过的对象数。"""
        return sum(1 for o in self.objects if o.edited)

    def total_count(self):
        return len(self.objects)


# ---------------------------------------------------------------------------
# 掩码栅格化：多边形 -> 位图（导出三种格式时用）
# ---------------------------------------------------------------------------

def rasterize_polygon(polygon, width, height):
    """
    把归一化多边形栅格化成一张二值位图（PIL Image, 模式 'L'）。

    用 OpenCV 的 fillPoly 做，效率高。
    依赖：opencv-python 的 cv2 + numpy 的 uint8 数组。
    """
    import cv2
    import numpy as np
    from PIL import Image

    mask = np.zeros((height, width), dtype=np.uint8)
    if len(polygon) < 3:
        return Image.fromarray(mask)

    # 归一化坐标 -> 像素坐标
    pts = np.array(
        [[int(round(p[0] * (width - 1))), int(round(p[1] * (height - 1)))]
         for p in polygon],
        dtype=np.int32,
    )
    cv2.fillPoly(mask, [pts], 1)
    return Image.fromarray(mask)


def semantic_from_objects(objects, width, height):
    """
    由实例掩码按类别合并，生成语义分割 PNG。

    语义分割的规则：同一个类别只管它是不是这个类，
    不区分类内不同个体（任务书附录明确写了这一点）。
    所以这里所有同类别的多边形都涂进同一个类别索引值里。

    返回：PIL Image（模式 'L'），每个像素存的是类别编号 0~9。
    类别编号 255 保留给「背景/未标注」，这是社区通用约定。

    【为什么用 'L' 而不是 'P'（调色板模式）】
    曾经写成 mode="P"，看起来没问题，但 PNG 存盘时会把像素值当成调色板索引，
    重新打开时索引被重新映射，类别编号全乱——自检立刻报"语义图只覆盖 2 个类别"。
    'L'（灰度）模式是逐像素存真实数值，存什么读出来就是什么，
    这才是我们需要的：像素值 == 类别编号。
    """
    import cv2
    import numpy as np
    from PIL import Image

    sem = np.full((height, width), 255, dtype=np.uint8)
    for obj in objects:
        if len(obj.polygon) < 3:
            continue
        cls_id = CLASS_TO_ID[obj.class_name]
        pts = np.array(
            [[int(round(p[0] * (width - 1))), int(round(p[1] * (height - 1)))]
             for p in obj.polygon],
            dtype=np.int32,
        )
        # 注意这里用 -1：语义图是「合并」，后面的同类目标也要写进来。
        # 用 1 会被后面的覆盖成 0，所以必须用 -1 表示「填这个值」。
        cv2.fillPoly(sem, [pts], int(cls_id), lineType=cv2.LINE_8)
    return Image.fromarray(sem, mode="L")


def instance_from_objects(objects, width, height):
    """
    生成实例掩码 PNG。

    社区常用做法：一张图里所有实例画在一张 PNG 上，
    不同实例用不同的灰度值区分（255 表示背景）。
    每个实例的类别和编号记录在同名 txt 里。

    返回：(PIL Image 灰度图, {实例值: (类别名, 序号)} 映射)

    【uint8 上限说明】
    灰度值上限 255，所以一张图最多存 254 个实例（0 留给背景）。
    超过就抛异常——宁可在导出阶段炸掉，也不要静默溢出把实例号写回绕，
    因为溢出会导致两个实例共用一个灰度值，训练时标签和掩码就对不上了。
    """
    import cv2
    import numpy as np
    from PIL import Image

    inst = np.zeros((height, width), dtype=np.uint8)
    mapping = {}
    value = 1
    for obj in objects:
        if len(obj.polygon) < 3:
            continue
        if value > 254:
            raise ValueError(
                "单张图的实例数超过 254（uint8 灰度值上限），"
                "请拆分该图或改用 16 位掩码")
        pts = np.array(
            [[int(round(p[0] * (width - 1))), int(round(p[1] * (height - 1)))]
             for p in obj.polygon],
            dtype=np.int32,
        )
        cv2.fillPoly(inst, [pts], value)
        mapping[value] = (obj.class_name, value - 1)
        value += 1
    return Image.fromarray(inst, mode="L"), mapping
