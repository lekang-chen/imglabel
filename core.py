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
# 图片扩展名白名单：全项目唯一的一处定义
#
# 【为什么必须只写一份】上传接口和「列出图片」曾经各写各的：上传收了
# .webp，列表只认 jpg/jpeg/png/bmp。手机相册上传 webp 时，文件确实
# 落进了 data/images/，但列表里看不到；紧接着自动预标注拿这个文件名
# 去和列表求交集，过滤完是空集，报「data/images/ 目录里没有图片」——
# 用户看到的就是这条自相矛盾的提示（明明刚提示上传成功）。
# 所以收口到这里，app / prelabel / train 一律引这一份。
# ---------------------------------------------------------------------------
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


# ---------------------------------------------------------------------------
# 进程探活：全项目唯一的一处定义（app 判训练进程、autolearn 判学习锁）
#
# 【为什么不能用 os.kill(pid, 0)】
# 那是 POSIX 的「发 0 号信号探活」语义，Windows 上没有这个语义，
# os.kill 会直接走 TerminateProcess —— 探活变成谋杀，训练当场被干掉。
# 改用 OpenProcess + WaitForSingleObject：等不到信号（WAIT_TIMEOUT=0x102）
# 才说明进程还活着。
# ---------------------------------------------------------------------------
def pid_alive(pid):
    """进程还在不在。用于区分「还在跑」和「进程没了但没写完成标记」。"""
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    try:
        if os.name == "nt":
            import ctypes

            SYNCHRONIZE = 0x00100000
            h = ctypes.windll.kernel32.OpenProcess(SYNCHRONIZE, False, pid)
            if not h:
                return False
            try:
                return ctypes.windll.kernel32.WaitForSingleObject(h, 0) == 0x102
            finally:
                ctypes.windll.kernel32.CloseHandle(h)
        os.kill(pid, 0)
        return True
    except OSError:
        return False


# ---------------------------------------------------------------------------
# 类别词表：存在 data/classes.json，界面上可以增、删、改名、换色
#
# 【为什么从写死的常量改成文件】
# 原来这里是十类常量，注释写着「任务书规定不得更改」。实际用起来第一个问题
# 就是「bed 怎么没标上」——床根本不在词表里。预标注用的是 YOLO-World，
# 开放词表检测器，它**只找你给它的词**，词表外的物体一律不输出。
# 所以「漏标」很多时候不是模型不行，是词表里没这个词。
#
# 【为什么改名和调顺序是安全的】
# 标注对象按**类别名**存，不按编号。改名只影响显示与导出映射（同步一遍已有
# 对象即可）；调顺序只影响导出的类别编号，重训一次就一致了。
#
# 【为什么删除要拦一道】
# 删掉一个还有对象在用的类别，那些对象导出时会被静默跳过——数据就这么没了，
# 而且没人会发现。所以删除前先数一遍，还在用就要求先把它们改到别的类。
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
CLASSES_PATH = os.path.join(DATA_DIR, "classes.json")

# 出厂十类（任务书词表）。只在 data/classes.json 不存在或损坏时用来兜底，
# 运行时判断「某个类别合不合法」一律走 get_classes()，不要用这个常量。
DEFAULT_CLASSES = [
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

# 出厂配色（十六进制 RGB）。新增类别由界面选色，没选就用中性灰。
DEFAULT_CLASS_COLORS = {
    "person":       "#3b82f6",   # 蓝
    "chair":        "#10b981",   # 绿
    "dining table": "#f59e0b",   # 橙
    "couch":        "#8b5cf6",   # 紫
    "tv":           "#ec4899",   # 粉
    "laptop":       "#0ea5e9",   # 天蓝
    "cup":          "#eab308",   # 黄
    "bottle":       "#14b8a6",   # 青
    "potted plant": "#84cc16",   # 黄绿
    "book":         "#d946ef",   # 品红
}

FALLBACK_CLASS_COLOR = "#94a3b8"   # 词表外/未知类别：中性灰

# 读盘缓存。键是文件 mtime——文件没变就不重复解析。
# 【为什么要缓存】预标注每处理一张图都要问一次「这个类别在不在词表里」，
# 150 张图就是 150 次读盘；训练导出时还要按类别统计。没必要反复解析 JSON。
_CLASS_CACHE = {"stamp": None, "items": None, "path": None}


def _default_class_items():
    return [{"name": n, "color": DEFAULT_CLASS_COLORS[n]} for n in DEFAULT_CLASSES]


def _clean_class_items(raw):
    """把任意形状的输入整理成 [{"name","color"}]，顺手去重、丢掉空名。"""
    if isinstance(raw, dict):
        raw = raw.get("classes", [])
    if not isinstance(raw, list):
        raise TypeError("classes 必须是数组")
    items, seen = [], set()
    for entry in raw:
        if isinstance(entry, str):
            name, color = entry, FALLBACK_CLASS_COLOR
        elif isinstance(entry, dict):
            name = str(entry.get("name", "")).strip()
            color = str(entry.get("color") or FALLBACK_CLASS_COLOR).strip()
        else:
            continue
        if not name or name in seen:
            continue
        if not color.startswith("#") or len(color) not in (4, 7):
            color = FALLBACK_CLASS_COLOR
        seen.add(name)
        items.append({"name": name, "color": color})
    return items


def load_class_items(path=None):
    """读当前词表，返回 [{"name": 类别名, "color": "#rrggbb"}]。

    文件不存在（还没改过词表）或内容坏了，都回落到出厂十类——
    词表读不出来不能让整个系统起不来，界面上还能改回去。
    """
    path = path or CLASSES_PATH
    try:
        stamp = os.path.getmtime(path)
    except OSError:
        return _default_class_items()

    cache = _CLASS_CACHE
    if cache["stamp"] == stamp and cache["path"] == path and cache["items"]:
        return [dict(it) for it in cache["items"]]

    try:
        with open(path, "r", encoding="utf-8") as f:
            items = _clean_class_items(json.load(f))
        if not items:
            return _default_class_items()
    except (OSError, ValueError, TypeError):
        return _default_class_items()

    cache.update({"stamp": stamp, "path": path,
                  "items": [dict(it) for it in items]})
    return [dict(it) for it in items]


def save_class_items(items, path=None):
    """写回词表（原子替换），并让缓存立刻失效。返回写入的规整结果。"""
    path = path or CLASSES_PATH
    clean = _clean_class_items(items)
    if not clean:
        raise ValueError("词表不能为空，至少要留一个类别")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"classes": clean}, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp, path)
    _CLASS_CACHE.update({"stamp": None, "items": None, "path": None})
    return clean


def get_classes(path=None):
    """当前词表的类别名列表（顺序即导出/训练时的类别编号）。"""
    return [it["name"] for it in load_class_items(path)]


def get_class_colors(path=None):
    """当前词表的 {类别名: (r, g, b)}。"""
    out = {}
    for it in load_class_items(path):
        out[it["name"]] = hex_to_rgb(it["color"])
    return out


def class_id_of(name, path=None):
    """类别名 -> 编号；不在词表里返回 -1（调用方自己决定是跳过还是报错）。"""
    try:
        return get_classes(path).index(name)
    except ValueError:
        return -1


def hex_to_rgb(value):
    """'#3b82f6' -> (59, 130, 246)；认不出来就给兜底灰。"""
    s = str(value or "").strip().lstrip("#")
    if len(s) == 3:
        s = "".join(ch * 2 for ch in s)
    if len(s) != 6:
        s = FALLBACK_CLASS_COLOR.lstrip("#")
    try:
        return tuple(int(s[i:i + 2], 16) for i in (0, 2, 4))
    except ValueError:
        return hex_to_rgb(FALLBACK_CLASS_COLOR)


def class_color(class_name, path=None):
    """取类别颜色；词表外的类别给中性灰（不会报错，画图要用）。"""
    return get_class_colors(path).get(class_name, hex_to_rgb(FALLBACK_CLASS_COLOR))


# 兼容旧代码：这两个常量等价于「出厂十类」，只用于工具脚本与默认值。
# 运行时判断合法性请用 class_id_of() / get_classes()，否则改过词表后会对不上。
CLASSES = list(DEFAULT_CLASSES)
CLASS_TO_ID = {name: i for i, name in enumerate(DEFAULT_CLASSES)}
CLASS_COLORS = {k: hex_to_rgb(v) for k, v in DEFAULT_CLASS_COLORS.items()}



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
      class_name: 类别名（必须是当前词表 data/classes.json 里的一个）
      polygon   : 归一化多边形点集 [[x,y], ...]  ← 唯一的几何真值
      source    : 'model' 表示模型预标注生成的，'human' 表示人工新建的
      edited    : True 表示这个对象被人改过（改类别/改形状/移动过）
                  修正率统计就靠这个字段
    """

    def __init__(self, class_name, polygon, source="model", edited=False,
                 obj_id=None, strict=True):
        # 【strict 为什么要留一个 False】
        # 人在界面上新建对象必须严格——类别一定来自当前词表。
        # 但读旧标注文件不能严格：词表被改过（比如删了 bed）之后，
        # 历史上标过 bed 的图再打开就会抛异常，整张图都读不出来。
        # 所以 from_dict 走 strict=False：先原样读进来，界面上把它标成
        # 「词表外类别」提醒处理，导出时跳过并计数——数据不会因为改词表就丢。
        if strict and class_id_of(class_name) < 0:
            raise ValueError(
                "类别 '%s' 不在当前词表内。可选：%s"
                % (class_name, ", ".join(get_classes()))
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
            strict=False,     # 旧文件里可能是已从词表移除的类别，先读进来
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
        # 词表外的类别（改过词表留下的历史对象）不画进语义图：
        # 给它编号只会污染导出的 mask，跳过并让 exporter 去记账。
        cls_id = class_id_of(obj.class_name)
        if cls_id < 0:
            continue
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
