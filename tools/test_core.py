"""
tools/test_core.py —— 核心逻辑自测

【为什么需要这个文件】
任务书第五部分「开发流程与工业界对照」里有一条：
    "文档导出前的格式校验 -> 自动化测试 / CI"

这个文件就是那条对照的落地：把「导出格式对不对」这件事，
从「人工肉眼看」变成「跑一条命令就有结论」。

【怎么用】
  python tools/test_core.py

【测什么】
  1. 几何工具：外接框、面积、点在多边形内、多边形抽稀
  2. 掩码转多边形：像素掩码 -> 归一化多边形
  3. 三格式一致性：同一份标注，导出的三种格式对象数必须一致
  4. 格式合法性：坐标在 [0,1]、类别编号在 0~9、顶点数 >= 3
  5. 边界情况：空标注、多边形点太少、非法类别要报错
"""

import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import (CLASSES, Annotation, Object, instance_from_objects,
                  point_in_polygon, polygon_area, polygon_bbox,
                  semantic_from_objects, simplify_polygon)
from exporter import export_dataset, export_yolo_label, validate_dataset

PASS = 0
FAIL = 0


def check(name, cond, extra=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("  [PASS] %s" % name)
    else:
        FAIL += 1
        print("  [FAIL] %s  %s" % (name, extra))


def approx(a, b, eps=1e-6):
    return abs(a - b) < eps


# ---------------------------------------------------------------------------
def test_geometry():
    print("\n[1] 几何工具")
    square = [[0.2, 0.2], [0.6, 0.2], [0.6, 0.6], [0.2, 0.6]]
    bbox = polygon_bbox(square)
    check("外接框 x 范围", approx(bbox[0], 0.2) and approx(bbox[2], 0.6))
    check("外接框 y 范围", approx(bbox[1], 0.2) and approx(bbox[3], 0.6))
    check("面积 = 0.16", approx(polygon_area(square), 0.16, 1e-4))
    check("点在多边形内", point_in_polygon(0.4, 0.4, square))
    check("点在多边形外", not point_in_polygon(0.9, 0.9, square))
    check("空多边形面积为 0", polygon_area([]) == 0)

    # 抽稀：共线的点应该被删掉
    line = [[i / 100.0, 0.5] for i in range(100)]
    line.append([0.99, 0.9])
    line.append([0.01, 0.9])
    simplified = simplify_polygon(line, max_points=20)
    check("抽稀后顶点数 <= 20", len(simplified) <= 20, "实际 %d" % len(simplified))
    check("抽稀后仍是闭合多边形", len(simplified) >= 3)


def test_mask_to_polygon():
    print("\n[2] 掩码转多边形")
    from prelabel import mask_to_polygon

    import numpy as np
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:60, 30:70] = 1

    poly = mask_to_polygon(mask)
    check("生成了多边形", len(poly) >= 3, "顶点数 %d" % len(poly))
    if poly:
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        check("坐标在 [0,1]", all(0 <= x <= 1 for x in xs) and all(0 <= y <= 1 for y in ys))
        check("x 范围约 0.3~0.7", approx(min(xs), 0.3, 0.05) and approx(max(xs), 0.7, 0.05))
        check("y 范围约 0.2~0.6", approx(min(ys), 0.2, 0.05) and approx(max(ys), 0.6, 0.05))

    empty = np.zeros((50, 50), dtype=np.uint8)
    check("空掩码返回空列表", mask_to_polygon(empty) == [])

    tiny = np.zeros((100, 100), dtype=np.uint8)
    tiny[0:2, 0:2] = 1
    check("极小碎片被过滤", mask_to_polygon(tiny) == [])


def test_three_formats():
    print("\n[3] 三格式一致性")
    objs = [
        Object("person", [[0.1, 0.1], [0.3, 0.1], [0.3, 0.5], [0.1, 0.5]]),
        Object("chair", [[0.5, 0.2], [0.8, 0.2], [0.8, 0.6], [0.5, 0.6]]),
        Object("cup", [[0.4, 0.05], [0.45, 0.05], [0.45, 0.12], [0.4, 0.12]]),
    ]
    ann = Annotation("test.jpg", 200, 150, objects=objs)

    # YOLO 标签
    label = export_yolo_label(ann, has_seg=True)
    lines = [x for x in label.strip().split("\n") if x]
    check("YOLO 标签 3 行", len(lines) == 3, "实际 %d" % len(lines))
    for ln in lines:
        parts = ln.split()
        cls_id = int(parts[0])
        coords = [float(x) for x in parts[1:]]
        ok_cls = 0 <= cls_id < 10
        ok_coord = all(0 <= c <= 1 for c in coords)
        ok_count = len(coords) >= 6 and len(coords) % 2 == 0
        check("标签行合法 cls=%d" % cls_id, ok_cls and ok_coord and ok_count)

    # 检测格式（由掩码外接矩形派生）
    det = export_yolo_label(ann, has_seg=False)
    det_lines = [x for x in det.strip().split("\n") if x]
    check("检测格式也是 3 行", len(det_lines) == 3)
    for ln in det_lines:
        parts = ln.split()
        check("检测格式 5 个数值", len(parts) == 5, "实际 %d" % len(parts))

    # 实例掩码
    inst, mapping = instance_from_objects(objs, 200, 150)
    check("实例掩码 3 个实例", len(mapping) == 3, "实际 %d" % len(mapping))
    check("实例值不为 0", all(v > 0 for v in mapping))

    # 语义 PNG
    sem = semantic_from_objects(objs, 200, 150)
    vals = set(sem.getdata())
    legal = all(v == 255 or v in range(10) for v in vals)
    check("语义 PNG 像素值合法", legal, str(sorted(vals)))
    check("语义 PNG 含背景 255", 255 in vals)
    # person 的类别编号应为 0
    check("语义图含 person 类(0)", 0 in vals)


def test_export_and_validate():
    print("\n[4] 导出 + 格式自检")
    tmp = tempfile.mkdtemp(prefix="imglabel_test_")
    try:
        img_dir = os.path.join(tmp, "images")
        os.makedirs(img_dir, exist_ok=True)
        from PIL import Image
        for i in range(4):
            Image.new("RGB", (120, 90), (200, 180, 160)).save(
                os.path.join(img_dir, "img%d.jpg" % i))

        anns = []
        for i in range(4):
            objs = [Object("person", [[0.1, 0.1], [0.5, 0.1], [0.5, 0.6], [0.1, 0.6]]),
                    Object("chair", [[0.55, 0.2], [0.9, 0.2], [0.9, 0.7], [0.55, 0.7]])]
            anns.append(Annotation("img%d.jpg" % i, 120, 90, objects=objs))

        out_dir = os.path.join(tmp, "dataset")
        stats = export_dataset(anns, out_dir, img_dir, val_ratio=0.25)
        check("导出 4 张", stats["total_images"] == 4)
        check("训练集 3 张", stats["train_images"] == 3, str(stats["train_images"]))
        check("验证集 1 张", stats["val_images"] == 1, str(stats["val_images"]))
        check("对象总数 8", stats["total_objects"] == 8, str(stats["total_objects"]))
        check("data.yaml 存在", os.path.exists(os.path.join(out_dir, "data.yaml")))

        for sub in ("images/train", "labels/train", "masks/train", "semantic/train"):
            p = os.path.join(out_dir, sub.replace("/", os.sep))
            check("%s 存在且非空" % sub,
                  os.path.isdir(p) and len(os.listdir(p)) > 0)

        ok, problems, counts = validate_dataset(out_dir)
        check("格式自检通过", ok, "; ".join(problems[:3]))
        check("标签对象数 = 8", counts["label_objects"] == 8, str(counts["label_objects"]))
        check("掩码实例数 = 8", counts["mask_instances"] == 8, str(counts["mask_instances"]))

        # 人为破坏一个标签，应被自检抓到
        lbl_dir = os.path.join(out_dir, "labels", "train")
        victim = os.path.join(lbl_dir, os.listdir(lbl_dir)[0])
        with open(victim, "a", encoding="utf-8") as f:
            f.write("99 1.5 1.5\n")
        ok2, problems2, _ = validate_dataset(out_dir)
        check("损坏标签被自检发现", not ok2 and len(problems2) > 0)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_edge_cases():
    print("\n[5] 边界情况")
    # 非法类别必须报错
    try:
        Object("dog", [[0, 0], [1, 0], [1, 1]])
        check("非法类别抛异常", False, "没抛异常")
    except ValueError:
        check("非法类别抛异常", True)

    # 空标注
    ann = Annotation("empty.jpg", 100, 100, objects=[])
    check("空标注对象数 0", ann.total_count() == 0)
    check("空标注导出为空字符串", export_yolo_label(ann) == "")

    # 顶点数不足的对象不写进标签
    ann2 = Annotation("x.jpg", 100, 100, objects=[Object("person", [[0.1, 0.1]])])
    check("顶点不足被跳过", export_yolo_label(ann2).strip() == "")

    # edited 标记统计
    objs = [Object("person", [[0, 0], [1, 0], [1, 1]]),
            Object("cup", [[0, 0], [1, 0], [1, 1]])]
    objs[0].edited = True
    ann3 = Annotation("y.jpg", 50, 50, objects=objs)
    check("修正计数正确", ann3.edited_count() == 1)
    check("总数正确", ann3.total_count() == 2)
    cc = ann3.class_counts()
    check("类别统计正确", cc.get("person") == 1 and cc.get("cup") == 1)


def test_png_roundtrip():
    """
    [7] PNG 存盘再读回，值必须不变。

    这是真实踩过的坑：语义 PNG 原来用 mode="P"（调色板模式）保存，
    PNG 把像素值当成调色板索引，重新打开时索引被重映射，类别编号全乱，
    自检直接报"语义图只覆盖 2 个类别"。

    这个测试就是防止它再犯：存盘 -> 读回 -> 值必须完全一致。
    """
    print("\n[7] PNG 存盘往返一致性")
    import tempfile
    from core import semantic_from_objects, instance_from_objects
    from PIL import Image

    objs = [Object("person", [[0.1, 0.1], [0.5, 0.1], [0.5, 0.5], [0.1, 0.5]]),
            Object("cup", [[0.6, 0.6], [0.9, 0.6], [0.9, 0.9], [0.6, 0.9]]),
            Object("book", [[0.2, 0.7], [0.4, 0.7], [0.4, 0.85], [0.2, 0.85]])]

    # --- 语义图 ---
    sem = semantic_from_objects(objs, 100, 100)
    check("语义图用 L 模式（不是调色板 P）", sem.mode == "L", "mode=%s" % sem.mode)
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "sem.png")
        sem.save(p)
        back = Image.open(p)
        check("语义图存回后 mode 仍为 L", back.mode == "L", "mode=%s" % back.mode)
        vals = set(back.getdata())
        check("语义图存回后类别编号未丢失",
              {0, 6, 9, 255} <= vals, str(sorted(vals)))
        check("语义图覆盖 3 个前景类别 + 背景",
              len([v for v in vals if v != 255]) == 3, str(sorted(vals)))

    # --- 实例图 ---
    inst, mapping = instance_from_objects(objs, 100, 100)
    check("实例映射表条数 = 对象数", len(mapping) == 3, str(mapping))
    check("实例映射表类别正确",
          mapping[1][0] == "person" and mapping[2][0] == "cup"
          and mapping[3][0] == "book", str(mapping))

    # --- 实例数超过 uint8 上限必须报错，而不是静默溢出 ---
    many = [Object("cup", [[0.0, 0.0], [0.01, 0.0], [0.01, 0.01], [0.0, 0.01]])
            for _ in range(300)]
    try:
        instance_from_objects(many, 100, 100)
        check("实例数超 254 时抛异常", False, "没抛异常")
    except ValueError:
        check("实例数超 254 时抛异常", True)


def test_classes():
    print("\n[6] 十类词表")
    expect = ["person", "chair", "dining table", "couch", "tv",
              "laptop", "cup", "bottle", "potted plant", "book"]
    check("词表完全匹配任务书", CLASSES == expect, str(CLASSES))
    check("共 10 类", len(CLASSES) == 10)


def test_simplify_closed_ring():
    """闭合环抽稀不能退化成矩形（真实踩过的坑）。

    背景：D-P 原本按开放折线设计，闭合环首尾同点会让首尾连线成为零长线段，
    所有点到它的距离都很大 -> 每轮都判定必须保留 -> 行为退化。
    实测 1000 点的圆被抽成 9 点（正八边形），面积误差 9.9%，
    视觉上就是个方块，正是「描边看起来是矩形」的根因之一。
    """
    print("\n[7] 闭合环抽稀精度")
    import math
    from core import simplify_polygon

    def area(p):
        a = 0.0
        for i in range(len(p)):
            x1, y1 = p[i]
            x2, y2 = p[(i + 1) % len(p)]
            a += x1 * y2 - x2 * y1
        return abs(a) / 2.0

    # 圆：1000 点 -> 应保留上百点，且面积几乎无损
    n, r = 1000, 0.3
    circle = [[0.5 + r * math.cos(2 * math.pi * i / n),
               0.5 + r * math.sin(2 * math.pi * i / n)] for i in range(n)]
    out = simplify_polygon([list(p) for p in circle], max_points=180)
    err = abs(area(circle) - area(out)) / area(circle)
    check("圆抽稀后点数 > 40（未被压成方形）", len(out) > 40, "实际 %d 点" % len(out))
    check("圆抽稀面积误差 < 1%", err < 0.01, "实际 %.3f%%" % (err * 100))

    # 不规则闭合环
    import random
    random.seed(1)
    blob = []
    for i in range(300):
        t = 2 * math.pi * i / 300
        blob.append([0.5 + 0.2 * math.cos(t) + 0.03 * random.random(),
                     0.5 + 0.35 * math.sin(t)])
    out2 = simplify_polygon([list(p) for p in blob], max_points=180)
    err2 = abs(area(blob) - area(out2)) / area(blob)
    check("不规则环抽稀后点数 > 60", len(out2) > 60, "实际 %d 点" % len(out2))
    check("不规则环面积误差 < 1%", err2 < 0.01, "实际 %.3f%%" % (err2 * 100))

    # 顶点数上限必须真的生效（180 是上限，不是目标）
    check("抽稀后不超过 max_points 上限",
          len(out) <= 180 and len(out2) <= 180, "%d/%d" % (len(out), len(out2)))

    # 简单多边形必须原样保留，不能被"简化"掉
    L = [[0.1, 0.1], [0.5, 0.1], [0.5, 0.3], [0.3, 0.3], [0.3, 0.7], [0.1, 0.7]]
    out3 = simplify_polygon([list(p) for p in L], max_points=180)
    check("6 点 L 形原样保留", len(out3) == 6, "实际 %d 点" % len(out3))

    # 首尾重复点的闭合环（cv2 轮廓常见）不能被吃掉
    dup = [list(p) for p in circle] + [[circle[0][0], circle[0][1]]]
    out4 = simplify_polygon(dup, max_points=180)
    check("首尾重复点已去除且形状正常",
          len(out4) > 40 and err < 0.01, "实际 %d 点" % len(out4))


def test_export_clamps_out_of_range():
    """导出时坐标越界必须被夹到 [0,1]（真实踩过的坑）。

    现象：界面点「导出数据集」报「格式校验 有问题」，
    列出 6 处 `坐标 -0.0009 超出 [0,1]`。

    根因：画笔/橡皮改形状走的是前端 geom.js 的 maskToPolygon，
    轮廓沿「像素缝隙」追踪，贴边的物体会产出 -0.5 或 w-0.5 的坐标
    （半个像素在画布外），除以宽度就变成 -0.0005 这种负数。
    YOLO 规范要求坐标在 [0,1]，越界会被校验判为有问题，
    部分训练器还会直接跳过该行（等于白标了一个物体）。

    修法：前端生成时夹取（治本）+ 导出时再兜一次（保住历史脏数据）。
    这里测的是导出侧的兜底：即使标注文件里已经存了负坐标，
    导出的 txt 也必须全部落在 [0,1]。
    """
    print("\n[8] 导出坐标越界夹取")
    tmp = tempfile.mkdtemp(prefix="imglabel_clamp_")
    try:
        img_dir = os.path.join(tmp, "images")
        os.makedirs(img_dir, exist_ok=True)
        from PIL import Image
        # 用 2 张图：导出器会强制留至少 1 张给训练集，
        # 只放 1 张会导致验证集为空、自检报「验证集为空」——
        # 那是切分策略的正常行为，不是坐标问题。
        for name in ("edge.jpg", "edge2.jpg"):
            Image.new("RGB", (120, 90), (180, 180, 180)).save(
                os.path.join(img_dir, name))

        # 故意构造越界多边形：模拟贴边物体的 -0.5 像素
        bad_poly = [[-0.0005, -0.0009], [0.5, -0.0026], [1.0005, 0.5],
                    [0.5, 1.0006], [-0.0023, 0.5]]
        anns = [Annotation(n, 120, 90, objects=[Object("book", list(bad_poly))])
                for n in ("edge.jpg", "edge2.jpg")]
        out_dir = os.path.join(tmp, "dataset")
        export_dataset(anns, out_dir, img_dir, val_ratio=0.5)

        lbl_dir = os.path.join(out_dir, "labels", "val")
        if not (os.path.isdir(lbl_dir) and os.listdir(lbl_dir)):
            lbl_dir = os.path.join(out_dir, "labels", "train")
        names = os.listdir(lbl_dir)
        check("导出了标签文件", len(names) >= 1, "实际 %d 个" % len(names))
        f = names[0]
        with open(os.path.join(lbl_dir, f), encoding="utf-8") as fh:
            nums = [float(v) for v in fh.read().split()[1:]]

        check("导出后坐标为 %.4f 个值" % len(nums), len(nums) == 10)
        check("全部坐标落在 [0,1]",
              all(0.0 <= v <= 1.0 for v in nums),
              "越界值: %s" % [v for v in nums if not 0.0 <= v <= 1.0])

        ok, problems, _ = validate_dataset(out_dir)
        check("越界标注导出后仍通过格式自检", ok, "; ".join(problems[:3]))

        # 检测格式（外接矩形）也要夹
        from exporter import export_yolo_label
        det_line = export_yolo_label(anns[0], has_seg=False)
        det_nums = [float(v) for v in det_line.split()[1:]]
        check("检测格式坐标也在 [0,1]",
              all(0.0 <= v <= 1.0 for v in det_nums),
              "越界值: %s" % [v for v in det_nums if not 0.0 <= v <= 1.0])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    print("=" * 60)
    print("核心逻辑自测")
    print("=" * 60)
    test_classes()
    test_geometry()
    test_mask_to_polygon()
    test_three_formats()
    test_export_and_validate()
    test_edge_cases()
    test_png_roundtrip()
    test_simplify_closed_ring()
    test_export_clamps_out_of_range()
    print("\n" + "=" * 60)
    print("结果: %d 通过, %d 失败" % (PASS, FAIL))
    print("=" * 60)
    sys.exit(1 if FAIL else 0)
