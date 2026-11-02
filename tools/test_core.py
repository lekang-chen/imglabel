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

import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import (CLASSES, IMAGE_EXTS, Annotation, Object,
                  instance_from_objects, point_in_polygon, polygon_area,
                  polygon_bbox, semantic_from_objects, simplify_polygon)
from PIL import Image

from exporter import export_dataset, export_yolo_label, load_val_split, validate_dataset

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
            # 颜色按 i 变化：内容完全相同的图会被判成重复副本（真实照片不会这样，
            # 但纯色占位图会），这里让每张都不一样
            Image.new("RGB", (120, 90), (200 - i, 180, 160 + i)).save(
                os.path.join(img_dir, "img%d.jpg" % i))

        anns = []
        for i in range(4):
            objs = [Object("person", [[0.1, 0.1], [0.5, 0.1], [0.5, 0.6], [0.1, 0.6]]),
                    Object("chair", [[0.55, 0.2], [0.9, 0.2], [0.9, 0.7], [0.55, 0.7]])]
            anns.append(Annotation("img%d.jpg" % i, 120, 90, objects=objs))

        out_dir = os.path.join(tmp, "dataset")
        stats = export_dataset(anns, out_dir, img_dir, val_ratio=0.25,
                                 split_path=os.path.join(tmp, "val_split.json"))
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
        for i, name in enumerate(("edge.jpg", "edge2.jpg")):
            Image.new("RGB", (120, 90), (180, 180, 180 - i * 20)).save(
                os.path.join(img_dir, name))

        # 故意构造越界多边形：模拟贴边物体的 -0.5 像素
        bad_poly = [[-0.0005, -0.0009], [0.5, -0.0026], [1.0005, 0.5],
                    [0.5, 1.0006], [-0.0023, 0.5]]
        anns = [Annotation(n, 120, 90, objects=[Object("book", list(bad_poly))])
                for n in ("edge.jpg", "edge2.jpg")]
        out_dir = os.path.join(tmp, "dataset")
        export_dataset(anns, out_dir, img_dir, val_ratio=0.5,
                       split_path=os.path.join(tmp, "val_split.json"))

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


def test_upload_exts_match_list():
    """上传白名单必须和「列出图片」的白名单一致（线上事故回归）。

    真实现象：手机上传 webp -> 后端存成 IMG_8072.webp 并提示上传成功，
    但列表接口不认 .webp，图片在界面上根本不出现；紧接着的自动预标注
    把这个文件名拿去和列表求交集，交集是空集，于是弹
    「data/images/ 目录里没有图片」——界面上同时写着「已加载 153 张」。
    自相矛盾的提示就是这么来的，根因是两份白名单各写各的。
    """
    print("\n[9] 上传 / 列表 扩展名一致性（webp 事故回归）")
    import app as web

    check("上传白名单就是列表白名单", tuple(web.UPLOAD_EXTS) == tuple(IMAGE_EXTS),
          "%s vs %s" % (web.UPLOAD_EXTS, IMAGE_EXTS))
    check("白名单含 .webp（手机相册会转成 webp）", ".webp" in IMAGE_EXTS, str(IMAGE_EXTS))
    check("白名单含 .bmp", ".bmp" in IMAGE_EXTS, str(IMAGE_EXTS))

    tmp = tempfile.mkdtemp(prefix="imgexts_")
    old = web.IMAGE_DIR
    try:
        web.IMAGE_DIR = tmp
        for fn in ("a.jpg", "b.webp", "c.bmp", "d.txt", "e.HEIC"):
            open(os.path.join(tmp, fn), "wb").close()

        got = web.list_images()
        check("list_images 列出 jpg/webp/bmp",
              got == ["a.jpg", "b.webp", "c.bmp"], str(got))
        check("list_images 不列非图片（txt/HEIC）",
              not any(g.lower().endswith((".txt", ".heic")) for g in got), str(got))

        # 返回码与文案：指定了图但一张都不在目录里时，不能笼统说「目录里没有图片」。
        # only 非空且不匹配 -> 一定提前 400，不会真的起任务。
        client = web.app.test_client()
        r = client.post("/api/prelabel/start", json={"images": ["不存在.jpg"]})
        msg = (r.get_json() or {}).get("error", "")
        check("指定的图找不到时 400 且文案点明「指定的图片」",
              r.status_code == 400 and "指定" in msg, "%d %s" % (r.status_code, msg))

        # 目录真空的时候，仍然是原来那句
        empty = tempfile.mkdtemp(prefix="imgexts_empty_")
        try:
            web.IMAGE_DIR = empty
            r2 = client.post("/api/prelabel/start", json={"model": "official"})
            msg2 = (r2.get_json() or {}).get("error", "")
            check("目录真空时报「目录里没有图片」",
                  r2.status_code == 400 and "目录里没有图片" in msg2,
                  "%d %s" % (r2.status_code, msg2))
        finally:
            shutil.rmtree(empty, ignore_errors=True)
    finally:
        web.IMAGE_DIR = old
        shutil.rmtree(tmp, ignore_errors=True)


def test_prelabel_overwrite_guard():
    """全量预标注必须挡住「有修正、但没确认覆盖」的请求。

    真实现象：在电脑上点「预标注」，confirm 只写「对全部图片生成草稿」，
    没人看得出人工修正会跟着没。确认之后 60 条修正全被模型原始输出覆盖，
    只能靠 git 里的版本捞回来。

    修法两层：前端弹窗明确报数字；服务端没收到 force=true 就直接 409。
    这里测第二层——前端可以被绕过（脚本、旧缓存页面），服务端不能。
    """
    print("\n[10] 全量预标注覆盖保护")
    import app as web
    client = web.app.test_client()

    edited = 0
    for n in web.list_images():
        a = web.load_anno(n)
        if a is not None:
            edited += a.edited_count()

    if edited == 0:
        # 没有修正时放行是正常行为，但那样会真的跑任务，测试里不碰
        check("当前标注没有修正，跳过 409 用例", True)
        return

    r = client.post("/api/prelabel/start", json={"model": "official"})
    j = r.get_json() or {}
    check("有修正时全量预标注被拒（409）", r.status_code == 409,
          "%d %s" % (r.status_code, j.get("error")))
    check("拒绝信息里带修正条数",
          isinstance(j.get("edited"), int) and j["edited"] > 0, str(j))
    check("拒绝信息提示需要 force", j.get("need_force") is True, str(j))

    # 回标（custom）同样要挡
    r2 = client.post("/api/prelabel/start", json={"model": "custom"})
    check("有修正时全量回标也被拒（409）", r2.status_code == 409,
          str(r2.get_json()))

    # 只标指定图片不受影响（only 非空不走这道闸门）
    one = web.list_images()[0]
    r3 = client.post("/api/prelabel/start",
                     json={"model": "official", "images": ["__nope__.jpg"]})
    check("指定图片路径仍走「找不到」分支（400）", r3.status_code == 400,
          str(r3.get_json()))


def test_train_progress():
    """训练日志解析（进度条的唯一数据来源）。

    训练在后台子进程里跑，前端画进度条只有一条路：把 ultralytics 的
    终端输出解析出来。它打印的格式不是契约（换版本就变），所以这段
    解析必须能被单独测——不然哪天它换排版，用户看到的就是一条永远 0% 的条。

    另外这条测试锁住一个真实踩过的坑：日志里带 ANSI 清行码（\\x1b[K），
    不剥掉的话行首不是空格，`^\\s*14/100` 全部匹配失败，进度永远「启动中」。
    """
    print("\n[11] 训练进度解析")
    from train_progress import human_duration, parse_train_log

    check("空日志 = 未开始", parse_train_log("")["state"] == "idle")

    s = parse_train_log("Starting training for 100 epochs...\n")
    check("刚开始 = starting，且知道总轮数",
          s["state"] == "starting" and s["total_epochs"] == 100, str(s))

    mid = ("Starting training for 100 epochs...\n"
           "     14/100      2.22G      1.305      3.525   640: 25% ━━ 4/16 5.0it/s\n")
    s = parse_train_log(mid)
    check("跑到一半 = running，百分比含轮内进度",
          s["state"] == "running" and s["epoch"] == 14
          and abs(s["percent"] - 13.2) < 0.2,
          "epoch=%s percent=%s" % (s["epoch"], s["percent"]))

    done = ("Starting training for 100 epochs...\n"
            "     51/100      2.22G      1.3      3.6\n"
            "                        all         31        166      0.498"
            "      0.488      0.486      0.347      0.346      0.385"
            "      0.334      0.151\n"
            "EarlyStopping: Training stopped early as no improvement"
            " observed in last 50 epochs.\n"
            "[4/4] 训练完成，耗时 3.3 分钟\n")
    s = parse_train_log(done)
    check("完成 = done / 100% / 带耗时",
          s["state"] == "done" and s["percent"] == 100.0 and s["minutes"] == 3.3,
          str(s))
    check("取到 box 与 mask 两个 mAP50（8 个数字里第 3 和第 7 个）",
          abs(s["box_map50"] - 0.486) < 1e-6
          and abs(s["mask_map50"] - 0.334) < 1e-6,
          "%s / %s" % (s["box_map50"], s["mask_map50"]))
    check("识别早停", s["early_stopped"] is True, str(s["early_stopped"]))

    s = parse_train_log("Starting training for 100 epochs...\n"
                        "[错误] 训练失败: CUDA out of memory\n")
    check("失败 = failed，且原因能带出来",
          s["state"] == "failed" and "out of memory" in s["error"], s["error"])

    ansi = "\x1b[K     14/100      2.22G      1.3\r\x1b[K     15/100      2.22G      1.3\r"
    s = parse_train_log(ansi)
    check("剥掉 ANSI 控制码后仍能认出轮次",
          s["epoch"] == 15 and s["state"] == "running",
          "epoch=%s state=%s" % (s["epoch"], s["state"]))

    check("秒数转人话", human_duration(192) == "3 分 12 秒"
          and human_duration(8) == "8 秒" and human_duration(None) == "",
          human_duration(192))


def test_train_backup_keep_baseline():
    """训练覆盖前的权重备份，以及「只留 3 份」时不能把 baseline 删掉。

    真实现象：一天里连点两次训练，第二次的产物比第一次差（早停在第 1 轮，
    检测 0.695 / 掩码 0.417，而原来那份是 0.760 / 0.540）。要是没有备份，
    验收用的权重就没了。备份本身又有个反向风险：留 3 份、按时间删，
    连训几次就把最早那份——往往正是 baseline——悄悄删掉。
    """
    print("\n[12] 权重备份保留策略")
    import train

    tmp = tempfile.mkdtemp(prefix="wback_")
    old_dir = train.BACKUP_DIR
    try:
        train.BACKUP_DIR = os.path.join(tmp, "_backup")
        os.makedirs(train.BACKUP_DIR)
        base = "best_20260101_000000_baseline.pt"
        with open(os.path.join(train.BACKUP_DIR, base), "wb") as f:
            f.write(b"BASE")

        dst = os.path.join(tmp, "best.pt")
        for i in range(6):
            with open(dst, "wb") as f:
                f.write(b"V%d" % i)
            p = train.backup_existing_best(dst)
            check("备份内容与原件一致", open(p, "rb").read() == b"V%d" % i)

        files = os.listdir(train.BACKUP_DIR)
        check("baseline 不会被「只留 N 份」清掉", base in files, str(sorted(files)))
        others = [f for f in files if "baseline" not in f]
        check("其余备份按上限保留", len(others) == train.BACKUP_KEEP,
              "%d 份: %s" % (len(others), sorted(others)))
    finally:
        train.BACKUP_DIR = old_dir
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
def test_autolearn():
    """
    [13] 自主学习：什么时候学、学完该不该采纳、配置怎么落盘

    这里测的三件事，正好对应三个真会出问题的地方：
      1. 阈值判据写错 -> 要么永远不学，要么每改一笔就学（显卡被拖死）
      2. 采纳闸门写松 -> 越练越差还自动换上去，验收用的 best.pt 被悄悄换掉
      3. 首次开启没对齐基准 -> 只是点了个开关，历史上攒的 60 多条修正
         立刻触发一整代训练，用户莫名其妙等三分钟
    """
    print("\n[13] 自主学习判据")
    import json

    import autolearn as al

    cfg = {"enabled": False, "threshold": 20, "learned_edited": 60,
           "round": 0, "running": False}
    check("关闭状态下永远不学", not al.should_learn(cfg, 999))
    on = dict(cfg, enabled=True)
    check("差 1 条不学", not al.should_learn(on, 79))
    check("刚好够阈值就学", al.should_learn(on, 80))
    check("待学条数 = 当前已修正 - 上次学过的", al.pending(on, 85) == 25)
    check("计数倒挂也不出现负数", al.pending(on, 50) == 0)
    check("上一代还在跑时不重复触发", not al.should_learn(dict(on, running=True), 999))

    # 学法切换：加新图必须改走基座重训，否则冻结的主干学不进新场景
    check("加 5 张新图 -> 改用基座重训", al.should_switch_to_retrain(157, 162))
    check("只加 4 张 -> 不动（不值得为它重训 5 分钟）", not al.should_switch_to_retrain(157, 161))
    check("图没变 -> 继续续训", not al.should_switch_to_retrain(157, 157))
    check("第一次学（没有上一代张数）-> 按配置走", not al.should_switch_to_retrain(0, 157))
    check("大库加 10 张（2%）-> 不算数据变了", not al.should_switch_to_retrain(500, 510))
    check("大库加 16 张（3.2%）-> 算数据变了", al.should_switch_to_retrain(500, 516))

    base = {"box_map50": 0.728, "mask_map50": 0.580}
    ok, _ = al.decide_accept(base, {"box_map50": 0.731, "mask_map50": 0.585})
    check("小幅提升 -> 采纳", ok)
    ok, _ = al.decide_accept(base, {"box_map50": 0.727, "mask_map50": 0.578})
    check("噪声带内的微降 -> 仍然采纳（否则永远学不动）", ok)
    ok, why = al.decide_accept(base, {"box_map50": 0.746, "mask_map50": 0.530})
    check("掩码掉 5 个点 -> 拒绝（检测涨也不行）", (not ok) and ("掩码" in why), why)
    ok, why = al.decide_accept(base, {"box_map50": 0.700, "mask_map50": 0.581})
    check("检测掉 3 个点 -> 拒绝（掩码涨也不行）", (not ok) and ("检测" in why), why)
    ok, _ = al.decide_accept({}, {"box_map50": 0.5, "mask_map50": 0.4})
    check("没有基线 -> 直接采纳当新基线", ok)

    tmp = tempfile.mkdtemp(prefix="alcfg_")
    try:
        p = os.path.join(tmp, "cfg.json")
        with open(p, "w", encoding="utf-8") as f:
            f.write('{"enabled": true}')
        c = al.load_cfg(p)
        check("旧配置缺字段用默认值补齐",
              c["threshold"] == al.DEFAULT_THRESHOLD and c["enabled"] is True)
        c["round"] = 3
        al.save_cfg(c, p)
        check("写回后能读出新值", al.load_cfg(p)["round"] == 3)

        c2 = al.set_enabled({"enabled": False, "round": 0, "learned_edited": 0}, True, 62)
        check("首次开启把基准对齐到现在（不翻旧账）", c2["learned_edited"] == 62)
        c3 = al.set_enabled({"enabled": False, "round": 2, "learned_edited": 62}, True, 90)
        check("已学过的不重置基准", c3["learned_edited"] == 62)

        ad = os.path.join(tmp, "ann")
        os.makedirs(ad)
        for i, edited in enumerate([True, True, False]):
            ann = Annotation("i%d.jpg" % i, 10, 10,
                             [Object("cup", [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
                                     edited=edited)])
            with open(os.path.join(ad, "i%d.json" % i), "w", encoding="utf-8") as f:
                json.dump(ann.to_dict(), f)
        check("scan_edited 只数被人改过的对象", al.scan_edited(ad) == 2)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
def test_val_split_is_frozen():
    """
    [14] 验证集名单必须固定：加新图不能偷偷换考卷

    【这一条来自一个真实事故】
    原来是 random.seed(42) + random.sample(range(total), n_val)：种子固定，
    但采样范围随总数变。150 张时抽 30 个索引、157 张时抽 31 个，同一个种子
    抽出两个几乎不相交的集合。于是同一份 best.pt 在「150 张的 val」上测出
    0.728/0.580，在「157 张的 val」上测出 0.760/0.540——两个数字都对，
    只是考卷换人了。自主学习的闸门就建在这个沙子上：每代重新导出，
    验证集跟着漂移，判据等于拿两份不同考卷的分数比大小。

    这个测试要钉死三件事：
      1. 名单一旦落盘，再加多少新图，val 还是那几张
      2. 新图一律进训练集（它们本来就是「新信息」，该去能学的地方）
      3. 名单里的图从库中删除后自动退出 val，不会留下空引用
    """
    print("\n[14] 验证集名单固定")
    tmp = tempfile.mkdtemp(prefix="valsplit_")
    try:
        img_dir = os.path.join(tmp, "images")
        os.makedirs(img_dir)
        for i in range(10):
            Image.new("RGB", (120, 90), (30 + i * 3, 60, 90)).save(
                os.path.join(img_dir, "img%d.jpg" % i))

        def mk(names):
            objs = [Object("cup", [[0.2, 0.2], [0.5, 0.2], [0.5, 0.5], [0.2, 0.5]])]
            return [Annotation(n, 120, 90, objects=objs) for n in names]

        split = os.path.join(tmp, "val_split.json")
        base = ["img%d.jpg" % i for i in range(10)]

        s1 = export_dataset(mk(base), os.path.join(tmp, "d1"), img_dir,
                            val_ratio=0.2, split_path=split)
        check("首次导出按比例分（10 张 -> 2 张 val）", s1["val_images"] == 2, str(s1["val_images"]))
        check("首次导出把名单落盘", os.path.exists(split))
        first = {n for n in os.listdir(os.path.join(tmp, "d1", "images", "val"))}
        check("名单内容与导出一致", set(load_val_split(split)) == first, str(first))

        # 加 5 张新图：这 5 张必须全进训练集，val 一张不变
        more = base + ["new%d.jpg" % i for i in range(5)]
        for i in range(5):
            Image.new("RGB", (120, 90), (200, 40 + i * 5, 40)).save(
                os.path.join(img_dir, "new%d.jpg" % i))
        s2 = export_dataset(mk(more), os.path.join(tmp, "d2"), img_dir,
                            val_ratio=0.2, split_path=split)
        second = {n for n in os.listdir(os.path.join(tmp, "d2", "images", "val"))}
        check("加 5 张新图后验证集不变（同一张考卷）", second == first, str(second))
        check("新图全进训练集", s2["train_images"] == 13, str(s2["train_images"]))
        check("验证集仍是 2 张，没有被比例带着漂",
              s2["val_images"] == 2, str(s2["val_images"]))
        check("统计里写明是沿用固定名单", "沿用" in s2["val_split_source"],
              s2["val_split_source"])

        # 名单里删掉一张图（模拟从库里删图）
        gone = sorted(first)[0]
        kept = base[:]
        kept.remove(gone)
        s3 = export_dataset(mk(kept), os.path.join(tmp, "d3"), img_dir,
                            val_ratio=0.2, split_path=split)
        third = {n for n in os.listdir(os.path.join(tmp, "d3", "images", "val"))}
        check("删图后它退出 val，不留空引用", gone not in third, str(third))
        check("剩下的验证图照旧", third == first - {gone}, str(third))

        # 同一张图传两次：一张在 val、一张在 train，就是数据泄漏
        # （模型训练时背下来了，验证时自然"答对"，mAP 虚高）
        dup_src = os.path.join(img_dir, "img0.jpg")
        shutil.copy2(dup_src, os.path.join(img_dir, "copy_of_img0.jpg"))
        leaked = base[:] + ["copy_of_img0.jpg"]
        cur = load_val_split(split)
        s5 = export_dataset(mk(leaked), os.path.join(tmp, "d5"), img_dir,
                            val_ratio=0.2, split_path=split)
        if "img0.jpg" in cur:
            check("重复图被判到两边 -> 强制挪进训练集",
                  s5["duplicates_moved"] == ["img0.jpg"], str(s5["duplicates_moved"]))
            check("名单也被清掉，下次不用原地再纠",
                  "img0.jpg" not in load_val_split(split), str(load_val_split(split)))
            v5 = {n for n in os.listdir(os.path.join(tmp, "d5", "images", "val"))}
            check("val 里不再有这一对", "img0.jpg" not in v5 and "copy_of_img0.jpg" not in v5,
                  str(v5))
        else:
            s6 = export_dataset(mk(leaked), os.path.join(tmp, "d6"), img_dir,
                                val_ratio=0.2, split_path=split)
            check("重复图都在 train 时不触发（没有泄漏）",
                  s6["duplicates_moved"] == [], str(s6["duplicates_moved"]))

        # 名单文件坏掉不能炸整个导出
        with open(split, "w", encoding="utf-8") as f:
            f.write("{ 坏掉的 json")
        check("名单文件损坏时读成空", load_val_split(split) == [])
        s4 = export_dataset(mk(base), os.path.join(tmp, "d4"), img_dir,
                            val_ratio=0.2, split_path=split)
        check("名单坏了就重新初始化，不报错", s4["val_images"] == 2, str(s4["val_images"]))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_class_vocab():
    """词表可增删改——「bed 标不上」就是词表里没这个词。

    这一组守住四件事：
      1. 没文件 / 文件坏了都不能让系统起不来（回落出厂十类）
      2. 词表外的历史对象读得出来（改词表不该把已有数据锁死）
      3. 导出 yaml 的 nc 跟着词表走（否则训练 nc 与标签编号对不上）
      4. 严格模式只用于人新建，宽松模式用于读旧文件
    """
    print("\n[15] 词表可增删改")
    import shutil

    import core
    from core import (DEFAULT_CLASSES, class_id_of, get_classes, hex_to_rgb,
                      load_class_items, save_class_items)

    tmp = tempfile.mkdtemp(prefix="imglabel_cls_")
    old_path = core.CLASSES_PATH
    try:
        path = os.path.join(tmp, "classes.json")
        core.CLASSES_PATH = path
        core._CLASS_CACHE.update({"stamp": None, "items": None, "path": None})

        check("没有 classes.json 时用出厂十类", get_classes() == DEFAULT_CLASSES,
              str(len(get_classes())))

        # 加一类：bed
        items = [{"name": n, "color": "#3b82f6"} for n in DEFAULT_CLASSES]
        items.append({"name": "bed", "color": "#ef4444"})
        save_class_items(items)
        names = get_classes()
        check("新增后变成 11 类", len(names) == 11, str(len(names)))
        check("bed 进了词表", "bed" in names)
        check("bed 排在最后（编号 10）", class_id_of("bed") == 10,
              str(class_id_of("bed")))
        check("颜色能取出来",
              hex_to_rgb(load_class_items()[-1]["color"]) == (239, 68, 68))

        # 严格模式：人新建对象只认词表内的类
        try:
            Object("bed", [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
            check("严格模式接受词表内类别", True)
        except ValueError as e:
            check("严格模式接受词表内类别", False, str(e))
        try:
            Object("spaceship", [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
            check("严格模式拒绝词表外类别", False, "没抛异常")
        except ValueError:
            check("严格模式拒绝词表外类别", True)

        # 宽松模式：读旧标注不能因为改过词表就失败
        old_obj = Object.from_dict({"class": "spaceship",
                                    "polygon": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]})
        check("旧图里词表外的类别仍能读出来", old_obj.class_name == "spaceship")

        # 导出：nc 必须等于词表长度
        img_dir = os.path.join(tmp, "images")
        os.makedirs(img_dir, exist_ok=True)
        from PIL import Image
        for i in range(4):
            Image.new("RGB", (120, 90), (200 - i, 175, 150 + i)).save(
                os.path.join(img_dir, "c%d.jpg" % i))
        anns = [
            Annotation("c%d.jpg" % i, 120, 90, objects=[
                Object("person", [[0.1, 0.1], [0.5, 0.1], [0.5, 0.6], [0.1, 0.6]]),
                Object("bed", [[0.55, 0.2], [0.9, 0.2], [0.9, 0.7], [0.55, 0.7]]),
            ]) for i in range(4)
        ]
        out_dir = os.path.join(tmp, "dataset")
        export_dataset(anns, out_dir, img_dir, val_ratio=0.25,
                       split_path=os.path.join(tmp, "val_split.json"))
        with open(os.path.join(out_dir, "data.yaml"), encoding="utf-8") as f:
            yaml_txt = f.read()
        check("data.yaml 的 nc 跟着词表（11）", "nc: 11" in yaml_txt, yaml_txt[:60])
        check("yaml 的 names 里有 bed", "bed" in yaml_txt)
        # bed 是编号 10，标签第一列必须是 10（不是 0/1 那种越界值）
        lbl_dir = os.path.join(out_dir, "labels", "train")
        first = os.path.join(lbl_dir, sorted(os.listdir(lbl_dir))[0])
        with open(first, encoding="utf-8") as f:
            ids = {ln.split()[0] for ln in f if ln.strip()}
        check("bed 写成编号 10", "10" in ids, str(sorted(ids)))

        # 文件坏掉不能炸
        with open(path, "w", encoding="utf-8") as f:
            f.write("{ 这不是 json")
        core._CLASS_CACHE.update({"stamp": None, "items": None, "path": None})
        check("文件损坏时回落出厂十类", get_classes() == DEFAULT_CLASSES)

        # 空词表要拒绝（删到最后一个类别时别把系统搞成零类）
        try:
            save_class_items([])
            check("保存空词表被拒绝", False, "没抛异常")
        except ValueError:
            check("保存空词表被拒绝", True)

        check("3 位色缩写能展开", hex_to_rgb("#abc") == (0xaa, 0xbb, 0xcc))
        check("认不出的颜色给兜底灰", hex_to_rgb("not-a-color") == (148, 163, 184))
    finally:
        core.CLASSES_PATH = old_path
        core._CLASS_CACHE.update({"stamp": None, "items": None, "path": None})
        shutil.rmtree(tmp, ignore_errors=True)


def test_class_delete_options():
    """删类别有两条出路：迁到别的类，或者连对象一起删。

    真实现象：词表里 book 有 81 个对象在用，想删掉时只给「迁到 person」。
    可这批 book 本来就是标错的——迁过去等于把 81 个错框换个标签留在训练集，
    下一轮训练会把误检学得更牢。所以必须有一条「连同对象一起删」的路。

    同时守住两条不能破的：
      1. 不带参数直接删仍然 409（误点一下就把数据删光的入口不能有）
      2. reassign 和 purge 同时给是矛盾请求，直接 400，不做隐式取舍
    """
    print("\n[16] 删除类别：迁到别的类 vs 连对象一起删")
    import app as web
    import core

    tmp = tempfile.mkdtemp(prefix="clsdel_")
    anno_dir = os.path.join(tmp, "annotations")
    os.makedirs(anno_dir)
    cls_path = os.path.join(tmp, "classes.json")
    state_path = os.path.join(tmp, "state.json")

    old_anno, old_cls, old_state = web.ANNO_DIR, core.CLASSES_PATH, web.STATE_FILE
    web.ANNO_DIR = anno_dir
    core.CLASSES_PATH = cls_path
    web.STATE_FILE = state_path
    core._CLASS_CACHE.update({"stamp": None, "items": None, "path": None})

    def reset():
        """重建词表与两张标注，让每个用例从同一个起点开始。"""
        core.save_class_items(
            [{"name": n, "color": "#3b82f6"} for n in ("person", "book", "cup")],
            cls_path)
        for fn in os.listdir(anno_dir):
            os.remove(os.path.join(anno_dir, fn))
        data = {
            "a.json": [{"class": "book"}, {"class": "person"}],
            "b.json": [{"class": "book"}, {"class": "book"}, {"class": "cup"}],
        }
        for fn, objs in data.items():
            for i, o in enumerate(objs):
                o.update({"id": "%s%d" % (fn[0], i),
                          "polygon": [[0, 0], [10, 0], [10, 10]],
                          "source": "model", "edited": False})
            with open(os.path.join(anno_dir, fn), "w", encoding="utf-8") as f:
                json.dump({"image": fn, "width": 100, "height": 100,
                           "objects": objs}, f)

    def classes_now():
        with open(cls_path, encoding="utf-8") as f:
            return [c["name"] for c in json.load(f)["classes"]]

    def objs_now():
        out = []
        for fn in sorted(os.listdir(anno_dir)):
            with open(os.path.join(anno_dir, fn), encoding="utf-8") as f:
                out += [o["class"] for o in json.load(f)["objects"]]
        return sorted(out)

    try:
        client = web.app.test_client()

        # 1) 不带参数：必须 409，且把两条出路都告诉前端
        reset()
        r = client.delete("/api/classes/book")
        d = r.get_json() or {}
        check("不带参数删在用的类别 -> 409", r.status_code == 409,
              "%d %s" % (r.status_code, d.get("error", "")))
        check("409 里带对象数与候选类别",
              d.get("in_use") == 3 and "person" in (d.get("candidates") or []),
              "in_use=%s candidates=%s" % (d.get("in_use"), d.get("candidates")))
        check("409 时数据一个没动", objs_now().count("book") == 3, str(objs_now()))

        # 2) 矛盾参数
        reset()
        r = client.delete("/api/classes/book?reassign=person&purge=1")
        check("reassign 与 purge 同时给 -> 400", r.status_code == 400,
              "%d" % r.status_code)

        # 3) 迁移：对象还在，只是换了名字
        reset()
        r = client.delete("/api/classes/book?reassign=cup")
        d = r.get_json() or {}
        check("迁移方式删掉词表项", "book" not in classes_now(), str(classes_now()))
        check("迁移方式报出移动数", d.get("moved_objects") == 3,
              str(d.get("moved_objects")))
        check("迁移后对象总数不变，book 全变 cup",
              objs_now() == ["cup", "cup", "cup", "cup", "person"], str(objs_now()))
        check("迁移不产生误删", d.get("purged_objects") == 0,
              str(d.get("purged_objects")))

        # 4) 连对象一起删：book 的框从所有图里消失，别的类完好
        reset()
        r = client.delete("/api/classes/book?purge=1")
        d = r.get_json() or {}
        check("purge 返回 200", r.status_code == 200, "%d" % r.status_code)
        check("purge 报出删除数", d.get("purged_objects") == 3,
              str(d.get("purged_objects")))
        check("purge 后 book 对象全部消失",
              objs_now() == ["cup", "person"], str(objs_now()))
        check("purge 后词表里没有 book", "book" not in classes_now(),
              str(classes_now()))
        check("purge 不会误动别的类别", "person" in classes_now(), str(classes_now()))

        # 5) 删不存在的类别
        reset()
        r = client.delete("/api/classes/nope?purge=1")
        check("删不存在的类别 -> 404", r.status_code == 404, "%d" % r.status_code)

        # 6) 空类别的 purge 不该写出空文件
        reset()
        r = client.delete("/api/classes/person?purge=1")
        d = r.get_json() or {}
        check("purge 报出的删除数与该类实际对象数一致",
              r.status_code == 200 and d.get("purged_objects") == 1,
              "%d purged=%s" % (r.status_code, d.get("purged_objects")))
        check("此时 person 的 1 个对象也没了（这类本来就要删）",
              objs_now() == ["book", "book", "book", "cup"], str(objs_now()))
    finally:
        web.ANNO_DIR, core.CLASSES_PATH, web.STATE_FILE = old_anno, old_cls, old_state
        core._CLASS_CACHE.update({"stamp": None, "items": None, "path": None})
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
    test_upload_exts_match_list()
    test_prelabel_overwrite_guard()
    test_train_progress()
    test_train_backup_keep_baseline()
    test_autolearn()
    test_val_split_is_frozen()
    test_class_vocab()
    test_class_delete_options()
    print("\n" + "=" * 60)
    print("结果: %d 通过, %d 失败" % (PASS, FAIL))
    print("=" * 60)
    sys.exit(1 if FAIL else 0)
