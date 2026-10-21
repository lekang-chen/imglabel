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


def test_classes():
    print("\n[6] 十类词表")
    expect = ["person", "chair", "dining table", "couch", "tv",
              "laptop", "cup", "bottle", "potted plant", "book"]
    check("词表完全匹配任务书", CLASSES == expect, str(CLASSES))
    check("共 10 类", len(CLASSES) == 10)


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
    print("\n" + "=" * 60)
    print("结果: %d 通过, %d 失败" % (PASS, FAIL))
    print("=" * 60)
    sys.exit(1 if FAIL else 0)
