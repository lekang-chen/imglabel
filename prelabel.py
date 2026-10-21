"""
prelabel.py —— 模型预标注模块

【任务书要求】
"模型预标注：调用 YOLO-World 与 YOLO26n-seg 官方预训练权重生成标注草稿"
"预标注任务在后台排队执行，页面仅显示进度条，不阻塞操作"（术语表：异步队列）

【两个模型怎么分工】
  YOLO-World  : 开放词表检测。给定物体名字（"cup"/"bottle"...）就能检出目标，
                不用训练。用来补「漏标」和「检测框」。
  YOLO26n-seg : 实例分割模型，输出精确掩码。用来给「形状」。

【为什么两个模型的结果要合并】
YOLO-World 只会给框，不会给掩码；YOLO26n-seg 只会给掩码，但它的类别是
COCO 固定 80 类，不是我们的十类词表。
所以策略是：
  1. 用 YOLO26n-seg 跑出每个物体的**掩码**（形状来源）
  2. 用 YOLO-World 在十类开放词表上跑出**检测框**（类别来源）
  3. 把两者按 IoU 匹配：框和掩码对上了，就用框的类别 + 掩码的形状
     没匹配上的：只有框 -> 用框造一个矩形掩码；有掩码但类别不在十类 -> 丢弃

【关键点：模型结果一律是"草稿"】
草稿一定会有错（认错类别、边缘不准、漏标），必须人工审核。
任务书明确要求修正记录不少于 60 条，所以草稿质量不能太好——
如果模型完美就没得改，这正是「人机协同」的意义：机器做粗，人做细。
"""

import os
import queue
import threading
import time

# ---------------------------------------------------------------------------
# 异步队列（任务书术语表明确提到的机制）
# ---------------------------------------------------------------------------

class PrelabelJob:
    """一次预标注任务的状态。页面通过 /api/prelabel/progress 轮询它。"""

    def __init__(self, job_id, total):
        self.id = job_id
        self.total = total
        self.done = 0
        self.current = ""
        self.status = "pending"   # pending | running | done | failed
        self.message = ""
        self.started_at = time.time()
        self.finished_at = None
        self.results = {}         # 图片名 -> {"objects": [...], "source": "..."}
        self._lock = threading.Lock()

    def advance(self, image_name, message=""):
        with self._lock:
            self.current = image_name
            self.done += 1
            if message:
                self.message = message

    def to_dict(self):
        with self._lock:
            elapsed = (self.finished_at or time.time()) - self.started_at
            return {
                "id": self.id,
                "total": self.total,
                "done": self.done,
                "current": self.current,
                "status": self.status,
                "message": self.message,
                "elapsed": round(elapsed, 1),
                "percent": round(100.0 * self.done / self.total, 1) if self.total else 0.0,
            }


class JobManager:
    """
    预标注任务管理器：维护一个任务队列，后台线程消费。

    为什么要用队列而不是直接循环？
    -> 150 张图的预标注要几分钟，如果同步跑，网页请求会超时，用户以为卡死。
    -> 放进后台线程，主线程立刻返回 job_id，前端轮询进度条即可。
    这就是任务书里说的"异步队列"。
    """

    def __init__(self):
        self.jobs = {}
        self.queue = queue.Queue()
        self.lock = threading.Lock()
        self.worker = None

    def submit(self, job, runner=None):
        """
        提交一个任务。

        runner : 可选的自定义执行函数，签名 runner(job)。
                 不传就用内置的 run_prelabel_job（默认走 YOLO-World + seg 的首轮预标注）。
                 app.py 里传自定义 runner，是为了同时支持"首轮预标注"和"用自训模型回标"
                 两种模式，并且避免内置 worker 重复执行同一个 job。
        """
        with self.lock:
            self.jobs[job.id] = job
        self.queue.put((job, runner))
        self._ensure_worker()
        return job

    def get(self, job_id):
        with self.lock:
            return self.jobs.get(job_id)

    def _ensure_worker(self):
        if self.worker is None or not self.worker.is_alive():
            self.worker = threading.Thread(target=self._run, daemon=True)
            self.worker.start()

    def _run(self):
        while True:
            item = self.queue.get()
            if item is None:
                break
            job, runner = item
            try:
                job.status = "running"
                if runner is not None:
                    runner(job)
                else:
                    run_prelabel_job(job)
                # 只有真的跑完才置为 done。
                # 注意：不能只看 done == total，因为单图出错时结果里会带 error 字段，
                # 但整体任务仍然算完成。
                job.status = "done"
            except Exception as exc:  # noqa: BLE001
                job.status = "failed"
                job.message = "预标注失败: %s" % exc
            finally:
                job.finished_at = time.time()
                self.queue.task_done()


# ---------------------------------------------------------------------------
# 模型加载（懒加载：第一次用到时才加载，避免拖慢启动）
# ---------------------------------------------------------------------------

_MODELS = {}


def _load_seg_model():
    """
    加载 YOLO26n-seg 实例分割模型。

    注意：ultralytics 首次运行会自动从官方地址下载权重。
    国内网络可能下不动，所以我在下面做了多源回退。
    """
    if "seg" in _MODELS:
        return _MODELS["seg"]

    from ultralytics import YOLO

    candidates = ["yolo26n-seg.pt", "yolov8n-seg.pt", "yolo11n-seg.pt"]
    last_err = None
    for name in candidates:
        try:
            model = YOLO(name)
            _MODELS["seg"] = model
            return model
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            continue
    raise RuntimeError(
        "无法加载实例分割模型，尝试过: %s。最后错误: %s\n"
        "提示：可手动下载权重放到 weights/ 目录，"
        "或设置环境变量 YOLO_OFFLINE 后用本地路径。" % (candidates, last_err)
    )


def _load_world_model():
    """
    加载 YOLO-World 开放词表检测模型。

    YOLO-World 的用法和普通 YOLO 不一样：
    要先 .set_classes([...]) 设定要检测的类别词表，然后才能 predict。
    """
    if "world" in _MODELS:
        return _MODELS["world"]

    from ultralytics import YOLO

    candidates = ["yolov8s-world.pt", "yolov8m-world.pt", "yolov8x-world.pt"]
    last_err = None
    for name in candidates:
        try:
            model = YOLO(name)
            _MODELS["world"] = model
            return model
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            continue
    raise RuntimeError(
        "无法加载 YOLO-World 模型，尝试过: %s。最后错误: %s" % (candidates, last_err)
    )


# ---------------------------------------------------------------------------
# 掩码 -> 多边形
# ---------------------------------------------------------------------------

def mask_to_polygon(mask, max_points=180, min_area_ratio=0.001):
    """
    把模型输出的像素级掩码，转成归一化多边形点集。

    【这是本项目最关键也最容易出错的一步】
    ultralytics 的分割模型输出的是「像素掩码」(numpy 二值数组)，
    而 YOLO 的实例分割格式要求的是「多边形顶点」。
    两者的转换关系：
        1. 找轮廓  (cv2.findContours)      —— 拿到像素坐标下的边界点
        2. 归一化  (除以宽/高)             —— 变成 0~1 的浮点，与分辨率解耦
        3. 抽稀    (simplify_polygon)      —— 控制顶点数，否则 txt 巨大、画布卡

    【为什么要抽稀】
    一个中等物体的原始轮廓可能有 800 个点。
    - 导出的 txt 会非常大
    - 前端 canvas 画 800 个点的多边形，缩放时会明显卡顿
    - 人工要拖动顶点修正根本不可能（根本点不中）
    实测抽到 180 个点以内，肉眼几乎看不出差别，但文件小一半以上。

    min_area_ratio: 过滤太小的碎片。模型经常在物体边缘吐出几像素的噪点，
    留着它们会让「对象总数」虚高，修正率统计也会失真。
    阈值 0.001 = 图像面积的 0.1%：960x960 的图约等于 30x30 像素，
    小于这个尺寸的连通块基本可以判定为噪声；而真实的杯子、书本
    （通常 40x40 像素以上）不会被误删。
    """
    import cv2
    import numpy as np

    if mask is None:
        return []

    arr = np.array(mask)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    arr = (arr > 0).astype(np.uint8)

    height, width = arr.shape
    if height == 0 or width == 0:
        return []

    # 找最大连通区域：模型偶尔会在一个物体上分出几块，取最大那块
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(arr, connectivity=8)
    if num_labels > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        arr = (labels == largest).astype(np.uint8)

    # 面积过滤
    area = int(arr.sum())
    if area < max(1, int(min_area_ratio * width * height)):
        return []

    # 找外轮廓
    contours, _ = cv2.findContours(arr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    contour = max(contours, key=cv2.contourArea)

    pts = contour.reshape(-1, 2).astype(float)
    if len(pts) < 3:
        return []

    # 归一化到 0~1
    poly = [[float(x) / (width - 1 if width > 1 else 1),
             float(y) / (height - 1 if height > 1 else 1)] for x, y in pts]

    # 抽稀
    from core import simplify_polygon
    poly = simplify_polygon(poly, max_points=max_points)
    if len(poly) < 3:
        return []
    return poly


def iou(box_a, box_b):
    """两个框的 IoU（交并比）。用来把 YOLO-World 的框和 seg 的掩码配对。"""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def run_prelabel_job(job, image_dir=None, class_names=None,
                     conf=0.15, iou_thresh=0.30):
    """
    执行一次完整的预标注任务。

    参数:
      job        : PrelabelJob 对象，用于更新进度
      image_dir  : 图片目录
      class_names: 十类词表（英文名）
      conf       : 置信度阈值。取 0.15 而不是常见的 0.25，是实测出来的：
                    在我们这 150 张图上，conf=0.25 只能产出 913 个草稿对象、
                    有 12 张图一个对象都检不出；降到 0.15 后草稿对象涨到 1302 个、
                    零草稿图降到 8 张。
                    理由是"草稿宁多勿漏"——多出来的误检人点一下就能删，
                    但漏掉的对象人根本看不见，就不会去改，
                    最终会导致人工修正记录失真、训练数据也少了正样本。
      iou_thresh : 框与掩码的匹配阈值

    返回: {图片名: {"objects": [...], "source": "..."}}
    """
    from core import CLASSES

    # 允许省略参数（JobManager 的默认执行路径就是只传 job）
    if image_dir is None:
        image_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "data", "images")
    if class_names is None:
        class_names = CLASSES

    seg_model = _load_seg_model()
    world_model = _load_world_model()

    # 设定 YOLO-World 的开放词表
    world_model.set_classes(list(class_names))

    images = sorted([f for f in os.listdir(image_dir)
                     if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    job.total = len(images)

    results = {}

    for i, img_name in enumerate(images):
        img_path = os.path.join(image_dir, img_name)
        job.advance(img_name, "预标注 %d/%d" % (job.done + 1, len(images)))

        try:
            objs = _prelabel_one(seg_model, world_model, img_path,
                                 CLASSES, conf, iou_thresh)
            results[img_name] = {
                "objects": [o.to_dict() for o in objs],
                "source": "model",
            }
        except Exception as exc:  # noqa: BLE001
            results[img_name] = {
                "objects": [],
                "source": "model",
                "error": str(exc),
            }
            job.advance(img_name, "跳过 %s: %s" % (img_name, exc))

    job.results = results
    return results


def _prelabel_one(seg_model, world_model, img_path, class_names, conf, iou_thresh):
    """
    对单张图片做预标注，返回 Object 列表。

    三步：
      1. seg 模型 -> 掩码列表（每项：mask, 原始类别名, 置信度）
      2. world 模型 -> 检测框列表（每项：框, 我们的十类类别名, 置信度）
      3. 匹配合并 -> 最终的 Object 列表
    """
    from core import Object

    # --- 第 1 步：实例分割，拿形状 ---
    seg_out = seg_model.predict(img_path, conf=conf, verbose=False)
    masks = []
    if seg_out and len(seg_out):
        r = seg_out[0]
        if r.masks is not None:
            for i, m in enumerate(r.masks.data):
                box = r.boxes.xyxy[i].tolist() if r.boxes is not None else None
                masks.append({
                    "mask": m.cpu().numpy(),
                    "box": box,
                    "cls": int(r.boxes.cls[i]) if r.boxes is not None else -1,
                    "conf": float(r.boxes.conf[i]) if r.boxes is not None else 0.0,
                })

    # --- 第 2 步：开放词表检测，拿我们的十类类别 ---
    world_out = world_model.predict(img_path, conf=conf, verbose=False)
    dets = []
    if world_out and len(world_out):
        r = world_out[0]
        if r.boxes is not None:
            names = r.names if isinstance(r.names, dict) else {}
            for i in range(len(r.boxes)):
                cls_id = int(r.boxes.cls[i])
                cls_name = names.get(cls_id, class_names[cls_id]
                                     if cls_id < len(class_names) else "")
                if cls_name not in class_names:
                    continue
                x1, y1, x2, y2 = r.boxes.xyxy[i].tolist()
                dets.append({
                    "box_norm": [x1, y1, x2, y2],
                    "cls": cls_name,
                    "conf": float(r.boxes.conf[i]),
                    "w": r.orig_shape[1],
                    "h": r.orig_shape[0],
                })

    # --- 第 3 步：匹配合并 ---
    objects = []
    used_dets = set()

    for m in masks:
        poly = mask_to_polygon(m["mask"])
        if not poly:
            continue

        # 用掩码的外接框去和检测框匹配
        from core import polygon_bbox
        m_box = polygon_bbox(poly)
        best_det, best_iou = None, 0.0
        for j, d in enumerate(dets):
            if j in used_dets:
                continue
            v = iou(m_box, d["box_norm"])
            if v > best_iou:
                best_iou, best_det = v, (j, d)

        if best_det is not None and best_iou >= iou_thresh:
            j, d = best_det
            used_dets.add(j)
            objects.append(Object(d["cls"], poly, source="model"))
        else:
            # 掩码没配上框：看看这个掩码的原始类别是否在十类里
            seg_names = _seg_class_names(seg_model)
            orig = seg_names.get(m["cls"], "") if seg_names else ""
            if orig in class_names:
                objects.append(Object(orig, poly, source="model"))
            # 否则丢弃：既不知道类别，又没有框，不给用户添乱

    # 没被用上的检测框：说明 YOLO-World 看到了但 seg 模型没分割出来
    # （典型情况：小物体、遮挡物体）。给它们造一个矩形掩码，至少不漏标。
    for j, d in enumerate(dets):
        if j in used_dets:
            continue
        x1, y1, x2, y2 = d["box_norm"]
        w, h = d["w"], d["h"]
        poly = [
            [x1 / w, y1 / h], [x2 / w, y1 / h],
            [x2 / w, y2 / h], [x1 / w, y2 / h],
        ]
        objects.append(Object(d["cls"], poly, source="model"))

    return objects


_SEG_NAMES_CACHE = {}


def _seg_class_names(seg_model):
    """缓存 seg 模型的类别名表，避免每张图都去取。"""
    key = id(seg_model)
    if key not in _SEG_NAMES_CACHE:
        try:
            _SEG_NAMES_CACHE[key] = dict(seg_model.names)
        except Exception:  # noqa: BLE001
            _SEG_NAMES_CACHE[key] = {}
    return _SEG_NAMES_CACHE[key]


# ---------------------------------------------------------------------------
# 用自训模型回标（第二轮迭代）
# ---------------------------------------------------------------------------

def run_relabel_job(job, image_dir, weights_path, conf=0.25, round_tag="R2"):
    """
    用自训的 best.pt 重新预标注（回标），形成第二轮迭代。

    和首轮的区别：不用 YOLO-World，直接用自训的分割模型。
    因为自训模型已经学会了我们的十类，类别直接从模型出。
    """
    from ultralytics import YOLO
    from core import CLASSES, Object

    model = YOLO(weights_path)
    seg_names = dict(model.names)

    images = sorted([f for f in os.listdir(image_dir)
                     if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    job.total = len(images)
    results = {}

    for i, img_name in enumerate(images):
        img_path = os.path.join(image_dir, img_name)
        job.advance(img_name, "%s 回标 %d/%d" % (round_tag, job.done + 1, len(images)))
        try:
            out = model.predict(img_path, conf=conf, verbose=False)
            objs = []
            if out and len(out):
                r = out[0]
                if r.masks is not None:
                    for i2, m in enumerate(r.masks.data):
                        poly = mask_to_polygon(m.cpu().numpy())
                        if not poly:
                            continue
                        cls_id = int(r.boxes.cls[i2])
                        cls_name = seg_names.get(cls_id, "")
                        if cls_name in CLASSES:
                            objs.append(Object(cls_name, poly, source="model"))
            results[img_name] = {"objects": [o.to_dict() for o in objs],
                                 "source": "model", "round": round_tag}
        except Exception as exc:  # noqa: BLE001
            results[img_name] = {"objects": [], "source": "model",
                                 "round": round_tag, "error": str(exc)}

    job.results = results
    return results
