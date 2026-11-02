"""
app.py —— Flask 主程序（后端入口）

【怎么启动】
  方式一（推荐，评委复现用）：双击 启动.bat
  方式二：python app.py

启动后浏览器自动打开 http://127.0.0.1:5000

【本文件负责什么】
1. 提供 HTTP 接口给前端画布调用
2. 管理标注数据的读写（存成 JSON 落盘，可断点续标）
3. 调度预标注任务（异步队列）
4. 触发导出和训练

【目录约定】
  data/images/       原始图片（你下载好的 150 张）
  data/annotations/  标注 JSON（每图一个文件，平台导出的可重建）
  dataset/           导出的三格式数据集
  model/             自训权重 best.pt
  records/           修正记录 + 对比结果
"""

import io
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import webbrowser

from flask import (Flask, jsonify, request, send_file,
                   send_from_directory, abort)

from core import (IMAGE_EXTS, Annotation, Object, class_color, class_id_of,
                  get_classes, load_class_items, pid_alive, point_in_polygon,
                  save_class_items)
from exporter import export_dataset, validate_dataset, format_report
from prelabel import JobManager, run_prelabel_job, run_relabel_job
from train_progress import human_duration, parse_train_log
import autolearn

# ---------------------------------------------------------------------------
# 路径配置（全部相对于项目根目录，保证换电脑也能跑）
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
IMAGE_DIR = os.path.join(DATA_DIR, "images")
ANNO_DIR = os.path.join(DATA_DIR, "annotations")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_DIR = os.path.join(BASE_DIR, "model")
RECORD_DIR = os.path.join(BASE_DIR, "records")
WEIGHT_DIR = os.path.join(BASE_DIR, "weights")

# 训练进度：日志给内容，meta 给 pid 与启动时刻（判断进程还活着没有）
TRAIN_LOG = os.path.join(DATA_DIR, "train.log")
TRAIN_META = os.path.join(DATA_DIR, "train_meta.json")

# 删除照片时不是 os.remove，而是把「图片 + 标注」成对搬到这里。
# 【为什么】data/images 里的图连着标注，标注里有 60 条人工修正，
# 手滑删掉就没了。搬到 _trash/<时间戳>/ 后文件名不变，拷回去即可完整恢复。
TRASH_DIR = os.path.join(DATA_DIR, "_trash")

for d in (DATA_DIR, IMAGE_DIR, ANNO_DIR, DATASET_DIR, MODEL_DIR, RECORD_DIR):
    os.makedirs(d, exist_ok=True)

app = Flask(__name__, static_folder="canvas", static_url_path="/static")
app.config["JSON_AS_ASCII"] = False
# 静态文件不缓存：这是本地开发服务器，每次都在改前端。
# 【为什么必须关掉】Flask 默认给静态文件打 12 小时 max-age，
# 改完 app.js / 测试脚本后浏览器仍跑旧代码，
# 表现为「代码明明改了，测试结果却和上次一模一样」——
# 极具欺骗性，会让人以为是新代码写错了。开发期一律走no-cache。
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
app.config["SEND_FILE_MAX_AGE"] = 0
# 上传接口的请求体上限（一次最多拖 5 张 20MB 的照片也够用）
app.config["MAX_CONTENT_LENGTH"] = 128 * 1024 * 1024

job_manager = JobManager()

# 全局运行状态：当前是第几轮、每轮的修正统计
STATE_FILE = os.path.join(DATA_DIR, "state.json")


def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {
        "round": 1,
        "current_model": None,      # 当前使用的模型（None = 官方预训练权重）
        "rounds": {},               # 每轮的统计
        "logs": [],                 # 操作流水
    }


def save_state(state):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def log_action(state, action, detail=""):
    """记录一条操作流水，用于答辩时展示工作过程。"""
    state.setdefault("logs", []).append({
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "action": action,
        "detail": detail,
    })


# ---------------------------------------------------------------------------
# 标注数据读写
# ---------------------------------------------------------------------------

def anno_path(image_name):
    return os.path.join(ANNO_DIR, os.path.splitext(image_name)[0] + ".json")


def load_anno(image_name):
    p = anno_path(image_name)
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return Annotation.from_dict(json.load(f))
    return None


def save_anno(ann):
    with open(anno_path(ann.image_name), "w", encoding="utf-8") as f:
        json.dump(ann.to_dict(), f, ensure_ascii=False, indent=1)


def list_images():
    if not os.path.isdir(IMAGE_DIR):
        return []
    # 扩展名白名单只在 core.IMAGE_EXTS 定义一次，别在这里另写一份：
    # 两边一旦分叉，就会出现「上传成功但列表里没有」的鬼故事（见 core.py）。
    return sorted([f for f in os.listdir(IMAGE_DIR)
                   if f.lower().endswith(IMAGE_EXTS)])


# ---------------------------------------------------------------------------
# 上传：把用户自己的照片放进 data/images/
# ---------------------------------------------------------------------------
UPLOAD_EXTS = IMAGE_EXTS   # 上传与列表共用同一份白名单，见 core.py 注释
MAX_UPLOAD_BYTES = 20 * 1024 * 1024   # 单张 20MB，手机照片足够
MAX_UPLOAD_SIDE = 1920                # 长边超过就等比缩小


def _safe_stem(filename):
    """
    把上传的文件名变成「安全且还能认出来」的文件名。

    【为什么不用 werkzeug 的 secure_filename】
    它会把中文整段删掉：'客厅照片.jpg' -> 'jpg'，两个中文名文件会撞成同一个名。
    这里保留中文、字母、数字、下划线、连字符，其余字符统一换成下划线。
    """
    stem = os.path.splitext(os.path.basename(filename or ""))[0]
    stem = re.sub(r"[^\w\u4e00-\u9fff-]+", "_", stem)
    stem = stem.strip("_")
    return stem or "image"


def _unique_image_name(stem, ext):
    """
    给新图找一个不冲突的文件名。

    【为什么必须防冲突】标注文件是按「文件名去后缀」存的（见 anno_path）。
    如果新上传的 001.jpg 和已有的 001.png 同名不同后缀，两者会共用
    一份 001.json，标注直接串在一起。所以这里连标注文件一起检查。
    """
    cand = stem + ext
    n = 2
    while (os.path.exists(os.path.join(IMAGE_DIR, cand))
           or os.path.exists(anno_path(cand))):
        cand = "%s_%d%s" % (stem, n, ext)
        n += 1
    return cand


def _save_upload(raw_bytes, filename):
    """
    校验 + 落盘一张图，返回 (保存的文件名, 宽, 高)。

    三件事必须做，否则后面整条链路都会出问题：
    1. 用 PIL 真正解一遍——只看后缀会被「改名的 .txt」骗过去；
    2. 按 EXIF 纠正方向——手机照片常带 Orientation 标记，
       不纠正的话图在床上躺着，人画出来的多边形也是歪的；
    3. 长边超过 1920 就等比缩小——4000px 的手机原图在 640 输入下
       推理慢好几倍，标注时浏览器也扛不住。
    """
    from PIL import Image, ImageOps, UnidentifiedImageError

    try:
        im = Image.open(io.BytesIO(raw_bytes))
        im.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("不是可识别的图片（%s）" % exc)

    fmt = (im.format or "").upper()
    if fmt == "JPG":
        fmt = "JPEG"
    if fmt not in ("JPEG", "PNG", "BMP", "WEBP"):
        raise ValueError("暂不支持 %s 格式" % (fmt or "未知"))

    im = ImageOps.exif_transpose(im)

    if max(im.size) > MAX_UPLOAD_SIDE:
        ratio = MAX_UPLOAD_SIDE / float(max(im.size))
        new_size = (max(1, int(im.size[0] * ratio)),
                    max(1, int(im.size[1] * ratio)))
        im = im.resize(new_size, Image.LANCZOS)

    # 存盘前统一成能落地的色彩模式
    if fmt == "JPEG":
        if im.mode != "RGB":
            im = im.convert("RGB")
    elif im.mode == "P":
        im = im.convert("RGBA" if "transparency" in im.info else "RGB")
    elif im.mode in ("CMYK", "LAB", "I;16"):
        im = im.convert("RGB")

    ext = {"JPEG": ".jpg", "PNG": ".png",
           "BMP": ".bmp", "WEBP": ".webp"}[fmt]
    name = _unique_image_name(_safe_stem(filename), ext)
    out_path = os.path.join(IMAGE_DIR, name)

    if fmt == "JPEG":
        im.save(out_path, "JPEG", quality=92, optimize=True)
    else:
        im.save(out_path, fmt)

    return name, im.size[0], im.size[1]


@app.route("/api/upload", methods=["POST"])
def api_upload():
    """
    上传照片。前端一次传一张（这样进度条能一张一张走），也支持一次多张。

    表单字段：
      file —— 文件本体，可重复（多选上传时同名多次）
    返回：
      {"ok": true, "saved": [{name,width,height}], "skipped": [{name,reason}]}
    """
    files = request.files.getlist("file")
    if not files:
        return jsonify({"error": "没有收到文件"}), 400

    saved, skipped = [], []
    for fs in files:
        raw_name = fs.filename or ""
        if not raw_name:
            continue
        ext = os.path.splitext(raw_name)[1].lower()
        if ext not in UPLOAD_EXTS:
            skipped.append({"name": raw_name,
                            "reason": "格式不支持（只收 jpg/png/bmp/webp）"})
            continue

        data = fs.read()
        if not data:
            skipped.append({"name": raw_name, "reason": "空文件"})
            continue
        if len(data) > MAX_UPLOAD_BYTES:
            skipped.append({
                "name": raw_name,
                "reason": "超过 %d MB" % (MAX_UPLOAD_BYTES // 1024 // 1024),
            })
            continue

        try:
            name, w, h = _save_upload(data, raw_name)
        except ValueError as exc:
            skipped.append({"name": raw_name, "reason": str(exc)})
            continue

        saved.append({"name": name, "width": w, "height": h})

    if saved:
        st = load_state()
        log_action(st, "上传图片", "%d 张：%s"
                   % (len(saved), "、".join(s["name"] for s in saved[:8])))
        save_state(st)

    return jsonify({
        "ok": True,
        "saved": saved,
        "skipped": skipped,
        "total_images": len(list_images()),
    })


# ---------------------------------------------------------------------------
# 删除：软删除到 data/_trash/，可恢复
# ---------------------------------------------------------------------------

def _trash_batches():
    """
    列出回收站里的批次（时间倒序）。

    每个批次是一个文件夹 data/_trash/<时间戳>/，里面装着这一次删除的
    「图片 + 对应标注 json」。文件名保持不变，所以恢复只是搬回去。
    """
    if not os.path.isdir(TRASH_DIR):
        return []
    out = []
    for name in sorted(os.listdir(TRASH_DIR), reverse=True):
        d = os.path.join(TRASH_DIR, name)
        if not os.path.isdir(d):
            continue
        files = sorted(f for f in os.listdir(d)
                       if os.path.isfile(os.path.join(d, f)))
        if files:
            out.append({"stamp": name, "files": files})
    return out


def _new_trash_dir():
    """开一个新的批次目录。同一秒内删两次时自动加 _2、_3，不覆盖上一批。"""
    base = os.path.join(TRASH_DIR, time.strftime("%Y%m%d_%H%M%S"))
    path, i = base, 2
    while os.path.isdir(path) and os.listdir(path):
        path = "%s_%d" % (base, i)
        i += 1
    return path


@app.route("/api/images/delete", methods=["POST"])
def api_delete_images():
    """
    删除照片（连同它的标注文件）。

    请求体：{"names": ["a.jpg", "b.png"]}

    【为什么是搬走而不是 os.remove】
    data/images 里那 150 张图配着标注，标注里有 60 条人工修正，
    真删了不可逆。所以图片和标注成对搬到 data/_trash/<时间戳>/，
    文件名不变——误删了点一下「恢复」就能完整拿回来（含修正标记）。
    """
    body = request.get_json(silent=True) or {}
    names = body.get("names")
    if isinstance(names, str):
        names = [names]
    if not isinstance(names, list) or not names:
        return jsonify({"error": "没有指定要删除的图片"}), 400

    # 只认「当前图片列表里真实存在的名字」。这既是白名单，
    # 也顺手挡住了 ../../ 这类路径穿越（拼出来的名字不会出现在 existing 里）。
    existing = set(list_images())
    trash_dir = _new_trash_dir()
    created = False

    deleted, missing, failed = [], [], []
    for raw in names:
        name = os.path.basename(str(raw))
        if name not in existing:
            missing.append(name)
            continue
        try:
            if not created:
                os.makedirs(trash_dir, exist_ok=True)
                created = True
            shutil.move(os.path.join(IMAGE_DIR, name),
                        os.path.join(trash_dir, name))
            ap = anno_path(name)
            if os.path.exists(ap):
                shutil.move(ap, os.path.join(trash_dir, os.path.basename(ap)))
            deleted.append(name)
        except OSError as exc:
            failed.append({"name": name, "reason": str(exc)})

    rel = None
    if deleted:
        rel = os.path.relpath(trash_dir, BASE_DIR).replace("\\", "/")
        st = load_state()
        log_action(st, "删除图片", "%d 张：%s（已移入 %s，可恢复）"
                   % (len(deleted), "、".join(deleted[:8]), rel))
        save_state(st)

    return jsonify({
        "ok": True,
        "deleted": deleted,
        "missing": missing,
        "failed": failed,
        "trash": rel,
        "total_images": len(list_images()),
    })


@app.route("/api/images/restore", methods=["POST"])
def api_restore_images():
    """
    把回收站里的最近一批（或指定批次）搬回去。

    请求体：{"stamp": "20261027_232000"}（省略则恢复最近一次）
    """
    body = request.get_json(silent=True) or {}
    stamp = body.get("stamp")
    batches = _trash_batches()
    if not batches:
        return jsonify({"error": "回收站是空的，没有可恢复的内容"}), 400

    batch = None
    if stamp:
        batch = next((b for b in batches if b["stamp"] == stamp), None)
        if batch is None:
            return jsonify({"error": "找不到这一批删除记录"}), 404
    else:
        batch = batches[0]

    src_dir = os.path.join(TRASH_DIR, batch["stamp"])
    restored, skipped = [], []
    for f in batch["files"]:
        is_anno = os.path.splitext(f)[1].lower() == ".json"
        dst = os.path.join(ANNO_DIR if is_anno else IMAGE_DIR, f)
        if os.path.exists(dst):
            # 用户又把同名文件传回来了。不覆盖，交给人自己判断。
            skipped.append(f)
            continue
        try:
            shutil.move(os.path.join(src_dir, f), dst)
            restored.append(f)
        except OSError:
            skipped.append(f)

    # 搬空的批次目录删掉，回收站列表里不残留空条目
    try:
        if not os.listdir(src_dir):
            os.rmdir(src_dir)
    except OSError:
        pass

    images_back = [f for f in restored
                   if os.path.splitext(f)[1].lower() != ".json"]
    if images_back:
        st = load_state()
        log_action(st, "恢复图片", "%d 张：%s（来自 data/_trash/%s）"
                   % (len(images_back), "、".join(images_back[:8]),
                      batch["stamp"]))
        save_state(st)

    return jsonify({
        "ok": True,
        "restored": images_back,
        "skipped": skipped,
        "stamp": batch["stamp"],
        "total_images": len(list_images()),
    })


# ---------------------------------------------------------------------------
# 基础接口
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


# --- 类别词表：可增删改 ------------------------------------------------
#
# 【为什么这几个接口要放在后端做，而不是前端改完存 localStorage】
# 词表决定三件事：导出的标签编号、训练的 nc、预标注 vocab。
# 三处必须同一份，所以只能有一个真值文件 data/classes.json，
# 前端只负责展示和发起修改。

# 新增类别没指定颜色时，按这个顺序取，避免一堆一样的灰点
_CLASS_PALETTE = [
    "#ef4444", "#f97316", "#eab308", "#84cc16", "#22c55e", "#14b8a6",
    "#06b6d4", "#3b82f6", "#6366f1", "#8b5cf6", "#d946ef", "#ec4899",
]


def _class_usage():
    """全库每个类别各有多少个标注对象在用。

    【为什么不用 Annotation 类去读】这里要的是极轻的计数：只读 json 的
    objects[].class，不做几何构造、不校验多边形。删类别前要立刻拿到数字
    给用户看，走一遍完整反序列化没必要。
    """
    usage = {}
    try:
        files = os.listdir(ANNO_DIR)
    except OSError:
        return usage
    for fn in files:
        if not fn.endswith(".json"):
            continue
        try:
            with open(os.path.join(ANNO_DIR, fn), "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            continue
        for o in data.get("objects") or []:
            c = o.get("class")
            if c:
                usage[c] = usage.get(c, 0) + 1
    return usage


def _rename_class_in_annotations(old, new):
    """把所有标注对象里的 old 类别改名成 new，返回改动对象数。

    直接改 json 而不走 Annotation 反序列化：这条路径只动 class 字段，
    多边形、edited 标记原样保留，也不该因为改名就把对象算成「人工修正」。
    """
    changed = 0
    try:
        files = os.listdir(ANNO_DIR)
    except OSError:
        return 0
    for fn in files:
        if not fn.endswith(".json"):
            continue
        p = os.path.join(ANNO_DIR, fn)
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            continue
        hit = 0
        for o in data.get("objects") or []:
            if o.get("class") == old:
                o["class"] = new
                hit += 1
        if hit:
            tmp = p + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
            os.replace(tmp, p)
            changed += hit
    return changed


def _purge_class_from_annotations(name):
    """把某类别的对象从所有标注里删掉，返回删掉的对象数。

    【为什么要有这条路】「迁到别的类」是保守做法，但用户经常是标错了：
    压根不该有这个类，那 81 个 book 框本来就是错的，迁到 person 上只是
    把错误数据换个标签留在训练集里。所以要给一条「连同对象一起删」的路。

    【为什么不能只删词表项】词表只是名字列表，真正的数据在 data/annotations。
    只删词表，那些对象会变成「词表外残留」——界面上报警、导出时被跳过，
    等于把垃圾扫到地毯下面。要么迁走，要么真删，不能留着。

    直接改 json 不走 Annotation 反序列化，理由同 _rename_class_in_annotations：
    这条路径只动 objects 数组，其他字段原样保留，也不该把人删对象算成
    「人工修正」（那会推高自主学习的待学计数，等于用删数据骗训练）。
    """
    if not name:
        return 0
    removed = 0
    try:
        files = os.listdir(ANNO_DIR)
    except OSError:
        return 0
    for fn in files:
        if not fn.endswith(".json"):
            continue
        p = os.path.join(ANNO_DIR, fn)
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            continue
        objs = data.get("objects") or []
        kept = [o for o in objs if o.get("class") != name]
        hit = len(objs) - len(kept)
        if hit:
            data["objects"] = kept
            tmp = p + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
            os.replace(tmp, p)
            removed += hit
    return removed


def _classes_payload():
    """词表 + 每类使用量 + 词表外残留。前端一次拿全，避免两次请求。"""
    usage = _class_usage()
    items = load_class_items()
    known = {it["name"] for it in items}
    return {
        "classes": [
            {"name": it["name"], "id": i, "color": it["color"],
             "count": usage.get(it["name"], 0)}
            for i, it in enumerate(items)
        ],
        # 词表外但仍有对象在用的类别（改过词表留下的历史数据）。
        # 界面上要提示「这些对象导出时会被跳过」，否则数据静默丢失。
        "unknown": [{"name": k, "count": v}
                    for k, v in sorted(usage.items()) if k not in known],
        "needs_retrain": True,
    }


@app.route("/api/classes")
def api_classes():
    """当前词表 + 每类颜色与使用量。"""
    return jsonify(_classes_payload())


@app.route("/api/classes", methods=["POST"])
def api_classes_add():
    """新增一个类别。"""
    body = request.get_json(silent=True) or {}
    name = str(body.get("name", "")).strip()
    color = str(body.get("color") or "").strip()

    if not name:
        return jsonify({"error": "类别名不能为空"}), 400
    if len(name) > 40:
        return jsonify({"error": "类别名太长（最多 40 字符）"}), 400
    if any(ch in name for ch in "\r\n\t"):
        return jsonify({"error": "类别名不能含换行或制表符"}), 400
    if class_id_of(name) >= 0:
        return jsonify({"error": "类别「%s」已经存在" % name}), 409

    items = load_class_items()
    if len(items) >= 60:
        return jsonify({"error": "类别太多（上限 60 个），先删几个不用的"}), 400
    if not color:
        color = _CLASS_PALETTE[len(items) % len(_CLASS_PALETTE)]

    items.append({"name": name, "color": color})
    save_class_items(items)

    state = load_state()
    log_action(state, "新增类别", name)
    save_state(state)
    payload = _classes_payload()
    payload["ok"] = True
    return jsonify(payload)


@app.route("/api/classes/<path:name>", methods=["PATCH"])
def api_classes_update(name):
    """改名或换色。改名会同步改掉所有标注对象里的类别。"""
    body = request.get_json(silent=True) or {}
    new_name = str(body.get("name", "") or "").strip()
    new_color = str(body.get("color", "") or "").strip()

    items = load_class_items()
    idx = next((i for i, it in enumerate(items) if it["name"] == name), -1)
    if idx < 0:
        return jsonify({"error": "词表里没有「%s」" % name}), 404

    if new_color:
        if not (new_color.startswith("#") and len(new_color) in (4, 7)):
            return jsonify({"error": "颜色要写成 #rrggbb"}), 400
        items[idx]["color"] = new_color

    moved = 0
    if new_name and new_name != name:
        if len(new_name) > 40:
            return jsonify({"error": "类别名太长（最多 40 字符）"}), 400
        if any(ch in new_name for ch in "\r\n\t"):
            return jsonify({"error": "类别名不能含换行或制表符"}), 400
        if class_id_of(new_name) >= 0:
            return jsonify({"error": "类别「%s」已经存在" % new_name}), 409
        items[idx]["name"] = new_name
        moved = _rename_class_in_annotations(name, new_name)

    save_class_items(items)

    state = load_state()
    if new_name and new_name != name:
        log_action(state, "类别改名", "%s -> %s（同步 %d 个对象）"
                   % (name, new_name, moved))
    if new_color:
        log_action(state, "类别换色", "%s -> %s" % (name, new_color))
    save_state(state)

    payload = _classes_payload()
    payload["ok"] = True
    payload["renamed_objects"] = moved
    return jsonify(payload)


@app.route("/api/classes/<path:name>", methods=["DELETE"])
def api_classes_delete(name):
    """删除一个类别。

    还有对象在用时默认拒绝（409 + 数量），给两条出路：
      ?reassign=另一类  —— 那批对象整体改到另一个类别上（不丢数据）
      ?purge=1          —— 连同那批对象一起删掉（数据本身就不要了）

    第二条是为「这一类标错了」准备的：book 根本不该出现在这批图里，那 81 个
    框迁到 person 只是把错数据换个名字继续喂训练集，误检会跟着教下去。
    """
    reassign = str(request.args.get("reassign", "") or "").strip()
    purge = str(request.args.get("purge", "") or "").strip().lower() in ("1", "true", "yes")

    items = load_class_items()
    if not any(it["name"] == name for it in items):
        return jsonify({"error": "词表里没有「%s」" % name}), 404
    if len(items) <= 1:
        return jsonify({"error": "至少得留一个类别"}), 400
    if reassign and purge:
        return jsonify({"error": "reassign 和 purge 只能选一个"}), 400

    usage = _class_usage().get(name, 0)
    if usage and not reassign and not purge:
        return jsonify({
            "error": "还有 %d 个对象用着「%s」。可以整体迁到别的类别，"
                     "或者连同这些对象一起删掉。" % (usage, name),
            "in_use": usage,
            "candidates": [it["name"] for it in items if it["name"] != name],
            "need_reassign": True,
        }), 409

    moved = 0
    purged = 0
    if usage and reassign:
        if reassign == name:
            return jsonify({"error": "不能迁到正在删除的类别"}), 400
        if class_id_of(reassign) < 0:
            return jsonify({"error": "目标类别「%s」不在词表里" % reassign}), 400
        moved = _rename_class_in_annotations(name, reassign)
    elif usage and purge:
        purged = _purge_class_from_annotations(name)

    items = [it for it in items if it["name"] != name]
    save_class_items(items)

    state = load_state()
    if purged:
        log_action(state, "删除类别",
                   "%s（连同 %d 个对象一起删除）" % (name, purged))
    else:
        log_action(state, "删除类别",
                   "%s（%d 个对象迁到 %s）" % (name, moved, reassign or "-"))
    save_state(state)

    payload = _classes_payload()
    payload["ok"] = True
    payload["moved_objects"] = moved
    payload["purged_objects"] = purged
    return jsonify(payload)


@app.route("/api/images")
def api_images():
    """图片列表 + 每张图的审核状态。"""
    state = load_state()
    images = list_images()
    items = []
    for name in images:
        p = anno_path(name)
        reviewed = os.path.exists(p)
        edited = 0
        total = 0
        if reviewed:
            try:
                with open(p, encoding="utf-8") as f:
                    d = json.load(f)
                objs = d.get("objects", [])
                total = len(objs)
                edited = sum(1 for o in objs if o.get("edited"))
            except Exception:  # noqa: BLE001
                pass
        items.append({
            "name": name,
            "reviewed": reviewed,
            "edited": edited,
            "total": total,
        })
    # 最近一次删除的批次，前端据此显示「恢复」提示条。
    # 只算图片数——标注 json 是跟着图一起删的，报数量只报图才符合直觉。
    last_trash = None
    for b in _trash_batches():
        imgs = [f for f in b["files"]
                if os.path.splitext(f)[1].lower() != ".json"]
        if imgs:
            last_trash = {"stamp": b["stamp"], "count": len(imgs)}
            break

    return jsonify({
        "images": items,
        "total": len(items),
        "reviewed": sum(1 for i in items if i["reviewed"]),
        "edited_total": sum(i["edited"] for i in items),
        "round": state.get("round", 1),
        "last_trash": last_trash,
    })


@app.route("/api/annotation/<path:image_name>")
def api_get_annotation(image_name):
    """取一张图的标注。"""
    ann = load_anno(image_name)
    if ann is None:
        return jsonify({"error": "该图还没有标注数据"}), 404
    return jsonify(ann.to_dict())


@app.route("/api/annotation/<path:image_name>", methods=["POST"])
def api_save_annotation(image_name):
    """
    保存一张图的标注。

    前端每次修改后调用。服务端会：
    1. 校验类别是否在十类词表内
    2. 落盘 JSON
    3. 记录一条修正流水（如果这个对象是模型生成且被改过）
    """
    data = request.get_json(force=True)
    try:
        ann = Annotation.from_dict(data)
    except (KeyError, ValueError) as exc:
        return jsonify({"error": "标注数据格式错误: %s" % exc}), 400

    if ann.image_name != image_name:
        ann.image_name = image_name

    # 类别校验：词表外的类别不拦，只报给前端。
    # 【为什么放行而不是 400】词表是可改的。改完词表再打开旧图，图上那些
    # 已经不在词表里的对象如果一保存就报错，用户连「把它们删掉」这个动作
    # 都做不成，等于数据被锁死。放行 + 返回清单，让界面提示「导出会跳过」，
    # 用户自己决定是改类还是把类别加回词表。
    unknown = sorted({o.class_name for o in ann.objects
                      if class_id_of(o.class_name) < 0})

    save_anno(ann)

    state = load_state()
    log_action(state, "保存标注", "%s (%d 个对象, %d 个已修正)"
               % (image_name, len(ann.objects), ann.edited_count()))
    save_state(state)

    return jsonify({
        "ok": True,
        "image": ann.image_name,
        "objects": len(ann.objects),
        "edited": ann.edited_count(),
        "unknown_classes": unknown,
    })


# ---------------------------------------------------------------------------
# 预标注接口
# ---------------------------------------------------------------------------

@app.route("/api/prelabel/start", methods=["POST"])
def api_prelabel_start():
    """
    启动预标注（异步）。

    请求体可选：{"model": "official" | "custom", "weights": "model/best.pt"}
      official -> 用 YOLO-World + YOLO26n-seg 官方权重（首轮）
      custom   -> 用自训 best.pt 回标（第二轮）
    """
    body = request.get_json(silent=True) or {}
    mode = body.get("model", "official")
    # images 可选：只处理这几张（上传后一键标注就只标新增的，
    # 不会把已有人工修正的其他图重新覆盖掉）
    only = body.get("images") or None
    images = list_images()
    if only:
        wanted = set(only)
        images = [n for n in images if n in wanted]
    if not images:
        # 两种空集要分开说：整目录空，和「你指定的这几张找不到」。
        # 混成一句「目录里没有图片」会让人对着 153 张图片发呆（真事）。
        if only:
            head = "、".join(sorted(wanted)[:3])
            more = "" if len(wanted) <= 3 else " 等 %d 张" % len(wanted)
            return jsonify({
                "error": "指定的图片不在 data/images/ 里：%s%s" % (head, more)
            }), 400
        return jsonify({"error": "data/images/ 目录里没有图片"}), 400

    # 全量预标注 / 回标都会重写全部标注文件，人工修正跟着没。
    # 之前只靠前端一句 confirm，用户在电脑上点「预标注」就把 60 条修正覆盖了
    # （2026-10-28 真事，最后靠 git 里的版本捞回来）。
    # 这里加一道服务端闸门：不带 force=true 的全量请求，只要存在人工修正就拒绝，
    # 把「有多少条会没」这个数字直接摆到调用方面前。
    force = bool(body.get("force"))
    if only is None and not force:
        edited = 0
        for n in images:
            a = load_anno(n)
            if a is not None:
                edited += a.edited_count()
        if edited > 0:
            return jsonify({
                "error": ("已有 %d 条人工修正，全量%s会覆盖它们。"
                          "确认要覆盖请加 force=true。" % (
                              edited, "回标" if mode == "custom" else "预标注")),
                "edited": edited,
                "need_force": True,
            }), 409

    state = load_state()
    job_id = "job_%d_%d" % (int(time.time()), len(job_manager.jobs))
    job = _new_job(job_id, len(images))

    def runner(job):
        """交给 JobManager 的后台 worker 线程执行。"""
        if mode == "custom":
            weights = body.get("weights") or os.path.join(MODEL_DIR, "best.pt")
            if not os.path.exists(weights):
                raise FileNotFoundError("找不到自训权重: %s" % weights)
            run_relabel_job(job, IMAGE_DIR, weights,
                            round_tag="R%d" % (state.get("round", 1) + 1),
                            only=only)
        else:
            run_prelabel_job(job, IMAGE_DIR, get_classes(), only=only)

        # 预标注结果写回标注文件（作为草稿）
        for name, res in job.results.items():
            p = anno_path(name)
            objs = res.get("objects", [])
            if objs:
                from PIL import Image
                img_path = os.path.join(IMAGE_DIR, name)
                try:
                    with Image.open(img_path) as im:
                        w, h = im.size
                except Exception:  # noqa: BLE001
                    w = h = 960
                ann = Annotation(name, w, h,
                                 objects=[Object.from_dict(o) for o in objs])
                save_anno(ann)
            elif not os.path.exists(p):
                from PIL import Image
                img_path = os.path.join(IMAGE_DIR, name)
                try:
                    with Image.open(img_path) as im:
                        w, h = im.size
                except Exception:  # noqa: BLE001
                    w = h = 960
                save_anno(Annotation(name, w, h, objects=[]))

        st = load_state()
        log_action(st, "预标注完成", "%s, %d 张图" % (mode, len(job.results)))
        save_state(st)

    # 交给 JobManager 的后台 worker 线程跑，主线程立刻返回 job_id，
    # 前端轮询 /api/prelabel/progress/<job_id> 画进度条。
    job_manager.submit(job, runner)
    return jsonify({"ok": True, "job_id": job_id, "total": len(images)})


def _new_job(job_id, total):
    from prelabel import PrelabelJob
    return PrelabelJob(job_id, total)


@app.route("/api/prelabel/progress/<job_id>")
def api_prelabel_progress(job_id):
    """轮询预标注进度。"""
    job = job_manager.get(job_id)
    if job is None:
        return jsonify({"error": "任务不存在"}), 404
    d = job.to_dict()
    d["results_count"] = len(job.results)
    return jsonify(d)


# ---------------------------------------------------------------------------
# 导出接口
# ---------------------------------------------------------------------------

@app.route("/api/export", methods=["POST"])
def api_export():
    """
    导出三格式数据集 + 跑格式自检。

    请求体可选：{"val_ratio": 0.2, "has_seg": true, "copy_images": true}
    """
    body = request.get_json(silent=True) or {}
    val_ratio = float(body.get("val_ratio", 0.2))
    has_seg = bool(body.get("has_seg", True))
    copy_images = bool(body.get("copy_images", True))

    anns = []
    for name in list_images():
        ann = load_anno(name)
        if ann is not None:
            anns.append(ann)

    if not anns:
        return jsonify({"error": "还没有任何标注数据，先跑预标注"}), 400

    # 不清空整个 dataset 目录。
    # 导出是幂等的：同名文件直接覆盖，exporter 只会删掉「上次有、这次没有」的残留。
    # 之前这里用 shutil.rmtree 整目录删，150 个文件的批量删除会被系统安全策略拦下，
    # 表现为页面请求直接断开、什么错误都看不到。
    os.makedirs(DATASET_DIR, exist_ok=True)

    stats = export_dataset(anns, DATASET_DIR, IMAGE_DIR,
                           val_ratio=val_ratio, has_seg=has_seg,
                           copy_images=copy_images)

    ok, problems, counts = validate_dataset(DATASET_DIR)

    state = load_state()
    log_action(state, "导出数据集", "训练%d/验证%d, 校验%s"
               % (stats["train_images"], stats["val_images"],
                  "通过" if ok else "有问题"))
    save_state(state)

    return jsonify({
        "ok": True,
        "stats": stats,
        "validation": {
            "passed": ok,
            "problems": problems,
            "report": format_report(counts),
        },
    })


# ---------------------------------------------------------------------------
# 训练接口
# ---------------------------------------------------------------------------

@app.route("/api/train", methods=["POST"])
def api_train():
    """
    触发训练（后台跑，因为要十几分钟）。

    任务书要求 "train.py 仅接受平台导出数据集"，
    所以这里固定调用 train.py，且 train.py 内部硬编码只读 ./dataset。
    """
    body = request.get_json(silent=True) or {}
    script = os.path.join(BASE_DIR, "train.py")
    if not os.path.exists(script):
        return jsonify({"error": "找不到 train.py"}), 400

    log_file = os.path.join(DATA_DIR, "train.log")
    cmd = [sys.executable, script]
    if body.get("epochs"):
        cmd += ["--epochs", str(int(body["epochs"]))]

    def worker():
        state = load_state()
        log_action(state, "开始训练", " ".join(cmd))
        save_state(state)
        with open(log_file, "w", encoding="utf-8") as f:
            proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT,
                                    cwd=BASE_DIR)
        # 记下 pid 和启动时刻。前端靠它区分「还在跑」和「进程没了但没写完成标记」，
        # 后者是最容易让人一直干等的情况（窗口被关、任务被系统杀）。
        with open(TRAIN_META, "w", encoding="utf-8") as f:
            json.dump({"pid": proc.pid, "started_at": time.time(),
                       "cmd": cmd}, f, ensure_ascii=False)

    threading.Thread(target=worker, daemon=True).start()
    return jsonify({"ok": True, "log": log_file})


def _train_busy():
    """训练进程是否还在跑（自主学习用它避让——同一块显卡不能同时跑两个）。"""
    if not os.path.exists(TRAIN_META):
        return False
    try:
        with open(TRAIN_META, encoding="utf-8") as f:
            meta = json.load(f)
    except (OSError, ValueError):
        return False
    return pid_alive(meta.get("pid"))


@app.route("/api/train/status")
def api_train_status():
    """训练进度快照，给前端画进度条。

    日志是唯一的事实来源（ultralytics 只往 stdout 写），这里把它翻译成
    「第几轮 / 百分之几 / 还要多久」。另外单独判断进程是否还活着：
    日志停在第 43 轮、进程却已经死了（关窗口、被杀）时，必须报失败，
    不能让它永远挂在 43% —— 那比没有进度条更糟。
    """
    text = ""
    if os.path.exists(TRAIN_LOG):
        with open(TRAIN_LOG, encoding="utf-8", errors="replace") as f:
            text = f.read()[-200000:]   # 只读尾部，100 轮的日志几百 KB 没必要全读
    snap = parse_train_log(text)

    meta = {}
    if os.path.exists(TRAIN_META):
        try:
            with open(TRAIN_META, encoding="utf-8") as f:
                meta = json.load(f) or {}
        except (OSError, ValueError):
            meta = {}

    alive = pid_alive(meta.get("pid"))
    # 只信 meta 里的启动时刻。不要拿日志文件的 ctime 兜底：训练开始时是
    # open(w) 截断重写，文件创建时间还停在很久以前，算出来会是「已用 17 分钟」这种假数。
    # 拿不到就不显示已用时间，宁可少一条信息也不编一条。
    started = meta.get("started_at")
    elapsed = (time.time() - started) if started else None
    if snap["state"] in ("idle", "starting") and alive:
        snap["state"] = "starting"   # 进程在，但还没打出第一轮的表格
    if snap["state"] in ("running", "starting") and not alive:
        snap["state"] = "failed"
        snap["error"] = snap["error"] or "训练进程已退出，日志里没有完成标记（可能被中断）"

    # 完成时以日志自己记的耗时为准（它不含模型加载那几十秒）
    if snap["state"] == "done" and snap.get("minutes"):
        elapsed = snap["minutes"] * 60

    # 预计剩余：按当前平均速度外推。进度太靠前时不给（误差会大到没意义）。
    eta = None
    if snap["state"] == "running" and elapsed and snap["percent"] > 2:
        eta = elapsed / snap["percent"] * (100 - snap["percent"])

    snap["alive"] = alive
    snap["elapsed_sec"] = round(elapsed, 1) if elapsed else None
    snap["elapsed_text"] = human_duration(elapsed)
    snap["eta_sec"] = round(eta, 1) if eta else None
    snap["eta_text"] = human_duration(eta)
    return jsonify(snap)


@app.route("/api/train/log")
def api_train_log():
    """看训练日志。"""
    p = os.path.join(DATA_DIR, "train.log")
    if not os.path.exists(p):
        return jsonify({"log": ""})
    with open(p, encoding="utf-8", errors="replace") as f:
        content = f.read()
    return jsonify({"log": content[-20000:]})


# ---------------------------------------------------------------------------
# 自主学习（持续微调）
#
# 和「训练」按钮的区别：
#   训练   —— 人点一下，从官方权重重来一遍，结果直接覆盖 best.pt
#   自主学习 —— 攒够修正自己跑一代，从当前 best.pt 续训，先出候选权重，
#               验证集过线才换上去。指挥权在人，执行权在机器。
# ---------------------------------------------------------------------------

def _prelabel_busy():
    """预标注/回标还在跑吗（它也吃显卡，不能和训练撞车）。"""
    for job in list(job_manager.jobs.values()):
        if getattr(job, "status", None) in ("pending", "running"):
            return True
    return False


def _gpu_busy():
    return _train_busy() or _prelabel_busy()


@app.route("/api/autolearn")
def api_autolearn_status():
    return jsonify(autolearn.status(busy=_gpu_busy()))


@app.route("/api/autolearn/config", methods=["POST"])
def api_autolearn_config():
    """开关与阈值。改完立刻生效，巡检线程下一轮就会按新设置判断。"""
    body = request.get_json(silent=True) or {}
    cfg = autolearn.load_cfg()

    if "threshold" in body:
        try:
            th = int(body["threshold"])
        except (TypeError, ValueError):
            return jsonify({"error": "阈值要是数字"}), 400
        cfg["threshold"] = max(5, min(200, th))

    if "strategy" in body:
        st = str(body["strategy"])
        if st not in (autolearn.STRATEGY_FT, autolearn.STRATEGY_RETRAIN):
            return jsonify({"error": "策略只能是 ft（快速微调）或 retrain（基座重训）"}), 400
        cfg["strategy"] = st

    if "enabled" in body:
        on = bool(body["enabled"])
        cfg = autolearn.set_enabled(cfg, on, autolearn.scan_edited())
        state = load_state()
        log_action(state, "自主学习", "开启" if on else "关闭")
        save_state(state)

    autolearn.save_cfg(cfg)
    return jsonify(autolearn.status(busy=_gpu_busy()))


@app.route("/api/autolearn/run", methods=["POST"])
def api_autolearn_run():
    """立刻学一代（不用等攒够）。跑在后台线程，前端轮询状态。"""
    if autolearn.read_lock():
        return jsonify({"error": "上一代还在学，等它结束"}), 409
    if _gpu_busy():
        return jsonify({"error": "训练或预标注正在占用显卡，等它跑完再学"}), 409

    def worker():
        autolearn.run_round()

    threading.Thread(target=worker, daemon=True).start()
    return jsonify({"ok": True, "message": "已开始学习第一代，进度看左侧状态"})


def autolearn_watcher():
    """后台巡检：每 60 秒问一次「攒够了吗」。

    为什么是 60 秒而不是实时：判据要扫全库 157 个标注文件，
    太频繁没必要；而学习一代本身要几分钟，多等一分钟无感。
    """
    while True:
        time.sleep(60)
        try:
            autolearn.tick(busy=_gpu_busy())
        except Exception as exc:  # noqa: BLE001
            # 巡检线程绝不能死：死了以后开关还亮着，但永远不会再学
            try:
                autolearn._log("[巡检异常] %s" % exc)
            except Exception:  # noqa: BLE001
                pass


# ---------------------------------------------------------------------------
# 修正记录与对比结果
# ---------------------------------------------------------------------------

@app.route("/api/records")
def api_records():
    """
    统计修正记录。

    修正率 = 实际修改的对象数 / 总对象数
    （任务书定义：仅确认、未修改的不计入分子）
    """
    state = load_state()
    total_objects = 0
    edited_objects = 0
    per_class = {}
    per_image = []
    detail = []

    for name in list_images():
        ann = load_anno(name)
        if ann is None:
            continue
        total_objects += ann.total_count()
        edited_objects += ann.edited_count()
        if ann.edited_count():
            per_image.append({
                "image": name,
                "edited": ann.edited_count(),
                "total": ann.total_count(),
            })
        for o in ann.objects:
            per_class.setdefault(o.class_name, {"total": 0, "edited": 0})
            per_class[o.class_name]["total"] += 1
            if o.edited:
                per_class[o.class_name]["edited"] += 1
                detail.append({
                    "image": name,
                    "object_id": o.id,
                    "class": o.class_name,
                    "source": o.source,
                })

    rate = (edited_objects / total_objects * 100) if total_objects else 0.0
    return jsonify({
        "total_objects": total_objects,
        "edited_objects": edited_objects,
        "correction_rate": round(rate, 2),
        "per_class": per_class,
        "per_image": per_image,
        "detail": detail,
        "rounds": state.get("rounds", {}),
    })


@app.route("/api/records/export", methods=["POST"])
def api_records_export():
    """把修正记录和对比结果导出成 CSV（任务书要求放在 records/）。"""
    import csv
    os.makedirs(RECORD_DIR, exist_ok=True)

    # 1) 修正记录明细
    rows = []
    for name in list_images():
        ann = load_anno(name)
        if ann is None:
            continue
        for o in ann.objects:
            if o.edited:
                rows.append({
                    "image": name,
                    "object_id": o.id,
                    "class": o.class_name,
                    "source": o.source,
                    "type": "类别修正" if o.source == "human" else "边界修正",
                })

    fix_path = os.path.join(RECORD_DIR, "修正记录.csv")
    with open(fix_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image", "object_id", "class",
                                          "source", "type"])
        w.writeheader()
        w.writerows(rows)

    # 2) 对比结果
    state = load_state()
    cmp_path = os.path.join(RECORD_DIR, "对比结果.csv")
    with open(cmp_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["轮次", "模型", "总对象数", "修正对象数", "修正率(%)", "mAP50"])
        for rnd, info in sorted(state.get("rounds", {}).items()):
            w.writerow([
                rnd, info.get("model", ""), info.get("total_objects", 0),
                info.get("edited_objects", 0), info.get("correction_rate", 0),
                info.get("map50", ""),
            ])

    return jsonify({"ok": True, "fix": fix_path, "compare": cmp_path,
                    "records": len(rows)})


@app.route("/api/comparison")
def api_comparison():
    """对比页数据：各轮 mAP 与修正率。"""
    state = load_state()
    return jsonify({
        "rounds": state.get("rounds", {}),
        "logs": state.get("logs", [])[-50:],
    })


@app.route("/api/state/save", methods=["POST"])
def api_state_save():
    """保存各轮统计（mAP 等），供对比页展示。"""
    body = request.get_json(force=True)
    state = load_state()
    rnd = body.get("round", "R1")
    state.setdefault("rounds", {})[rnd] = {
        "model": body.get("model", ""),
        "total_objects": body.get("total_objects", 0),
        "edited_objects": body.get("edited_objects", 0),
        "correction_rate": body.get("correction_rate", 0),
        "map50": body.get("map50", ""),
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    log_action(state, "记录轮次统计", "%s mAP=%s 修正率=%s%%"
               % (rnd, body.get("map50", ""), body.get("correction_rate", 0)))
    save_state(state)
    return jsonify({"ok": True})


# ---------------------------------------------------------------------------
# 静态资源
# ---------------------------------------------------------------------------

@app.route("/images/<path:filename>")
def serve_image(filename):
    return send_from_directory(IMAGE_DIR, filename)


@app.route("/model/<path:filename>")
def serve_model(filename):
    return send_from_directory(MODEL_DIR, filename)


# ---------------------------------------------------------------------------
# 启动
# ---------------------------------------------------------------------------

def pick_port(preferred=5000, tries=20, host="127.0.0.1"):
    """
    找一个能用端口。

    【为什么需要】一键启动时如果 5000 正被占用（比如上次的窗口没关，
    或者别的软件占了），app.run 会直接抛 OSError 闪退，
    用户看到的就是"双击了但窗口一闪而过"，根本不知道发生了什么。
    这里主动往后找一个空端口，并把实际地址打印出来。

    host 要和实际监听地址一致：手机访问时监听 0.0.0.0，
    如果还按 127.0.0.1 探端口，别的网卡占着同一个端口就探不出来。
    """
    import socket
    for port in range(preferred, preferred + tries):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.bind((host, port))
            return port
        except OSError:
            continue
        finally:
            s.close()
    return preferred


def lan_ip():
    """
    拿本机在局域网里的 IP，给手机访问用。

    【为什么不是 gethostbyname(gethostname())】
    那套在装了虚拟机/VPN 的机器上经常返回 127.0.0.1 或虚拟网卡地址。
    往 8.8.8.8 建一个 UDP socket 不会真的发包，只是让系统按路由表
    选出"默认出口网卡"的地址——这正是手机该连的那个。
    """
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        return ip if ip and not ip.startswith("127.") else "127.0.0.1"
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def open_browser(url):
    """延迟 1.2 秒后自动打开浏览器（等服务起来再打开）。"""
    time.sleep(1.2)
    try:
        webbrowser.open(url)
    except Exception:  # noqa: BLE001
        pass


if __name__ == "__main__":
    # --lan：监听 0.0.0.0，让同一 Wi-Fi 下的手机/平板也能打开。
    # 默认只监听本机，避免把标注系统暴露到公共网络上。
    lan = ("--lan" in sys.argv) or (os.environ.get("IMGLABEL_LAN") == "1")
    host = "0.0.0.0" if lan else "127.0.0.1"

    # --port N：指定首选端口。tools/tunnel.py 用它挑一个确实空闲的端口再起服务，
    # 否则 pick_port 一旦顺延，隧道就会连到错的地方。被占用时仍然自动顺延。
    prefer = 5000
    if "--port" in sys.argv:
        try:
            prefer = int(sys.argv[sys.argv.index("--port") + 1])
        except (IndexError, ValueError):
            print("  [警告] --port 后面要跟一个数字，已改用 5000")

    port = pick_port(prefer, host=host)
    url = "http://127.0.0.1:%d" % port

    print("=" * 64)
    print("  人机协同图像标注系统")
    print("=" * 64)
    print("  图片目录 : %s" % IMAGE_DIR)
    print("  已有图片 : %d 张" % len(list_images()))
    print("  访问地址 : %s" % url)
    if port != 5000:
        print("  说明     : 5000 端口被占用，已自动改用 %d" % port)
    if lan:
        ip = lan_ip()
        if ip == "127.0.0.1":
            print("  手机访问 : 没找到局域网地址，检查是否连了 Wi-Fi")
        else:
            print("-" * 64)
            print("  手机访问 : http://%s:%d" % (ip, port))
            print("             ↑ 手机连同一个 Wi-Fi，浏览器输这个地址")
            print("             （Windows 弹防火墙提示要点「允许」）")
            print("-" * 64)
    print("  打开页面后，点左上角「上传照片并标注」即可开始。")
    print("  关闭这个窗口即停止服务。")
    print("=" * 64)

    threading.Thread(target=open_browser, args=(url,), daemon=True).start()
    # 自主学习巡检：开关打开后，攒够修正就自己学一代（默认关闭，界面上开）
    threading.Thread(target=autolearn_watcher, daemon=True).start()
    app.run(host=host, port=port, debug=False, threaded=True)
