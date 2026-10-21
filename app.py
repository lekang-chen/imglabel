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

import json
import os
import shutil
import subprocess
import sys
import threading
import time
import webbrowser

from flask import (Flask, jsonify, request, send_file,
                   send_from_directory, abort)

from core import (CLASSES, Annotation, Object, class_color,
                  point_in_polygon)
from exporter import export_dataset, validate_dataset, format_report
from prelabel import JobManager, run_prelabel_job, run_relabel_job

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

for d in (DATA_DIR, IMAGE_DIR, ANNO_DIR, DATASET_DIR, MODEL_DIR, RECORD_DIR):
    os.makedirs(d, exist_ok=True)

app = Flask(__name__, static_folder="canvas", static_url_path="/static")
app.config["JSON_AS_ASCII"] = False

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
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    return sorted([f for f in os.listdir(IMAGE_DIR)
                   if f.lower().endswith(exts)])


# ---------------------------------------------------------------------------
# 基础接口
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/classes")
def api_classes():
    """十类词表 + 每类颜色，前端渲染用。"""
    return jsonify({
        "classes": [
            {"name": c, "id": i, "color": "#%02x%02x%02x" % class_color(c)}
            for i, c in enumerate(CLASSES)
        ]
    })


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
    return jsonify({
        "images": items,
        "total": len(items),
        "reviewed": sum(1 for i in items if i["reviewed"]),
        "edited_total": sum(i["edited"] for i in items),
        "round": state.get("round", 1),
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

    # 类别校验
    for o in ann.objects:
        if o.class_name not in CLASSES:
            return jsonify({"error": "非法类别: %s" % o.class_name}), 400

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
    images = list_images()
    if not images:
        return jsonify({"error": "data/images/ 目录里没有图片"}), 400

    state = load_state()
    job_id = "job_%d_%d" % (int(time.time()), len(job_manager.jobs))
    job = job_manager.submit(_new_job(job_id, len(images)))

    def worker():
        if mode == "custom":
            weights = body.get("weights") or os.path.join(MODEL_DIR, "best.pt")
            if not os.path.exists(weights):
                job.status = "failed"
                job.message = "找不到自训权重: %s" % weights
                return
            run_relabel_job(job, IMAGE_DIR, weights,
                            round_tag="R%d" % (state.get("round", 1) + 1))
        else:
            run_prelabel_job(job, IMAGE_DIR, CLASSES)

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

    threading.Thread(target=worker, daemon=True).start()
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

    # 清空旧的导出（保留目录本身）
    if os.path.isdir(DATASET_DIR):
        for sub in ("images", "labels", "masks", "semantic"):
            p = os.path.join(DATASET_DIR, sub)
            if os.path.isdir(p):
                shutil.rmtree(p)
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
            subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT,
                             cwd=BASE_DIR)

    threading.Thread(target=worker, daemon=True).start()
    return jsonify({"ok": True, "log": log_file})


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

def open_browser():
    """延迟 1.2 秒后自动打开浏览器（等服务起来再打开）。"""
    time.sleep(1.2)
    try:
        webbrowser.open("http://127.0.0.1:5000")
    except Exception:  # noqa: BLE001
        pass


if __name__ == "__main__":
    print("=" * 60)
    print("人机协同图像标注系统")
    print("图片目录: %s" % IMAGE_DIR)
    print("访问地址: http://127.0.0.1:5000")
    print("=" * 60)
    threading.Thread(target=open_browser, daemon=True).start()
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
