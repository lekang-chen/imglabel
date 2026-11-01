#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
autolearn.py —— 自主学习：攒够人工修正，自动微调一代，过线才启用

【它解决什么】
原来的闭环是人推着走的：改标注 → 点导出 → 点训练 → 点回标。
人一懒，模型就停在原地。这个模块把「攒够修正 → 学一代」变成自动的。

【为什么不能「每改一笔就学」】
单条修正的信息量太小。一张图改 3 笔，对 126 张训练集来说是 0.1% 的扰动，
微调出来的模型基本等于原样；而且要跑几分钟，人改一笔等几分钟，体验不可用。
所以按批：攒够 threshold 条（默认 20）才学一代。

【为什么不会「越练越差」】
这是持续学习最经典的坑：模型在新数据上学到东西的同时，会把旧的忘掉
（灾难性遗忘），表现为验证集指标下滑。三道闸门：

  1. 续训而非重训：从当前 best.pt 出发、学习率压到 0.002，
     不是从官方权重重来——重来等于把之前学的全丢了。
  2. 全量回放：每代都拿「库里全部标注」重新导出训练，不是只喂新增的几张。
     旧数据一直在场，遗忘就慢得多。
  3. 指标闸门：新权重先当候选（model/_cand/），在同一个验证集上评估，
     检测或掩码任一指标掉超过 TOLERANCE 就判失败，候选挪进 model/_backup/
     留档，best.pt 一个字节都不动。
  闸门不放行时界面上会写清「这一代未采纳，因为掩码 mAP 掉了 0.031」——
  不是悄悄失败，也不是硬着头皮换上去。

【状态机】
  idle → export（导数据集）→ train（候选权重）→ eval（验证集打分）
       → gate（比对基线）→ adopt / reject → idle

【文件】
  data/autolearn.json         开关、阈值、第几代、上次结果
  data/autolearn.log          每一代的过程日志（人可读）
  data/autolearn_eval.json    本代候选权重的评估明细
  records/自主学习记录.csv     每代一行，可 Excel 打开
  model/_cand/genN.pt         候选权重（采纳后拷成 best.pt）
  model/_backup/              旧权重与被否掉的候选
"""

import csv
import json
import os
import shutil
import subprocess
import sys
import time

from core import Annotation, pid_alive
from exporter import export_dataset, validate_dataset
from train import backup_existing_best

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
ANNO_DIR = os.path.join(DATA_DIR, "annotations")
IMAGE_DIR = os.path.join(DATA_DIR, "images")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
RECORDS_DIR = os.path.join(BASE_DIR, "records")
MODEL_DIR = os.path.join(BASE_DIR, "model")
CAND_DIR = os.path.join(MODEL_DIR, "_cand")

CONFIG_PATH = os.path.join(DATA_DIR, "autolearn.json")
LOCK_PATH = os.path.join(DATA_DIR, "autolearn.lock")
LOG_PATH = os.path.join(DATA_DIR, "autolearn.log")
EVAL_JSON = os.path.join(DATA_DIR, "autolearn_eval.json")
BEST_METRICS = os.path.join(DATA_DIR, "best_metrics.json")
HISTORY_CSV = os.path.join(RECORDS_DIR, "自主学习记录.csv")
BEST_PT = os.path.join(MODEL_DIR, "best.pt")

DEFAULT_THRESHOLD = 20      # 攒多少条修正学一代
FT_EPOCHS = 120             # 续训轮数：和 iter3_long 一致，早停会提前收（实测 3 分钟左右）
RETRAIN_EPOCHS = 100        # 基座重训的轮数：和「训练」按钮、R4 那一轮同一配方
RETRAIN_PATIENCE = 20
# 容差不是「越小越严格」。同配置换随机种子，掩码 mAP 实测就抖 ±0.035，
# 卡到 0.01 等于要求「每次都比上次好」，那是噪声判据，会把好模型全否掉。
# 取 0.02：比噪声小，能拦住真退化，又不会把持平的一代当失败。
TOLERANCE = 0.02

# 两种学法，实测差别很大：
#   ft      —— 从当前 best.pt 续训，小学习率 + 关掉 mosaic/copy_paste。快（1~2 分钟），
#              只在新修正信息量大时才涨得动。
#   retrain —— 从官方基座重训一遍全量数据。慢（4~5 分钟），但历史上 R2/R3/R4 的
#              提升全走这条路。新修正攒得多（比如一次导入几十张新图）时切它更划算。
# 默认 ft：日常边标边学要的是「快、且不把现有水平弄坏」，不是每次冲新高。
STRATEGY_FT = "ft"
STRATEGY_RETRAIN = "retrain"
DEFAULT_STRATEGY = STRATEGY_FT

DEFAULTS = {
    "enabled": False,
    "threshold": DEFAULT_THRESHOLD,
    "round": 0,
    "learned_edited": 0,     # 上次学完时的「全库已修正对象数」
    "running": False,
    "stage": "idle",
    "last_run_at": None,
    "strategy": DEFAULT_STRATEGY,
    # 上一代导出时的图片总数：用来判断「这一代数据是不是真的变多了」
    "last_train_images": 0,
    "last_result": None,     # {gen, accepted, reason, old, new, at}
    "history": [],           # 最近若干代摘要
}


# ---------------------------------------------------------------------------
# 配置读写
# ---------------------------------------------------------------------------

def load_cfg(path=None):
    """读配置，缺字段用默认值补齐（旧版本文件不会因为少字段炸掉）。

    path 默认在调用时取当前的 CONFIG_PATH，而不是写在参数默认值里——
    默认值只在函数定义时求值一次，测试想把路径改到沙箱就会改不动。
    """
    path = path or CONFIG_PATH
    cfg = dict(DEFAULTS)
    cfg["history"] = []
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                cfg.update(data)
        except (OSError, ValueError):
            pass
    if not isinstance(cfg.get("history"), list):
        cfg["history"] = []
    return cfg


def save_cfg(cfg, path=None):
    """先写临时文件再替换：中途被打断也不会留下半截 JSON。"""
    path = path or CONFIG_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# 数据侧：全库已修正对象数（判据是「人改过多少笔」，不是「新传多少张」）
# ---------------------------------------------------------------------------

def load_annos(anno_dir=None):
    """读出目录下全部标注（顺序固定，保证导出划分可复现）。"""
    anno_dir = anno_dir or ANNO_DIR
    anns = []
    if not os.path.isdir(anno_dir):
        return anns
    for fn in sorted(os.listdir(anno_dir)):
        if not fn.endswith(".json"):
            continue
        try:
            with open(os.path.join(anno_dir, fn), encoding="utf-8") as f:
                anns.append(Annotation.from_dict(json.load(f)))
        except (OSError, ValueError, KeyError):
            continue
    return anns


def scan_edited(anno_dir=None):
    """全库「被人改过」的对象总数。"""
    return sum(a.edited_count() for a in load_annos(anno_dir))


def pending(cfg, edited_total):
    """离下一次学习还差多少条（已攒的）。"""
    return max(0, int(edited_total) - int(cfg.get("learned_edited", 0)))


def should_switch_to_retrain(prev_images, now_images):
    """训练集明显变多时，这一代该改成基座重训而不是续训。

    【判据为什么是「张数」而不是「修正条数」】
    ft 那条路的 freeze=10 冻结了主干，它只会把检测头往当前数据上调一点点。
    同一批图上把框挪准，这够用；但训练集里多了模型没见过的场景时，
    冻结的主干根本学不进去——新图喂多少都是白喂。
    所以只看「图片数有没有明显变多」：至少 5 张，且至少 3%。
    只看 5 张是为了小库不至于「加 3 张就重训 5 分钟」；再看 3% 是防止
    大库加 5 张也被判成「数据变了」。
    3% 这个数按现在的库算：157 张时门槛是 5 张（一张新图都没白加），
    1000 张时门槛是 30 张。
    """
    try:
        prev_images = int(prev_images or 0)
        now_images = int(now_images or 0)
    except (TypeError, ValueError):
        return False
    if prev_images <= 0:
        return False
    return (now_images - prev_images) >= max(5, int(prev_images * 0.03))


def should_learn(cfg, edited_total):
    if not cfg.get("enabled"):
        return False
    if cfg.get("running"):
        return False
    return pending(cfg, edited_total) >= int(cfg.get("threshold", DEFAULT_THRESHOLD))


def set_enabled(cfg, enabled, edited_total=None):
    """打开开关时把计数基准对齐到现在。

    否则第一次开启会因为「历史上攒的 62 条修正」立刻触发一整代训练——
    用户只是点了个开关，显卡却突然占满几分钟。基准对齐后从这一刻起算，
    想立刻学就用「立即学一代」。
    """
    cfg = dict(cfg)
    cfg["enabled"] = bool(enabled)
    if (enabled and int(cfg.get("round", 0)) == 0
            and int(cfg.get("learned_edited", 0)) == 0
            and edited_total is not None):
        cfg["learned_edited"] = int(edited_total)
    return cfg


# ---------------------------------------------------------------------------
# 闸门：这一代该不该采纳（纯函数，测试重点）
# ---------------------------------------------------------------------------

def decide_accept(old, new, tol=TOLERANCE):
    """比对基线与候选的验证集指标，返回 (是否采纳, 人话理由)。

    old/new 都是 {"box_map50": float, "mask_map50": float} 结构。
    掩码是本项目的核心标注形式，检测是从掩码派生的，所以两个都要看：
    只有掩码涨、检测暴跌的情况，说明模型在丢框，同样不能放行。
    """
    if not old or old.get("mask_map50") is None or old.get("box_map50") is None:
        return True, "没有基线指标，直接采用（这一代就是新基线）"

    d_mask = float(new["mask_map50"]) - float(old["mask_map50"])
    d_box = float(new["box_map50"]) - float(old["box_map50"])

    if d_mask < -tol:
        return False, "掩码 mAP 掉了 %.3f（超过容差 %.2f），判为退化" % (-d_mask, tol)
    if d_box < -tol:
        return False, "检测 mAP 掉了 %.3f（超过容差 %.2f），判为退化" % (-d_box, tol)
    return True, "掩码 %+0.3f / 检测 %+0.3f，未退化" % (d_mask, d_box)


# ---------------------------------------------------------------------------
# 运行锁：一个进程同时在学两代没有意义，还会互相覆盖日志与候选权重
# ---------------------------------------------------------------------------

def read_lock():
    if not os.path.exists(LOCK_PATH):
        return None
    try:
        with open(LOCK_PATH, encoding="utf-8") as f:
            info = json.load(f)
    except (OSError, ValueError):
        return None
    if pid_alive(info.get("pid")):
        return info
    try:
        os.remove(LOCK_PATH)
    except OSError:
        pass
    return None


def write_lock(pid=None, stage="export"):
    try:
        with open(LOCK_PATH, "w", encoding="utf-8") as f:
            json.dump({"pid": pid or os.getpid(), "stage": stage,
                       "started_at": time.time()}, f, ensure_ascii=False)
    except OSError:
        pass


def clear_lock():
    try:
        os.remove(LOCK_PATH)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# 日志与记录
# ---------------------------------------------------------------------------

def _log(msg, log_path=None):
    log_path = log_path or LOG_PATH
    line = "[%s] %s" % (time.strftime("%H:%M:%S"), msg)
    print(line, flush=True)
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


def _append_csv(row, path=None):
    path = path or HISTORY_CSV
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new = not os.path.exists(path)
    header = ["代数", "时间", "本次新增修正", "训练轮数", "学法", "训练集张数",
              "旧检测mAP50", "新检测mAP50", "旧掩码mAP50", "新掩码mAP50",
              "结论", "原因"]
    try:
        with open(path, "a", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(header)
            w.writerow(row)
    except OSError:
        pass


def _read_json(path, default=None):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def _run(cmd, log_file):
    """跑子进程，输出原样进日志文件。返回 returncode。"""
    with open(log_file, "a", encoding="utf-8") as f:
        f.write("\n$ " + " ".join(cmd) + "\n")
        f.flush()
        proc = subprocess.run(cmd, cwd=BASE_DIR, stdout=f,
                              stderr=subprocess.STDOUT)
    return proc.returncode


# ---------------------------------------------------------------------------
# 一代学习的完整流程
# ---------------------------------------------------------------------------

def run_round(edited_total=None, epochs=None):
    """跑一代自主学习。同步阻塞（几秒到几分钟），调用方自己开线程。

    返回 {"ok": bool, "accepted": bool, "reason": str, ...}
    """
    cfg = load_cfg()

    if cfg.get("running") or read_lock():
        return {"ok": False, "error": "上一代还在学，别同时跑两个"}

    edited_total = scan_edited() if edited_total is None else int(edited_total)
    strategy = cfg.get("strategy", DEFAULT_STRATEGY)
    auto_epochs = epochs is None
    if auto_epochs:
        epochs = RETRAIN_EPOCHS if strategy == STRATEGY_RETRAIN else FT_EPOCHS
    gen = int(cfg.get("round", 0)) + 1
    started_edited = edited_total
    added = pending(cfg, edited_total)

    cfg["running"] = True
    cfg["stage"] = "export"
    save_cfg(cfg)
    write_lock(stage="export")

    old_metrics = _read_json(BEST_METRICS, {}) or {}
    _log("========== 第 %d 代自主学习开始（新增修正 %d 条，基线检测 %s / 掩码 %s）"
         % (gen, pending(cfg, edited_total),
            old_metrics.get("box_map50", "—"), old_metrics.get("mask_map50", "—")))

    try:
        # --- 1. 导出数据集：每代都用「全部标注」重导，旧数据一直在场，防遗忘 ---
        anns = load_annos()
        if not anns:
            raise RuntimeError("还没有任何标注，无法学习")
        stats = export_dataset(anns, DATASET_DIR, IMAGE_DIR)
        ok, problems, _counts = validate_dataset(DATASET_DIR)
        if not ok:
            raise RuntimeError("数据集自检未通过：%s" % "；".join(problems[:3]))
        _log("导出完成：训练 %d 张 / 验证 %d 张（%s）"
             % (stats["train_images"], stats["val_images"],
                stats.get("val_split_source", "")))

        # 【为什么这里要换学法】
        # ft 是「从 best.pt 续训 + freeze=10 冻结主干」：只调检测头，
        # 适合「同一批图上把框挪准一点」。可一旦训练集里出现了模型没见过的
        # 场景，冻结的主干根本学不进去——看多少新图都是白看。
        # 所以只要图片数明显变多（>=5 张且 >=5%），就改走基座重训：
        # 慢一点，但新场景真能进模型。
        prev_n = int(cfg.get("last_train_images") or 0)
        if should_switch_to_retrain(prev_n, stats["total_images"]) \
                and strategy != STRATEGY_RETRAIN:
            strategy = STRATEGY_RETRAIN
            if auto_epochs:
                epochs = RETRAIN_EPOCHS
            _log("训练集从 %d 张涨到 %d 张，自动改用基座重训"
                 "（续训冻结主干，学不进新场景）"
                 % (prev_n, stats["total_images"]))
        cfg["last_train_images"] = stats["total_images"]
        save_cfg(cfg)

        # --- 2. 续训出候选权重（不碰 best.pt） ---
        cfg["stage"] = "train"
        save_cfg(cfg)
        write_lock(stage="train")
        os.makedirs(CAND_DIR, exist_ok=True)
        cand = os.path.join(CAND_DIR, "gen%02d.pt" % gen)
        cmd = [sys.executable, os.path.join(BASE_DIR, "train.py"),
               "--out", cand, "--epochs", str(epochs)]
        if strategy == STRATEGY_RETRAIN:
            # 基座重训：和「训练」按钮同一套配方，只是不覆盖 best.pt。
            # 慢（几分钟），但这是历史上唯一被验证能把指标推上去的路。
            cmd += ["--patience", str(RETRAIN_PATIENCE)]
            _log("开始训练候选权重（基座重训 %d 轮）…" % epochs)
        else:
            # 快速微调：从当前 best.pt 续训，小学习率、关增强
            cmd += ["--weights", BEST_PT, "--ft"]
            _log("开始训练候选权重（快速微调 %d 轮）…" % epochs)
        rc = _run(cmd, LOG_PATH)
        if rc != 0 or not os.path.exists(cand):
            raise RuntimeError("训练进程退出码 %s，没有产出候选权重" % rc)
        _log("候选权重已生成：%s" % cand)

        # --- 3. 在同一个验证集上打分 ---
        cfg["stage"] = "eval"
        save_cfg(cfg)
        write_lock(stage="eval")
        if os.path.exists(EVAL_JSON):
            os.remove(EVAL_JSON)
        cmd = [sys.executable, os.path.join(BASE_DIR, "tools", "eval_best.py"),
               "--model", cand, "--out", EVAL_JSON]
        _log("在验证集上评估候选权重…")
        rc = _run(cmd, LOG_PATH)
        new_metrics = _read_json(EVAL_JSON, None)
        if rc != 0 or not new_metrics:
            raise RuntimeError("评估失败，没有拿到候选指标")

        # --- 4. 闸门 ---
        cfg["stage"] = "gate"
        save_cfg(cfg)
        accepted, reason = decide_accept(old_metrics, new_metrics)
        _log("闸门判定：%s —— %s" % ("采纳" if accepted else "不采纳", reason))

        # --- 5. 采纳或留档 ---
        if accepted:
            old_backup = backup_existing_best(BEST_PT)
            shutil.copy2(cand, BEST_PT)
            if old_backup:
                _log("旧权重已备份：%s" % os.path.basename(old_backup))
            _log("已启用新权重：model/best.pt ← gen%02d" % gen)
            # 新基线写回，下一代跟它比
            with open(BEST_METRICS, "w", encoding="utf-8") as f:
                json.dump(new_metrics, f, ensure_ascii=False, indent=2)
        else:
            reject_dir = os.path.join(MODEL_DIR, "_backup")
            os.makedirs(reject_dir, exist_ok=True)
            reject = os.path.join(
                reject_dir, "rejected_gen%02d_%s.pt"
                % (gen, time.strftime("%Y%m%d_%H%M%S")))
            shutil.move(cand, reject)
            _log("候选已留档作废：%s" % os.path.basename(reject))

        # --- 6. 记账 ---
        cfg = load_cfg()
        cfg["round"] = gen
        cfg["learned_edited"] = started_edited
        cfg["last_run_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        cfg["last_result"] = {
            "gen": gen,
            "accepted": bool(accepted),
            "reason": reason,
            "old": {"box_map50": old_metrics.get("box_map50"),
                    "mask_map50": old_metrics.get("mask_map50")},
            "new": {"box_map50": new_metrics.get("box_map50"),
                    "mask_map50": new_metrics.get("mask_map50")},
            "at": cfg["last_run_at"],
        }
        cfg.setdefault("history", []).append({
            "gen": gen, "at": cfg["last_run_at"], "accepted": bool(accepted),
            "reason": reason,
            "box_map50": new_metrics.get("box_map50"),
            "mask_map50": new_metrics.get("mask_map50"),
        })
        cfg["history"] = cfg["history"][-20:]
        cfg["running"] = False
        cfg["stage"] = "idle"
        save_cfg(cfg)

        _append_csv([
            gen, cfg["last_run_at"], added, epochs,
            "基座重训" if strategy == STRATEGY_RETRAIN else "续训",
            stats.get("total_images", ""),
            old_metrics.get("box_map50", ""), new_metrics.get("box_map50", ""),
            old_metrics.get("mask_map50", ""), new_metrics.get("mask_map50", ""),
            "采纳" if accepted else "不采纳", reason,
        ])
        _log("========== 第 %d 代结束：%s" % (gen, "已采纳" if accepted else "未采纳"))
        return {"ok": True, "accepted": bool(accepted), "reason": reason,
                "gen": gen, "new": new_metrics, "old": old_metrics}

    except Exception as exc:  # noqa: BLE001
        _log("[失败] 第 %d 代中断：%s" % (gen, exc))
        cfg = load_cfg()
        cfg["running"] = False
        cfg["stage"] = "idle"
        cfg["last_result"] = {"gen": gen, "accepted": False,
                              "reason": "中断：%s" % exc,
                              "at": time.strftime("%Y-%m-%d %H:%M:%S")}
        save_cfg(cfg)
        return {"ok": False, "error": str(exc)}
    finally:
        clear_lock()


# ---------------------------------------------------------------------------
# 巡检：后台线程每隔一会儿问一次「该学了吗」
# ---------------------------------------------------------------------------

def tick(busy=False):
    """该学就学一代，否则什么也不做。busy = 训练/预标注正在占显卡。"""
    cfg = load_cfg()
    if read_lock() or cfg.get("running"):
        return None
    if busy:
        return None
    edited_total = scan_edited()
    if not should_learn(cfg, edited_total):
        return None
    _log("巡检触发：累积修正 %d 条 ≥ 阈值 %d"
         % (pending(cfg, edited_total), cfg.get("threshold")))
    return run_round(edited_total)


def status(busy=False):
    """给前端的一份快照（不含 history 全量，只给最近几代）。"""
    cfg = load_cfg()
    edited_total = scan_edited()
    return {
        "enabled": bool(cfg.get("enabled")),
        "threshold": int(cfg.get("threshold", DEFAULT_THRESHOLD)),
        "round": int(cfg.get("round", 0)),
        "edited_total": edited_total,
        "learned_edited": int(cfg.get("learned_edited", 0)),
        "pending": pending(cfg, edited_total),
        "running": bool(cfg.get("running")) or bool(read_lock()),
        "stage": cfg.get("stage", "idle"),
        "last_run_at": cfg.get("last_run_at"),
        "last_result": cfg.get("last_result"),
        "history": cfg.get("history", [])[-5:],
        "busy": bool(busy),
        "strategy": cfg.get("strategy", DEFAULT_STRATEGY),
        "epochs": RETRAIN_EPOCHS if cfg.get("strategy") == STRATEGY_RETRAIN else FT_EPOCHS,
        "tolerance": TOLERANCE,
    }
