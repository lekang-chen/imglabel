"""解析 train.py 的输出日志，算出训练进度。

为什么单独成一个文件：
- ultralytics 的终端输出不是 API，格式随时可能变。解析规则集中在这一处，
  将来它换了排版，只改这里，前端和后端都不用动。
- 纯函数好测：喂一段字符串就能验，不用真跑一次训练（那是 4~5 分钟 + 覆盖权重）。

日志里能用的锚点（实测 train.log）：
    [2/4] 自动选择的训练配置：
            轮数      : 100
    Starting training for 100 epochs...
         14/100      2.22G      1.305      3.525  ...  640: 25% ━━━ 4/16 5.0it/s
                      Class     Images  Instances   Box(P  R  mAP50  mAP50-95)  Mask(...)
                        all         31        166   0.498 0.488 0.486 0.347 ...
    [4/4] 训练完成，耗时 4.6 分钟
    [错误] 训练失败: ...
"""

import re

# 轮次行：行首 "14/100" 后面跟显存（2.22G）——验证时的 "1/2 2.2it/s" 不会误匹配
_EPOCH_RE = re.compile(r"^\s*(\d{1,4})/(\d{1,4})\s+\d+(?:\.\d+)?G")
# 同一行里的 batch 进度："640: 25% ━━━"
_BATCH_RE = re.compile(r"640:\s*(\d{1,3})%")
_START_RE = re.compile(r"Starting training for (\d+) epochs")
_CFG_RE = re.compile(r"轮数\s*:\s*(\d{1,4})")
_DONE_RE = re.compile(r"\[4/4\]\s*训练完成，耗时\s*([\d.]+)\s*分钟")
_ERR_RE = re.compile(r"\[错误\]\s*(.+)")
# 验证汇总行：all + images + instances + 8 个数字
# box P/R/mAP50/mAP50-95 在前四个，mask 的在后四个（没有掩码任务时只有前四个）
# 注意用 [ \t] 而不是 \s：\s 会吃换行，把下一行的数字也吞进来
_ALL_RE = re.compile(r"^[ \t]*all[ \t]+(\d+)[ \t]+(\d+)((?:[ \t]+[\d.]+){4,10})",
                     re.M)
_NUMS_RE = re.compile(r"[\d.]+")
_EARLY_RE = re.compile(r"EarlyStopping|early stopping|Stopping training early", re.I)

# ultralytics 的进度条会往文件里写 ANSI 控制码（如行首的 \x1b[K 清行）。
# 不清掉的话，行首是转义序列而不是空格，"^\s*14/100" 这类锚点全都对不上。
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

# 正常结束、可以安全停轮询的状态
TERMINAL = ("done", "failed")


def parse_train_log(text):
    """把训练日志解析成一份进度快照。

    返回字段：
        state        idle | starting | running | done | failed
        epoch        当前轮次（done 时等于实际跑完的轮数）
        total_epochs 计划轮数
        batch_percent 当前轮内部进度 0~100
        percent      总进度 0~100（含轮内小数，所以比"第几轮/总轮数"更连续）
        minutes      完成时的耗时（分钟）
        error        失败原因
        box_map50 / mask_map50  最后一次验证的指标
        early_stopped 是否触发了早停
    """
    out = {
        "state": "idle", "epoch": 0, "total_epochs": 0, "batch_percent": 0,
        "percent": 0.0, "minutes": None, "error": "",
        "box_map50": None, "mask_map50": None, "early_stopped": False,
    }
    if not text:
        return out

    # ultralytics 的进度条用 \r 原地刷新，重定向到文件后两种换行混在一起；
    # 顺带剥掉 ANSI 控制码，否则行首锚点会被转义序列挡住
    text = _ANSI_RE.sub("", text.replace("\r\n", "\n").replace("\r", "\n"))

    # 1) 计划轮数：优先取 ultralytics 自己那行，退回我们打印的配置
    m = _START_RE.search(text)
    if m:
        out["total_epochs"] = int(m.group(1))
    else:
        m = _CFG_RE.search(text)
        if m:
            out["total_epochs"] = int(m.group(1))

    # 2) 已跑到第几轮、轮内到百分之几
    for line in text.split("\n"):
        m = _EPOCH_RE.match(line)
        if not m:
            continue
        out["epoch"] = int(m.group(1))
        # 分子分母以行内为准（早停或 --epochs 变更时更准）
        if int(m.group(2)):
            out["total_epochs"] = int(m.group(2))
        b = _BATCH_RE.search(line)
        out["batch_percent"] = int(b.group(1)) if b else 0

    # 3) 指标：取最后一条 all 汇总行
    for m in _ALL_RE.finditer(text):
        nums = _NUMS_RE.findall(m.group(3))
        if len(nums) >= 8:
            out["box_map50"] = float(nums[2])
            out["mask_map50"] = float(nums[6])

    # 4) 终态
    m = _DONE_RE.search(text)
    if m:
        out["state"] = "done"
        out["minutes"] = float(m.group(1))
    else:
        m = _ERR_RE.search(text)
        if m:
            out["state"] = "failed"
            out["error"] = m.group(1).strip()
        elif out["epoch"] > 0:
            out["state"] = "running"
        elif _START_RE.search(text):
            out["state"] = "starting"

    # 5) 总进度：把轮内进度摊进当前轮，进度条才不会一轮一跳
    if out["state"] == "done":
        out["percent"] = 100.0
    elif out["total_epochs"] > 0 and out["epoch"] > 0:
        done = (out["epoch"] - 1) + out["batch_percent"] / 100.0
        out["percent"] = round(min(done / out["total_epochs"] * 100.0, 99.9), 1)

    out["early_stopped"] = bool(_EARLY_RE.search(text))
    return out


def human_duration(seconds):
    """秒 → 「3 分 12 秒」。进度条上不要出现 192.4s 这种。"""
    if seconds is None:
        return ""
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return "%d 小时 %d 分" % (h, m)
    if m:
        return "%d 分 %02d 秒" % (m, s)
    return "%d 秒" % s


if __name__ == "__main__":
    # 手工验一下：python train_progress.py data/train.log
    import io
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "data/train.log"
    with io.open(path, encoding="utf-8", errors="replace") as f:
        snap = parse_train_log(f.read())
    for k, v in snap.items():
        print("%-14s %s" % (k, v))
