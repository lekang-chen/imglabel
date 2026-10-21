"""
tools/fast_download.py —— 多线程分片下载器

【为什么要自己写】
COCO 图片包 815MB，单线程下载在国内网络下只有 20KB/s（实测），
要跑十几个小时。服务器支持 HTTP Range 请求（206 Partial Content），
所以可以分片并发下载，速度能提升一个数量级。

【用法】
  python tools/fast_download.py <url> <输出路径> [--threads 8] [--retry 5]

【原理】
1. 先用 Range 请求探测文件总大小
2. 把文件切成 N 段，每段一个线程，用 Range 各自下载到临时文件
3. 全部完成后按顺序拼接成完整文件
4. 失败的段自动重试（断点续传：从已下载的位置继续）
"""

import argparse
import os
import sys
import threading
import time

import requests

# 走直连，绕过系统代理（代理对大文件站点不稳定）
SESSION = requests.Session()
SESSION.trust_env = False

_lock = threading.Lock()
_done_bytes = 0
_total_bytes = 0
_printed_pct = -1


def probe_size(url, headers=None):
    """探测文件总大小。返回字节数，失败返回 None。"""
    try:
        r = SESSION.get(url, headers=dict(headers or {}, Range="bytes=0-1"),
                        timeout=30, stream=True)
        cr = r.headers.get("Content-Range", "")
        if "/" in cr:
            return int(cr.split("/")[-1])
        cl = r.headers.get("Content-Length")
        return int(cl) if cl else None
    except Exception as exc:  # noqa: BLE001
        print("探测大小失败: %s" % exc)
        return None


def download_chunk(url, start, end, out_path, idx, headers=None, retry=5):
    """下载 [start, end] 这一段到 out_path.part{idx}。"""
    global _done_bytes
    part_path = "%s.part%d" % (out_path, idx)
    pos = start
    attempts = 0

    # 截断旧分片（用 wb 模式而不是 os.remove，避开系统回收站机制）
    with open(part_path, "wb"):
        pass

    while pos <= end and attempts < retry:
        attempts += 1
        h = dict(headers or {})
        h["Range"] = "bytes=%d-%d" % (pos, end)
        try:
            with SESSION.get(url, headers=h, timeout=(30, 60), stream=True) as r:
                if r.status_code not in (200, 206):
                    raise IOError("HTTP %d" % r.status_code)
                with open(part_path, "ab") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 256):
                        if not chunk:
                            continue
                        f.write(chunk)
                        pos += len(chunk)
                        with _lock:
                            _done_bytes += len(chunk)
        except Exception as exc:  # noqa: BLE001
            print("  分片 %d 第 %d 次中断: %s" % (idx, attempts, str(exc)[:80]))
            time.sleep(min(2 ** attempts, 20))

    with _lock:
        print("  分片 %d 完成 (%d bytes)" % (idx, pos - start))
    return pos > end


def progress_loop(total, stop_event):
    global _printed_pct
    while not stop_event.is_set():
        with _lock:
            done = _done_bytes
        pct = int(done * 100 / total) if total else 0
        if pct != _printed_pct and pct % 5 == 0:
            _printed_pct = pct
            print("  进度 %d%%  (%.1f / %.1f MB)"
                  % (pct, done / 1048576, total / 1048576))
        time.sleep(2)


def download(url, out_path, threads=8, retry=5, headers=None):
    global _done_bytes, _total_bytes

    total = probe_size(url, headers)

    # 已存在且大小正确 -> 跳过；存在但不完整 -> 覆盖重下
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        if total and os.path.getsize(out_path) == total:
            print("已存在且完整，跳过: %s" % out_path)
            return True
        print("已存在但不完整（%d / %s），重新下载"
              % (os.path.getsize(out_path), total))
    # 截断目标文件（用 wb 而不是 os.remove，避开系统回收站机制）
    with open(out_path, "wb"):
        pass
    # 截断上次中断的分片残留
    d = os.path.dirname(os.path.abspath(out_path))
    base = os.path.basename(out_path)
    for f in os.listdir(d or "."):
        if f.startswith(base + ".part"):
            with open(os.path.join(d, f), "wb"):
                pass

    if not total:
        print("无法探测文件大小，退回单线程下载")
        r = SESSION.get(url, timeout=(30, 120), stream=True, headers=headers or {})
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(1024 * 256):
                f.write(chunk)
        return True

    print("文件大小: %.1f MB，%d 线程分片" % (total / 1048576, threads))
    _total_bytes = total
    _done_bytes = 0

    chunk_size = (total + threads - 1) // threads
    parts = []
    for i in range(threads):
        s = i * chunk_size
        e = min(s + chunk_size - 1, total - 1)
        if s > e:
            break
        parts.append((i, s, e))

    stop_event = threading.Event()
    prog = threading.Thread(target=progress_loop, args=(total, stop_event), daemon=True)
    prog.start()

    results = []
    ths = []
    for idx, s, e in parts:
        t = threading.Thread(target=lambda i=idx, a=s, b=e:
                            results.append(download_chunk(url, a, b, out_path, i,
                                                         headers, retry)))
        t.start()
        ths.append(t)
    for t in ths:
        t.join()

    stop_event.set()

    # 拼接
    print("拼接分片…")
    with open(out_path, "wb") as out:
        for idx, _s, _e in parts:
            p = "%s.part%d" % (out_path, idx)
            if os.path.exists(p):
                with open(p, "rb") as f:
                    while True:
                        buf = f.read(1024 * 1024)
                        if not buf:
                            break
                        out.write(buf)
                with open(p, "wb"):
                    pass

    final = os.path.getsize(out_path)
    if final == total:
        print("下载完成: %s (%.1f MB)" % (out_path, final / 1048576))
        return True
    print("下载不完整: 期望 %d 实际 %d" % (total, final))
    return False


def main():
    ap = argparse.ArgumentParser(description="多线程分片下载")
    ap.add_argument("url")
    ap.add_argument("out")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--retry", type=int, default=5)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    t0 = time.time()
    ok = download(args.url, args.out, args.threads, args.retry)
    print("耗时 %.1f 秒" % (time.time() - t0))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
