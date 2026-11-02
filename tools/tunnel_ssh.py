#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tools/tunnel_ssh.py —— 备用外网通道：SSH 反向隧道（pinggy.io）

【为什么需要备用】
cloudflared 的 quick tunnel 要连边缘节点的 7844 端口。校园网（以及部分运营商）
把这个端口封了，症状很好认：地址能正常打印出来，但一访问就是 530，日志里写

    ERR Unable to establish connection with Cloudflare edge
    error="dial tcp 198.41.200.43:7844: i/o timeout"

2026-11-02 在本机实测：198.41.200.43 / 198.41.192.167 / 198.41.192.7 /
198.41.200.13 四个边缘 IP 的 7844 全部超时。cloudflared 自己的 precheck 反而
显示 region1/region2「PASS」，容易误导——那一步走的不是同一条路径。
结论：这个网络下 cloudflared 走不通，换通道。

【这条通道怎么绕】
SSH 反向隧道只需要一个出站 443。远端服务器不直接连你，而是把域名上的请求
顺着「你已经连出去的那条 TCP 连接」推回来，本机再转给 127.0.0.1:5000。
出站 443 几乎不会被封，所以这条通道在校园网里最稳。

【代价，必须知道】
- 免费版单条隧道 60 分钟，到点自己断；重跑就换一个新地址。
- 地址随机、不可枚举，但拿到地址的人能改标注、删图、起训练。别外传。
- 演示到一半断了，重跑本脚本即可，只是二维码要重新扫。

【用法】
    python tools/tunnel_ssh.py               # 服务没跑就顺手起一个
    python tools/tunnel_ssh.py --no-server   # 只开隧道，复用已启动的服务
    python tools/tunnel_ssh.py --port 5001   # 指定端口
"""

import argparse
import os
import re
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PNG_PATH = os.path.join(BASE_DIR, "手机外网二维码.png")
URL_PATH = os.path.join(BASE_DIR, "data", "phone_url.txt")
KNOWN_HOSTS = os.path.join(BASE_DIR, "data", "_ssh_known_hosts")

# pinggy 免费版会给两个域名，优先用 run.pinggy-free.link（走 443，最稳）
_URL_RE = re.compile(
    r"https://[a-z0-9][a-z0-9.-]*\.(?:run\.pinggy-free\.link|free\.pinggy\.net|pinggy\.link)"
)


def say(msg=""):
    print(msg, flush=True)


def port_open(port, host="127.0.0.1"):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(0.4)
    try:
        return s.connect_ex((host, port)) == 0
    finally:
        s.close()


def http_ok(port, path="/api/classes", timeout=2.0):
    try:
        with urllib.request.urlopen(
            "http://127.0.0.1:%d%s" % (port, path), timeout=timeout
        ) as r:
            return r.status == 200
    except Exception:
        return False


def show_qr(url):
    """把地址画成二维码存成 PNG，手机直接扫，省得手打。"""
    try:
        import qrcode
    except ImportError:
        say("  （没装 qrcode，跳过二维码：pip install qrcode）")
        return False
    try:
        qrcode.make(url).save(PNG_PATH)
        say("  · 二维码已另存：%s" % PNG_PATH)
        return True
    except Exception as e:
        say("  （二维码生成失败：%s）" % e)
        return False


def start_server(port):
    """服务没跑时顺手起一个（--lan，手机和电脑同 WiFi 也能用）。"""
    say("[1/3] 启动本地服务（未检测到 :%d 在跑）…" % port)
    proc = subprocess.Popen(
        [sys.executable, os.path.join(BASE_DIR, "app.py"), "--lan", "--port", str(port)],
        cwd=BASE_DIR,
    )
    for _ in range(60):
        if http_ok(port):
            say("      服务就绪：http://127.0.0.1:%d" % port)
            return proc
        time.sleep(0.5)
    return proc


def main(argv=None):
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--no-server", action="store_true")
    args = ap.parse_args(argv)

    say()
    say("人机协同图像标注系统 · 外网访问（SSH 备用通道）")
    say("=" * 62)

    server = None
    if args.no_server or port_open(args.port):
        say("[1/3] 复用已在运行的服务 :%d" % args.port)
        if not http_ok(args.port):
            say("      [警告] 端口在，但 /api/classes 没回应，确认那个进程是本系统")
    else:
        server = start_server(args.port)

    os.makedirs(os.path.dirname(KNOWN_HOSTS), exist_ok=True)
    cmd = [
        "ssh",
        "-p", "443",
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=%s" % KNOWN_HOSTS,
        "-o", "ServerAliveInterval=20",
        "-o", "ServerAliveCountMax=3",
        "-o", "ExitOnForwardFailure=yes",
        "-R0:localhost:%d" % args.port,
        "a.pinggy.io",
    ]

    say("[2/3] 连接 pinggy.io（走 443，首次约 5~10 秒）…")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    holder = {"url": None}
    done = threading.Event()
    lines = []

    def pump():
        for line in proc.stdout:
            line = line.rstrip()
            lines.append(line)
            m = _URL_RE.search(line)
            if m and not holder["url"]:
                holder["url"] = m.group(0)
                done.set()
        done.set()

    threading.Thread(target=pump, daemon=True).start()

    deadline = time.time() + 40
    while time.time() < deadline and not done.is_set():
        time.sleep(0.3)

    url = holder["url"]
    if not url:
        say("      [失败] 没拿到地址，下面是 ssh 的原始输出：")
        for line in lines[-12:]:
            say("      " + line)
        say("      常见原因：本机 ssh 不可用（Windows 设置里装 OpenSSH 客户端）、")
        say("      或者 443 出站也被拦（换手机热点再试）。")
        if proc.poll() is None:
            proc.terminate()
        if server:
            server.terminate()
        return 1

    try:
        with open(URL_PATH, "w", encoding="utf-8") as f:
            f.write(url + "\n")
    except OSError:
        pass

    say("[3/3] 隧道已建立")
    say()
    say("=" * 62)
    say("  外网地址（手机用流量 / 任何网络都能打开）")
    say()
    say("    %s" % url)
    say()
    say("=" * 62)
    show_qr(url)
    say()
    say("  · 本机仍然用 http://127.0.0.1:%d" % args.port)
    say("  · 免费版 60 分钟后自动断，断了重跑本脚本（地址会变）")
    say("  · 拿到地址的人能改标注、删图，别外传")
    say("  · 地址已记录：%s" % URL_PATH)
    say()
    say("隧道运行中，按 Ctrl+C 停止。")
    say()

    try:
        proc.wait()
    except KeyboardInterrupt:
        say()
        say("已停止。")
    finally:
        for p in (proc, server):
            if p and p.poll() is None:
                try:
                    p.terminate()
                except Exception:
                    pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
