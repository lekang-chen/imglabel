#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tools/tunnel.py —— 用 cloudflared 临时隧道，让手机用流量也能打开界面

【解决什么】
手机在外面用 4G/5G，访问不了 172.24.190.99 这种内网地址：那是私有地址
（RFC1918），公网上没有任何路由能把它送到你家里那台电脑。所以「同一 Wi-Fi」
不是配置问题，是网络拓扑决定的。

内网穿透反过来做：让本机的服务主动连出去，由 Cloudflare 给一个公网域名，
手机访问那个域名，流量经 Cloudflare 转回这台电脑。手机在哪个网络都行。

【为什么是 quick tunnel】
不用注册账号、不用买域名、不用写配置，一条命令给 URL。
代价：URL 每次重启都变；只在这个窗口活着的时候有效。

【用法】
    python tools/tunnel.py              # 服务没跑就顺手起一个
    python tools/tunnel.py --port 5001  # 指定端口
    python tools/tunnel.py --no-server  # 只开隧道，复用已启动的服务

【安全边界】
quick tunnel 的域名是随机的、不可枚举，但**拿到 URL 的人都能完全操作**这个
系统（改标注、删图、起训练）。演示完按 Ctrl+C 或关掉窗口，链路立刻断。
不要把 URL 发到群里。
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
CLOUDFLARED = os.path.join(BASE_DIR, "tools", "cloudflared.exe")

# cloudflared 打印的临时域名，形如 https://three-words-here.trycloudflare.com
_URL_RE = re.compile(r"https://[a-z0-9][a-z0-9-]*\.trycloudflare\.com")


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
    """服务真的能应答，而不只是端口连着（Flask 起来到能响应之间有个空档）。"""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open("http://127.0.0.1:%d%s" % (port, path), timeout=timeout) as r:
            return r.status == 200
    except (urllib.error.URLError, OSError, ValueError):
        return False


def pick_free_port(prefer, tries=20):
    """从 prefer 往后找一个确实没被占的端口，再交给 app.py。
    app.py 自己有 pick_port，但它的结果只打印在启动横幅里，管道里拿不到；
    这里先探一遍，两边就对齐了。"""
    for p in range(prefer, prefer + tries):
        if not port_open(p):
            return p
    return prefer


def find_cloudflared():
    for p in (CLOUDFLARED, os.path.join(BASE_DIR, "cloudflared.exe")):
        if os.path.isfile(p):
            return p
    # 最后试 PATH
    from shutil import which
    return which("cloudflared")


def start_server(port):
    """起一个只监听本机的 Flask。不开 --lan：隧道从 127.0.0.1 转发进来，
    局域网里其他设备仍然访问不到，暴露面只有 Cloudflare 那一条。"""
    say("[1/3] 启动标注服务（仅本机 127.0.0.1:%d）…" % port)
    py = sys.executable
    proc = subprocess.Popen(
        [py, "app.py", "--port", str(port)],
        cwd=BASE_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    def pump():
        for line in proc.stdout:
            line = line.rstrip()
            # 只转发有意义的行，别把 werkzeug 的每条访问日志都刷上来
            if line and ("Running on" in line or "Error" in line or "Traceback" in line
                         or "Address already" in line):
                say("      " + line)

    threading.Thread(target=pump, daemon=True).start()
    return proc


def wait_ready(port, proc=None, timeout=45):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if http_ok(port):
            return True
        if proc is not None and proc.poll() is not None:
            return False
        time.sleep(0.6)
    return False


def start_tunnel(port, url_holder, err_holder):
    say("[2/3] 连接 Cloudflare（首次约 3~10 秒）…")
    cmd = [find_cloudflared(), "tunnel", "--url", "http://127.0.0.1:%d" % port,
           "--no-autoupdate"]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    def pump():
        for line in proc.stderr:
            line = line.rstrip()
            m = _URL_RE.search(line)
            if m and not url_holder.get("url"):
                url_holder["url"] = m.group(0)
                url_holder["evt"].set()
            if "ERR" in line or "error" in line.lower():
                err_holder.append(line)

    threading.Thread(target=pump, daemon=True).start()
    return proc


def show_qr(url):
    try:
        import qrcode
    except ImportError:
        return False
    qr = qrcode.QRCode(border=1)
    qr.add_data(url)
    qr.make()
    qr.print_ascii(invert=True)
    return True


def save_qr_png(url):
    """终端里的半块字符在部分字体下会花，另存一张 PNG：
    手机微信/相册都能直接长按识别，不用手输 40 位域名。"""
    try:
        import qrcode
    except ImportError:
        return None
    path = os.path.join(BASE_DIR, "外网二维码.png")
    try:
        qr = qrcode.QRCode(border=4, box_size=10)
        qr.add_data(url)
        qr.make()
        qr.make_image().save(path)
        return path
    except Exception:
        return None


def save_url(url):
    """地址每次重启都变，写一份到 data/ 里，万一窗口被关掉还能找回刚才那个。"""
    path = os.path.join(BASE_DIR, "data", "tunnel_url.txt")
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(url + "\n")
        return path
    except OSError:
        return None


def banner(url, port):
    line = "=" * 62
    png = save_qr_png(url)
    txt = save_url(url)
    say()
    say(line)
    say("  外网地址（手机用流量 / 任何网络都能打开）")
    say()
    say("    " + url)
    say()
    say(line)
    say("  手机上：相机扫下面的码，或在浏览器里输入上面的地址")
    say()
    if not show_qr(url):
        say("  （未安装 qrcode，无法显示二维码，手输地址即可）")
        say()
    say(line)
    say("  · 本机仍然用 http://127.0.0.1:%d" % port)
    say("  · 关掉这个窗口 = 链接立刻失效（地址下次会变）")
    say("  · 拿到地址的人能改标注、删图，别外传")
    if png:
        say("  · 二维码已另存：%s" % png)
    if txt:
        say("  · 地址已记录：%s" % txt)
    say(line)
    say()


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--no-server", action="store_true",
                    help="不起服务，复用一个已经在跑的")
    ap.add_argument("--timeout", type=int, default=90, help="等域名的最长秒数")
    a = ap.parse_args()

    exe = find_cloudflared()
    if not exe:
        say("找不到 cloudflared.exe。")
        say("它应该放在 tools/cloudflared.exe，或已加入 PATH。")
        say("下载地址（选 windows-amd64）：")
        say("  https://github.com/cloudflare/cloudflared/releases/latest")
        return 2

    say()
    say("人机协同图像标注系统 · 外网访问")
    say("=" * 62)

    server = None
    port = a.port
    if a.no_server:
        if not http_ok(port):
            say("本机 %d 端口上没有能应答的服务。" % port)
            say("先双击「启动.bat」，或去掉 --no-server 让本脚本自己起。")
            return 3
        say("[1/3] 复用已在运行的服务 :%d" % port)
    elif http_ok(port):
        say("[1/3] 检测到 %d 端口已有服务在跑，直接复用" % port)
    else:
        port = pick_free_port(port)
        server = start_server(port)
        if not wait_ready(port, server):
            say()
            say("服务没能起来，看上面的报错。")
            if server and server.poll() is None:
                server.terminate()
            return 4
        say("      服务就绪")

    url_holder = {"url": None, "evt": threading.Event()}
    errors = []
    tun = start_tunnel(port, url_holder, errors)

    ok = url_holder["evt"].wait(a.timeout)
    if not ok:
        say()
        say("等了 %d 秒没拿到公网地址。常见原因：" % a.timeout)
        say("  · 电脑本身没联网（隧道要连 cloudflare 的服务器）")
        say("  · 公司/学校网络拦截了 7844 端口的外连")
        say("  · 代理软件把 cloudflared 的流量拦了")
        for e in errors[-4:]:
            say("  ! " + e)
        try:
            tun.terminate()
        except OSError:
            pass
        if server:
            server.terminate()
        return 5

    url = url_holder["url"]
    banner(url, port)

    say("隧道运行中，按 Ctrl+C 停止。")
    say()
    try:
        while True:
            if tun.poll() is not None:
                say("隧道进程已退出（code %s）。" % tun.returncode)
                break
            time.sleep(1)
    except KeyboardInterrupt:
        say()
        say("正在关闭…")
    finally:
        for p in (tun, server):
            if p is not None and p.poll() is None:
                try:
                    p.terminate()
                except OSError:
                    pass
        say("已停止。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
