# -*- coding: utf-8 -*-
"""
在桌面创建启动器（.bat），并保证项目里的 start.bat 编码正确。

为什么是 .bat 而不是 .lnk：
  - .lnk 的二进制格式要手写（MS-SHLLINK 规范），容易在偏移量、
    时间戳字节数、ANSI/Unicode 编码上出错；
    本机环境又禁用了 COM（WScript.Shell / Shell.Application），
    用不了最省事的现成方案。
  - .bat 是纯文本，资源管理器双击就会自动用 cmd 执行，行为最可预期。

为什么必须是 GBK 编码 + CRLF：
  cmd.exe 按系统 ANSI 代码页(GBK)解析 .bat。
  存成 UTF-8 的话中文注释会被截断，
  表现是 '@echo off' 变成 'ho'、URL 变成 '.cn'，脚本直接崩。
  这个坑实测踩过。

用法：
  python tools/make_shortcut.py
  python tools/make_shortcut.py --name 我的名字
"""
import argparse
import os
import sys

try:
    import winreg
except ImportError:
    print("这个脚本只在 Windows 上用")
    sys.exit(1)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BAT = os.path.join(BASE_DIR, "start.bat")
BAT_CN = os.path.join(BASE_DIR, "启动.bat")

# start.bat 的内容：全英文 + 纯 ASCII 字符，
# 这样不论 cmd 用什么代码页解析都不会出错。
BAT_CONTENT = r"""@echo off
chcp 65001 >nul
title Human-Machine Collaborative Image Labeling System
cd /d "%~dp0"

echo ============================================================
echo   Human-Machine Collaborative Image Labeling System
echo ============================================================
echo.

REM ---------- pick interpreter ----------
REM Prefer the project .venv: it has the CUDA build of torch,
REM which is required to train on GPU. Fall back to system python.
set "PY="
if exist ".venv\Scripts\python.exe" (
    set "PY=.venv\Scripts\python.exe"
) else (
    where python >nul 2>nul
    if not errorlevel 1 set "PY=python"
)

if "%PY%"=="" (
    echo [ERROR] Python not found.
    echo.
    echo Create the environment first:
    echo   python -m venv .venv
    echo   .venv\Scripts\activate
    echo   pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

echo [ENV] Interpreter: %PY%

REM ---------- check dependencies ----------
"%PY%" -c "import ultralytics, flask, psutil, cv2, numpy" >nul 2>nul
if errorlevel 1 (
    echo [INFO] Missing dependencies, installing...
    "%PY%" -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
    if errorlevel 1 (
        echo [ERROR] Install failed. Try the Aliyun mirror:
        echo   pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
        pause
        exit /b 1
    )
)

REM ---------- check image folder ----------
if not exist "data\images" (
    mkdir "data\images"
)
dir /b "data\images" >nul 2>nul
if errorlevel 1 (
    echo [INFO] data\images is empty.
    echo        Put the selected indoor images into this folder.
    echo.
)

echo [START] Opening http://127.0.0.1:5000 ...
echo [TIP] Close this window to stop the server
echo.
"%PY%" app.py

if errorlevel 1 (
    echo.
    echo [ERROR] Server exited unexpectedly. See the log above.
    pause
)
"""


def desktop_dir():
    """拿真实桌面路径，兼容 OneDrive 重定向。"""
    try:
        key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"SOFTWARE\Microsoft\Windows\CurrentVersion\Explorer\User Shell Folders")
        val, _ = winreg.QueryValueEx(key, "Desktop")
        return os.path.expandvars(val)
    except OSError:
        return os.path.join(os.path.expanduser("~"), "Desktop")


def write_bat(path, content):
    """按 GBK + CRLF 写 .bat，保证 cmd 能正确解析。"""
    data = content.replace("\n", "\r\n").encode("gbk")
    with open(path, "wb") as f:
        f.write(data)
    return len(data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="人机协同图像标注系统")
    args = ap.parse_args()

    # 1) 生成项目里的 start.bat（GBK 编码）
    n = write_bat(BAT, BAT_CONTENT)
    print("✓ start.bat 已生成 (GBK/CRLF, %d 字节)" % n)
    # 同步一份中文名，评委按 README 找得到
    write_bat(BAT_CN, BAT_CONTENT)
    print("✓ 启动.bat 已同步")

    # 2) 桌面启动器
    desk = desktop_dir()
    os.makedirs(desk, exist_ok=True)
    out_path = os.path.join(desk, args.name + ".bat")

    # 内容全英文：cmd 解析 100% 可靠，不受代码页影响
    content = '@echo off\r\nstart "" "%s"\r\n' % BAT
    with open(out_path, "wb") as f:
        f.write(content.encode("gbk"))

    print("\n✓ 桌面启动器已创建")
    print("  位置    : %s" % out_path)
    print("  指向    : %s" % BAT)
    print("  编码    : GBK / CRLF")
    print("\n双击即可启动，会自动打开 http://127.0.0.1:5000")
    print("（如果桌面看不到图标，把这个 .bat 右键「发送到 → 桌面快捷方式」）")


if __name__ == "__main__":
    main()
