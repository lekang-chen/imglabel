@echo off
chcp 65001 >nul
title Image Labeling System - Public Access (SSH fallback, pinggy.io)
cd /d "%~dp0"

REM 【什么时候用这个】
REM cloudflared 那条路被校园网封了 7844 端口时用：地址能打印出来，
REM 但手机一打开就是 530，日志写 "dial tcp ...:7844: i/o timeout"。
REM 本脚本改用 ssh 反向隧道（出站 443），在校园网里更稳。
REM
REM 代价：免费版 60 分钟断一次，地址每次都不一样。演示够用。
REM 不要外传地址：拿到的人能改标注、删图。

set "PYEXE=.venv\Scripts\python.exe"
if not exist "%PYEXE%" set "PYEXE=python"

"%PYEXE%" tools\tunnel_ssh.py %*

if errorlevel 1 (
    echo.
    echo Tunnel exited. See messages above.
    pause
)
