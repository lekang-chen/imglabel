@echo off
chcp 65001 >nul
title 修复 GitHub 访问
REM 双击（或右键以管理员身份运行）本文件即可。
REM 实际逻辑在 fix_github_hosts.ps1，这里只负责提权与转达。

net session >nul 2>&1
if errorlevel 1 (
    echo 需要管理员权限，正在请求提权...
    powershell -NoProfile -Command "Start-Process -FilePath '%~f0' -Verb RunAs"
    exit /b
)

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0fix_github_hosts.ps1"
echo.
pause
