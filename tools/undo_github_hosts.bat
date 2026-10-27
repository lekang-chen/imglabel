@echo off
chcp 65001 >nul
title 撤销 GitHub hosts 修改
REM 删除 fix_github_hosts.ps1 写入的整段记录，hosts 恢复原状。
REM 可重复运行；不会碰系统原有的其他记录。

net session >nul 2>&1
if errorlevel 1 (
    echo 需要管理员权限，正在请求提权...
    powershell -NoProfile -Command "Start-Process -FilePath '%~f0' -Verb RunAs"
    exit /b
)

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0fix_github_hosts.ps1" -Remove
echo.
pause
