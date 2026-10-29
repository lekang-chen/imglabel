@echo off
chcp 65001 >nul
title Image Labeling System - Public Access (Cloudflare Tunnel)
cd /d "%~dp0"

REM Exposes the local server to a temporary public URL, so a phone on
REM mobile data (or any other network) can open it.
REM The URL changes every run and dies when this window closes.
REM Do not share the URL: anyone who has it can edit your labels.

set "PYEXE=.venv\Scripts\python.exe"
if not exist "%PYEXE%" set "PYEXE=python"

if not exist "tools\cloudflared.exe" (
    echo [ERROR] tools\cloudflared.exe not found.
    echo         Download cloudflared-windows-amd64.exe and put it there:
    echo         https://github.com/cloudflare/cloudflared/releases/latest
    echo.
    pause
    exit /b 1
)

"%PYEXE%" tools\tunnel.py %*

if errorlevel 1 (
    echo.
    echo Tunnel exited. See messages above.
    pause
)
