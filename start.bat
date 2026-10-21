@echo off
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
