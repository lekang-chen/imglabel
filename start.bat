@echo off
setlocal enabledelayedexpansion
title Image Labeling System - One Click Start
cd /d "%~dp0"

echo ============================================================
echo   Human-Machine Collaborative Image Labeling System
echo   One-click start
echo ============================================================
echo.
echo   Steps: check Python -^> prepare .venv -^> check deps -^> run
echo   First run may take 5-10 min for dependencies.
echo   Keep this window open. Close it to stop the server.
echo.

REM ============================================================
REM  1. Reuse the project .venv if it already exists
REM ============================================================
if exist ".venv\Scripts\python.exe" (
    set "PYEXE=.venv\Scripts\python.exe"
    echo [1/3] Python environment found: .venv
    goto :check_deps
)

REM ============================================================
REM  2. No .venv - find a real Python to build one
REM ============================================================
set "BASEPY="
for %%P in (py python python3) do (
    if not defined BASEPY (
        %%P -c "import sys" >nul 2>nul
        if not errorlevel 1 (
            for /f "delims=" %%V in ('%%P -c "import sys;print(sys.executable)" 2^>nul') do (
                if not defined BASEPY set "BASEPY=%%V"
            )
        )
    )
)

if not defined BASEPY (
    echo [ERROR] Python not found.
    echo.
    echo   Install Python 3.9 or newer from https://www.python.org/downloads/
    echo   IMPORTANT: tick "Add python.exe to PATH" during installation.
    echo.
    pause
    exit /b 1
)

echo [1/3] Python found: %BASEPY%
echo       Creating virtual environment .venv ...
"%BASEPY%" -m venv .venv
if errorlevel 1 (
    echo [ERROR] Failed to create .venv
    pause
    exit /b 1
)
set "PYEXE=.venv\Scripts\python.exe"

:check_deps
REM ============================================================
REM  3. Dependencies
REM ============================================================
echo [2/3] Checking dependencies ...
"%PYEXE%" -c "import ultralytics, flask, psutil, cv2, numpy, PIL" >nul 2>nul
if not errorlevel 1 goto :run

echo       Missing packages. Installing (first run only) ...
"%PYEXE%" -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple >nul 2>nul
"%PYEXE%" -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
if errorlevel 1 (
    echo       Tsinghua mirror failed, retrying with Aliyun ...
    "%PYEXE%" -m pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
)
if errorlevel 1 (
    echo.
    echo [ERROR] Dependency installation failed.
    echo   Check your network, then run this line manually:
    echo     .venv\Scripts\python.exe -m pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

:run
REM ============================================================
REM  4. Run
REM ============================================================
if not exist "data\images" mkdir "data\images"

echo [3/3] Starting server, browser will open automatically ...
echo.
echo   Address : http://127.0.0.1:5000
echo   Upload photos with the button at top-left of the page.
echo.
"%PYEXE%" app.py %*

if errorlevel 1 (
    echo.
    echo [ERROR] Server exited unexpectedly. See messages above.
    pause
)
endlocal
