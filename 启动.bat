@echo off
chcp 65001 >nul
title 人机协同图像标注系统
cd /d "%~dp0"

echo ============================================================
echo   人机协同图像标注系统
echo ============================================================
echo.

REM ---------- 检查 Python ----------
where python >nul 2>nul
if errorlevel 1 (
    echo [错误] 找不到 python，请先安装 Miniforge 并创建 imglabel 环境。
    echo.
    echo 安装步骤：
    echo   conda create -n imglabel python=3.10 -y
    echo   conda activate imglabel
    echo   pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

REM ---------- 检查依赖 ----------
python -c "import ultralytics, flask, psutil, cv2, numpy" >nul 2>nul
if errorlevel 1 (
    echo [提示] 依赖不完整，正在安装...
    python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
    if errorlevel 1 (
        echo [错误] 依赖安装失败，可改用阿里云源：
        echo   pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
        pause
        exit /b 1
    )
)

REM ---------- 检查图片目录 ----------
if not exist "data\images" (
    mkdir "data\images"
)
dir /b "data\images" >nul 2>nul
if errorlevel 1 (
    echo [提示] data\images 里还没有图片。
    echo        请把筛选好的室内场景图片放进这个目录。
    echo.
)

echo [启动] 正在打开 http://127.0.0.1:5000 ...
echo [提示] 关闭此窗口即可停止服务
echo.
python app.py

if errorlevel 1 (
    echo.
    echo [错误] 服务异常退出，请查看上方日志。
    pause
)
