@echo off
REM 安装OFA代码以解决 ModuleNotFoundError: No module named 'utils'

echo ========================================
echo 安装OFA代码
echo ========================================
echo.

cd /d C:\Users\Aiur\Hateful-Image-Project

REM 检查OFA目录是否存在
if exist "OFA" (
    echo OFA目录已存在，跳过克隆
) else (
    echo 正在尝试从GitHub克隆OFA仓库...
    git clone https://github.com/OFA-Sys/OFA.git
    if errorlevel 1 (
        echo.
        echo [错误] GitHub连接失败！
        echo.
        echo 请手动下载OFA代码：
        echo 1. 访问: https://github.com/OFA-Sys/OFA
        echo 2. 点击 "Code" 按钮 -^> "Download ZIP"
        echo 3. 解压到当前目录，确保有 OFA 文件夹
        echo 4. 然后重新运行此脚本
        echo.
        pause
        exit /b 1
    )
)

echo.
echo 正在安装OFA...
cd OFA
call conda activate hateful-image-ofa
pip install -e .

if errorlevel 1 (
    echo [错误] 安装失败
    pause
    exit /b 1
)

echo.
echo ========================================
echo 安装完成！
echo ========================================
echo.
echo 现在可以返回notebook继续运行了
echo.
pause


