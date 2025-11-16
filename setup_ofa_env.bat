@echo off
REM OFA图像标题生成项目的Conda环境安装脚本 (Windows)
REM 使用Miniconda创建独立环境

echo =========================================
echo 创建OFA图像标题生成项目环境
echo =========================================

REM 检查conda是否安装
where conda >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo 错误: 未找到conda命令。请先安装Miniconda。
    echo 下载地址: https://docs.conda.io/en/latest/miniconda.html
    exit /b 1
)

REM 从environment.yml创建环境
echo 正在从environment.yml创建conda环境...
conda env create -f environment_ofa.yml

if %ERRORLEVEL% EQU 0 (
    echo =========================================
    echo 环境创建成功！
    echo =========================================
    echo.
    echo 激活环境...
    call conda activate hateful-image-ofa
    
    echo.
    echo 正在安装fairseq...
    pip install git+https://github.com/pytorch/fairseq.git
    
    if %ERRORLEVEL% NEQ 0 (
        echo fairseq安装失败，尝试安装特定版本...
        pip install fairseq==0.12.2
    )
    
    echo.
    echo =========================================
    echo 基础环境安装完成！
    echo =========================================
    echo.
    echo 接下来还需要执行以下步骤:
    echo 1. 环境已激活，当前在 hateful-image-ofa 环境中
    echo 2. 安装OFA模型代码:
    echo    - git clone https://github.com/OFA-Sys/OFA.git
    echo    - cd OFA
    echo    - pip install -e .
    echo    - cd ..
    echo.
    echo 3. 下载预训练模型权重到 checkpoints/caption.pt
    echo    参考: https://github.com/OFA-Sys/OFA
    echo.
    echo 验证安装:
    echo   python -c "import torch; import fairseq; print('安装成功！')"
    echo.
) else (
    echo 环境创建失败，请检查错误信息。
    exit /b 1
)

pause

