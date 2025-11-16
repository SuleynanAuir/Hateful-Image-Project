@echo off
REM 将OFA的utils、tasks、models目录复制到项目根目录

echo ========================================
echo 复制OFA模块到项目根目录
echo ========================================
echo.

cd /d C:\Users\Aiur\Hateful-Image-Project

if not exist "OFA" (
    echo [错误] OFA目录不存在！
    echo 请先下载OFA代码到: %CD%\OFA
    pause
    exit /b 1
)

echo 正在复制utils、tasks、models目录...

if exist "OFA\utils" (
    if exist "utils" (
        echo utils目录已存在，跳过...
    ) else (
        xcopy /E /I /Y "OFA\utils" "utils"
        echo [OK] utils目录已复制
    )
) else (
    echo [错误] OFA\utils目录不存在！
)

if exist "OFA\tasks" (
    if exist "tasks" (
        echo tasks目录已存在，跳过...
    ) else (
        xcopy /E /I /Y "OFA\tasks" "tasks"
        echo [OK] tasks目录已复制
    )
) else (
    echo [错误] OFA\tasks目录不存在！
)

if exist "OFA\models" (
    if exist "models" (
        echo models目录已存在，跳过...
    ) else (
        xcopy /E /I /Y "OFA\models" "models"
        echo [OK] models目录已复制
    )
) else (
    echo [错误] OFA\models目录不存在！
)

echo.
echo ========================================
echo 复制完成！
echo ========================================
echo.
echo 现在可以在notebook中导入OFA模块了
echo.
pause

