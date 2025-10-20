@echo off
REM 完整实验脚本 - 遍历所有数据集并进行10折交叉验证
echo ========================================
echo 运行完整特征选择实验
echo 警告: 这可能需要较长时间！
echo ========================================
echo.

set /p confirm="确认要运行完整实验吗？(Y/N): "
if /i "%confirm%" NEQ "Y" (
    echo 已取消
    pause
    exit /b
)

echo.
echo 开始实验...
echo.

python test.py > results_%date:~0,4%%date:~5,2%%date:~8,2%_%time:~0,2%%time:~3,2%%time:~6,2%.log 2>&1

echo.
echo ========================================
echo 实验完成！结果已保存到日志文件
echo ========================================
pause
