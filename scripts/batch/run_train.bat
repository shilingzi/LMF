@echo off
echo 开始训练 LMF-LTE 模型...

:: 创建新的保存目录
set SAVE_DIR=save/edsr-b_lm-lmlte_new
if not exist %SAVE_DIR% mkdir %SAVE_DIR%

:: 启动训练
python scripts/model/train_windows.py --config scripts/config/train-lmf/train_edsr-baseline-lmlte_small.yaml --gpu 0 --save_path %SAVE_DIR%

echo 训练完成
pause 