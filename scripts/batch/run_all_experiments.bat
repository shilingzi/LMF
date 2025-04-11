@echo off
echo 开始运行所有实验...

:: 训练模型
echo 开始训练模型...
call run_train.bat

:: 评估模型
echo 开始评估模型...
python scripts/eval/eval_all_datasets.py

echo 所有实验完成
pause 