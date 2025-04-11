@echo off
echo Starting training SwinIR model...

set SAVE_DIR=save/swinir-b_lm-lmlte_new
if not exist %SAVE_DIR% mkdir %SAVE_DIR%

python train_windows.py --config configs/train-lmf/train_swinir-baseline-lmlte_small.yaml --gpu 0 --save_path %SAVE_DIR%

echo Training completed
pause 