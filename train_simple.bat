cd /d E:\lmf_project\lmf_temp
call conda activate lian
python train_windows.py --config configs/train-lmf/train_swinir-baseline-lmlte_small.yaml --gpu 0 --save_path save/swinir-b_lm-lmlte_new
pause 