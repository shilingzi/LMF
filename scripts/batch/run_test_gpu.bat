@echo off
echo 使用GPU模式运行Set5数据集测试
python scripts/data/test.py --config scripts/config/test-lmf/test-set5.yaml --model save/edsr-b_lm-lmlte/epoch-best.pth --gpu 0
echo 测试完成
pause 