@echo off
echo 使用GPU模式运行超分辨率测试
python scripts/web/demo.py --input load/Set5/butterfly.png --model save/edsr-b_lm-lmlte/epoch-best.pth --scale 4 --output butterfly_sr.png
echo 处理完成
pause 