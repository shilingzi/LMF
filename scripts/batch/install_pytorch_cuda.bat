@echo off
echo 正在安装支持CUDA的PyTorch版本...
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
echo 安装完成，请检查GPU是否可用
pause 