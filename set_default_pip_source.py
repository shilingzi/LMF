import os
import sys
from pathlib import Path

# 获取用户主目录
home_dir = Path.home()

# pip配置文件路径
pip_conf_dir = home_dir / "pip"
pip_conf_file = pip_conf_dir / "pip.conf" if sys.platform != "win32" else pip_conf_dir / "pip.ini"

# 创建目录(如果不存在)
pip_conf_dir.mkdir(exist_ok=True)

# 写入配置
with open(pip_conf_file, "w") as f:
    f.write("[global]\n")
    f.write("index-url = https://pypi.org/simple\n")
    f.write("trusted-host = pypi.org\n")

print(f"已将默认pip源设置为官方PyPI源")
print(f"配置文件位置: {pip_conf_file}") 