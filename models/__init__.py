from .models import register, make

# 基础模型
from . import edsr, rdn, rcan, swinir
from . import mlp, lmmlp

# 高级模型（有些可能不存在）
try:
    from . import misc, liif, lte, ltep, ciaosr
    from . import lmliif, lmlte, lmciaosr
except ImportError as e:
    print(f"警告: 部分模型导入失败 - {str(e)}") 