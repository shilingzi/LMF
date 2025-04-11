from .models import register, make

# 基础模型
from . import edsr, rdn, rcan
from . import mlp, lmmlp

# 高级模型
try:
    from . import misc, lte, ltep
    from . import lmlte, lmlte_rdn, lmlte_edsr
    from . import metasr, swinir
except ImportError as e:
    print(f"警告: 部分模型导入失败 - {str(e)}")

# 确保模型被正确导入
import models.edsr
import models.mlp
import models.misc
import models.rdn
import models.rcan
import models.lmlte
import models.lmlte_rdn
import models.lmlte_edsr
import models.metasr
import models.swinir

# 注意: 请使用models.py中定义的register和make函数 