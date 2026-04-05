"""
DeepRibo 重构版本

一个基于深度学习的原核生物基因注释工具
结合核糖体图谱信号和Shine-Dalgarno序列模式
实现对开放阅读框（ORF）的精确识别

主要特性：
- CNN+RNN混合模型架构
- 完整的数据处理流程
- 规范的训练和预测接口
- 详细的中文文档


作者:李文煜
日期: 2025-04-01
"""

__version__ = "2.0.0"
__author__ = "DeepRibo Team"
__email__ = "team@deepribo.org"
__license__ = "GNU General Public License v3.0"

# 导出主要的公共接口
from .models import ModelFactory
from .training import Trainer, train_model, predict

__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "__license__",
    "ModelFactory",
    "Trainer",
    "train_model",
    "predict",
]
