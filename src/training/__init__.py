"""
训练模块

包含训练和预测功能：
- Trainer: 模型训练器
- train_model: 训练接口函数
- predict: 预测接口函数
- load_database: 数据加载函数

作者: 李文煜
日期: 2025-04-01
"""

from .trainer import Trainer, train_model, predict, load_database

__all__ = [
    "Trainer",
    "train_model",
    "predict",
    "load_database",
]
