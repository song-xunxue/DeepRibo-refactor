"""
命令行接口模块

提供DeepRibo的命令行工具：
- train: 训练模型
- predict: 使用模型预测
- data: 解析数据

作者: 李文煜
日期: 2025-04-02
"""

from .train import main as train_main
from .predict import main as predict_main
from .data import main as data_main

__all__ = [
    "train_main",
    "predict_main",
    "data_main",
]
