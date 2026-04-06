"""
日志和进度条模块

包含训练日志记录和进度条显示功能。

作者: 李文煜
日期: 2025-04-01

2026-04-06
变更说明：
  1. log_metrics 添加 NaN 检测保护，预测结果含 NaN 时跳过指标计算并记录警告
"""

import sys
import json
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from typing import Dict, List, Any


class ProgressBar:
    """
    进度条类

    用于显示训练或处理进度。

    Attributes:
        n (int): 总进度数
        nf (float): 总进度数的浮点表示
        length (int): 进度条长度（字符数）
        verbose (bool): 是否显示详细信息
        ticks (List[int]): 应触发写入操作的预计算i值

    Example:
        >>> pb = ProgressBar(100, length=40)
        >>> for i in range(100):
        ...     pb.bar(i)
        >>> pb.close()
        [========================================] 100% Complete
    """

    def __init__(self, n: int, length: int = 40, verbose: bool = True) -> None:
        """
        初始化进度条

        Args:
            n (int): 总项目数
            length (int, optional): 进度条长度，默认为40
            verbose (bool, optional): 是否显示详细信息，默认为True
        """
        # 防止除以零
        self.n = max(1, n)
        self.nf = float(n)
        self.length = length
        self.verbose = verbose

        # 预计算应触发写入操作的i值
        self.ticks = [round(i / 100.0 * n) for i in range(101)]
        self.ticks.append(n - 1)
        self.bar(0)

    def bar(self, i: int, message: str = "") -> None:
        """
        更新进度条

        Args:
            i (int): 当前进度（范围[0, n-1]）
            message (str, optional): 要显示的附加消息

        Note:
            该方法假定i范围从0到n-1
        """
        if i in self.ticks:
            if self.verbose:
                b = int(np.ceil(((i + 1) / self.nf) * self.length))
                sys.stdout.write(
                    "\r[{0}{1}] {2}%\t{3}".format(
                        "=" * b,
                        " " * (self.length - b),
                        int(100 * ((i + 1) / self.nf)),
                        message
                    )
                )
            else:
                sys.stdout.write("=")
            sys.stdout.flush()

    def close(self, message: str = "") -> None:
        """
        关闭进度条

        Args:
            message (str, optional): 要在进度条完成后显示的消息
        """
        # 将进度条移至100%再关闭
        self.bar(self.n - 1)
        sys.stdout.write(f"{message}\n")
        sys.stdout.flush()


class Logger:
    """
    训练日志记录器

    用于存储和计算神经网络训练期间产生的指标。

    Attributes:
        i (Dict[str, int]): 每个数据集的计数器
        log_acc (bool): 是否记录准确率
        log_auc (bool): 是否记录AUC
        log_p_r (bool): 是否记录P-R曲线下面积
        metrics (Dict[str, Dict[str, List[float]]]): 指标字典
        keys (List[str]): 数据集键列表

    Example:
        >>> logger = Logger(args, ['acc', 'AUC'], valid=True, test_keys=['test'])
        >>> logger.log_loss(0.5)
        >>> logger.log_metrics(y_true, y_pred)
        >>> print(logger.metrics)
        {
            'train': {'acc': [0.85], 'AUC': [0.92], 'loss': [0.5]},
            'valid': {'acc': [0.82], 'AUC': [0.89], 'loss': [0.52]},
            'test': {'acc': [0.80], 'AUC': [0.88], 'loss': [0.55]}
        }
    """

    def __init__(
        self,
        args: Dict[str, Any],
        metrics: List[str],
        valid: bool = True,
        test_keys: List[str] = None
    ) -> None:
        """
        初始化日志记录器

        Args:
            args (Dict[str, Any]): 命令行参数字典
            metrics (List[str]): 要记录的指标列表，可选值：
                - 'acc': 准确率
                - 'AUC': ROC曲线下面积
                - 'P-R': P-R曲线下面积
                - 'loss': 损失值
            valid (bool, optional): 是否记录验证集指标，默认为True
            test_keys (List[str], optional): 测试数据集标签列表，默认为None
        """
        self.i = {'train': 0}
        self.log_acc, self.log_auc, self.log_p_r = False, False, False
        self.metrics = {'train': {}}

        if valid:
            self.metrics['valid'] = {}
            self.i['valid'] = 0

        if test_keys is not None:
            for test_key in test_keys:
                self.metrics[test_key] = {}
                self.i[test_key] = 0

        # 根据指标配置初始化
        if 'acc' in metrics:
            self.log_acc = True
            for key in self.metrics:
                self.metrics[key].update({'acc': []})

        if 'AUC' in metrics:
            self.log_auc = True
            for key in self.metrics:
                self.metrics[key].update({'auc': []})

        if 'P-R' in metrics:
            self.log_p_r = True
            for key in self.metrics:
                self.metrics[key].update({'p-r': []})

        self.keys = list(self.metrics.keys())
        for key in self.keys:
            self.metrics[key].update({'loss': [0]})

        # 保存参数
        self.metrics['args'] = args

    def log_loss(self, loss: float, key: str = 'train') -> None:
        """
        记录损失指标

        Args:
            loss (float): 训练损失值
            key (str, optional): 数据集标签，默认为'train'

        Note:
            该方法计算运行平均值并更新损失列表
        """
        self.i[key] += 1
        update = (
            self.metrics[key]['loss'][-1] * (self.i[key] - 1) + loss
        ) / self.i[key]
        self.metrics[key]['loss'].append(update)

    def log_metrics(
        self,
        y_true: np.ndarray,
        y_hat: np.ndarray,
        key: str = 'train'
    ) -> None:
        """
        记录非损失指标

        Args:
            y_true (np.ndarray): 包含真实标签的数组
            y_hat (np.ndarray): 包含预测logits的数组
            key (str, optional): 数据集标签，默认为'train'

        Note:
            该方法计算并记录AUC、准确率和P-R曲线下面积
        """
        # 检测NaN，跳过本轮指标计算
        if np.any(np.isnan(y_hat)) or np.any(np.isnan(y_true)):
            if self.log_auc:
                self.metrics[key]['auc'].append(float('nan'))
            if self.log_acc:
                self.metrics[key]['acc'].append(float('nan'))
            if self.log_p_r:
                self.metrics[key]['p-r'].append(float('nan'))
            return

        # 检测单类别，跳过AUC/P-R计算（避免sklearn警告）
        n_classes = len(np.unique(y_true))
        if n_classes < 2:
            if self.log_auc:
                self.metrics[key]['auc'].append(float('nan'))
            if self.log_acc:
                acc = np.sum(np.argmax(y_hat, axis=1) == y_true) / len(y_true)
                self.metrics[key]['acc'].append(acc)
            if self.log_p_r:
                self.metrics[key]['p-r'].append(float('nan'))
            return

        if self.log_auc:
            auc = roc_auc_score(y_true, y_hat[:, 1])
            self.metrics[key]['auc'].append(auc)

        if self.log_acc:
            acc = np.sum(np.argmax(y_hat, axis=1) == y_true) / len(y_true)
            self.metrics[key]['acc'].append(acc)

        if self.log_p_r:
            p_r = average_precision_score(y_true, y_hat[:, 1])
            self.metrics[key]['p-r'].append(p_r)

    def output_metric(self, key: str = 'train', metric: str = 'loss') -> float:
        """
        打印最后记录的指标值

        Args:
            key (str, optional): 数据集标签，默认为'train'
            metric (str, optional): 要打印的指标键，默认为'loss'

        Returns:
            float: 最后记录的指标值
        """
        return self.metrics[key][metric][-1]

    def output_metrics(self) -> None:
        """
        打印所有数据集的最后记录指标值

        该方法输出格式化的指标摘要。
        """
        print('')
        for key in sorted(self.keys):
            print(f'{key}:', end='')
            for k, v in self.metrics[key].items():
                print(f'\t{k}: {v[-1]:5.3f}', end='')
            print('\n', end='')
