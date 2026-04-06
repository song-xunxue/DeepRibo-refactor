"""
工具函数模块

包含各种辅助函数，如类型转换、指标计算等。

作者: 李文煜
日期: 2025-04-01

2026-04-06
变更说明：
  1. 修复 Python 3.10+ 兼容性问题：collections.Mapping/Sequence 改为 collections.abc.Mapping/Sequence
  2. 移除 default_collate 中已弃用的 _use_shared_memory 共享内存代码，消除 TypedStorage 和 tensor resize 警告
"""

import argparse
from collections.abc import Mapping, Sequence
import re
import numpy as np
import pandas as pd
import torch
from typing import Union, Tuple, List


def str2bool(v: str) -> bool:
    """
    将字符串转换为布尔值

    Args:
        v (str): 字符串值

    Returns:
        bool: 对应的布尔值

    Raises:
        argparse.ArgumentTypeError: 当字符串不是有效的布尔表示时抛出异常

    Example:
        >>> str2bool('yes')
        True
        >>> str2bool('false')
        False
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def default_collate(batch: List) -> Union[torch.Tensor, Tuple]:
    """
    将每个数据字段放入具有外维批量大小的张量中

    该函数处理多种输入类型：
    - 张量：堆叠为批次
    - 字典：递归处理每个键
    - 序列：转置并递归处理

    Args:
        batch (List): 数据批次列表

    Returns:
        Union[torch.Tensor, Tuple]: 处理后的批次数据

    Raises:
        TypeError: 当batch包含不支持的类型时抛出异常

    Example:
        >>> batch = [
        ...     {'sequence': torch.randn(4, 30, 1), 'label': 0},
        ...     {'sequence': torch.randn(4, 30, 1), 'label': 1}
        ... ]
        >>> collated = default_collate(batch)
    """
    numpy_type_map = {
        'float64': torch.DoubleTensor,
        'float32': torch.FloatTensor,
        'float16': torch.HalfTensor,
        'int64': torch.LongTensor,
        'int32': torch.IntTensor,
        'int16': torch.ShortTensor,
        'int8': torch.CharTensor,
        'uint8': torch.ByteTensor,
    }

    error_msg = 'batch必须包含张量、数字、字典或列表；找到 {}'
    string_classes = (str, bytes)
    int_classes = int

    elem_type = type(batch[0])

    # 处理张量
    if torch.is_tensor(batch[0]):
        pad = False

        # 检查是否需要填充
        if batch[0].shape[0] != 4:
            pad = True
            batch_lens = np.sort([b.shape[0] for b in batch])[::-1].copy()
            sort_order = np.argsort([b.shape[0] for b in batch])[::-1].copy()
            batch = torch.nn.utils.rnn.pad_sequence(
                [batch[idx] for idx in sort_order]
            )
            batch.unsqueeze_(2).contiguous()

        if pad:
            return (batch, batch_lens, sort_order)
        else:
            return torch.stack(batch, dim=0)

    # 处理NumPy数组
    elif elem_type.__module__ == 'numpy' and \
            elem_type.__name__ != 'str_' and \
            elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # 检查字符串类和对象数组
            if re.search('[SaUO]', elem.dtype.str) is not None:
                raise TypeError(error_msg.format(elem.dtype))
            return torch.stack([torch.from_numpy(b) for b in batch], 0)

        if elem.shape == ():  # 标量
            py_type = float if elem.dtype.name.startswith('float') else int
            return numpy_type_map[elem.dtype.name](list(map(py_type, batch)))

    # 处理整数
    elif isinstance(batch[0], int_classes):
        return torch.LongTensor(batch)

    # 处理浮点数
    elif isinstance(batch[0], float):
        return torch.DoubleTensor(batch)

    # 处理字符串
    elif isinstance(batch[0], string_classes):
        return batch

    # 处理字典
    elif isinstance(batch[0], Mapping):
        return {
            key: default_collate([d[key] for d in batch])
            for key in batch[0]
        }

    # 处理序列
    elif isinstance(batch[0], Sequence):
        transposed = zip(*batch)
        return [default_collate(samples) for samples in transposed]

    raise TypeError(error_msg.format(type(batch[0])))


def extend_lib(df: pd.DataFrame, pred: np.ndarray) -> pd.DataFrame:
    """
    使用预测结果扩展data_list.csv对象

    该函数处理预测结果，添加以下列：
    - label: 布尔标签
    - pred: 预测概率
    - pred_rank: 预测排名
    - SS: 是否为起始位点
    - dist: 距离最近预测起始位点的距离
    - SS_pred_rank: 起始位点预测排名

    Args:
        df (pd.DataFrame): 包含data_list.csv表格的DataFrame
        pred (np.ndarray): 神经网络输出的预测数组

    Returns:
        pd.DataFrame: 扩展后的DataFrame

    Note:
        该函数实现了起始位点选择算法，用于从预测中选择最可能的ORF
    """
    df = df.copy()
    df['label'] = df['label'].values.astype(bool)
    df['in_gene'] = df['in_gene'].values.astype(bool)
    df['pred'] = pred[:, 1]

    # 按预测值排序
    sort_idx = np.argsort(df['pred'].values)
    df.loc[sort_idx, 'pred_rank'] = np.arange(len(df))[::-1]

    SS = np.full(len(df), False)
    dist = np.zeros(len(df))

    # 对每条链和每个终止位点处理
    for strand in df['strand'].unique():
        for stop in df[df['strand'] == strand]['stop_site'].unique():
            mask = np.where(
                (df['strand'] == strand) & (df['stop_site'] == stop)
            )[0]

            if len(mask) > 1:
                # 多个ORF共享同一个终止位点
                SS[df.loc[mask, 'pred'].idxmax()] = True
                dist_mask = df.loc[mask, 'label'] == False

                if np.any(dist_mask == False):
                    right = df.loc[mask].loc[dist_mask == False, 'start_site'].iloc[0]
                    left = df.loc[mask].loc[dist_mask, 'start_site'].values
                    dist[mask[dist_mask]] = left - right
                else:
                    dist[mask[dist_mask]] = -1
            else:
                SS[mask] = True
                if not df.loc[mask, 'label'].values:
                    dist[mask] = -1

    df['SS'] = SS
    df['dist'] = dist.astype(np.int)

    # 起始位点预测排名
    SS_pred_rank = np.full(len(df), 999999, dtype=np.int)
    sort_idx = df[df['SS']].sort_values(by='pred').index.values[::-1]
    SS_pred_rank[sort_idx] = np.arange(len(df[df['SS']]))
    df['SS_pred_rank'] = SS_pred_rank

    return df


def auc_from_tensors(y_hat: torch.Tensor, y_true: torch.Tensor) -> float:
    """
    从张量计算AUC值

    Args:
        y_hat (torch.Tensor): 预测logits
        y_true (torch.Tensor): 真实标签

    Returns:
        float: AUC值

    Example:
        >>> y_hat = torch.tensor([[0.2, 0.8], [0.7, 0.3]])
        >>> y_true = torch.tensor([1, 0])
        >>> auc = auc_from_tensors(y_hat, y_true)
        >>> print(auc)
        0.75
    """
    y_true_np = y_true.numpy()
    y_hat_np = y_hat.numpy()
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y_true_np, y_hat_np[:, 1])
    return auc
