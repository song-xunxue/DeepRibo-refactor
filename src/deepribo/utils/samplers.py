"""
数据采样器模块

包含自定义的数据采样器，用于批次采样和桶排序。

作者: DeepRibo Team
日期: 2025-04-01
"""

import numpy as np
from torch.utils.data.sampler import Sampler
from typing import Iterator, List, Any


class BatchSampler:
    """
    批次采样器

    包装另一个采样器以产生小批量的索引。

    Attributes:
        sampler (Sampler): 基础采样器
        batch_size (int): 批次大小
        drop_last (bool): 是否丢弃小于batch_size的最后一批

    Example:
        >>> sampler = list(range(10))
        >>> batch_sampler = BatchSampler(sampler, batch_size=3, drop_last=False)
        >>> for batch in batch_sampler:
        ...     print(batch)
        [0, 1, 2]
        [3, 4, 5]
        [6, 7, 8]
        [9]
    """

    def __init__(
        self,
        sampler: Sampler,
        batch_size: int,
        drop_last: bool
    ) -> None:
        """
        初始化批次采样器

        Args:
            sampler (Sampler): 基础采样器
            batch_size (int): 小批次的尺寸
            drop_last (bool): 如果为True，当批次大小小于batch_size时丢弃该批次
        """
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self) -> Iterator[List[int]]:
        """
        返回批次的迭代器

        Yields:
            List[int]: 包含批次索引的列表

        Note:
            如果drop_last为False，最后一个批次可能小于batch_size
        """
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []

        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self) -> int:
        """
        返回批次数

        Returns:
            int: 批次数
        """
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size


class BucketSampler(Sampler):
    """
    桶采样器

    按照ORF长度对样本进行分组，在每个epoch需要调用bucketShuffle。

    Attributes:
        data_source: 数据源DataFrame
        batch_size (int): 批次大小
        lens (np.ndarray): ORF长度数组
        s_idx (np.ndarray): 长度排序索引
        sort_idx (np.ndarray): 排序后的索引
        idx_list (np.ndarray): 最终索引列表

    Note:
        桶采样有助于减少padding，提高训练效率
    """

    def __init__(
        self,
        data_source: Any,
        index: List[int],
        batch_size: int
    ) -> None:
        """
        初始化桶采样器

        Args:
            data_source: 数据集（DataFrame格式）
            index (List[int]): 要采样的索引列表
            batch_size (int): 批次大小
        """
        starts = data_source.loc[index, 'start_site']
        stops = data_source.loc[index, 'stop_site']
        self.lens = np.abs(starts - stops)
        self.batch_size = batch_size
        self.s_idx = np.argsort(self.lens)
        self.sort_idx = np.array(self.lens.index[self.s_idx])
        self.bucketShuffle()

    def __iter__(self) -> Iterator[int]:
        """
        返回索引的迭代器

        Yields:
            int: 样本索引
        """
        return iter(self.idx_list)

    def __len__(self) -> int:
        """
        返回样本总数

        Returns:
            int: 样本总数
        """
        return len(self.lens)

    def bucketShuffle(self) -> None:
        """
        对桶内的样本进行洗牌

        该方法实现以下洗牌策略：
        1. 在区域内洗牌行
        2. 在数据集中洗牌批次

        Note:
            应该在每个epoch开始时调用此方法
        """
        # 在区域内洗牌行
        region_size = np.maximum(
            int(len(self.lens) / self.batch_size // 12), 100
        )
        inc_batch_reg = len(self.lens) % region_size
        index_list = np.array(self.sort_idx[inc_batch_reg:])
        np.random.shuffle(np.reshape(index_list, (-1, region_size)).T)
        shuffled_index = np.hstack((index_list, self.sort_idx[:inc_batch_reg]))

        # 在数据集中洗牌批次
        inc_batch = len(self.lens) % self.batch_size
        shuffled_index_list = shuffled_index[inc_batch:]
        np.random.shuffle(np.reshape(shuffled_index_list, (-1, self.batch_size)))

        # 设置数据集的新顺序
        self.idx_list = np.hstack((
            shuffled_index_list,
            shuffled_index_list[:inc_batch]
        ))
