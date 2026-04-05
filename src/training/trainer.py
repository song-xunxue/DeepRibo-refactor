"""
训练器模块

包含DeepRibo模型的训练和预测功能。

作者: 李文煜
日期: 2025-04-01
"""

import datetime
import json
import csv
import os
import shutil
import argparse
from typing import Tuple, Union

import torch
import numpy as np
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence

from ..models.deepribo_v3 import ModelFactory
from ..utils.samplers import BatchSampler, BucketSampler
from ..utils.helpers import default_collate, extend_lib, auc_from_tensors
from ..utils.logging import ProgressBar, Logger
from ..utils.optimizers import Adam


class Trainer:
    """
    模型训练器

    管理模型的完整训练流程，包括：
    - 训练循环
    - 验证
    - 测试
    - 模型保存
    - 日志记录

    Attributes:
        model (Module): 要训练的模型
        device (torch.device): 训练设备（CPU或GPU）

    Example:
        # >>> from deepribo.models import DeepRiboV3
        # >>> model = DeepRiboV3(config)
        # >>> trainer = Trainer(model)
        # >>> trainer.train(train_loader, valid_loader, epochs=50)
    """

    def __init__(self, model: Module) -> None:
        """
        初始化训练器

        Args:
            model (Module): PyTorch模型实例
        """
        self.model = model

    def fit(
        self,
        device: torch.device,
        train_loader: DataLoader,
        valid_loader: DataLoader = None,
        test_loaders: list = None,
        test_keys: list = None,
        scheduler=None,
        epochs: int = 50,
        initial_epoch: int = 0,
        seed: int = None,
        loss=None,
        optimizer=None,
        log: Logger = None,
        dest: str = 'default',
        verbose: int = 1,
        GPU: bool = False,
        metadata: dict = None
    ) -> Logger:
        """
        训练模型（类似Keras的.fit()方法）

        Args:
            device (torch.device): 训练设备
            train_loader (DataLoader): 训练数据加载器
            valid_loader (DataLoader, optional): 验证数据加载器，默认为None
            test_loaders (list, optional): 测试数据加载器列表，默认为None
            test_keys (list, optional): 测试数据集标签列表，默认为None
            scheduler (object, optional): 学习率调度器，默认为None
            epochs (int, optional): 训练轮数，默认为50
            initial_epoch (int, optional): 开始训练的epoch，默认为0
            seed (int, optional): 随机种子，默认为None
            loss (object, optional): 训练损失函数，默认为None
            optimizer (object, optional): 训练优化器，默认为None
            log (Logger, optional): 日志记录器对象，默认为None
            dest (str, optional): 模型保存路径，默认为'default'
            verbose (int, optional): 详细模式，0=静默，1=详细，默认为1
            GPU (bool, optional): 是否使用GPU，默认为False
            metadata (dict, optional): 训练元信息，默认为None

        Returns:
            Logger: 包含训练指标的Logger对象

        Note:
            该方法实现完整的训练循环，包括早停、模型保存等
        """
        # 创建时间戳文件夹（格式：2026-4-1-20-42）
        ts = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
        ts_dir = f'{dest}/{ts}/'

        # 创建时间戳目录
        import os
        os.makedirs(ts_dir, exist_ok=True)

        if seed and seed >= 0:
            torch.manual_seed(seed)

        # 准备测试数据
        if GPU:
            dtypeX = torch.cuda.FloatTensor
            dtypeY = torch.cuda.LongTensor
        else:
            dtypeX = torch.FloatTensor
            dtypeY = torch.LongTensor

        # 编译优化器
        opt = optimizer

        # 运行训练循环
        epoch_records = []
        best_valid_auc = -1.0
        best_epoch = 0

        for t in range(initial_epoch, epochs):
            if scheduler and t != initial_epoch:
                scheduler.step()

            print(f'Epoch {t + 1} / {epochs}')

            # 设置日志记录器
            pb = ProgressBar(len(train_loader), verbose=verbose)
            epoch_loss = 0.0

            # 运行批次
            self.model.train()
            for b_i, b_data in enumerate(train_loader):
                # 反向传播
                opt.zero_grad()

                sort_order, X_batch_RNN_len = b_data[1][2], b_data[1][1]
                y_batch = b_data[2][sort_order].type(dtypeY).to(device)
                X_batch_conv = b_data[0][sort_order].type(dtypeX).to(device)
                X_batch_RNN = b_data[1][0].type(dtypeX).to(device)
                X_batch_RNN = pack_padded_sequence(X_batch_RNN, X_batch_RNN_len)

                y_batch_pred, hidden = self.model((X_batch_conv, X_batch_RNN))

                batch_loss = loss(y_batch_pred, y_batch)
                batch_loss.backward()
                opt.step()

                # 更新状态
                epoch_loss += batch_loss.item()
                log.log_loss(batch_loss.item())

                pb.bar(b_i, log.output_metric())

            pb.close()

            # 重新洗牌桶采样器
            if hasattr(train_loader, 'batch_sampler'):
                train_loader.batch_sampler.sampler.bucketShuffle()

            # 运行指标计算
            y_pred, y_true = self.predict(
                device, train_loader, log=log, GPU=GPU, verbose=verbose
            )
            log.log_metrics(y_true.cpu().numpy(), y_pred.cpu().numpy())

            if valid_loader is not None:
                y_pred, y_true = self.predict(
                    device, valid_loader, loss=loss,
                    key='valid', log=log, GPU=GPU, verbose=verbose
                )
                log.log_metrics(
                    y_true.cpu().numpy(), y_pred.cpu().numpy(), 'valid'
                )

            if test_loaders and test_loaders[0] is not None:
                for test_loader, test_key in zip(test_loaders, test_keys):
                    y_pred, y_true = self.predict(
                        device, test_loader, loss=loss,
                        key=test_key, log=log, GPU=GPU, verbose=verbose
                    )
                    log.log_metrics(
                        y_true.cpu().numpy(), y_pred.cpu().numpy(), test_key
                    )

            # 保存模型到时间戳目录
            torch.save(
                self.model.state_dict(),
                f'{ts_dir}model_epoch_{t + 1}.pt'
            )
            with open(f'{ts_dir}metrics_epoch_{t}.json', 'w') as fp:
                json.dump(log.metrics, fp)

            log.output_metrics()

            # 记录本轮指标
            record = {'epoch': t + 1}
            for key in log.keys:
                record[f'{key}_loss'] = log.metrics[key].get('loss', [None])[-1]
                if log.log_auc and log.metrics[key].get('auc'):
                    record[f'{key}_auc'] = log.metrics[key]['auc'][-1]
                if log.log_acc and log.metrics[key].get('acc'):
                    record[f'{key}_acc'] = log.metrics[key]['acc'][-1]
                if log.log_p_r and log.metrics[key].get('p-r'):
                    record[f'{key}_pr'] = log.metrics[key]['p-r'][-1]
            epoch_records.append(record)

            # 跟踪最佳模型（按valid_auc或train_auc）
            current_auc = record.get('valid_auc', record.get('train_auc', -1))
            if current_auc > best_valid_auc:
                best_valid_auc = current_auc
                best_epoch = t + 1

        # 保存最佳模型副本
        best_src = f'{ts_dir}model_epoch_{best_epoch}.pt'
        if os.path.exists(best_src):
            shutil.copy2(best_src, f'{ts_dir}best_model.pt')
            print(f'Best model: epoch {best_epoch} (AUC={best_valid_auc:.4f})')

        # 保存汇总训练日志CSV
        self._save_training_log(epoch_records, ts_dir, metadata)

        return log

    @staticmethod
    def _save_training_log(
        records: list,
        ts_dir: str,
        metadata: dict = None
    ) -> None:
        """
        保存汇总训练日志为CSV文件

        生成 training_log.csv，包含每轮的训练/验证/测试指标，
        可直接用于R语言 ggplot2 绘图。

        Args:
            records (list): 每轮指标的字典列表
            ts_dir (str): 时间戳目录路径
            metadata (dict, optional): 训练元信息
        """
        if not records:
            return

        log_path = f'{ts_dir}training_log.csv'
        headers = list(records[0].keys())

        with open(log_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            # 元信息作为注释行（R可用 comment.char='#' 跳过）
            if metadata:
                for k, v in metadata.items():
                    writer.writerow([f'# {k}={v}'])

            writer.writerow(headers)
            for rec in records:
                writer.writerow([rec.get(h, '') for h in headers])

        print(f'Training log saved to {log_path}')

    def predict(
        self,
        device: torch.device,
        loader: DataLoader,
        loss=None,
        key: str = None,
        log: Logger = None,
        GPU: bool = False,
        verbose: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成输入样本的预测输出

        计算以批次方式进行。

        Args:
            device (torch.device): 计算设备
            loader (DataLoader): 数据加载器
            loss (object, optional): 训练损失对象，默认为None
            key (str, optional): 数据集标签，默认为None
            log (Logger, optional): 日志记录器对象，默认为None
            GPU (bool, optional): 是否使用GPU，默认为False
            verbose (bool, optional): 是否显示详细信息，默认为False

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - 预测logits
                - 真实标签

        Note:
            该方法处理可变长度序列，使用pack_padded_sequence
        """
        # 构建数据类型
        if GPU:
            dtypeX = torch.cuda.FloatTensor
            dtypeY = torch.cuda.LongTensor
        else:
            dtypeX = torch.FloatTensor
            dtypeY = torch.LongTensor

        self.model.eval()
        r = 0
        n = len(loader.batch_sampler.sampler)
        batch_size = loader.batch_sampler.batch_size
        pb = ProgressBar(len(loader), verbose=verbose)

        for b_i, b_data in enumerate(loader):
            # 预测批次
            with torch.no_grad():
                sort_order, X_batch_RNN_len = b_data[1][2], b_data[1][1]
                y_batch = b_data[2][sort_order].type(dtypeY).to(device)
                X_batch_conv = b_data[0][sort_order].type(dtypeX).to(device)
                X_batch_RNN = b_data[1][0].type(dtypeX).to(device)
                X_batch_RNN = pack_padded_sequence(X_batch_RNN, X_batch_RNN_len)

                y_batch_pred, hidden = self.model((X_batch_conv, X_batch_RNN))

                if key:
                    batch_loss = loss(y_batch_pred, y_batch)
                    log.log_loss(batch_loss.item(), key)

                # 推断预测形状
                y_batch_pred = y_batch_pred.data

                if r == 0:
                    y_pred = torch.zeros((n,) + y_batch_pred.size()[1:])
                    y_true = torch.zeros((n,) + y_batch_pred.data.size()[1:])

                # 添加到预测张量
                unsort_idx = np.argsort(sort_order)
                y_pred[r:min(n, r + batch_size)] = y_batch_pred[unsort_idx]
                y_true[r:min(n, r + batch_size)] = y_batch.data[unsort_idx]
                r += batch_size

                pb.bar(b_i)

        # 解桶排序
        if hasattr(loader, 'batch_sampler'):
            unbucket_idx = np.argsort(loader.batch_sampler.sampler.idx_list)

        pb.close()

        return y_pred[unbucket_idx], y_true[unbucket_idx]


def load_database(
    data_path: str,
    data: list,
    cutoff: tuple,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    valid_size: float = 0
) -> Union[DataLoader, tuple]:
    """
    使用自定义加载器加载数据

    Args:
        data_path (str): 包含所有实验数据的主目录路径
        data (list): data_path中连接为一个数据集的文件夹名称列表
        cutoff (tuple): 包含两个列表的元组，分别列出data中每个数据集的
            最小RPKM（[0]）和覆盖率（[1]）截止值
        batch_size (int): 批次大小
        num_workers (int): 用于数据加载的CPU单元数
        pin_memory (bool): 使用分配的GPU内存进行更快处理
        valid_size (float, optional): 用于验证集的数据分数，默认为0

    Returns:
        Union[DataLoader, tuple]:
            - 如果valid_size为0：返回train_loader
            - 如果valid_size>0：返回(train_loader, valid_loader)

    Note:
            - 标签按基因组数据库细分以分层组织和生物体的标签
            - 使用桶采样器按ORF长度分组样本
    """
    # 这里应该导入CustomLoader，但由于简化，我们暂时使用占位符
    from ..data.dataset import DeepRiboDataset

    dataset = DeepRiboDataset(data_path, data, cutoff)

    idx = np.arange(len(dataset.masked_list))
    dfs = dataset.masked_list.iloc[:, 0].str.split('/').str[0].value_counts()

    # 标签按基因组数据库细分，以分层组织和生物体的标签
    labels = np.hstack([np.full(x, i) for i, x in enumerate(dfs.values)])
    labels[dataset.masked_list['label'] == 1] += len(dfs)

    if valid_size > 0:
        # 分割训练集和验证集
        from sklearn.model_selection import train_test_split
        train_idx, valid_idx = train_test_split(idx, test_size=valid_size, stratify=labels)

        valid_sampler = BucketSampler(dataset.masked_list, valid_idx, 256)
        valid_batch_loader = BatchSampler(valid_sampler, 256, False)

        train_sampler = BucketSampler(dataset.masked_list, train_idx, batch_size)
        train_batch_loader = BatchSampler(train_sampler, batch_size, False)

        valid_loader = DataLoader(
            dataset,
            batch_sampler=valid_batch_loader,
            num_workers=num_workers,
            collate_fn=default_collate,
            pin_memory=pin_memory
        )

        train_loader = DataLoader(
            dataset,
            batch_sampler=train_batch_loader,
            num_workers=num_workers,
            collate_fn=default_collate,
            pin_memory=pin_memory
        )

        return train_loader, valid_loader
    else:
        train_sampler = BucketSampler(
            dataset.masked_list, np.arange(len(dataset.masked_list)), batch_size
        )
        train_batch_loader = BatchSampler(train_sampler, batch_size, False)

        train_loader = DataLoader(
            dataset,
            batch_sampler=train_batch_loader,
            num_workers=num_workers,
            collate_fn=default_collate,
            pin_memory=pin_memory
        )

        return train_loader


def train_model(
    args: dict,
    data_path: str,
    train_data: list,
    valid_size: float,
    test_data: list,
    train_cutoff: tuple,
    test_cutoff: tuple,
    dest: str,
    batch_size: int,
    epochs: int,
    hidden_size: int,
    layers: int,
    bidirect: bool,
    motif_count: int,
    nodes: list,
    model_type: str,
    num_workers: int,
    GPU: bool,
    verbose: bool
) -> None:
    """
    使用DeepRibo方法训练模型

    Args:
        args (dict): 包含所有参数的字典
        data_path (str): 包含所有实验数据的主目录路径
        train_data (list): data_path中用于训练的文件夹名称列表
        valid_size (float): train_data中用作验证集的数据分数
        test_data (list): data_path中用作测试的文件夹名称列表
        train_cutoff (tuple): 包含两个列表的元组，分别列出train_data中每个数据集的
            最小RPKM（[0]）和覆盖率（[1]）截止值
        test_cutoff (tuple): 包含两个列表的元组，分别列出test_data中每个数据集的
            最小RPKM（[0]）和覆盖率（[1]）截止值
        dest (str): 保存模型的文件夹路径
        batch_size (int): 批次大小
        epochs (int): 训练轮数
        hidden_size (int): 分配给GRU的权重
        layers (int): GRU层数
        bidirect (bool): 模型使用双向GRU
        motif_count (int): CNN层使用的motif（卷积核）数量
        nodes (list): 构成DeepRibo全连接层的每层节点数和层数的整数数组
        model_type (str): 用于训练的模型类型（CNNRNN、CNN或RNN）
        num_workers (int): 用于数据加载的CPU单元数
        GPU (bool): 使用GPU训练模型
        verbose (bool): 使用简单（False）或复杂（True）训练输出

    Note:
        该方法实现了完整的训练流程，包括数据加载、模型创建、训练和测试
    """
    # 加载训练和验证数据
    if valid_size > 0:
        valid_bool = True
        train_loader, valid_loader = load_database(
            data_path, train_data, train_cutoff, batch_size,
            num_workers, GPU, valid_size
        )
    else:
        valid_bool = False
        train_loader = load_database(
            data_path, train_data, train_cutoff,
            batch_size, num_workers, GPU, valid_size
        )

    # 加载测试数据
    if test_data is not None:
        test_loader = load_database(
            data_path, test_data, test_cutoff,
            64, num_workers, GPU, 0
        )
    else:
        test_loader = None

    print(test_loader)
    sample_train = len(train_loader.batch_sampler.sampler)
    sample_valid = len(valid_loader.batch_sampler.sampler)
    print(f"{sample_train} samples in train data")
    print(f"{sample_valid} samples in valid data")

    # 选择设备
    if GPU:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # 创建加权损失（严重不平衡的数据）
    ratio = sum(train_loader.dataset.y_train) / len(train_loader.dataset.y_train)
    weights = torch.FloatTensor([ratio, 1 - ratio]).to(device)

    # 初始化模型
    model = ModelFactory.create_model(
        model_type=model_type,
        motif_count=motif_count,
        hidden_size=hidden_size,
        layers=layers,
        bidirect=bidirect,
        nodes=nodes
    )
    model.to(device)

    # 定义损失和优化器
    loss = torch.nn.CrossEntropyLoss(weights)
    optimizer = Adam(
        model.parameters(),
        lr=0.001,
        weight_decay=1e-4,
        amsgrad=True
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 9, gamma=0.1)

    # 记录loss、准确率、AUC、测试集上的P-R曲线下面积
    log = Logger(
        vars(args),
        ["loss", "AUC", "P-R", "acc"],
        valid_bool,
        test_data
    )

    # 构建训练元信息
    metadata = {
        'dataset': '+'.join(train_data) if train_data else 'unknown',
        'model_type': model_type,
        'epochs': epochs,
        'batch_size': batch_size,
        'GRU_nodes': hidden_size,
        'GRU_layers': layers,
        'GRU_bidirect': bidirect,
        'COV_motifs': motif_count,
        'FC_nodes': '+'.join(map(str, nodes)) if nodes else 'unknown',
        'train_rpkm': '+'.join(map(str, train_cutoff[0])) if train_cutoff[0] else 'none',
        'train_coverage': '+'.join(map(str, train_cutoff[1])) if train_cutoff[1] else 'none',
        'valid_size': valid_size,
        'GPU': GPU,
    }

    # 训练模型
    trainer = Trainer(model)
    trainer.fit(
        device=device,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loaders=[test_loader],
        test_keys=test_data,
        scheduler=scheduler,
        epochs=epochs,
        loss=loss,
        optimizer=optimizer,
        log=log,
        dest=dest,
        GPU=GPU,
        verbose=verbose,
        metadata=metadata
    )


def predict(
    data_path: str,
    pred_data: list,
    pred_cutoff: tuple,
    model_name: str,
    dest: str,
    batch_size: int,
    hidden_size: int,
    layers: int,
    bidirect: bool,
    motif_count: int,
    nodes: list,
    model_type: str,
    num_workers: int,
    GPU: bool,
    verbose: bool
) -> None:
    """
    使用模型进行预测

    Args:
        data_path (str): 包含实验数据的主目录路径
        pred_data (list): data_path中用于预测的文件夹名称列表
        pred_cutoff (tuple): 包含两个列表的元组，分别列出pred_data中每个数据集的
            最小RPKM（[0]）和覆盖率（[1]）截止值
        model_name (str): 训练模型的路径
        dest (str): 保存模型的文件夹路径
        batch_size (int): 批次大小
        hidden_size (int): 分配给GRU的权重
        layers (int): GRU层数
        bidirect (bool): 模型使用双向GRU
        motif_count (int): CNN层使用的motif（卷积核）数量
        nodes (list): 构成DeepRibo全连接层的每层节点数和层数的整数数组
        model_type (str): 用于预测的模型类型（CNNRNN、CNN或RNN）
        num_workers (int): 用于数据加载的CPU单元数
        GPU (bool): 使用GPU进行预测
        verbose (bool): 使用简单（False）或复杂（True）训练输出

    Note:
        该方法加载模型并在新数据上进行预测
    """
    # 加载预测数据
    pred_loader = load_database(
        data_path, pred_data, pred_cutoff, batch_size, num_workers, GPU
    )

    # 选择设备
    if GPU:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # 创建模型
    model = ModelFactory.create_model(
        model_type=model_type,
        motif_count=motif_count,
        hidden_size=hidden_size,
        layers=layers,
        bidirect=bidirect,
        nodes=nodes
    )
    model.to(device)

    # 加载预训练权重
    model.load_state_dict(torch.load(model_name, map_location=device))

    # 进行预测
    trainer = Trainer(model)
    pred, true = trainer.predict(
        device=device,
        loader=pred_loader,
        GPU=GPU,
        verbose=verbose
    )

    # 扩展数据列表并保存
    df_pred = extend_lib(pred_loader.dataset.masked_list, pred)
    df_pred.to_csv(dest)
