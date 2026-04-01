"""
优化器模块

包含自定义的Adam优化器实现。

作者: DeepRibo Team
日期: 2025-04-01
"""

import torch
from torch.optim import Optimizer
import math


class Adam(Optimizer):
    """
    Adam优化器实现

    实现了Adam算法，结合了动量法和自适应学习率的优点。
    适用于大多数深度学习任务。

    算法基于论文：
    "Adam: A Method for Stochastic Optimization"
    Kingma & Ba, ICLR 2015
    https://arxiv.org/abs/1412.6980

    AMSGrad变体基于论文：
    "On the Convergence of Adam and Beyond"
    Reddi et al., ICLR 2018
    https://openreview.net/forum?id=ryQu7f-RZ

    Attributes:
        defaults (Dict[str, Any]): 默认超参数配置
        state (Dict): 优化器状态字典

    Example:
        >>> model = nn.Linear(10, 2)
        >>> optimizer = Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        >>> loss = criterion(output, target)
        >>> loss.backward()
        >>> optimizer.step()
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        amsgrad: bool = False
    ) -> None:
        """
        初始化Adam优化器

        Args:
            params (Iterable): 要优化的参数或参数组字典
            lr (float, optional): 学习率，默认为1e-3
            betas (Tuple[float, float], optional):
                用于计算梯度和梯度平方的运行平均值的系数
                默认为(0.9, 0.999)
            eps (float, optional):
                添加到分母以提高数值稳定性的项
                默认为1e-8
            weight_decay (float, optional):
                权重衰减（L2正则化），默认为0
            amsgrad (bool, optional):
                是否使用AMSGrad变体，默认为False

        Note:
            - betas[0]控制一阶矩的指数衰减率
            - betas[1]控制二阶矩的指数衰减率
            - weight_decay实现L2正则化
        """
        defaults = {
            'lr': lr,
            'betas': betas,
            'eps': eps,
            'weight_decay': weight_decay,
            'amsgrad': amsgrad
        }
        super(Adam, self).__init__(params, defaults)

    def step(self, closure=None):
        """
        执行单步优化

        Args:
            closure (Callable, optional): 重新评估模型并返回损失的闭包函数

        Returns:
            Optional[float]: 如果提供了closure，则返回损失值

        Note:
            该方法实现了Adam算法的完整更新步骤：
            1. 计算偏差校正后的梯度和梯度平方
            2. 更新参数
            3. 更新动量变量
        """
        loss = None
        if closure is not None:
            loss = closure()

        # 遍历每个参数组
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        'Adam不支持稀疏梯度，请考虑使用SparseAdam'
                    )

                amsgrad = group['amsgrad']
                state = self.state[p]

                # 状态初始化
                if len(state) == 0:
                    state['step'] = 0
                    # 梯度的指数移动平均
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # 梯度平方的指数移动平均
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # 维护所有二阶矩的运行平均值最大值
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']

                beta1, beta2 = group['betas']

                state['step'] += 1

                # 权重衰减（L2正则化）
                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # 衰减一阶和二阶矩的运行平均系数
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                if amsgrad:
                    # 维护所有二阶矩的运行平均值最大值
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # 使用最大值进行归一化
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                # 偏差校正
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # 计算步长
                step_size = group['lr'] * (
                    math.sqrt(bias_correction2) / bias_correction1
                )

                # 更新参数
                p.data.addcdiv_(-step_size, exp_avg, denom)

        return loss
