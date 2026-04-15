"""
预测命令行接口

提供模型预测的命令行工具。

作者: 李文煜
日期: 2025-04-02

2026-04-12
变更说明：
  1. -M/--model 参数改为可选，省略时自动查找 models/{pred_data}/ 下最新的 best_model.pt
  2. 修复 pred_cutoff 格式：将 (float, float) 包装为 ([float], [float])，匹配 dataset.py 的预期格式
  3. -d/--dest 参数改为可选，省略时自动生成与模型时间戳对应的路径 predictions/{pred_data}/{时间戳}/predictions.csv

2026-04-14
变更说明：
  1. --model_type 新增 CNNRNN_V11 选项（阶段一改进模型）
"""

import argparse
import glob as glob_mod
import os
from ..training import predict


def _find_latest_model(pred_data: str, model_dir: str = 'models') -> str:
    """
    在 models/{pred_data}/ 下查找最新的 best_model.pt

    参数:
        pred_data: 菌种名称（如 bac, eco）
        model_dir: 模型根目录，默认 'models'

    返回:
        最新 best_model.pt 的完整路径

    异常:
        FileNotFoundError: 未找到任何 best_model.pt
    """
    species_dir = os.path.join(model_dir, pred_data)
    if not os.path.isdir(species_dir):
        raise FileNotFoundError(
            f"模型目录不存在: {species_dir}\n"
            f"请先训练模型，或通过 -M 手动指定模型路径"
        )

    # 查找所有时间戳子目录下的 best_model.pt
    pattern = os.path.join(species_dir, '*', 'best_model.pt')
    candidates = glob_mod.glob(pattern)

    if not candidates:
        raise FileNotFoundError(
            f"在 {species_dir} 下未找到任何 best_model.pt\n"
            f"请先训练模型，或通过 -M 手动指定模型路径"
        )

    # 按修改时间排序，取最新的
    latest = max(candidates, key=os.path.getmtime)
    return latest


def _extract_timestamp(model_path: str) -> str:
    """
    从模型路径中提取时间戳目录名

    模型路径格式: models/{菌种}/{时间戳}/best_model.pt
    或: {任意前缀}/{时间戳}/best_model.pt

    参数:
        model_path: 模型文件路径

    返回:
        时间戳目录名（如 '2026-04-06-23-24'）
    """
    # 取父目录名即为时间戳
    ts_dir = os.path.basename(os.path.dirname(model_path))
    return ts_dir


def _auto_dest(pred_data: str, model_path: str) -> str:
    """
    根据菌种名和模型时间戳自动生成预测输出路径

    格式: predictions/{pred_data}/{时间戳}/predictions.csv

    参数:
        pred_data: 菌种名称
        model_path: 模型文件路径

    返回:
        自动生成的输出路径
    """
    ts = _extract_timestamp(model_path)
    return os.path.join('predictions', pred_data, ts, 'predictions.csv')


def main() -> None:
    """
    预测命令的主函数

    该函数定义了预测命令的参数解析和执行。
    """
    parser = argparse.ArgumentParser(
        description='使用预训练模型进行预测',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # 位置参数
    parser.add_argument(
        'data_path',
        type=str,
        help="包含预测数据文件夹的路径"
    )

    # 必需参数
    parser.add_argument(
        '--pred_data',
        type=str,
        required=True,
        help="data_path中用于预测的文件夹名称"
    )
    parser.add_argument(
        '-r', '--rpkm',
        type=float,
        required=True,
        help="过滤用于预测的RPKM值的最小截止值"
    )
    parser.add_argument(
        '-c', '--coverage',
        type=float,
        required=True,
        help="过滤用于预测的覆盖率值的最小截止值，"
                 "这些值按相同顺序给出"
    )
    parser.add_argument(
        '-M', '--model',
        type=str,
        default=None,
        help="预训练模型的路径；省略时自动查找 models/{pred_data}/ 下最新的 best_model.pt"
    )
    parser.add_argument(
        '-d', '--dest',
        default=None,
        type=str,
        help="保存预测结果的文件路径；省略时自动生成 predictions/{pred_data}/{时间戳}/predictions.csv"
    )

    # 可选参数
    parser.add_argument(
        '-g', '--GRU_nodes',
        default=128,
        type=int,
        help="GRU单元隐藏状态的大小"
    )
    parser.add_argument(
        '-l', '--GRU_layers',
        default=2,
        choices=[1, 2],
        type=int,
        help="顺序GRU层的数量"
    )
    parser.add_argument(
        '-B', '--GRU_bidirect',
        default=True,
        type=lambda x: x.lower() in ('true', '1', 'yes'),
        help="使用双向GRU单元"
    )
    parser.add_argument(
        '-m', '--COV_motifs',
        default=32,
        type=int,
        help="卷积层使用的motifs（卷积核）数量"
    )
    parser.add_argument(
        '-n', '--FC_nodes',
        default=[1024, 512],
        type=int,
        nargs='+',
        help="DeepRibo全连接层中每层的节点数"
    )
    parser.add_argument(
        '--model_type',
        default='CNNRNN',
        type=str,
        choices=['CNNRNN', 'CNN', 'RNN', 'CNNRNN_V11'],
        help="使用CNNRNN、CNN、RNN或CNNRNN_V11（阶段一改进）架构"
    )
    parser.add_argument(
        '--num_workers',
        default=0,
        type=int,
        help="用于数据加载的CPU单元数"
    )
    parser.add_argument(
        '--GPU',
        action='store_true',
        help="使用GPU"
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help="更详细的进度条"
    )

    args = parser.parse_args()

    # 自动查找最新模型
    model_path = args.model
    if model_path is None:
        model_path = _find_latest_model(args.pred_data)
        print(f"自动选择最新模型: {model_path}")

    # 自动生成输出路径
    dest = args.dest
    if dest is None:
        dest = _auto_dest(args.pred_data, model_path)
        print(f"自动输出路径: {dest}")

    # 确保输出目录存在
    os.makedirs(os.path.dirname(dest), exist_ok=True)

    print(f"使用模型 {model_path} 创建预测\n使用参数: {args}")
    predict(
        data_path=args.data_path,
        pred_data=[args.pred_data],
        pred_cutoff=([args.rpkm], [args.coverage]),
        model_name=model_path,
        dest=dest,
        batch_size=256,
        hidden_size=args.GRU_nodes,
        layers=args.GRU_layers,
        bidirect=args.GRU_bidirect,
        motif_count=args.COV_motifs,
        nodes=args.FC_nodes,
        model_type=args.model_type,
        num_workers=args.num_workers,
        GPU=args.GPU,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()
