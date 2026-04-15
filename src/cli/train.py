"""
训练命令行接口

提供模型训练的命令行工具。

作者: 李文煜
日期: 2025-04-02

2026-04-14
变更说明：
  1. --model_type 新增 CNNRNN_V11 选项（阶段一改进模型）
"""

import argparse
from ..training import train_model


def main() -> None:
    """
    训练命令的主函数

    该函数定义了训练命令的参数解析和执行。
    """
    parser = argparse.ArgumentParser(
        description='训练DeepRibo模型',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # 位置参数
    parser.add_argument(
        'data_path',
        type=str,
        help="包含训练和测试数据文件夹的路径"
    )

    # 可选参数
    parser.add_argument(
        '--train_data',
        default=[],
        nargs='+',
        type=str,
        required=True,
        help="data_path中用于训练的文件夹名称列表"
    )
    parser.add_argument(
        '--valid_size',
        type=float,
        default=0.05,
        help="用作验证集的训练数据比例，"
                 "数据分割在标签和所有训练数据集之间分层"
    )
    parser.add_argument(
        '--test_data',
        default=None,
        nargs='*',
        type=str,
        help="data_path中用作测试的文件夹名称列表"
    )
    parser.add_argument(
        '-r', '--rpkm',
        nargs='+',
        type=float,
        required=True,
        help="过滤训练数据的最小RPKM截止值"
    )
    parser.add_argument(
        '-c', '--coverage',
        nargs='+',
        type=float,
        required=True,
        help="过滤训练数据的最小覆盖率截止值，"
                 "这些值按相同顺序给出"
    )
    parser.add_argument(
        '-ct', '--coverage_test',
        nargs='*',
        type=float,
        default=None,
        help="过滤训练数据的最小覆盖率截止值，"
                 "这些值按相同顺序给出"
    )
    parser.add_argument(
        '-rt', '--rpkm_test',
        nargs='*',
        type=float,
        default=None,
        help="过滤训练数据的最小RPKM截止值"
    )
    parser.add_argument(
        '-d', '--dest',
        default='models/trained',
        type=str,
        help="保存模型的文件夹路径"
    )
    parser.add_argument(
        '-b', '--batch_size',
        type=int,
        default=256,
        help="训练批次大小"
    )
    parser.add_argument(
        '-e', '--epochs',
        default=20,
        type=int,
        help="训练轮数"
    )
    parser.add_argument(
        '-g', '--GRU_nodes',
        default=128,
        type=int,
        help="GRU单元隐藏状态的大小"
    )
    parser.add_argument(
        '-l', '--GRU_layers',
        default=2,
        choices=[1, 2, 3, 4],
        type=int,
        help="顺序GRU层的数量"
    )
    parser.add_argument(
        '-B', '--GRU_bidirect',
        type=lambda x: x.lower() in ('true', '1', 'yes'),
        nargs='?',
        const=True,
        default=True,
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
        help="使用GPU（推荐）"
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help="更详细的进度条"
    )

    args = parser.parse_args()

    print(f'使用参数训练模型: {args}')
    train_model(
        args=args,
        data_path=args.data_path,
        train_data=args.train_data,
        valid_size=args.valid_size,
        test_data=args.test_data,
        train_cutoff=(args.rpkm, args.coverage),
        test_cutoff=(args.rpkm_test, args.coverage_test),
        dest=args.dest,
        batch_size=args.batch_size,
        epochs=args.epochs,
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
