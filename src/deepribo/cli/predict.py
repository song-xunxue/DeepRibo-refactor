"""
预测命令行接口

提供模型预测的命令行工具。

作者: 李文煜
日期: 2025-04-02
"""

import argparse
from ..training import predict


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
        required=True,
        help="预训练模型的路径"
    )
    parser.add_argument(
        '-d', '--dest',
        default='predictions.csv',
        type=str,
        required=True,
        help="保存预测结果的文件路径"
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
        type=str,
        help="顺序GRU层的数量"
    )
    parser.add_argument(
        '-B', '--GRU_bidirect',
        default=True,
        type=bool,
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
        choices=['CNNRNN', 'CNN', 'RNN'],
        help="使用CNNRNN、CNN或RNN架构"
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

    print(f"使用模型 {args.model} 创建预测\n使用参数: {args}")
    predict(
        data_path=args.data_path,
        pred_data=[args.pred_data],
        pred_cutoff=(args.rpkm, args.coverage),
        model_name=args.model,
        dest=args.dest,
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
