"""
数据解析命令行接口

提供数据解析的命令行工具。

作者: 李文煜
日期: 2025-04-02
"""

import argparse
from ..data.parser import DataParser


def main() -> None:
    """
    数据解析命令的主函数

    该函数定义了数据解析命令的参数解析和执行。
    """
    parser = argparse.ArgumentParser(
        description="解析核糖体测序数据为DeepRibo使用的文件",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # 位置参数
    parser.add_argument(
        'sense_cov',
        type=str,
        help="包含有义链核糖体数据（覆盖率）的bedgraph路径"
    )
    parser.add_argument(
        'asense_cov',
        type=str,
        help="包含反义链核糖体数据（覆盖率）的bedgraph路径"
    )
    parser.add_argument(
        'sense_elo',
        type=str,
        help="包含有义链核糖体数据（延伸）的bedgraph路径"
    )
    parser.add_argument(
        'asense_elo',
        type=str,
        help="包含反义链核糖体数据（延伸）的bedgraph路径"
    )
    parser.add_argument(
        'fasta',
        type=str,
        help="包含基因组序列的fasta路径"
    )
    parser.add_argument(
        'destination',
        type=str,
        help="输出目标路径。此路径必须包含两个名为0和1的文件夹"
    )

    # 可选参数
    parser.add_argument(
        '-g', '--gtf',
        type=str,
        help="包含注释的gtf/gff路径"
    )
    parser.add_argument(
        '-s', '--start_trips',
        nargs='+',
        type=str,
        default=['ATG', 'GTG', 'TTG'],
        help="被视为可能起始密码子的三联体列表"
    )
    parser.add_argument(
        '-p', '--stop_trips',
        nargs='+',
        type=str,
        default=['TAA', 'TGA', 'TAG'],
        help="被视为可能终止密码子的三联体列表"
    )

    args = parser.parse_args()

    # 创建数据解析器并执行
    parser = DataParser(
        fasta_path=args.fasta,
        ribo_cov_sense=args.sense_cov,
        ribo_cov_asense=args.asense_cov,
        ribo_elo_sense=args.sense_elo,
        ribo_elo_asense=args.asense_elo,
        dest_path=args.destination,
        start_trips=args.start_trips,
        stop_trips=args.stop_trips
    )
    parser.parse(args.gtf)


if __name__ == "__main__":
    main()
