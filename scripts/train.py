#!/usr/bin/env python
"""
快速训练脚本

简化版本的训练脚本，用于快速开始训练。

用法：
    python scripts/train.py --data_path DATA_PATH --train_data DATA1 DATA2
"""

import sys
import os
from pathlib import Path

# 添加src到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from deepribo.cli import train as train_module


def main():
    """主函数"""
    # 检查参数
    if len(sys.argv) < 3:
        print("使用方法：")
        print("  python scripts/train.py --data_path DATA_PATH --train_data DATA1 DATA2 [其他参数]")
        print("\n或者使用完整命令行接口：")
        print("  python -m deepribo.cli.train --help")
        sys.exit(1)

    # 使用train模块的main函数
    train_module.main()


if __name__ == "__main__":
    main()
