#!/usr/bin/env python
"""
快速预测脚本

简化版本的预测脚本，用于快速进行预测。

用法：
    python scripts/predict.py --data_path DATA_PATH --pred_data DATASET --model MODEL_PATH
"""

import sys
import os
from pathlib import Path

# 添加src到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from deepribo.cli import predict as predict_module


def main():
    """主函数"""
    # 检查参数
    if len(sys.argv) < 5:
        print("使用方法：")
        print("  python scripts/predict.py --data_path DATA_PATH --pred_data DATASET --model MODEL_PATH")
        print("\n或者使用完整命令行接口：")
        print("  python -m deepribo.cli.predict --help")
        sys.exit(1)

    # 使用predict模块的main函数
    predict_module.main()


if __name__ == "__main__":
    main()
