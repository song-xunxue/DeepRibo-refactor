# DeepRibo 重构版本

## 项目简介

DeepRibo是一个基于深度学习的原核生物基因注释工具，结合核糖体图谱信号和Shine-Dalgarno序列模式实现对开放阅读框（ORF）的精确识别。

本项目是对原始DeepRibo的Python 3.12重构版本，采用干净的、规范的、教学向的、易维护的代码风格。

## 主要特性

- 🚀 **Python 3.12.2** + PyTorch 2.0+ 现代化环境
- 🧠 **基于DeepRibo3.py的模型架构**
- 📝 **完整的中文注释**：所有代码都有详细的中文文档字符串
- 📐 **PEP 8规范**：遵循Python官方代码风格指南
- 🏗️ **模块化设计**：清晰的职责分离，易于维护
- 🧪 **类型提示**：完整的类型注解，提升代码质量

## 项目结构

```
DeepRibo-refactor/
├── src/                        # 核心代码包
│   ├── cli/                    # 命令行接口
│   │   ├── train.py            #   训练命令入口
│   │   ├── predict.py          #   预测命令入口（支持自动模型发现）
│   │   └── data.py             #   数据解析命令入口
│   ├── data/                   # 数据处理
│   │   ├── dataset.py          #   PyTorch Dataset，加载/过滤 .pt 样本
│   │   └── parser.py           #   核糖体图谱原始数据 → 训练格式解析器
│   ├── models/                 # 模型实现
│   │   └── deepribo.py      #    模型（DualComplex/CNNComplex/RNNComplex）
│   ├── training/               # 训练与预测
│   │   └── trainer.py          #   Trainer 类：训练循环、验证、模型保存/加载/预测
│   └── utils/                  # 工具函数
│       ├── helpers.py          #   类型转换、AUC 计算、extend_lib 预测后处理
│       ├── logging.py          #   训练日志、ProgressBar 进度条
│       ├── optimizers.py       #   自定义 Adam 优化器
│       └── samplers.py         #   BucketSampler / BatchSampler
├── models/                     # 训练产出：按 {菌种}/{时间戳}/ 组织的模型文件
├── predictions/                # 预测产出：按 {菌种}/{时间戳}/ 组织的 CSV 文件
├── raw/                        # 各菌种训练记录文档
├── tests/                      # 测试代码
├── configs/                    # 配置文件
│   ├── environment.yml         #   Conda 环境配置
│   └── requirements.txt        #   pip 依赖
├── pyproject.toml              # 项目元数据
└── README.md                   # 本文件
```

### 模块职责说明

| 模块 | 路径 | 职责 |
|------|------|------|
| **cli** | `src/cli/` | 命令行入口，解析用户参数并调用对应功能。`train.py` 训练模型、`predict.py` 加载模型预测 ORF、`data.py` 将原始核糖体图谱数据转换为训练格式 |
| **data** | `src/data/` | 数据层。`parser.py` 从 FASTA + bedgraph 提取所有候选 ORF 并生成 .pt 文件；`dataset.py` 负责 Dataset 加载与三级过滤（RPKM/覆盖度/短 ORF/共享终止位点） |
| **models** | `src/models/` | 模型定义。`deepribo_v3.py` 实现三种架构：`DualComplex`（CNN+RNN 混合，主模型）、`CNNComplex`（纯 CNN）、`RNNComplex`（纯 RNN），均继承自 `BaseModel` |
| **training** | `src/training/` | 训练引擎。`trainer.py` 中的 `Trainer` 类封装了完整生命周期：训练循环、验证评估、early stopping、best_model 保存、模型加载与批量预测 |
| **utils** | `src/utils/` | 基础设施。`helpers.py` 提供预测后处理（`extend_lib` 起始位点选择算法）和 AUC 计算；`logging.py` 进度条；`optimizers.py` 自定义 Adam；`samplers.py` 按长度分桶采样 |

## 安装指南

### 方法1：使用Conda（推荐）

```bash
# 创建Conda环境
conda env create -f environment.yml

# 激活环境
conda activate deepribo-refactor

# 安装项目（开发模式）
pip install -e .
```

### 方法2：使用pip

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt

# 安装项目
pip install -e .
```

## 使用指南

### 数据准备

```bash
# 解析原始核糖体图谱数据
python -m deepribo.cli.data \
    --sense-cov data/sense_cov.bedgraph \
    --asense-cov data/asense_cov.bedgraph \
    --sense-elo data/sense_elo.bedgraph \
    --asense-elo data/asense_elo.bedgraph \
    --fasta data/genome.fasta \
    --destination data/processed/ \
    --gtf data/annotations.gff \
    --start_trips ATG GTG TTG \
    --stop_trips TAA TGA TAG
```

### 模型训练

```bash
# 训练模型
python -m deepribo.cli.train \
    data_path data/processed/ \
    --train_data dataset1 dataset2 \
    --valid_size 0.05 \
    --rpkm 0.12 0.15 \
    --coverage 0.12 0.15 \
    --dest models/trained/ \
    --batch_size 256 \
    --epochs 20 \
    --GRU_nodes 128 \
    --GRU_layers 2 \
    --GRU_bidirect \
    --COV_motifs 32 \
    --FC_nodes 1024 512 \
    --model_type CNNRNN \
    --GPU
```

### 模型预测

```bash
# 使用预训练模型进行预测
python -m deepribo.cli.predict \
    data_path data/processed/ \
    --pred_data dataset3 \
    --rpkm 0.12 \
    --coverage 0.15 \
    --model models/trained/best_model.pt \
    --dest predictions.csv \
    --GRU_nodes 128 \
    --GRU_layers 2 \
    --GRU_bidirect \
    --COV_motifs 32 \
    --FC_nodes 1024 512 \
    --model_type CNNRNN \
    --GPU
```

## 开发指南

### 代码风格

项目遵循[PEP 8](https://peps.python.org/pep-0008/)规范，使用以下工具：

```bash
# 格式化代码
black refactor_model/

# 检查代码质量
flake8 refactor_model/

# 类型检查
mypy refactor_model/
```

### 运行测试

```bash
# 运行所有测试
pytest tests/

# 运行特定测试
pytest tests/test_models.py -v

# 生成覆盖率报告
pytest --cov=refactor_model tests/
```

## 核心组件

### 模型架构（src/models/deepribo_v3.py）

- **DualComplex**: CNN+RNN混合模型（默认）
  - CNN分支：从DNA序列中提取特征
  - RNN分支（GRU）：处理核糖体图谱信号
  - 全连接层：融合双分支特征并输出二分类结果
- **CNNComplex**: 纯CNN模型，仅使用序列数据
- **RNNComplex**: 纯RNN模型，仅使用信号数据

### 数据处理（src/data/）

- **parser.py**: 完整的数据解析流程
  - 加载 FASTA 基因组序列
  - 加载 bedgraph 核糖体图谱信号
  - 查找所有候选 ORF 并生成 .pt 训练文件
- **dataset.py**: PyTorch Dataset 实现
  - 加载 data_list.csv 索引并动态读取 .pt 文件
  - 三级样本过滤：RPKM/覆盖度阈值 → 短ORF（≤30nt）→ 共享终止位点清理

### 训练工具（src/training/trainer.py）

- **Trainer**: 管理完整的训练/预测生命周期
  - 训练循环 + 梯度裁剪（max_norm=5.0）
  - 验证集 AUC 评估 + early stopping
  - best_model 自动保存与加载
  - 批量预测 + `extend_lib` 起始位点选择算法

## 代码规范

项目遵循以下规范：

1. **PEP 8代码风格**
   - 使用4个空格缩进
   - 最大行长度88字符
   - 清晰的命名规范

2. **Google风格docstring**
   - 所有函数和类都有文档字符串
   - 详细的参数说明
   - 返回值文档
   - 使用示例说明用法

3. **类型提示**
   - 完整的类型注解
   - 使用typing模块的类型
   - 提升代码可读性和IDE支持

4. **中文注释**
   - 所有注释使用中文
   - 清晰简洁的说明
   - 教学向的解释

详细规范请参考：`docs/代码规范.md`

## 依赖管理

### Conda环境

项目使用Conda环境，主要依赖：

- Python 3.12.2
- PyTorch 2.0+
- NumPy, Pandas, SciPy
- scikit-learn
- Biopython

### pip依赖

如果使用pip安装，依赖在`requirements.txt`中。

## 版本信息

- **版本**: 2.0.1
- **Python**: 3.12+
- **PyTorch**: 2.0+
- **重构日期**: 2025-04-01

## 原始项目

本项目基于原始DeepRibo：
- 原始作者：Jim Clauwaert, Gerben Menschaert
- 原始论文：[Nucleic Acids Research, 2019](https://doi.org/10.1093/nar/gkz061)
- 原始仓库：https://github.com/Biobix/DeepRibo

## 主要改进

### 代码质量

- ✅ 完整的中文文档字符串
- ✅ PEP 8代码风格
- ✅ 类型提示
- ✅ 模块化设计
- ✅ 错误处理

### 工程化

- ✅ 配置文件管理
- ✅ 测试框架
- ✅ 日志记录
- ✅ 进度显示
- ✅ 命令行接口

### 可维护性

- ✅ 清晰的目录结构
- ✅ 分离的关注点
- ✅ 可重用的组件
- ✅ 详细的注释

## 许可证

本项目使用GNU General Public License v3.0许可证，与原始DeepRibo保持一致。

## 联系方式

- 作者：李文煜 1120233042@bit.edu.cn

## 致谢

感谢原始DeepRibo团队的开源工作，以及所有贡献者对Python深度学习生态的贡献。
