# 大肠杆菌 (Escherichia coli) - 训练说明

## 菌种信息

- **学名**: Escherichia coli
- **菌株**: str. K-12 substr. MG1655
- **分类**: 革兰氏阴性菌
- **基因组大小**: 约4.6 Mb
- **染色体数量**: 1条环形染色体

## 数据文件

本项目包含大肠杆菌的以下数据文件：

| 文件名 | 说明 | 文件大小 |
|--------|------|----------|
| `eco.fna` | 基因组序列（FASTA格式） | 4.6 MB |
| `eco.gff` | 基因注释（GFF3格式） | 2.3 MB |
| `eco_cov_sense.bedgraph` | 正义链覆盖度信号 | 23.5 MB |
| `eco_cov_asense.bedgraph` | 反义链覆盖度信号 | 26.6 MB |
| `eco_elo_sense.bedgraph` | 正义链延伸信号 | 16.9 MB |
| `eco_elo_asense.bedgraph` | 反义链延伸信号 | 18.5 MB |

## 数据解析

### 步骤1：激活环境

```bash
conda activate deepribo-refactor
```

### 步骤2：运行数据解析

```bash
python -m deepribo.cli.data \
    raw/eco/eco_cov_sense.bedgraph \
    raw/eco/eco_cov_asense.bedgraph \
    raw/eco/eco_elo_sense.bedgraph \
    raw/eco/eco_elo_asense.bedgraph \
    raw/eco/eco.fna \
    data/processed/ \
    --gtf raw/eco/eco.gff \
    --start_trips ATG GTG TTG \
    --stop_trips TAA TGA TAG
```

### 解析后数据结构

解析完成后，会在`data/processed/eco/`下创建以下结构：

```
data/processed/eco/
├── 0/                    # 负样本（非基因ORF）
├── 1/                    # 正样本（真实基因ORF）
└── data_list.csv        # 样本列表
```

## 模型训练

### 基础训练命令

```bash
conda activate deepribo-refactor

python -m deepribo.cli.train \
    data/processed/ \
    --train_data eco \
    --valid_size 0.05 \
    --rpkm 0.12 0.15 \
    --coverage 0.12 0.15 \
    --dest models/eco/ \
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

### 高性能训练命令（推荐）

由于大肠杆菌数据量较大，可以使用更大的批次和更多epoch：

```bash
python -m deepribo.cli.train \
    data/processed/ \
    --train_data eco \
    --valid_size 0.05 \
    --rpkm 0.12 0.15 \
    --coverage 0.12 0.15 \
    --dest models/eco/ \
    --batch_size 512 \
    --epochs 30 \
    --GRU_nodes 128 \
    --GRU_layers 2 \
    --GRU_bidirect \
    --COV_motifs 32 \
    --FC_nodes 1024 512 \
    --model_type CNNRNN \
    --GPU
```

### 参数说明

| 参数 | 基础值 | 推荐值 | 说明 |
|------|--------|--------|------|
| `--train_data` | `eco` | `eco` | 训练数据集名称 |
| `--valid_size` | `0.05` | `0.05` | 验证集比例（5%） |
| `--rpkm` | `0.12 0.15` | `0.12 0.15` | RPKM过滤阈值 |
| `--coverage` | `0.12 0.15` | `0.12 0.15` | 覆盖率过滤阈值 |
| `--dest` | `models/eco/` | `models/eco/` | 模型保存路径 |
| `--batch_size` | `256` | `512` | 批次大小（数据量大时可用更大值） |
| `--epochs` | `20` | `30` | 训练轮数（数据量大时需要更多轮次） |
| `--GRU_nodes` | `128` | `128` | GRU隐藏层大小 |
| `--GRU_layers` | `2` | `2` | GRU层数 |
| `--COV_motifs` | `32` | `32` | CNN卷积核数量 |
| `--FC_nodes` | `1024 512` | `1024 512` | 全连接层节点数 |
| `--model_type` | `CNNRNN` | `CNNRNN` | 模型类型 |

### 输出文件

训练完成后，模型文件保存在`models/eco/`目录，每次训练会创建一个以时间戳命名的文件夹：

```
models/eco/
└── 2026-4-1-20-42/          # 时间戳文件夹（示例）
    ├── model_epoch_1.pt     # 第1个epoch的模型
    ├── model_epoch_2.pt     # 第2个epoch的模型
    ├── ...
    ├── model_epoch_30.pt    # 第30个epoch的模型
    ├── metrics_epoch_0.json # 第1个epoch的指标
    ├── metrics_epoch_1.json # 第2个epoch的指标
    ├── ...
    └── metrics_epoch_29.json # 第30个epoch的指标
```

**说明**：
- 每次训练都会创建一个新的时间戳文件夹
- 文件夹格式：`年-月-日-小时-分钟`（如：2026-4-1-20-42）
- 同一次训练的所有epoch模型都保存在同一个时间戳文件夹内

## 模型预测

### 预测命令

```bash
conda activate deepribo-refactor

python -m deepribo.cli.predict \
    data/processed/ \
    --pred_data eco \
    --rpkm 0.12 \
    --coverage 0.15 \
    --model models/eco/2026-4-1-20-42/model_epoch_20.pt \
    --dest predictions/eco/predictions.csv \
    --GRU_nodes 128 \
    --GRU_layers 2 \
    --GRU_bidirect \
    --COV_motifs 32 \
    --FC_nodes 1024 512 \
    --model_type CNNRNN \
    --GPU
```

**注意**：`--model` 参数需要指定完整的时间戳路径，如 `models/eco/2026-4-1-20-42/model_epoch_20.pt`

### 参数说明

| 参数 | 值 | 说明 |
|------|---|------|
| `--pred_data` | `eco` | 预测数据集名称 |
| `--model` | `models/eco/model_epoch_20.pt` | 预训练模型路径 |
| `--dest` | `predictions/eco/predictions.csv` | 预测结果保存路径 |

## 性能预期

大肠杆菌是模式生物，有高质量的基因注释和核糖体图谱数据，预期性能：

- **AUC**: >0.98
- **准确率**: >95%
- **召回率**: >90%

## 注意事项

1. **数据量大**: 大肠杆菌数据文件较大（总共约90MB），确保磁盘空间充足
2. **内存需求**: 训练时需要较大内存，建议使用GPU
3. **训练时间**: 由于数据量大，训练时间较长（可能数小时）
4. **早停策略**: 建议使用早停策略避免过拟合
5. **批次大小**: 根据GPU内存调整batch_size（256-512）

## 常见问题

### Q: 训练时间太长怎么办？

A:
- 使用GPU加速
- 增大batch_size（如果内存充足）
- 减少epoch数量
- 使用数据并行训练

### Q: 内存不足怎么办？

A:
- 减小`--batch_size`参数
- 使用混合精度训练（`--mixed_precision`）
- 关闭不必要的后台程序

### Q: 如何验证模型效果？

A:
1. 查看训练曲线（loss和AUC）
2. 在验证集上评估性能
3. 与已知基因注释对比
4. 计算统计指标（precision, recall, F1-score）

### Q: 可以用大肠杆菌模型预测其他菌种吗？

A: 不推荐。虽然都是肠道菌，但基因表达模式和调控机制存在差异，建议使用各自菌种训练的模型。

## 扩展应用

### 迁移学习

如果有其他大肠杆菌菌株的数据，可以用该模型作为预训练模型：

```bash
# 使用预训练权重初始化
python -m deepribo.cli.train \
    data/processed/ \
    --train_data other_eco_strain \
    --pretrained_model models/eco/model_epoch_20.pt \
    --epochs 10 \
    ...
```

### 多数据集训练

如果有多株大肠杆菌的数据，可以合并训练：

```bash
python -m deepribo.cli.train \
    data/processed/ \
    --train_data eco_strain1 eco_strain2 eco_strain3 \
    --valid_size 0.05 \
    ...
```

## 参考资料

- 原始DeepRibo论文：https://doi.org/10.1093/nar/gkz061
- E. coli K-12 MG1655基因组：https://www.ncbi.nlm.nih.gov/genome/167
- EcoCyc数据库：https://ecocyc.org/
