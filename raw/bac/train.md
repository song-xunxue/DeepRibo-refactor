# 枯草芽孢杆菌 (Bacillus subtilis) - 训练说明

## 菌种信息

- **学名**: Bacillus subtilis
- **菌株**: ASM904v1
- **分类**: 革兰氏阳性菌
- **基因组大小**: 约4.2 Mb
- **染色体数量**: 1条环形染色体

## 数据文件

本项目包含枯草芽孢杆菌的以下数据文件：

| 文件名 | 说明 | 文件大小 |
|--------|------|----------|
| `bac.fa` | 基因组序列（FASTA格式） | 6.2 KB |
| `bac.gff` | 基因注释（GFF3格式） | 2.4 MB |
| `bac_cov_sense.bedgraph` | 正义链覆盖度信号 | 839 B |
| `bac_cov_asense.bedgraph` | 反义链覆盖度信号 | 882 B |
| `bac_elo_sense.bedgraph` | 正义链延伸信号 | 768 B |
| `bac_elo_asense.bedgraph` | 反义链延伸信号 | 882 B |

## 数据解析

### 步骤1：激活环境

```bash
conda activate deepribo-refactor
```

### 步骤2：运行数据解析

```bash
python -m deepribo.cli.data \
    raw/bac/bac_cov_sense.bedgraph \
    raw/bac/bac_cov_asense.bedgraph \
    raw/bac/bac_elo_sense.bedgraph \
    raw/bac/bac_elo_asense.bedgraph \
    raw/bac/bac.fa \
    data/processed/ \
    --gtf raw/bac/bac.gff \
    --start_trips ATG GTG TTG \
    --stop_trips TAA TGA TAG
```

### 解析后数据结构

解析完成后，会在`data/processed/bac/`下创建以下结构：

```
data/processed/bac/
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
    --train_data bac \
    --valid_size 0.05 \
    --rpkm 0.12 0.15 \
    --coverage 0.12 0.15 \
    --dest models/bac/ \
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

### 参数说明

| 参数 | 值 | 说明 |
|------|---|------|
| `--train_data` | `bac` | 训练数据集名称 |
| `--valid_size` | `0.05` | 验证集比例（5%） |
| `--rpkm` | `0.12 0.15` | RPKM过滤阈值 |
| `--coverage` | `0.12 0.15` | 覆盖率过滤阈值 |
| `--dest` | `models/bac/` | 模型保存路径 |
| `--batch_size` | `256` | 批次大小 |
| `--epochs` | `20` | 训练轮数 |
| `--GRU_nodes` | `128` | GRU隐藏层大小 |
| `--GRU_layers` | `2` | GRU层数 |
| `--COV_motifs` | `32` | CNN卷积核数量 |
| `--FC_nodes` | `1024 512` | 全连接层节点数 |
| `--model_type` | `CNNRNN` | 模型类型 |

### 输出文件

训练完成后，模型文件保存在`models/bac/`目录，每次训练会创建一个以时间戳命名的文件夹：

```
models/bac/
└── 2026-4-1-20-42/          # 时间戳文件夹（示例）
    ├── model_epoch_1.pt     # 第1个epoch的模型
    ├── model_epoch_2.pt     # 第2个epoch的模型
    ├── ...
    ├── model_epoch_20.pt    # 第20个epoch的模型
    ├── metrics_epoch_0.json # 第1个epoch的指标
    ├── metrics_epoch_1.json # 第2个epoch的指标
    ├── ...
    └── metrics_epoch_19.json # 第20个epoch的指标
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
    --pred_data bac \
    --rpkm 0.12 \
    --coverage 0.15 \
    --model models/bac/2026-4-1-20-42/model_epoch_20.pt \
    --dest predictions/bac/predictions.csv \
    --GRU_nodes 128 \
    --GRU_layers 2 \
    --GRU_bidirect \
    --COV_motifs 32 \
    --FC_nodes 1024 512 \
    --model_type CNNRNN \
    --GPU
```

**注意**：`--model` 参数需要指定完整的时间戳路径，如 `models/bac/2026-4-1-20-42/model_epoch_20.pt`

### 参数说明

| 参数 | 值 | 说明 |
|------|---|------|
| `--pred_data` | `bac` | 预测数据集名称 |
| `--model` | `models/bac/model_epoch_20.pt` | 预训练模型路径 |
| `--dest` | `predictions/bac/predictions.csv` | 预测结果保存路径 |

## 性能预期

根据文献和实验经验，枯草芽孢杆菌的DeepRibo模型预期性能：

- **AUC**: >0.95
- **准确率**: >90%
- **召回率**: >85%

## 注意事项

1. 枯草芽孢杆菌基因组较小，训练速度相对较快
2. 建议使用较小的batch_size（128-256）以获得更好的收敛性
3. 如果数据量较少，可以适当减少epoch数量
4. 关注验证集AUC，选择最佳模型进行预测

## 常见问题

### Q: 如何选择最佳模型？

A: 查看各个epoch的`metrics_epoch_N.json`文件，选择验证集AUC最高的模型。

### Q: 训练过程中出现内存不足怎么办？

A: 减小`--batch_size`参数，或使用CPU训练。

### Q: 可以使用其他菌种的模型进行预测吗？

A: 不建议。不同菌种的模式特征不同，应该使用对应菌种训练的模型。

## 参考资料

- 原始DeepRibo论文：https://doi.org/10.1093/nar/gkz061
- Bacillus subtilis基因组数据库：https://www.ncbi.nlm.nih.gov/genome/224
