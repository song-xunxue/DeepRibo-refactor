# 沙门氏菌 (Salmonella enterica) - 训练说明

## 菌种信息

- **学名**: Salmonella enterica
- **血清型**: Typhimurium
- **菌株**: SL1344
- **分类**: 革兰氏阴性菌
- **基因组大小**: 约4.9 Mb
- **染色体数量**: 1条环形染色体
- **基因组版本**: GCA_000210855.2

## 数据文件

本项目包含沙门氏菌的以下数据文件：

| 文件名 | 说明 | 文件大小 |
|--------|------|----------|
| `sal.fa` | 基因组序列（FASTA格式） | 6.1 KB |
| `sal.gff` | 基因注释（GFF3格式） | 2.3 MB |
| `sal_cov_sense.bedgraph` | 正义链覆盖度信号 | 746 B |
| `sal_cov_asense.bedgraph` | 反义链覆盖度信号 | 810 B |
| `sal_elo_sense.bedgraph` | 正义链延伸信号 | 724 B |
| `sal_elo_asense.bedgraph` | 反义链延伸信号 | 818 B |

## 数据解析

### 步骤1：激活环境

```bash
conda activate deepribo-refactor
```

### 步骤2：运行数据解析

```bash
python -m deepribo.cli.data \
    raw/sal/sal_cov_sense.bedgraph \
    raw/sal/sal_cov_asense.bedgraph \
    raw/sal/sal_elo_sense.bedgraph \
    raw/sal/sal_elo_asense.bedgraph \
    raw/sal/sal.fa \
    data/processed/ \
    --gtf raw/sal/sal.gff \
    --start_trips ATG GTG TTG \
    --stop_trips TAA TGA TAG
```

### 解析后数据结构

解析完成后，会在`data/processed/sal/`下创建以下结构：

```
data/processed/sal/
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
    --train_data sal \
    --valid_size 0.05 \
    --rpkm 0.12 0.15 \
    --coverage 0.12 0.15 \
    --dest models/sal/ \
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
| `--train_data` | `sal` | 训练数据集名称 |
| `--valid_size` | `0.05` | 验证集比例（5%） |
| `--rpkm` | `0.12 0.15` | RPKM过滤阈值 |
| `--coverage` | `0.12 0.15` | 覆盖率过滤阈值 |
| `--dest` | `models/sal/` | 模型保存路径 |
| `--batch_size` | `256` | 批次大小 |
| `--epochs` | `20` | 训练轮数 |
| `--GRU_nodes` | `128` | GRU隐藏层大小 |
| `--GRU_layers` | `2` | GRU层数 |
| `--COV_motifs` | `32` | CNN卷积核数量 |
| `--FC_nodes` | `1024 512` | 全连接层节点数 |
| `--model_type` | `CNNRNN` | 模型类型 |

### 输出文件

训练完成后，模型文件保存在`models/sal/`目录，每次训练会创建一个以时间戳命名的文件夹：

```
models/sal/
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
    --pred_data sal \
    --rpkm 0.12 \
    --coverage 0.15 \
    --model models/sal/2026-4-1-20-42/model_epoch_20.pt \
    --dest predictions/sal/predictions.csv \
    --GRU_nodes 128 \
    --GRU_layers 2 \
    --GRU_bidirect \
    --COV_motifs 32 \
    --FC_nodes 1024 512 \
    --model_type CNNRNN \
    --GPU
```

**注意**：`--model` 参数需要指定完整的时间戳路径，如 `models/sal/2026-4-1-20-42/model_epoch_20.pt`

### 参数说明

| 参数 | 值 | 说明 |
|------|---|------|
| `--pred_data` | `sal` | 预测数据集名称 |
| `--model` | `models/sal/2026-4-1-20-42/model_epoch_20.pt` | 预训练模型路径（包含时间戳） |
| `--dest` | `predictions/sal/predictions.csv` | 预测结果保存路径 |

## 性能预期

沙门氏菌与大肠杆菌亲缘关系较近，预期性能良好：

- **AUC**: >0.97
- **准确率**: >93%
- **召回率**: >88%

## 注意事项

1. **亲缘关系**: 沙门氏菌与大肠杆菌亲缘关系较近，可以考虑使用迁移学习
2. **基因组大小**: 沙门氏菌基因组较大（约4.9Mb），训练时间可能较长
3. **数据量**: 信号文件较小，确保数据完整性
4. **批次大小**: 可以使用与枯草芽孢杆菌相似的批次大小
5. **早停策略**: 建议监控验证集性能，避免过拟合

## 常见问题

### Q: 沙门氏菌和大肠杆菌可以使用相同模型吗？

A: 虽然亲缘关系较近，但不同菌种的基因调控机制存在差异，建议各自训练专用模型。

### Q: 如何进行跨菌种迁移学习？

A: 可以用训练好的大肠杆菌模型作为预训练模型：

```bash
python -m deepribo.cli.train \
    data/processed/ \
    --train_data sal \
    --pretrained_model models/eco/model_epoch_20.pt \
    --epochs 10 \
    ...
```

这样可以加速收敛并提升性能。

### Q: 训练不稳定怎么办？

A:
- 检查学习率，适当调小
- 使用梯度裁剪
- 增加批次归一化层
- 确保数据预处理正确

### Q: 如何评估预测结果的生物学意义？

A:
1. 检查预测ORF的起始密码子使用（ATG/GTG/TTG）
2. 分析Shine-Dalgarno序列特征
3. 与已知注释对比
4. 检查核糖体图谱信号强度

## 扩展应用

### 多血清型训练

如果有不同血清型的沙门氏菌数据，可以构建多血清型模型：

```bash
python -m deepribo.cli.train \
    data/processed/ \
    --train_data sal_typhimurium sal_enteritidis \
    --valid_size 0.05 \
    ...
```

### 病原体特异性研究

沙门氏菌是重要病原体，可以研究：
- 毒力基因的识别
- 抗生素抗性基因的注释
- 宿主适应性基因的预测

## 生物学意义

沙门氏菌（Salmonella enterica serovar Typhimurium）的研究价值：

1. **重要病原体**: 引起人类和动物的沙门氏菌病
2. **模式生物**: 广泛用于细胞内感染机制研究
3. **肠道菌群**: 与宿主免疫系统的相互作用
4. **基因调控**: 复杂的应激反应和毒力调控网络

## 参考资料

- 原始DeepRibo论文：https://doi.org/10.1093/nar/gkz061
- Salmonella enterica Typhimurium LT2基因组：https://www.ncbi.nlm.nih.gov/genome/115
- Salmonella数据库：https://www.salmonellagenome.org/
- SL1344菌株研究：https://www.ncbi.nlm.nih.gov/bioproject/39929
