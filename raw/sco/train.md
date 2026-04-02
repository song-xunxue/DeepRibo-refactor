# 天蓝色链霉菌 (Streptomyces coelicolor) - 训练说明

## 菌种信息

- **学名**: Streptomyces coelicolor
- **菌株**: A3(2) strain CFB_NBC_0001
- **分类**: 放线菌门（Actinobacteria）
- **基因组大小**: 约8.7 Mb
- **染色体数量**: 1条线性染色体
- **特点**: 重要的抗生素生产菌，次级代谢产物丰富
- **基因组版本**: GCA_000009225.1

## 数据文件

本项目包含天蓝色链霉菌的以下数据文件：

| 文件名 | 说明 | 文件大小 |
|--------|------|----------|
| `sco.fna` | 基因组序列（FASTA格式） | 8.4 MB |
| `sco.gff` | 基因注释（GFF3格式） | 3.5 MB |
| `sco_cov_sense.bedgraph` | 正义链覆盖度信号 | 2.8 MB |
| `sco_cov_asense.bedgraph` | 反义链覆盖度信号 | 2.8 MB |
| `sco_elo_sense.bedgraph` | 正义链延伸信号 | 1.7 MB |
| `sco_elo_asense.bedgraph` | 反义链延伸信号 | 1.7 MB |

**数据特点**：
- 基因组相对较大，适合作为中等规模测试数据集
- 放线菌具有独特的基因调控模式
- 丰富的次级代谢基因簇
- 高GC含量（约72%）

## 数据解析

### 步骤1：激活环境

```bash
conda activate deepribo-refactor
```

### 步骤2：运行数据解析

```bash
python -m src.cli.data \
    raw/sco/sco_cov_sense.bedgraph \
    raw/sco/sco_cov_asense.bedgraph \
    raw/sco/sco_elo_sense.bedgraph \
    raw/sco/sco_elo_asense.bedgraph \
    raw/sco/sco.fna \
    data/processed/ \
    --gtf raw/sco/sco.gff \
    --start_trips ATG GTG TTG \
    --stop_trips TAA TGA TAG
```

### 解析后数据结构

解析完成后，会在`data/processed/sco/`下创建以下结构：

```
data/processed/sco/
├── 0/                    # 负样本（非基因ORF）
├── 1/                    # 正样本（真实基因ORF）
└── data_list.csv        # 样本列表
```

### 解析参数说明

| 参数 | 值 | 说明 |
|------|---|------|
| `--start_trips` | `ATG GTG TTG` | 起始密码子（放线菌常用） |
| `--stop_trips` | `TAA TGA TAG` | 终止密码子 |
| `--gtf` | `raw/sco/sco.gff` | 基因注释文件路径 |

**注意事项**：
- 放线菌的GC含量高，可能影响序列特征
- 起始密码子偏好可能与普通细菌不同
- 建议先小规模测试解析结果

## 模型训练

### 基础训练命令

```bash
conda activate deepribo-refactor

python -m src.cli.train \
    data/processed/ \
    --train_data sco \
    --valid_size 0.05 \
    --rpkm 0.15 0.20 \
    --coverage 0.15 0.20 \
    --dest models/sco/ \
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

### 快速测试训练（5个epoch）

```bash
python -m src.cli.train \
    data/processed/ \
    --train_data sco \
    --valid_size 0.05 \
    --rpkm 0.15 0.20 \
    --coverage 0.15 0.20 \
    --dest models/sco/ \
    --batch_size 256 \
    --epochs 5 \
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
| `--train_data` | `sco` | 训练数据集名称 |
| `--valid_size` | `0.05` | 验证集比例（5%） |
| `--rpkm` | `0.15 0.20` | RPKM过滤阈值（放线菌可能需要更高） |
| `--coverage` | `0.15 0.20` | 覆盖率过滤阈值 |
| `--dest` | `models/sco/` | 模型保存路径 |
| `--batch_size` | `256` | 批次大小 |
| `--epochs` | `20` | 训练轮数（快速测试可用5） |
| `--GRU_nodes` | `128` | GRU隐藏层大小 |
| `--GRU_layers` | `2` | GRU层数 |
| `--COV_motifs` | `32` | CNN卷积核数量 |
| `--FC_nodes` | `1024 512` | 全连接层节点数 |
| `--model_type` | `CNNRNN` | 模型类型 |

### 放线菌特定参数调整建议

| 参数 | 建议值 | 原因 |
|------|--------|------|
| `--rpkm` | `0.15-0.25` | 放线菌表达模式可能不同 |
| `--coverage` | `0.15-0.25` | 考虑GC含量影响 |
| `--COV_motifs` | `32-48` | 可能需要更多motif捕捉特征 |
| `--epochs` | `25-30` | 大基因组可能需要更多训练 |

### 输出文件

训练完成后，模型文件保存在`models/sco/`目录，每次训练会创建一个以时间戳命名的文件夹：

```
models/sco/
└── 2026-4-2-16-30/          # 时间戳文件夹（示例）
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
- 文件夹格式：`年-月-日-小时-分钟`（如：2026-4-2-16-30）
- 同一次训练的所有epoch模型都保存在同一个时间戳文件夹内

## 模型预测

### 预测命令

```bash
conda activate deepribo-refactor

python -m src.cli.predict \
    data/processed/ \
    --pred_data sco \
    --rpkm 0.15 \
    --coverage 0.20 \
    --model models/sco/2026-4-2-16-30/model_epoch_20.pt \
    --dest predictions/sco/predictions.csv \
    --GRU_nodes 128 \
    --GRU_layers 2 \
    --GRU_bidirect \
    --COV_motifs 32 \
    --FC_nodes 1024 512 \
    --model_type CNNRNN \
    --GPU
```

**注意**：`--model` 参数需要指定完整的时间戳路径，如 `models/sco/2026-4-2-16-30/model_epoch_20.pt`

### 参数说明

| 参数 | 值 | 说明 |
|------|---|------|
| `--pred_data` | `sco` | 预测数据集名称 |
| `--model` | `models/sco/2026-4-2-16-30/model_epoch_20.pt` | 预训练模型路径 |
| `--dest` | `predictions/sco/predictions.csv` | 预测结果保存路径 |

## 性能预期

根据放线菌的特点和DeepRibo在其他原核生物上的表现，天蓝色链霉菌的预期性能：

- **AUC**: >0.93（放线菌可能略低于一般细菌）
- **准确率**: >88%
- **召回率**: >82%
- **特异性**: >85%

**影响因素**：
- 高GC含量可能影响序列特征学习
- 复杂的次级代谢调控可能增加预测难度
- 大基因组需要充分训练

## 注意事项

### 数据特异性
1. **高GC含量**: 天蓝色链霉菌GC含量约72%，可能影响序列编码和模型学习
2. **基因密度**: 放线菌基因密度相对较低，需要注意正负样本平衡
3. **长基因**: 次级代谢基因簇可能包含较长的ORF

### 训练策略
1. **监控验证集**: 密切关注验证集AUC变化，防止过拟合
2. **学习率调整**: 高GC含量可能需要调整学习率
3. **早停策略**: 建议设置早停，当验证集性能不再提升时停止训练

### 硬件需求
1. **内存需求**: 21MB数据集，建议至少8GB内存
2. **GPU推荐**: 有GPU可显著加速训练（约2-3小时）
3. **CPU训练**: 无GPU时，完整训练可能需要8-12小时

### 参数调优
1. **batch_size**: 可尝试128-512范围
2. **GRU_nodes**: 复杂基因调控可能需要更大的隐藏层（128-256）
3. **COV_motifs**: 放线菌可能需要更多motif（32-48）

## 常见问题

### Q: 为什么预测AUC比其他菌种低？

A: 放线菌有特殊的生物学特征：
- 高GC含量影响序列编码
- 复杂的次级代谢调控网络
- 基因密度较低
- 建议增加训练epochs和调整参数

### Q: 训练过程中出现梯度消失/爆炸？

A:
- 降低学习率
- 使用梯度裁剪
- 检查数据预处理
- 尝试不同的batch_size

### Q: 如何提高预测准确性？

A:
1. 增加训练epochs（25-30）
2. 调整RPKM和coverage阈值
3. 使用更大的模型（增加GRU_nodes和COV_motifs）
4. 尝试不同的model_type（CNN、RNN、CNNRNN）

### Q: 放线菌的Shine-Dalgarno序列有什么特点？

A:
- 放线菌的SD序列可能不如典型细菌保守
- 某些基因可能使用非传统的翻译起始机制
- 建议检查预测ORF的起始位点分布

## 放线菌特殊考虑

### 基因组特征
1. **线性染色体**: 与大多数细菌的环形染色体不同
2. **高GC含量**: 约72%，影响序列特征
3. **核心区与臂区**: 染色体两端含有次级代谢基因簇

### 次级代谢基因
1. **基因簇**: 抗生素合成基因通常成簇排列
2. **调控复杂**: 涉及多层次调控网络
3. **表达特异性**: 不同条件下表达模式差异大

### 训练建议
1. **数据筛选**: 重点关注高表达基因的训练
2. **特征工程**: 考虑GC含量相关的特征
3. **模型复杂度**: 可能需要更复杂的模型结构

## 扩展应用

### 抗生素生产研究
天蓝色链霉菌是重要的抗生素生产菌，可以应用：
- 抗生素合成基因簇的识别
- 生物合成基因的注释
- 调控元件的预测

### 次级代谢预测
- 新型天然产物基因簇发现
- 代谢途径重构
- 菌种改良靶点识别

### 比较基因组学
- 与其他链霉菌的比较分析
- 进化关系研究
- 基因水平转移检测

## 测试建议

作为测试数据集，建议：

### 快速验证测试
```bash
# 使用5个epoch快速验证功能
python -m src.cli.train \
    data/processed/ \
    --train_data sco \
    --epochs 5 \
    --batch_size 256 \
    --dest models/sco/test_run/
```

### 完整功能测试
```bash
# 完整训练和预测流程
# 1. 数据解析
python -m src.cli.data [参数...]

# 2. 模型训练（20 epochs）
python -m src.cli.train \
    data/processed/ \
    --train_data sco \
    --epochs 20 \
    --dest models/sco/

# 3. 模型预测
python -m src.cli.predict [参数...]
```

## 参考资料

- 原始DeepRibo论文：https://doi.org/10.1093/nar/gkz061
- Streptomyces coelicolor A3(2)基因组：https://www.ncbi.nlm.nih.gov/genome/196
- 放线菌数据库：https://actinobase.org/
- 链霉菌资源：https://streptomyces.org.uk/

## 与其他数据集对比

| 数据集 | 大小 | 基因组类型 | 用途 |
|--------|------|------------|------|
| sco | 21 MB | 放线菌，线性染色体 | 测试数据集，中等规模 |
| bac | 2.4 MB | 革兰氏阳性菌，环形染色体 | 小型测试集 |
| eco | 90 MB | 革兰氏阴性菌，环形染色体 | 大型生产数据集 |
| sal | 2.3 MB | 革兰氏阴性菌，环形染色体 | 小型测试集 |

**sco数据集优势**：
- 适中的数据量，测试时间合理
- 代表放线菌这一重要微生物类群
- 包含次级代谢基因簇，具有特殊生物学意义
- 适合作为中等规模的测试基准