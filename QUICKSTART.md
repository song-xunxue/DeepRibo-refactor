# DeepRibo 快速开始

## 安装环境

```bash
conda env create -f configs/environment.yml
conda activate DeepRibo-refactor
pip install -e .
```

## 数据目录

解析后的数据存放在同级目录 `../DeepRibo-data/processed/` 下，与代码仓库分离，避免编辑器卡顿。

### 数据集总览

| 菌种 | 代号 | 原始数据 | 基因组 | 特点 |
|------|------|----------|--------|------|
| 枯草芽孢杆菌 | bac | 2.4 MB | 4.2 Mb | 最小测试集，秒级解析 |
| 沙门氏菌 | sal | 2.3 MB | 4.9 Mb | 最小测试集，秒级解析 |
| 天蓝色链霉菌 | sco | 21 MB | 8.7 Mb | 中等规模，适合功能测试 |
| 大肠杆菌 | eco | 90 MB | 4.6 Mb | 最大数据集，建议GPU |

建议先用 bac 或 sal 验证流程，再跑 sco、eco。

```
DeepRibo-data/                # 数据目录（与代码仓库同级）
└── processed/
    ├── bac/                   # 各菌种解析数据
    ├── eco/
    ├── sal/
    └── sco/
```

## 使用流程

选择要训练的菌种，参考对应的 `raw/{菌种}/train.md` 执行命令：

- `raw/bac/train.md` — 枯草芽孢杆菌
- `raw/eco/train.md` — 大肠杆菌
- `raw/sal/train.md` — 沙门氏菌
- `raw/sco/train.md` — 天蓝色链霉菌

每个 train.md 包含该菌种的解析、训练、预测三条命令，直接复制执行即可。

## 参数说明

以下参数适用于所有菌种，无需在每个 train.md 中重复。

### 数据解析参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `sense_cov` | 正义链覆盖度文件 | 必需 |
| `asense_cov` | 反义链覆盖度文件 | 必需 |
| `sense_elo` | 正义链延伸信号文件 | 必需 |
| `asense_elo` | 反义链延伸信号文件 | 必需 |
| `fasta` | 基因组FASTA文件 | 必需 |
| `destination` | 输出目录 | 必需 |
| `-g` / `--gtf` | GFF注释文件 | None |
| `-s` / `--start_trips` | 起始密码子 | ATG GTG TTG |
| `-p` / `--stop_trips` | 终止密码子 | TAA TGA TAG |

### 训练参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `data_path` | 解析后数据根目录 | 必需 |
| `--train_data` | 训练数据集名称 | 必需 |
| `-r` / `--rpkm` | 最小RPK过滤阈值 | 必需 |
| `-c` / `--coverage` | 最小覆盖率过滤阈值 | 必需 |
| `-d` / `--dest` | 模型保存路径 | models/trained |
| `-e` / `--epochs` | 训练轮数 | 20 |
| `-b` / `--batch_size` | 批次大小 | 256 |
| `--valid_size` | 验证集比例 | 0.05 |
| `-g` / `--GRU_nodes` | GRU隐藏层大小 | 128 |
| `-l` / `--GRU_layers` | GRU层数 | 2 |
| `-B` / `--GRU_bidirect` | 双向GRU | True |
| `-m` / `--COV_motifs` | CNN卷积核数量 | 32 |
| `-n` / `--FC_nodes` | 全连接层节点数 | 1024 512 |
| `--model_type` | 模型架构 | CNNRNN |
| `--GPU` | 使用GPU | False |
| `--test_data` | 测试数据集 | None |
| `-rt` / `--rpkm_test` | 测试集RPK阈值 | None |
| `-ct` / `--coverage_test` | 测试集覆盖率阈值 | None |

### 预测参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `data_path` | 解析后数据根目录 | 必需 |
| `--pred_data` | 预测数据集名称 | 必需 |
| `-r` / `--rpkm` | 最小RPK过滤阈值 | 必需 |
| `-c` / `--coverage` | 最小覆盖率过滤阈值 | 必需 |
| `-M` / `--model` | 模型文件路径 | 必需 |
| `-d` / `--dest` | 预测结果保存路径 | 必需 |
| `-g` / `--GRU_nodes` | GRU隐藏层大小 | 128 |
| `-l` / `--GRU_layers` | GRU层数 | 2 |
| `-m` / `--COV_motifs` | CNN卷积核数量 | 32 |
| `-n` / `--FC_nodes` | 全连接层节点数 | 1024 512 |
| `--model_type` | 模型架构 | CNNRNN |
| `--GPU` | 使用GPU | False |

## 输出结构

训练完成后，模型按时间戳保存：

```
models/{菌种}/
└── 2026-4-5-14-30/              # 时间戳目录
    ├── model_epoch_1.pt         # 各epoch模型
    ├── metrics_epoch_0.json     # 各epoch指标
    └── ...
```

预测结果保存在 `predictions/{菌种}/predictions.csv`。

### DeepRibo 的预测目标

DeepRibo 利用核糖体图谱（Ribo-seq）数据和核糖体结合位点序列模式，预测原核生物基因组中的**开放阅读框（ORF）**，特别是识别：

- **翻译起始位点（TIS）**：确定基因的正确起始密码子位置
- **蛋白质变体（Proteoforms）**：与已注释基因共享终止密码子但具有不同起始位点的ORF，即同一基因的替代翻译起始变体
- **全新蛋白质（Novel proteins）**：在已注释基因区域之外发现的全新ORF，包括大量**小开放阅读框（sORFs）**
- **反义链ORF**：在基因组反义链上发现的翻译事件

模型结合两种输入：CNN处理起始密码子附近30nt的DNA序列（覆盖Shine-Dalgarno基序区域），RNN（双向GRU）处理覆盖整个ORF的核糖体图谱信号。

### 预测结果字段说明

| 字段 | 说明 |
|------|------|
| `filename` | 序列特征文件路径（`.pt` 张量），用于CNN的30nt结合位点序列 |
| `filename_counts` | 测序覆盖度特征文件路径（`.pt` 张量），用于RNN的核糖体图谱信号 |
| `label` | 基因组注释标签。`True`=已注释基因（正样本），`False`=候选新ORF（负样本） |
| `in_gene` | 该ORF是否位于已注释基因内部（重叠基因区域的常见情况） |
| `strand` | 所在链方向。`+`=正义链，`-`=反义链 |
| `coverage` | 核糖体图谱覆盖度（原始值），表示ORF中有信号覆盖的碱基比例 |
| `coverage_elo` | 核糖体图谱覆盖度（elo标准化值） |
| `rpk` | 每千碱基读数（原始值），衡量ORF区域的表达水平 |
| `rpk_elo` | 每千碱基读数（elo标准化值） |
| `start_site` | 预测的翻译起始位点（TIS）坐标 |
| `start_codon` | 起始密码子（ATG/GTG/TTG，细菌几乎只使用这三种） |
| `stop_site` | 终止位点坐标 |
| `stop_codon` | 终止密码子（TAA/TGA/TAG） |
| `locus` | 基因座标签，格式 `染色体:起始-终止` |
| `prot_seq` | 翻译后的蛋白质序列 |
| `nuc_seq` | 核苷酸序列 |
| **`pred`** | **模型预测分数**（softmax输出的正类概率），值越大表示该ORF越可能是真实表达的基因 |
| `pred_rank` | 全局预测排名（按pred降序），0=最可能为真实基因 |
| `SS` | **单起始位点选择（Single Start）**。`True`表示该ORF是其终止位点下预测概率最高的起始位点（后处理结果） |
| `dist` | 与最近已注释基因TIS的距离。`0`=与注释一致；`>0`=在注释TIS下游；`<0`=在注释TIS上游；`-1`=该区域无已注释基因 |
| `SS_pred_rank` | 在所有SS=True的ORF中按pred排序的排名（越小越可信，999999=非最优起始位点） |

### 解读指南

#### 1. 单起始位点（SS）后处理

原核生物基因组中，同一终止密码子上游可能存在多个候选起始密码子（89.5%的候选ORF与其他ORF共享终止位点）。后处理步骤对每组共享终止位点的ORF，仅保留预测概率最高的起始位点（`SS=True`），与基因组注释惯例一致（每个终止位点仅注释一个TIS）。

#### 2. 发现新基因

筛选 `SS=True` 且 `label=False` 的行，按 `SS_pred_rank` 升序排列，这些是模型最确信的新发现ORF。

#### 3. 区分蛋白质变体和全新蛋白质

- **蛋白质变体（Proteoform）**：`label=True` 且 `dist != 0` 和 `dist != -1`，表示与已注释基因共享终止密码子但起始位点不同，可能是替代翻译起始变体
- **全新蛋白质（Novel protein）**：`dist = -1`，表示该终止位点区域无任何已注释基因，是完全新的发现

#### 4. 小开放阅读框（sORFs）

DeepRibo特别擅长发现传统序列比对方法遗漏的sORFs。论文中，新发现ORF的中位长度远短于已注释基因（如E. coli中270nt vs 827nt），这些短蛋白在基因组注释中经常被忽略。

#### 5. 预测阈值建议

论文中将正预测数量设为与已注释基因数量相等（top-k策略）。实际使用时：
- `pred` 越高越可信，可根据所需的精确率/召回率平衡自行设定阈值
- 建议优先关注 `SS=True` 且 `SS_pred_rank` 靠前的预测结果
