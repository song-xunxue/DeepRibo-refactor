# 天蓝色链霉菌 (Streptomyces coelicolor)

放线菌门，基因组约8.7 Mb，线性染色体，高GC含量（~72%），重要的抗生素生产菌。

**原始数据**: 21 MB（中等规模，适合功能测试）

## 解析数据

```bash
python -m src.cli.data raw/sco/sco_cov_sense.bedgraph raw/sco/sco_cov_asense.bedgraph raw/sco/sco_elo_sense.bedgraph raw/sco/sco_elo_asense.bedgraph raw/sco/sco.fa ../DeepRibo-data/processed/sco -g raw/sco/sco.gff
```

## 训练模型

> cutoff=0.15，过滤后约 3846 个样本（正=187, 负=3659）。
> 已验证：best model valid AUC=0.9734（epoch 9）。

```bash
python -m src.cli.train ../DeepRibo-data/processed --train_data sco -r 0.15 -c 0.15 -e 20 -b 256 -d models/sco --GPU
```

## 训练模型（显式验证集划分）

> 显式指定 `--valid_size 0.1`（10%），增大验证集比例以获得更可靠的验证指标。

```bash
python -m src.cli.train ../DeepRibo-data/processed --train_data sco -r 0.15 -c 0.15 -e 20 -b 256 -d models/sco-valid10 --valid_size 0.1 --GPU
```

## 预测

```bash
python -m src.cli.predict ../DeepRibo-data/processed --pred_data sco -r 0.15 -c 0.15 --GPU
```
