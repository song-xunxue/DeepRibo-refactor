# 大肠杆菌 (Escherichia coli)

革兰氏阴性菌，基因组约4.6 Mb，环形染色体，菌株 K-12 MG1655。

**原始数据**: 90 MB（bedgraph信号文件各17-27 MB，为最大数据集，建议GPU训练）

## 解析数据

```bash
python -m src.cli.data raw/eco/eco_cov_sense.bedgraph raw/eco/eco_cov_asense.bedgraph raw/eco/eco_elo_sense.bedgraph raw/eco/eco_elo_asense.bedgraph raw/eco/eco.fa ../DeepRibo-data/processed/eco -g raw/eco/eco.gff
```

## 训练模型

```bash
python -m src.cli.train ../DeepRibo-data/processed --train_data eco -r 0.12 -c 0.12 -e 20 -b 256 -d models/eco --GPU
```

## 预测

```bash
python -m src.cli.predict ../DeepRibo-data/processed --pred_data eco -r 0.12 -c 0.15 -M models/eco/{时间戳目录}/model_epoch_20.pt -d predictions/eco/predictions.csv --GPU
```
