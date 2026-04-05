# 枯草芽孢杆菌 (Bacillus subtilis)

革兰氏阳性菌，基因组约4.2 Mb，环形染色体，菌株 ASM904v1。

**原始数据**: 2.4 MB（bedgraph信号文件仅数KB，为最小测试集）

## 解析数据

```bash
python -m src.cli.data raw/bac/bac_cov_sense.bedgraph raw/bac/bac_cov_asense.bedgraph raw/bac/bac_elo_sense.bedgraph raw/bac/bac_elo_asense.bedgraph raw/bac/bac.fa ../DeepRibo-data/processed/bac -g raw/bac/bac.gff
```

## 训练模型

```bash
python -m src.cli.train ../DeepRibo-data/processed --train_data bac -r 0.12 -c 0.12 -e 20 -b 256 -d models/bac --GPU
```

## 预测

```bash
python -m src.cli.predict ../DeepRibo-data/processed --pred_data bac -r 0.12 -c 0.15 -M models/bac/{时间戳目录}/model_epoch_20.pt -d predictions/bac/predictions.csv --GPU
```
