# 沙门氏菌 (Salmonella enterica)

革兰氏阴性菌，基因组约4.9 Mb，环形染色体，血清型 Typhimurium，菌株 SL1344。

**原始数据**: 2.3 MB（bedgraph信号文件仅数KB，为最小测试集）

## 解析数据

```bash
python -m src.cli.data raw/sal/sal_cov_sense.bedgraph raw/sal/sal_cov_asense.bedgraph raw/sal/sal_elo_sense.bedgraph raw/sal/sal_elo_asense.bedgraph raw/sal/sal.fa ../DeepRibo-data/processed/sal -g raw/sal/sal.gff
```

## 训练模型

```bash
python -m src.cli.train ../DeepRibo-data/processed --train_data sal -r 0.12 -c 0.12 -e 20 -b 256 -d models/sal --GPU
```

## 预测

```bash
python -m src.cli.predict ../DeepRibo-data/processed --pred_data sal -r 0.12 -c 0.15 -M models/sal/{时间戳目录}/model_epoch_20.pt -d predictions/sal/predictions.csv --GPU
```
