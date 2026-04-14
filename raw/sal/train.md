# 沙门氏菌 (Salmonella enterica)

革兰氏阴性菌，基因组约4.9 Mb，环形染色体，血清型 Typhimurium，菌株 SL1344。

**原始数据**: 2.3 MB（bedgraph信号文件仅数KB，为最小测试集）

**注意**: sal 数据集正样本极少（仅4个），属于极小测试集，训练效果有限。建议仅用于流程验证。

## 解析数据

```bash
python -m src.cli.data raw/sal/sal_cov_sense.bedgraph raw/sal/sal_cov_asense.bedgraph raw/sal/sal_elo_sense.bedgraph raw/sal/sal_elo_asense.bedgraph raw/sal/sal.fa ../DeepRibo-data/processed/sal -g raw/sal/sal.gff
```

## 训练模型

> cutoff 必须设为 0.0，否则正样本会被过滤掉导致训练失败（NaN）。
> 过滤后约 386 个样本（正=4, 负=382），极度不平衡。

```bash
python -m src.cli.train ../DeepRibo-data/processed --train_data sal -r 0.0 -c 0.0 -e 20 -b 256 -d models/sal --GPU
```

## 预测

```bash
python -m src.cli.predict ../DeepRibo-data/processed --pred_data sal -r 0.0 -c 0.0 --GPU
```
