# Security Detection: 入侵检测与对抗防御系统

本项目围绕“数据采集—威胁检测—对抗防御—安全评估—可视化展示”的闭环，实现核心代码与训练流程：

- 数据集：支持以 CSV 为输入（如 CICIDS2017、UNSW-NB15 的处理后表格）。
- 模型层：提供 CNN、BiLSTM、Transformer 三种基础检测模型。
- 攻防层：实现 FGSM、PGD 攻击与对抗训练流程。
- 评估层：提供 Accuracy、Precision、Recall、F1、ROC-AUC 与混淆矩阵。
- 应用层：后续将加入 Flask + Vue + ECharts 的可视化平台（暂未启用）。

## 目录结构
```
security_detection/
├── attacks/            # 对抗攻击（FGSM/PGD）
├── defense/            # 防御（对抗训练）
├── models/             # 检测模型（CNN/BiLSTM/Transformer）
├── utils/              # 数据加载与评估指标
├── train.py            # 训练与评估主脚本
├── evaluate.py         # 预留的评估脚本（待拓展）
└── data/               # 数据目录（raw/processed）
```

## 环境准备（Conda）
若使用独立环境，建议：

```
conda env create -f environment.yml
conda activate security_detection
```

现有解释器：`D:\anaconda\envs\bqs1\python.exe`。

## 数据准备
- 将处理后的 CSV 放到 `data/processed/` 下，或自定义路径。
- 标签列名（如 `Label` 或 `label`）请与你的数据一致。

## 训练与评估
示例（普通训练）：
```
python train.py --csv data/processed/unsw_nb15.csv --label Label --model cnn --epochs 10 --batch_size 256
```
开启对抗训练（FGSM）：
```
python train.py --csv data/processed/unsw_nb15.csv --label Label --model bilstm --attack fgsm --epsilon 0.02 --epochs 5
```
开启对抗训练（PGD）：
```
python train.py --csv data/processed/cicids2017.csv --label Label --model transformer --attack pgd --epsilon 0.02 --alpha 0.005 --iters 10 --epochs 5
```

输出将包含指标与混淆矩阵：
- Metrics: accuracy / precision / recall / f1 / roc_auc
- Confusion Matrix

## 注意事项
- 若为二分类，`train.py` 会使用 `BCEWithLogitsLoss`（单输出）；多分类使用 `CrossEntropyLoss`。
- `utils/dataset.py` 默认对特征做标准化（`StandardScaler`），按 stratify 切分 train/val/test。
- 攻击强度 `epsilon` 与步长 `alpha` 需结合数据尺度调整；标准化后一般无需 clamp。
- 后续可在 `evaluate.py` 中加入更详细的对比评估与可视化导出。

## 下一步（可视化平台）
- 增加 Flask 后端：提供实时检测与评估接口。
- 前端使用 Vue + ECharts 可视化流量、风险评分与威胁等级。
