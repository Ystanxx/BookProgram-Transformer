# Transformer拉曼光谱混合物分析

## 项目简介

本项目实现了基于**Encoder-only Transformer**的拉曼光谱混合物分析系统，用于**药物成分识别和浓度预测**的多任务学习。

### 核心特点

- **Transformer架构**: Encoder-only结构，利用自注意力机制捕捉光谱特征
- **位置编码**: 正弦余弦位置编码，保留波长位置信息
- **多任务学习**: 同时进行成分识别（多标签分类）和浓度预测（多目标回归）
- **药物混合物**: 支持阿司匹林、对乙酰氨基酚、咖啡因、布洛芬、萘普生等药物成分
- **数据预处理**: 基线校正+归一化

### 模型架构

```
输入 (1738维拉曼光谱)
    ↓
嵌入层 (1 → d_model)
    ↓
位置编码 (Positional Encoding)
    ↓
Transformer Encoder (4层)
├─ Multi-Head Self-Attention (8头)
├─ Add & Norm
├─ Feed-Forward Network
└─ Add & Norm
    ↓
全局池化
    ↓
输出层
├─ 成分分类头 (多标签分类)
└─ 浓度回归头 (多目标回归)
```

## 项目结构

```
Transformer模型/
│
├── config.py                 # 配置文件
├── quick_start.py            # 快速开始脚本
├── requirements.txt          # Python依赖包
├── README.md                 # 项目说明（本文件）
├── 使用指南.md               # 详细使用指南
├── .gitignore                # Git配置
│
├── models/                   # 模型定义
│   ├── __init__.py
│   ├── transformer.py        # Transformer模型
│   └── positional_encoding.py # 位置编码
│
├── data/                     # 数据处理
│   ├── __init__.py
│   └── data_generator.py     # 拉曼光谱数据生成器
│
├── utils/                    # 工具函数
│   ├── __init__.py
│   ├── dataset.py            # PyTorch数据集
│   ├── metrics.py            # 评估指标
│   └── preprocessing.py      # 数据预处理
│
├── saved_models/             # 保存的模型
├── results/                  # 结果输出
└── logs/                     # 日志文件
```

## 快速开始

### 1. 环境配置

详细的安装步骤请参考 **[使用指南.md](使用指南.md)**

```bash
# 安装依赖
pip install -r requirements.txt
```

### 2. 快速演示

```bash
# 运行快速开始脚本
python quick_start.py
```

## 技术细节

### 数据集

- **混合物数量**: 80种
- **光谱维度**: 1738个采样点 (400-2000 cm⁻¹)
- **药物成分**: 5种（阿司匹林、对乙酰氨基酚、咖啡因、布洛芬、萘普生）
- **混合范围**: 每个混合物包含1-3种成分
- **数据类型**: 模拟拉曼光谱（带基线漂移和噪声）

### 模型配置

- **d_model**: 128 (模型维度)
- **nhead**: 8 (注意力头数)
- **num_layers**: 4 (编码器层数)
- **dim_feedforward**: 512 (前馈网络维度)
- **dropout**: 0.1
- **总参数**: 810,506

### 任务

1. **成分识别** (多标签分类)
   - 识别混合物中包含哪些药物成分
   - 损失函数: Binary Cross Entropy (BCE)

2. **浓度预测** (多目标回归)
   - 预测每种成分的相对浓度
   - 损失函数: Mean Squared Error (MSE)

### 评估指标

**分类指标**:
- Exact Match Ratio (所有成分完全正确的比例)
- Hamming Loss
- Precision, Recall, F1-score

**回归指标**:
- MSE, RMSE, MAE
- R² (决定系数)

## 系统要求

- **操作系统**: Windows / Linux / macOS
- **Python版本**: 3.8, 3.9, 或 3.10
- **内存**: 建议 ≥4GB
- **硬盘空间**: 约500MB
- **GPU**: 可选（默认使用CPU）

## 详细文档

请参阅 **[使用指南.md](使用指南.md)** 获取：

- 详细的环境安装步骤（Conda / 本地Python）
- 完整的使用教程
- 配置说明和参数调整
- 常见问题解答
- 错误排查指南

## 与其他项目对比

| 特性 | CNN | RNN/LSTM | Autoencoder | Transformer |
|------|-----|----------|-------------|-------------|
| **学习类型** | 监督 | 监督 | 无监督 | 监督 |
| **数据类型** | 拉曼 | NIR | NIR | 拉曼混合物 |
| **输入维度** | 1015 | 700 | 737 | 1738 |
| **任务** | 分类+回归 | 回归 | 降维+重构 | 成分识别+浓度 |
| **模型特色** | 局部特征 | 时序依赖 | 数据压缩 | 自注意力 |

## 作者与引用

本项目为教学示例，展示了Transformer在光谱分析中的应用。

如果本项目对您的研究或教学有帮助，欢迎引用和分享。

## 许可证

本项目仅供学习和研究使用。

