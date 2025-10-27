# Transformer拉曼光谱混合物分析

> 基于Encoder-only Transformer的拉曼光谱药物成分识别与浓度预测系统

## 项目简介

本项目实现了基于**Transformer**的拉曼光谱混合物分析系统，用于**药物成分识别和浓度预测**。通过自注意力机制捕捉光谱特征，实现端到端的多任务学习。

**主要特点**：
- ✅ Transformer架构：Encoder-only结构，8头自注意力机制
- ✅ 位置编码：正弦余弦位置编码，保留波长位置信息
- ✅ 多任务学习：同时进行成分识别（多标签分类）和浓度预测（多目标回归）
- ✅ 端到端训练：从原始光谱到成分和浓度的直接预测
- ✅ 完整的中文文档：详细的配置说明和使用指南

## 快速开始

### 环境要求

- **Python**: 3.8 - 3.11
- **操作系统**: Windows / Linux / macOS
- **内存**: 建议4GB以上
- **GPU**: 可选（默认使用CPU）

### 安装步骤

**1. 创建环境**

```bash
conda create -n use python=3.11 -y
conda activate use
```

**2. 安装依赖**

```bash
pip install -r requirements.txt
```

**3. 快速测试**

```bash
python quick_start.py
```

### 使用示例

```bash
# 快速演示（数据生成、模型创建、训练演示）:)
python quick_start.py
```


详细使用说明请参考 **[使用指南.md](使用指南.md)**

## 项目结构

```
Transformer模型/
├── config.py              # 配置文件
├── quick_start.py         # 快速演示
├── models/                # 模型定义
│   ├── transformer.py     # Transformer模型
│   └── positional_encoding.py # 位置编码
├── data/                  # 数据处理
│   └── data_generator.py  # 数据生成器
├── utils/                 # 工具函数
│   ├── dataset.py         # 数据集类
│   ├── metrics.py         # 评估指标
│   └── preprocessing.py   # 数据预处理
├── requirements.txt       # 依赖列表
├── docs/                  # 文档目录
│   └── 深度学习项目文档模板.md
├── 环境配置.md            # 环境配置指南
├── 使用指南.md            # 使用指南
└── README.md              # 本文件
```

## 模型性能

| 任务 | 指标 | 值 | 说明 |
|------|------|-----|------|
| **成分识别** | Exact Match Ratio | - | 所有成分完全正确的比例 |
|  | Hamming Loss | - | 平均错误标签比例 |
|  | F1-score | - | 精确率和召回率的调和平均 |
| **浓度预测** | RMSE | - | 均方根误差 |
|  | MAE | - | 平均绝对误差 |
|  | R² | - | 决定系数 |
| **模型参数** | 总参数量 | 810,506 | 约81万参数 |

*运行quick_start.py可查看实际性能*

## 详细文档

- 📖 **[环境配置.md](环境配置.md)** - 详细的环境搭建指南
- 📖 **[使用指南.md](使用指南.md)** - 完整的项目使用文档

## 模型架构

```
输入 (1738维拉曼光谱)
    ↓
嵌入层 (1 → 128维)
    ↓
位置编码 (正弦余弦编码)
    ↓
Transformer Encoder × 4层
    ├─ 多头自注意力 (8头)
    ├─ Add & LayerNorm
    ├─ 前馈网络 (512维)
    └─ Add & LayerNorm
    ↓
全局平均池化
    ↓
双分支输出
    ├─ 成分分类头 → 5维 (多标签分类)
    └─ 浓度回归头 → 5维 (多目标回归)
```

**关键参数**：
- `d_model`: 128（模型维度）
- `nhead`: 8（注意力头数）
- `num_layers`: 4（编码器层数）
- `dim_feedforward`: 512（前馈网络维度）
- `dropout`: 0.1（Dropout率）

## 数据说明

### 拉曼光谱数据
- **光谱维度**: 1738个采样点
- **波数范围**: 400-2000 cm⁻¹
- **混合物数量**: 80种（可配置）
- **成分范围**: 每个混合物包含1-3种成分

### 药物成分
1. 阿司匹林（Aspirin）
2. 对乙酰氨基酚（Acetaminophen）
3. 咖啡因（Caffeine）
4. 布洛芬（Ibuprofen）
5. 萘普生（Naproxen）

## 适用场景

- 药物成分快速识别
- 混合物定量分析
- 光谱数据深度学习应用
- Transformer在科学数据中的应用
- 多任务学习研究

## 技术栈

- **深度学习框架**: PyTorch 1.10+
- **数值计算**: NumPy, SciPy
- **数据处理**: scikit-learn
- **可视化**: Matplotlib
- **进度显示**: tqdm

## 与其他模型对比

| 特性 | CNN | LSTM | Autoencoder | **Transformer** |
|------|-----|------|-------------|-----------------|
| **架构特点** | 局部特征提取 | 序列建模 | 特征压缩 | **自注意力机制** |
| **全局依赖** | 弱 | 中 | 弱 | **强** |
| **并行计算** | 高 | 低 | 高 | **高** |
| **位置信息** | 隐式 | 显式 | 无 | **显式编码** |
| **多任务** | 支持 | 支持 | 受限 | **原生支持** |
| **本项目应用** | - | - | - | **成分识别+浓度预测** |

## 常见问题

### 如何切换GPU/CPU？

修改`config.py`：
```python
DEVICE_MODE = 'auto'  # 自动检测
DEVICE_MODE = 'cpu'   # 强制CPU
DEVICE_MODE = 'cuda'  # 强制GPU
```

### 如何调整模型复杂度？

修改`config.py`中的`MODEL_CONFIG`：
```python
MODEL_CONFIG = {
    'd_model': 128,        # 减小以降低复杂度
    'nhead': 8,            # 需是d_model的因子
    'num_layers': 4,       # 减少层数加快训练
    'dim_feedforward': 512,
    'dropout': 0.1,
}
```

### 如何增加数据量？

修改`config.py`：
```python
NUM_MIXTURES = 160  # 增加混合物数量
```

更多问题请参考 **[使用指南.md](使用指南.md)** 的"常见问题"章节。

## 系统要求

- **操作系统**: Windows 10/11 / Linux / macOS
- **Python**: 3.8 - 3.11
- **内存**: 建议4GB以上
- **硬盘**: 约500MB
- **GPU**: 可选（自动检测）

## 许可

本项目仅供学习和研究使用。

## 致谢

本项目为教学示例，展示了Transformer在光谱分析中的应用。感谢PyTorch团队提供的深度学习框架。

---

**版本**: v1.1  
**更新日期**: 2025-10-27  
**适用领域**: 光谱分析、深度学习、多任务学习
