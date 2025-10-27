"""
配置文件
定义Transformer拉曼光谱混合物分析系统的所有超参数和配置
"""
import os
import torch

# ==================== 数据配置 ====================
# 拉曼光谱数据配置
INPUT_DIM = 1738  # 拉曼光谱采样点数量
RAMAN_SHIFT_RANGE = (400, 2000)  # 拉曼位移范围 (cm⁻¹)
NUM_MIXTURES = 80  # 混合光谱组合数量

# 药物成分配置
COMPONENT_NAMES = [
    '阿司匹林',          # Aspirin
    '对乙酰氨基酚',      # Acetaminophen (Paracetamol)
    '咖啡因',            # Caffeine
    '布洛芬',            # Ibuprofen
    '萘普生',            # Naproxen
]
NUM_COMPONENTS = len(COMPONENT_NAMES)

# 浓度范围（归一化后的比例）
CONCENTRATION_MIN = 0.0
CONCENTRATION_MAX = 1.0

# ==================== 模型配置 ====================
# Transformer模型配置
MODEL_CONFIG = {
    'd_model': 128,              # 模型维度
    'nhead': 8,                  # 多头注意力头数
    'num_layers': 4,             # Transformer编码器层数
    'dim_feedforward': 512,      # 前馈网络维度
    'dropout': 0.1,              # Dropout比率
    'max_len': INPUT_DIM,        # 最大序列长度
    'activation': 'gelu',        # 激活函数
}

# ==================== 训练配置 ====================
# 基本训练参数
BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 1e-4

# 早停策略
EARLY_STOPPING_PATIENCE = 15

# 学习率调度
LR_SCHEDULER = {
    'type': 'ReduceLROnPlateau',
    'factor': 0.5,
    'patience': 8,
    'min_lr': 1e-6,
}

# 多任务学习权重
TASK_WEIGHTS = {
    'classification': 1.0,   # 成分识别权重
    'regression': 1.0,       # 浓度预测权重
}

# ==================== 数据集配置 ====================
# 数据划分比例
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# 模拟数据生成参数
DATA_CONFIG = {
    'num_mixtures': NUM_MIXTURES,
    'noise_level': 0.02,           # 噪声水平
    'baseline_drift': 0.1,         # 基线漂移程度
    'random_seed': 42,             # 随机种子
    'min_components': 1,           # 每个混合物的最少成分数
    'max_components': 3,           # 每个混合物的最多成分数
}

# 数据预处理配置
PREPROCESSING_CONFIG = {
    'baseline_correction': True,   # 是否进行基线校正
    'normalization': 'minmax',     # 归一化方法：'minmax', 'standard', 'none'
    'baseline_method': 'poly',     # 基线校正方法：'poly', 'als'
    'poly_degree': 3,              # 多项式拟合的阶数
}

# ==================== 路径配置 ====================
# 基础路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'saved_models')
LOG_DIR = os.path.join(BASE_DIR, 'logs')
RESULT_DIR = os.path.join(BASE_DIR, 'results')

# 创建必要的目录
for directory in [DATA_DIR, MODEL_DIR, LOG_DIR, RESULT_DIR]:
    os.makedirs(directory, exist_ok=True)

# 模型保存路径
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, 'transformer_best.pth')

# ==================== 设备配置 ====================
# 设备选择配置
DEVICE_MODE = 'cpu'  # 默认使用CPU以确保兼容性

def get_device():
    """智能检测并返回可用设备"""
    global DEVICE_MODE
    
    # 检查环境变量
    env_force_cpu = os.environ.get('FORCE_CPU', '').lower() in ('1', 'true', 'yes')
    if env_force_cpu:
        print("[*] 环境变量FORCE_CPU已设置，使用CPU模式")
        return torch.device('cpu')
    
    # 强制CPU模式
    if DEVICE_MODE == 'cpu':
        print("[*] 配置设定为CPU模式")
        return torch.device('cpu')
    
    # 强制CUDA模式
    if DEVICE_MODE == 'cuda':
        if torch.cuda.is_available():
            print("[*] 配置设定为CUDA模式")
            return torch.device('cuda')
        else:
            print("[!] 警告: CUDA不可用，切换到CPU模式")
            return torch.device('cpu')
    
    # 自动检测模式
    if torch.cuda.is_available():
        try:
            test_tensor = torch.zeros(1).cuda()
            del test_tensor
            torch.cuda.empty_cache()
            print("[√] CUDA兼容性测试通过，使用GPU加速")
            return torch.device('cuda')
        except Exception as e:
            print(f"\n[!] 警告: CUDA设备不兼容")
            print(f"   原因: {str(e)[:80]}...")
            print("   自动切换到CPU模式")
            print("   提示: 如需使用GPU，请确保PyTorch版本支持您的显卡")
            return torch.device('cpu')
    else:
        print("[*] CUDA不可用，使用CPU模式")
        return torch.device('cpu')

DEVICE = get_device()

# ==================== 可视化配置 ====================
PLOT_CONFIG = {
    'dpi': 150,
    'figsize': (12, 8),
    'save_format': 'png',
}

# ==================== 日志配置 ====================
LOG_CONFIG = {
    'log_interval': 5,
    'save_interval': 5,
}

print(f"配置加载完成，使用设备: {DEVICE}")
if DEVICE.type == 'cpu':
    print("[*] 提示: 如果您有兼容的GPU，可以在config.py中修改 DEVICE_MODE = 'auto' 来启用GPU加速")

