"""
Transformer模型定义
用于拉曼光谱混合物成分识别和浓度预测
"""
import torch
import torch.nn as nn
import os
import sys

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from models.positional_encoding import PositionalEncoding


class SpectrumTransformer(nn.Module):
    """基于Encoder-only Transformer的光谱分析模型"""
    
    def __init__(self,
                 input_dim=1738,
                 num_components=5,
                 d_model=128,
                 nhead=8,
                 num_layers=4,
                 dim_feedforward=512,
                 dropout=0.1,
                 max_len=2000,
                 activation='gelu'):
        """
        初始化Transformer模型
        
        Args:
            input_dim: 输入光谱维度
            num_components: 药物成分数量
            d_model: 模型维度
            nhead: 注意力头数
            num_layers: 编码器层数
            dim_feedforward: 前馈网络维度
            dropout: Dropout比率
            max_len: 最大序列长度
            activation: 激活函数
        """
        super(SpectrumTransformer, self).__init__()
        
        self.input_dim = input_dim
        self.num_components = num_components
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        
        # 输入嵌入层：将1738维光谱映射到d_model维
        self.embedding = nn.Linear(1, d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True  # 使用batch_first格式
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 全局池化：将序列表示聚合为单个向量
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # 成分分类头（多标签分类）
        self.classification_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_components),
            nn.Sigmoid()  # 多标签分类使用Sigmoid
        )
        
        # 浓度回归头（多目标回归）
        self.regression_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_components),
            nn.Sigmoid()  # 浓度范围在[0, 1]
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x, return_attention=False):
        """
        前向传播
        
        Args:
            x: 输入光谱 (batch_size, input_dim)
            return_attention: 是否返回注意力权重
        
        Returns:
            component_pred: 成分预测 (batch_size, num_components)
            concentration_pred: 浓度预测 (batch_size, num_components)
            attention_weights: 注意力权重（如果return_attention=True）
        """
        batch_size = x.size(0)
        
        # 将输入reshape为序列格式 (batch_size, seq_len, 1)
        x = x.unsqueeze(-1)  # (batch_size, input_dim, 1)
        
        # 嵌入 (batch_size, input_dim, d_model)
        x = self.embedding(x)
        
        # 添加位置编码
        x = self.pos_encoder(x)
        
        # Transformer编码器
        # 注意：如果需要获取注意力权重，需要修改这里
        encoded = self.transformer_encoder(x)  # (batch_size, input_dim, d_model)
        
        # 全局池化：(batch_size, d_model, input_dim) -> (batch_size, d_model, 1) -> (batch_size, d_model)
        pooled = self.global_pool(encoded.transpose(1, 2)).squeeze(-1)
        
        # 成分分类
        component_pred = self.classification_head(pooled)
        
        # 浓度回归
        concentration_pred = self.regression_head(pooled)
        
        if return_attention:
            # 获取第一层的注意力权重（简化版）
            # 实际应用中可以获取所有层的注意力
            attention_weights = None  # 需要修改TransformerEncoder来提取
            return component_pred, concentration_pred, attention_weights
        
        return component_pred, concentration_pred
    
    def get_model_info(self):
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        embedding_params = sum(p.numel() for p in self.embedding.parameters())
        encoder_params = sum(p.numel() for p in self.transformer_encoder.parameters())
        classification_params = sum(p.numel() for p in self.classification_head.parameters())
        regression_params = sum(p.numel() for p in self.regression_head.parameters())
        
        info = {
            'model_name': 'SpectrumTransformer',
            'input_dim': self.input_dim,
            'num_components': self.num_components,
            'd_model': self.d_model,
            'nhead': self.nhead,
            'num_layers': self.num_layers,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'embedding_parameters': embedding_params,
            'encoder_parameters': encoder_params,
            'classification_parameters': classification_params,
            'regression_parameters': regression_params,
        }
        return info


def create_model(config_dict=None):
    """
    创建Transformer模型的工厂函数
    
    Args:
        config_dict: 配置字典，如果为None则使用默认配置
    
    Returns:
        model: SpectrumTransformer模型实例
    """
    if config_dict is None:
        config_dict = config.MODEL_CONFIG
    
    model = SpectrumTransformer(
        input_dim=config.INPUT_DIM,
        num_components=config.NUM_COMPONENTS,
        d_model=config_dict['d_model'],
        nhead=config_dict['nhead'],
        num_layers=config_dict['num_layers'],
        dim_feedforward=config_dict['dim_feedforward'],
        dropout=config_dict['dropout'],
        max_len=config_dict['max_len'],
        activation=config_dict['activation']
    )
    
    return model


if __name__ == '__main__':
    # 测试模型
    print("测试Transformer模型...")
    
    # 创建模型
    model = create_model()
    model.eval()
    
    # 打印模型信息
    info = model.get_model_info()
    print("\n模型信息:")
    for key, value in info.items():
        print(f"  {key}: {value:,}" if isinstance(value, int) else f"  {key}: {value}")
    
    # 测试前向传播
    batch_size = 4
    test_input = torch.randn(batch_size, config.INPUT_DIM)
    
    with torch.no_grad():
        component_pred, concentration_pred = model(test_input)
    
    print(f"\n前向传播测试:")
    print(f"  输入形状: {test_input.shape}")
    print(f"  成分预测形状: {component_pred.shape}")
    print(f"  浓度预测形状: {concentration_pred.shape}")
    print(f"  成分预测示例: {component_pred[0].numpy()}")
    print(f"  浓度预测示例: {concentration_pred[0].numpy()}")
    
    print("\n模型结构:")
    print(model)


