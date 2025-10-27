"""
位置编码模块
实现Transformer的正弦余弦位置编码
"""
import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """位置编码模块"""
    
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        """
        初始化位置编码
        
        Args:
            d_model: 模型维度
            max_len: 最大序列长度
            dropout: Dropout比率
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # 正弦和余弦函数
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 添加batch维度
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        # 注册为buffer，不会被视为模型参数
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量 (batch_size, seq_len, d_model)
        
        Returns:
            加上位置编码后的张量
        """
        # 添加位置编码
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


if __name__ == '__main__':
    # 测试位置编码
    print("测试位置编码模块...")
    
    d_model = 128
    max_len = 1738
    batch_size = 4
    
    # 创建位置编码
    pos_encoder = PositionalEncoding(d_model, max_len)
    
    # 创建测试输入
    test_input = torch.randn(batch_size, max_len, d_model)
    
    # 添加位置编码
    output = pos_encoder(test_input)
    
    print(f"\n输入形状: {test_input.shape}")
    print(f"输出形状: {output.shape}")
    print(f"位置编码矩阵形状: {pos_encoder.pe.shape}")
    
    # 可视化前几个位置的编码
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 6))
    
    # 显示前100个位置的所有维度
    pe_sample = pos_encoder.pe[0, :100, :].numpy()
    plt.imshow(pe_sample.T, cmap='RdBu', aspect='auto')
    plt.xlabel('Position')
    plt.ylabel('Dimension')
    plt.title('Positional Encoding Visualization (First 100 positions)')
    plt.colorbar()
    
    plt.tight_layout()
    plt.savefig('results/positional_encoding_test.png', dpi=150)
    print("\n位置编码可视化已保存到: results/positional_encoding_test.png")
    plt.close()
    
    print("\n位置编码测试完成！")

