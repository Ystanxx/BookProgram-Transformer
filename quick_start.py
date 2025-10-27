"""
快速开始脚本
演示数据生成、模型创建、位置编码和简单训练
"""
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 导入项目模块
import config
from models import create_model
from data import RamanMixtureGenerator


def demo_data_generation():
    """演示数据生成"""
    print("=" * 80)
    print("1. 数据生成演示")
    print("=" * 80)
    
    # 创建数据生成器
    generator = RamanMixtureGenerator(num_mixtures=10)
    
    # 生成数据
    spectra, component_labels, concentrations = generator.generate_dataset()
    
    print(f"\n生成的数据:")
    print(f"  光谱形状: {spectra.shape}")
    print(f"  成分标签形状: {component_labels.shape}")
    print(f"  浓度形状: {concentrations.shape}")
    print(f"  光谱范围: [{spectra.min():.3f}, {spectra.max():.3f}]")
    
    # 显示前3个混合物的成分
    print(f"\n前3个混合物的成分:")
    for i in range(3):
        active_idx = np.where(component_labels[i] == 1)[0]
        print(f"\n  混合物{i+1}:")
        for idx in active_idx:
            print(f"    {config.COMPONENT_NAMES[idx]}: {concentrations[i, idx]:.3f}")
    
    return spectra, component_labels, concentrations


def demo_model_creation():
    """演示模型创建"""
    print("\n" + "=" * 80)
    print("2. 模型创建演示")
    print("=" * 80)
    
    # 创建模型
    model = create_model()
    
    # 打印模型信息
    model_info = model.get_model_info()
    print(f"\n模型信息:")
    for key, value in model_info.items():
        print(f"  {key}: {value:,}" if isinstance(value, int) else f"  {key}: {value}")
    
    return model


def demo_forward_pass(model, spectra):
    """演示前向传播"""
    print("\n" + "=" * 80)
    print("3. 前向传播演示")
    print("=" * 80)
    
    # 设置为评估模式
    model.eval()
    model = model.to(config.DEVICE)
    
    # 选择几个样本
    batch_size = 4
    batch_spectra = spectra[:batch_size]
    
    # 转换为tensor
    batch_tensor = torch.FloatTensor(batch_spectra).to(config.DEVICE)
    
    print(f"\n输入形状: {batch_tensor.shape}")
    
    # 前向传播
    with torch.no_grad():
        component_pred, concentration_pred = model(batch_tensor)
    
    print(f"成分预测形状: {component_pred.shape}")
    print(f"浓度预测形状: {concentration_pred.shape}")
    print(f"\n成分预测示例（样本1）: {component_pred[0].cpu().numpy()}")
    print(f"浓度预测示例（样本1）: {concentration_pred[0].cpu().numpy()}")


def demo_simple_training():
    """演示简单的训练循环"""
    print("\n" + "=" * 80)
    print("4. 简单训练演示 (5个epoch)")
    print("=" * 80)
    
    # 生成小量数据
    generator = RamanMixtureGenerator(num_mixtures=20)
    spectra, component_labels, concentrations = generator.generate_dataset()
    
    # 简单预处理
    from utils.preprocessing import preprocess_spectrum
    spectra = preprocess_spectrum(spectra, 'poly', 'minmax', degree=3)
    
    # 创建模型
    model = create_model()
    model = model.to(config.DEVICE)
    
    # 转换为tensor
    X = torch.FloatTensor(spectra).to(config.DEVICE)
    Y_comp = torch.FloatTensor(component_labels).to(config.DEVICE)
    Y_conc = torch.FloatTensor(concentrations).to(config.DEVICE)
    
    # 定义优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 训练
    model.train()
    losses = []
    
    print("\n开始训练...")
    for epoch in range(1, 6):
        optimizer.zero_grad()
        
        # 前向传播
        comp_pred, conc_pred = model(X)
        
        # 计算损失
        from utils.metrics import calculate_loss
        loss, class_loss, reg_loss = calculate_loss(comp_pred, Y_comp, conc_pred, Y_conc)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        print(f"Epoch {epoch}/5 - Loss: {loss.item():.4f} (Class: {class_loss.item():.4f}, Reg: {reg_loss.item():.4f})")
    
    print(f"\n训练完成！损失从 {losses[0]:.4f} 降到 {losses[-1]:.4f}")


def main():
    """主函数"""
    print("欢迎使用Transformer拉曼光谱混合物分析系统")
    print(f"使用设备: {config.DEVICE}\n")
    
    # 1. 数据生成演示
    spectra, component_labels, concentrations = demo_data_generation()
    
    # 2. 模型创建演示
    model = demo_model_creation()
    
    # 3. 前向传播演示
    demo_forward_pass(model, spectra)
    
    # 4. 简单训练演示
    demo_simple_training()
    
    print("\n" + "=" * 80)
    print("快速开始演示完成！")
    print("=" * 80)
    print("\n下一步:")
    print("  1. 查看 '使用指南.md' 了解详细使用方法")
    print("  2. 运行完整训练、预测和可视化脚本")


if __name__ == '__main__':
    main()


