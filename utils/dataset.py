"""
PyTorch数据集类定义
"""
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import sys

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.preprocessing import preprocess_spectrum


class RamanDataset(Dataset):
    """拉曼光谱混合物数据集类"""
    
    def __init__(self, spectra, component_labels, concentrations, 
                 preprocess=True, transform=None):
        """
        初始化数据集
        
        Args:
            spectra: 拉曼光谱数据 (N, input_dim)
            component_labels: 成分标签 (N, num_components)
            concentrations: 浓度 (N, num_components)
            preprocess: 是否进行预处理
            transform: 数据变换函数
        """
        if preprocess:
            # 预处理光谱
            baseline_method = config.PREPROCESSING_CONFIG.get('baseline_method', 'poly')
            norm_method = config.PREPROCESSING_CONFIG.get('normalization', 'minmax')
            poly_degree = config.PREPROCESSING_CONFIG.get('poly_degree', 3)
            
            spectra = preprocess_spectrum(
                spectra,
                baseline_correction_method=baseline_method if config.PREPROCESSING_CONFIG['baseline_correction'] else None,
                normalization_method=norm_method,
                degree=poly_degree
            )
        
        self.spectra = torch.FloatTensor(spectra)
        self.component_labels = torch.FloatTensor(component_labels)
        self.concentrations = torch.FloatTensor(concentrations)
        self.transform = transform
        
    def __len__(self):
        return len(self.spectra)
    
    def __getitem__(self, idx):
        """
        获取单个样本
        
        Returns:
            spectrum: 光谱数据
            component_label: 成分标签
            concentration: 浓度
        """
        spectrum = self.spectra[idx]
        component_label = self.component_labels[idx]
        concentration = self.concentrations[idx]
        
        if self.transform:
            spectrum = self.transform(spectrum)
        
        return spectrum, component_label, concentration


class RamanDataModule:
    """数据模块，管理数据加载和预处理"""
    
    def __init__(self, data_dir, batch_size=16, num_workers=0):
        """
        初始化数据模块
        
        Args:
            data_dir: 数据目录
            batch_size: 批次大小
            num_workers: 数据加载工作进程数
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # 加载数据
        self.load_data()
        
    def load_data(self):
        """加载数据文件"""
        from data.data_generator import load_or_generate_data
        
        self.spectra, self.component_labels, self.concentrations = load_or_generate_data(
            self.data_dir,
            num_mixtures=config.NUM_MIXTURES,
            regenerate=False
        )
        
        print(f"\n数据加载完成:")
        print(f"  光谱形状: {self.spectra.shape}")
        print(f"  成分标签形状: {self.component_labels.shape}")
        print(f"  浓度形状: {self.concentrations.shape}")
    
    def get_train_val_test_dataloaders(self, train_ratio=0.7, val_ratio=0.15, random_state=42):
        """
        获取训练集、验证集和测试集的数据加载器
        
        Args:
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            random_state: 随机种子
        
        Returns:
            train_loader, val_loader, test_loader
        """
        # 首先划分训练集和临时集（验证+测试）
        temp_ratio = val_ratio + (1 - train_ratio - val_ratio)
        train_spectra, temp_spectra, train_labels, temp_labels, train_conc, temp_conc = train_test_split(
            self.spectra, self.component_labels, self.concentrations,
            test_size=temp_ratio,
            random_state=random_state
        )
        
        # 然后将临时集划分为验证集和测试集
        val_size = val_ratio / temp_ratio
        val_spectra, test_spectra, val_labels, test_labels, val_conc, test_conc = train_test_split(
            temp_spectra, temp_labels, temp_conc,
            test_size=(1 - val_size),
            random_state=random_state
        )
        
        # 创建数据集（预处理在Dataset初始化时完成）
        train_dataset = RamanDataset(train_spectra, train_labels, train_conc, preprocess=True)
        val_dataset = RamanDataset(val_spectra, val_labels, val_conc, preprocess=True)
        test_dataset = RamanDataset(test_spectra, test_labels, test_conc, preprocess=True)
        
        # 创建数据加载器
        use_pin_memory = (config.DEVICE.type == 'cuda')
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=use_pin_memory
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=use_pin_memory
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=use_pin_memory
        )
        
        print(f"\n数据集划分:")
        print(f"  训练集: {len(train_dataset)} 样本")
        print(f"  验证集: {len(val_dataset)} 样本")
        print(f"  测试集: {len(test_dataset)} 样本")
        
        return train_loader, val_loader, test_loader


if __name__ == '__main__':
    # 测试数据集类
    data_module = RamanDataModule(
        data_dir=config.DATA_DIR,
        batch_size=config.BATCH_SIZE
    )
    
    # 测试数据划分
    train_loader, val_loader, test_loader = data_module.get_train_val_test_dataloaders()
    
    # 测试一个batch
    for spectra, component_labels, concentrations in train_loader:
        print(f"\nBatch测试:")
        print(f"  光谱形状: {spectra.shape}")
        print(f"  成分标签形状: {component_labels.shape}")
        print(f"  浓度形状: {concentrations.shape}")
        print(f"  光谱范围: [{spectra.min():.3f}, {spectra.max():.3f}]")
        print(f"  成分标签示例: {component_labels[0].numpy()}")
        print(f"  浓度示例: {concentrations[0].numpy()}")
        break

