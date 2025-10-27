"""
评估指标
包含多标签分类和多目标回归的评估指标
"""
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, hamming_loss
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import sys

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def calculate_loss(component_pred, component_true, concentration_pred, concentration_true, 
                   class_weight=1.0, reg_weight=1.0):
    """
    计算多任务损失
    
    Args:
        component_pred: 成分预测 (batch_size, num_components)
        component_true: 真实成分 (batch_size, num_components)
        concentration_pred: 浓度预测 (batch_size, num_components)
        concentration_true: 真实浓度 (batch_size, num_components)
        class_weight: 分类任务权重
        reg_weight: 回归任务权重
    
    Returns:
        total_loss: 总损失
        class_loss: 分类损失
        reg_loss: 回归损失
    """
    # 多标签分类损失（BCE）
    bce_criterion = nn.BCELoss()
    class_loss = bce_criterion(component_pred, component_true)
    
    # 多目标回归损失（MSE），只计算实际存在的成分
    # 使用mask只计算component_true==1的位置
    mask = component_true > 0.5
    if mask.sum() > 0:
        mse_criterion = nn.MSELoss()
        reg_loss = mse_criterion(concentration_pred[mask], concentration_true[mask])
    else:
        reg_loss = torch.tensor(0.0).to(component_pred.device)
    
    # 总损失
    total_loss = class_weight * class_loss + reg_weight * reg_loss
    
    return total_loss, class_loss, reg_loss


class MultiTaskMetrics:
    """多任务评估指标计算器"""
    
    def __init__(self, threshold=0.5):
        """
        初始化指标计算器
        
        Args:
            threshold: 分类阈值
        """
        self.threshold = threshold
        self.reset()
    
    def reset(self):
        """重置所有累积的数据"""
        self.component_preds = []
        self.component_trues = []
        self.concentration_preds = []
        self.concentration_trues = []
    
    def update(self, component_pred, component_true, concentration_pred, concentration_true):
        """
        更新指标
        
        Args:
            component_pred: 成分预测
            component_true: 真实成分
            concentration_pred: 浓度预测
            concentration_true: 真实浓度
        """
        # 转换为numpy数组
        if torch.is_tensor(component_pred):
            component_pred = component_pred.detach().cpu().numpy()
        if torch.is_tensor(component_true):
            component_true = component_true.detach().cpu().numpy()
        if torch.is_tensor(concentration_pred):
            concentration_pred = concentration_pred.detach().cpu().numpy()
        if torch.is_tensor(concentration_true):
            concentration_true = concentration_true.detach().cpu().numpy()
        
        self.component_preds.append(component_pred)
        self.component_trues.append(component_true)
        self.concentration_preds.append(concentration_pred)
        self.concentration_trues.append(concentration_true)
    
    def compute(self):
        """
        计算所有指标
        
        Returns:
            metrics: 包含所有指标的字典
        """
        if len(self.component_preds) == 0:
            return {}
        
        # 合并所有batch
        component_pred = np.concatenate(self.component_preds, axis=0)
        component_true = np.concatenate(self.component_trues, axis=0)
        concentration_pred = np.concatenate(self.concentration_preds, axis=0)
        concentration_true = np.concatenate(self.concentration_trues, axis=0)
        
        # 二值化分类预测
        component_pred_binary = (component_pred > self.threshold).astype(int)
        component_true_binary = component_true.astype(int)
        
        # ==================== 分类指标 ====================
        # 计算每个样本的准确率（所有成分都正确才算正确）
        exact_match = np.all(component_pred_binary == component_true_binary, axis=1).mean()
        
        # Hamming Loss（错误预测的成分比例）
        hamming = hamming_loss(component_true_binary, component_pred_binary)
        
        # 其他分类指标（macro平均）
        precision = precision_score(component_true_binary, component_pred_binary, average='macro', zero_division=0)
        recall = recall_score(component_true_binary, component_pred_binary, average='macro', zero_division=0)
        f1 = f1_score(component_true_binary, component_pred_binary, average='macro', zero_division=0)
        
        # ==================== 回归指标 ====================
        # 只计算实际存在的成分的浓度
        mask = component_true > 0.5
        if mask.sum() > 0:
            valid_pred = concentration_pred[mask]
            valid_true = concentration_true[mask]
            
            mse = mean_squared_error(valid_true, valid_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(valid_true, valid_pred)
            r2 = r2_score(valid_true, valid_pred)
        else:
            mse = rmse = mae = r2 = 0.0
        
        metrics = {
            # 分类指标
            'exact_match_ratio': float(exact_match),
            'hamming_loss': float(hamming),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            # 回归指标
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2),
        }
        
        return metrics
    
    def compute_and_reset(self):
        """计算指标并重置"""
        metrics = self.compute()
        self.reset()
        return metrics
    
    def print_metrics(self, metrics, prefix=''):
        """
        打印指标
        
        Args:
            metrics: 指标字典
            prefix: 打印前缀
        """
        print(f"{prefix}分类指标:")
        print(f"{prefix}  Exact Match: {metrics['exact_match_ratio']:.4f}")
        print(f"{prefix}  Hamming Loss: {metrics['hamming_loss']:.4f}")
        print(f"{prefix}  Precision: {metrics['precision']:.4f}")
        print(f"{prefix}  Recall: {metrics['recall']:.4f}")
        print(f"{prefix}  F1-score: {metrics['f1_score']:.4f}")
        print(f"{prefix}回归指标:")
        print(f"{prefix}  MSE: {metrics['mse']:.6f}")
        print(f"{prefix}  RMSE: {metrics['rmse']:.6f}")
        print(f"{prefix}  MAE: {metrics['mae']:.6f}")
        print(f"{prefix}  R²: {metrics['r2']:.4f}")


def evaluate_model(model, dataloader, device):
    """
    评估模型性能
    
    Args:
        model: 模型
        dataloader: 数据加载器
        device: 设备
    
    Returns:
        metrics: 评估指标字典
        total_loss: 平均总损失
        class_loss: 平均分类损失
        reg_loss: 平均回归损失
    """
    model.eval()
    metrics_calculator = MultiTaskMetrics()
    total_loss_sum = 0.0
    class_loss_sum = 0.0
    reg_loss_sum = 0.0
    num_batches = 0
    
    class_weight = config.TASK_WEIGHTS['classification']
    reg_weight = config.TASK_WEIGHTS['regression']
    
    with torch.no_grad():
        for spectra, component_labels, concentrations in dataloader:
            spectra = spectra.to(device)
            component_labels = component_labels.to(device)
            concentrations = concentrations.to(device)
            
            # 前向传播
            component_pred, concentration_pred = model(spectra)
            
            # 计算损失
            total_loss, class_loss, reg_loss = calculate_loss(
                component_pred, component_labels,
                concentration_pred, concentrations,
                class_weight, reg_weight
            )
            
            total_loss_sum += total_loss.item()
            class_loss_sum += class_loss.item()
            reg_loss_sum += reg_loss.item()
            num_batches += 1
            
            # 更新指标
            metrics_calculator.update(
                component_pred, component_labels,
                concentration_pred, concentrations
            )
    
    # 计算平均损失
    avg_total_loss = total_loss_sum / num_batches if num_batches > 0 else 0.0
    avg_class_loss = class_loss_sum / num_batches if num_batches > 0 else 0.0
    avg_reg_loss = reg_loss_sum / num_batches if num_batches > 0 else 0.0
    
    # 计算指标
    metrics = metrics_calculator.compute()
    
    return metrics, avg_total_loss, avg_class_loss, avg_reg_loss


if __name__ == '__main__':
    # 测试指标计算
    print("测试多任务指标计算...")
    
    # 创建模拟数据
    np.random.seed(42)
    batch_size = 10
    num_components = 5
    
    # 模拟多标签分类
    component_true = np.random.rand(batch_size, num_components)
    component_true = (component_true > 0.7).astype(float)
    component_pred = component_true + np.random.normal(0, 0.1, (batch_size, num_components))
    component_pred = np.clip(component_pred, 0, 1)
    
    # 模拟多目标回归
    concentration_true = np.random.rand(batch_size, num_components) * component_true
    concentration_pred = concentration_true + np.random.normal(0, 0.05, (batch_size, num_components))
    concentration_pred = np.clip(concentration_pred, 0, 1)
    
    # 计算指标
    metrics_calc = MultiTaskMetrics()
    metrics_calc.update(component_pred, component_true, concentration_pred, concentration_true)
    metrics = metrics_calc.compute()
    
    print("\n指标结果:")
    metrics_calc.print_metrics(metrics, prefix="  ")


