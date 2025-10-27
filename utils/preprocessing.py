"""
数据预处理工具
包含基线校正和归一化函数
"""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def baseline_als(y, lam=1e5, p=0.01, niter=10):
    """
    使用非对称最小二乘法（ALS）进行基线校正
    
    Args:
        y: 输入光谱
        lam: 平滑参数
        p: 非对称参数
        niter: 迭代次数
    
    Returns:
        baseline: 拟合的基线
    """
    L = len(y)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L-2))
    w = np.ones(L)
    
    for i in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = spsolve(Z, w*y)
        w = p * (y > z) + (1-p) * (y < z)
    
    return z


def baseline_poly(y, degree=3):
    """
    使用多项式拟合进行基线校正
    
    Args:
        y: 输入光谱
        degree: 多项式阶数
    
    Returns:
        baseline: 拟合的基线
    """
    x = np.arange(len(y))
    coeffs = np.polyfit(x, y, degree)
    baseline = np.polyval(coeffs, x)
    return baseline


def baseline_correction(spectrum, method='poly', degree=3, **kwargs):
    """
    基线校正
    
    Args:
        spectrum: 输入光谱 (N,) 或 (batch, N)
        method: 'poly' 或 'als'
        degree: 多项式阶数（method='poly'时使用）
        **kwargs: 其他参数
    
    Returns:
        corrected: 校正后的光谱
    """
    is_single = (spectrum.ndim == 1)
    if is_single:
        spectrum = spectrum.reshape(1, -1)
    
    corrected = np.zeros_like(spectrum)
    
    for i in range(spectrum.shape[0]):
        if method == 'poly':
            baseline = baseline_poly(spectrum[i], degree)
        elif method == 'als':
            baseline = baseline_als(spectrum[i], **kwargs)
        else:
            raise ValueError(f"未知的基线校正方法: {method}")
        
        corrected[i] = spectrum[i] - baseline
    
    if is_single:
        corrected = corrected.squeeze(0)
    
    return corrected


def normalize_spectrum(spectrum, method='minmax'):
    """
    光谱归一化
    
    Args:
        spectrum: 输入光谱 (N,) 或 (batch, N)
        method: 'minmax', 'standard', 或 'none'
    
    Returns:
        normalized: 归一化后的光谱
    """
    if method == 'none':
        return spectrum
    
    is_single = (spectrum.ndim == 1)
    if is_single:
        spectrum = spectrum.reshape(1, -1)
    
    if method == 'minmax':
        # Min-Max归一化到[0, 1]
        spec_min = spectrum.min(axis=1, keepdims=True)
        spec_max = spectrum.max(axis=1, keepdims=True)
        spec_range = spec_max - spec_min
        
        # 避免除零
        spec_range[spec_range < 1e-10] = 1.0
        normalized = (spectrum - spec_min) / spec_range
        
    elif method == 'standard':
        # 标准化（零均值，单位方差）
        mean = spectrum.mean(axis=1, keepdims=True)
        std = spectrum.std(axis=1, keepdims=True)
        
        # 避免除零
        std[std < 1e-10] = 1.0
        normalized = (spectrum - mean) / std
    else:
        raise ValueError(f"未知的归一化方法: {method}")
    
    if is_single:
        normalized = normalized.squeeze(0)
    
    return normalized


def preprocess_spectrum(spectrum, baseline_correction_method='poly', 
                        normalization_method='minmax', **kwargs):
    """
    完整的光谱预处理流程
    
    Args:
        spectrum: 输入光谱
        baseline_correction_method: 基线校正方法
        normalization_method: 归一化方法
        **kwargs: 其他参数
    
    Returns:
        processed: 处理后的光谱
    """
    # 基线校正
    if baseline_correction_method is not None:
        spectrum = baseline_correction(spectrum, method=baseline_correction_method, **kwargs)
    
    # 归一化
    if normalization_method is not None:
        spectrum = normalize_spectrum(spectrum, method=normalization_method)
    
    return spectrum


if __name__ == '__main__':
    # 测试预处理函数
    print("测试预处理函数...")
    
    # 生成测试光谱（带基线漂移）
    x = np.linspace(0, 10, 1738)
    # 真实信号：几个高斯峰
    signal = np.exp(-(x-3)**2/0.5) + 0.5*np.exp(-(x-7)**2/0.3)
    # 基线漂移
    baseline = 0.5 + 0.1*x
    # 添加噪声
    noise = np.random.normal(0, 0.01, len(x))
    # 原始光谱
    spectrum = signal + baseline + noise
    
    # 基线校正
    corrected_poly = baseline_correction(spectrum, method='poly', degree=3)
    
    # 归一化
    normalized = normalize_spectrum(corrected_poly, method='minmax')
    
    print(f"\n光谱统计:")
    print(f"  原始光谱范围: [{spectrum.min():.3f}, {spectrum.max():.3f}]")
    print(f"  基线校正后范围: [{corrected_poly.min():.3f}, {corrected_poly.max():.3f}]")
    print(f"  归一化后范围: [{normalized.min():.3f}, {normalized.max():.3f}]")
    
    # 可视化
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(15, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(x, spectrum, 'b-', linewidth=1)
    plt.title('原始光谱（带基线漂移）')
    plt.xlabel('Raman Shift')
    plt.ylabel('Intensity')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.plot(x, corrected_poly, 'g-', linewidth=1)
    plt.title('基线校正后')
    plt.xlabel('Raman Shift')
    plt.ylabel('Intensity')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    plt.plot(x, normalized, 'r-', linewidth=1)
    plt.title('归一化后')
    plt.xlabel('Raman Shift')
    plt.ylabel('Intensity')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/preprocessing_test.png', dpi=150)
    print("\n预处理可视化已保存到: results/preprocessing_test.png")
    plt.close()
    
    print("\n预处理测试完成！")

