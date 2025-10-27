"""
拉曼光谱混合物数据生成器
生成药物成分拉曼光谱及其混合物
"""
import numpy as np
import os
import sys

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class RamanMixtureGenerator:
    """拉曼光谱混合物数据生成器"""
    
    def __init__(self, num_mixtures=80, random_seed=42):
        """
        初始化数据生成器
        
        Args:
            num_mixtures: 混合物数量
            random_seed: 随机种子
        """
        self.num_mixtures = num_mixtures
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        # 从配置文件获取参数
        self.input_dim = config.INPUT_DIM
        self.raman_shift_range = config.RAMAN_SHIFT_RANGE
        self.component_names = config.COMPONENT_NAMES
        self.num_components = config.NUM_COMPONENTS
        self.min_components = config.DATA_CONFIG['min_components']
        self.max_components = config.DATA_CONFIG['max_components']
        self.noise_level = config.DATA_CONFIG['noise_level']
        self.baseline_drift = config.DATA_CONFIG['baseline_drift']
        
        # 生成拉曼位移数组
        self.raman_shifts = np.linspace(
            self.raman_shift_range[0],
            self.raman_shift_range[1],
            self.input_dim
        )
        
        # 定义每种药物成分的特征峰位置（拉曼位移 cm⁻¹）
        self.component_peaks = {
            '阿司匹林': [650, 850, 1050, 1200, 1600],
            '对乙酰氨基酚': [630, 800, 970, 1170, 1560],
            '咖啡因': [670, 900, 1040, 1240, 1650],
            '布洛芬': [640, 820, 1000, 1190, 1580],
            '萘普生': [660, 880, 1030, 1220, 1620],
        }
    
    def _raman_shift_to_index(self, raman_shift):
        """将拉曼位移转换为数组索引"""
        idx = int((raman_shift - self.raman_shift_range[0]) / 
                  (self.raman_shift_range[1] - self.raman_shift_range[0]) * 
                  self.input_dim)
        return np.clip(idx, 0, self.input_dim - 1)
    
    def _generate_gaussian_peak(self, center, intensity, width=20):
        """
        生成高斯峰
        
        Args:
            center: 峰中心（拉曼位移）
            intensity: 峰强度
            width: 峰宽度
        
        Returns:
            peak: 高斯峰数组
        """
        center_idx = self._raman_shift_to_index(center)
        x = np.arange(self.input_dim)
        peak = intensity * np.exp(-((x - center_idx) ** 2) / (2 * width ** 2))
        return peak
    
    def generate_pure_component_spectrum(self, component_name):
        """
        生成单个药物成分的纯拉曼光谱
        
        Args:
            component_name: 药物成分名称
        
        Returns:
            spectrum: 拉曼光谱
        """
        spectrum = np.zeros(self.input_dim)
        
        # 获取该成分的特征峰位置
        peak_positions = self.component_peaks[component_name]
        
        # 添加特征峰
        for i, pos in enumerate(peak_positions):
            # 峰强度随位置变化
            intensity = np.random.uniform(0.5, 1.0) * (1 - 0.2 * i / len(peak_positions))
            width = np.random.uniform(15, 25)
            spectrum += self._generate_gaussian_peak(pos, intensity, width)
        
        # 添加一些次要峰
        num_minor_peaks = np.random.randint(3, 6)
        for _ in range(num_minor_peaks):
            pos = np.random.uniform(self.raman_shift_range[0] + 100, 
                                   self.raman_shift_range[1] - 100)
            intensity = np.random.uniform(0.05, 0.15)
            width = np.random.uniform(15, 30)
            spectrum += self._generate_gaussian_peak(pos, intensity, width)
        
        # 添加基线漂移
        baseline = self._generate_baseline()
        spectrum += baseline
        
        # 添加噪声
        noise = np.random.normal(0, self.noise_level, self.input_dim)
        spectrum += noise
        
        # 确保光谱值为正
        spectrum = np.maximum(spectrum, 0)
        
        return spectrum
    
    def _generate_baseline(self):
        """生成基线漂移"""
        x = np.linspace(0, 1, self.input_dim)
        # 多项式基线
        baseline = (self.baseline_drift * 
                   (0.5 + 0.3*x + 0.1*x**2 + 0.05*np.sin(5*np.pi*x)))
        return baseline
    
    def generate_mixture_spectrum(self, component_labels, concentrations):
        """
        生成混合光谱
        
        Args:
            component_labels: 成分标签 (num_components,)，二值向量
            concentrations: 浓度 (num_components,)
        
        Returns:
            mixture_spectrum: 混合光谱
        """
        mixture_spectrum = np.zeros(self.input_dim)
        
        for i, (label, conc) in enumerate(zip(component_labels, concentrations)):
            if label == 1 and conc > 0:
                # 生成该成分的光谱
                component_name = self.component_names[i]
                pure_spectrum = self.generate_pure_component_spectrum(component_name)
                # 按浓度加权
                mixture_spectrum += conc * pure_spectrum
        
        # 添加混合物特有的噪声
        extra_noise = np.random.normal(0, self.noise_level * 0.5, self.input_dim)
        mixture_spectrum += extra_noise
        
        # 确保为正
        mixture_spectrum = np.maximum(mixture_spectrum, 0)
        
        return mixture_spectrum
    
    def generate_dataset(self):
        """
        生成完整数据集
        
        Returns:
            spectra: 拉曼光谱数组 (num_mixtures, input_dim)
            component_labels: 成分标签 (num_mixtures, num_components)
            concentrations: 浓度 (num_mixtures, num_components)
        """
        spectra = []
        component_labels = []
        concentrations = []
        
        for i in range(self.num_mixtures):
            # 随机决定有几种成分
            num_components_in_mixture = np.random.randint(
                self.min_components, 
                self.max_components + 1
            )
            
            # 随机选择成分
            selected_components = np.random.choice(
                self.num_components,
                size=num_components_in_mixture,
                replace=False
            )
            
            # 创建成分标签（多标签）
            component_label = np.zeros(self.num_components, dtype=np.float32)
            component_label[selected_components] = 1
            
            # 生成浓度（归一化，使总和为1）
            concentration = np.zeros(self.num_components, dtype=np.float32)
            raw_concentrations = np.random.dirichlet(
                np.ones(num_components_in_mixture) * 2
            )
            concentration[selected_components] = raw_concentrations
            
            # 生成混合光谱
            spectrum = self.generate_mixture_spectrum(component_label, concentration)
            
            spectra.append(spectrum)
            component_labels.append(component_label)
            concentrations.append(concentration)
        
        spectra = np.array(spectra, dtype=np.float32)
        component_labels = np.array(component_labels, dtype=np.float32)
        concentrations = np.array(concentrations, dtype=np.float32)
        
        # 打乱顺序
        indices = np.random.permutation(self.num_mixtures)
        spectra = spectra[indices]
        component_labels = component_labels[indices]
        concentrations = concentrations[indices]
        
        return spectra, component_labels, concentrations
    
    def save_dataset(self, save_dir):
        """
        生成并保存数据集
        
        Args:
            save_dir: 保存目录
        """
        print(f"正在生成 {self.num_mixtures} 个拉曼光谱混合物...")
        spectra, component_labels, concentrations = self.generate_dataset()
        
        # 保存为numpy文件
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, 'spectra.npy'), spectra)
        np.save(os.path.join(save_dir, 'component_labels.npy'), component_labels)
        np.save(os.path.join(save_dir, 'concentrations.npy'), concentrations)
        
        print(f"数据集已保存到 {save_dir}")
        print(f"光谱形状: {spectra.shape}")
        print(f"成分标签形状: {component_labels.shape}")
        print(f"浓度形状: {concentrations.shape}")
        
        # 统计信息
        print(f"\n混合物统计:")
        for i in range(1, self.max_components + 1):
            count = np.sum(component_labels.sum(axis=1) == i)
            print(f"  含{i}种成分: {count} 个混合物")
        
        return spectra, component_labels, concentrations


def load_or_generate_data(data_dir, num_mixtures=80, regenerate=False):
    """
    加载或生成数据集
    
    Args:
        data_dir: 数据目录
        num_mixtures: 混合物数量
        regenerate: 是否重新生成数据
    
    Returns:
        spectra, component_labels, concentrations
    """
    spectra_path = os.path.join(data_dir, 'spectra.npy')
    labels_path = os.path.join(data_dir, 'component_labels.npy')
    conc_path = os.path.join(data_dir, 'concentrations.npy')
    
    # 检查数据是否已存在
    if not regenerate and os.path.exists(spectra_path) and os.path.exists(labels_path):
        print("加载已有数据...")
        spectra = np.load(spectra_path)
        component_labels = np.load(labels_path)
        concentrations = np.load(conc_path)
        print(f"加载完成: {len(spectra)} 个混合物")
    else:
        print("生成新数据...")
        generator = RamanMixtureGenerator(num_mixtures=num_mixtures)
        spectra, component_labels, concentrations = generator.save_dataset(data_dir)
    
    return spectra, component_labels, concentrations


if __name__ == '__main__':
    # 测试数据生成器
    generator = RamanMixtureGenerator(num_mixtures=10)
    spectra, component_labels, concentrations = generator.generate_dataset()
    
    print(f"\n数据统计:")
    print(f"光谱形状: {spectra.shape}")
    print(f"光谱范围: [{spectra.min():.3f}, {spectra.max():.3f}]")
    print(f"成分标签形状: {component_labels.shape}")
    print(f"浓度形状: {concentrations.shape}")
    
    # 显示前5个混合物的信息
    print(f"\n前5个混合物的成分和浓度:")
    for i in range(5):
        active_components = np.where(component_labels[i] == 1)[0]
        print(f"\n混合物 {i+1}:")
        for idx in active_components:
            comp_name = config.COMPONENT_NAMES[idx]
            conc = concentrations[i, idx]
            print(f"  {comp_name}: {conc:.3f}")

