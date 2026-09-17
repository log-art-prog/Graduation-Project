"""
衰落模型模块
实现阴影衰落和小尺度莱斯/瑞利衰落生成
"""
import numpy as np


def generate_shadow_fading(
    num_samples: int,
    sigma_db: float = 3.0,
    decorrelation_distance_m: float = 50.0,
    velocity_ms: float = 7600.0,
    fs: float = 10.0,
    rng: np.random.Generator =None,
):
    """
    生成阴影衰落 (对数正态分布)
    使用一阶AR模型模拟空间相关性
    sigma_db: 阴影衰落标准差
    decorrelation_distance_m: 去相关距离
    velocity_ms: 移动速度 (m/s)
    fs: 采样率 (Hz)
    rng: 随机数生成器，传入固定种子的生成器可保证结果可复现
    """
    if rng is None:
        rng = np.random.default_rng()

    # 相关系数：距离间隔d处的相关系数为 exp(-d/d_corr)
    dt = 1.0 / fs
    d_sample = velocity_ms * dt
    rho = np.exp(-d_sample / decorrelation_distance_m)
    
    # AR(1)过程生成相关对数正态阴影
    shadow = np.zeros(num_samples)
    
    shadow[0] = rng.standard_normal() * sigma_db
    
    for i in range(1, num_samples):
        shadow[i] = rho * shadow[i-1] + np.sqrt(1 - rho**2) * rng.standard_normal() * sigma_db
    
    return shadow


def generate_rayleigh_fading(
    num_samples: int,
    fd_hz: float,
    fs: float,
    num_sinusoids: int = 20,
    rng: np.random.Generator =None,
):
    """
    生成瑞利衰落 (Jakes模型/正弦叠加法)
    fd_hz: 最大多普勒频移
    fs: 采样率
    rng: 随机数生成器，传入固定种子的生成器可保证结果可复现
    """
    if rng is None:
        rng=np.random.default_rng()

    t = np.arange(num_samples) / fs
    
    # Clarke/Jakes模型：多个等幅、随机相位、均匀到达角的正弦波叠加
    x = np.zeros(num_samples)
    y = np.zeros(num_samples)
    
    theta = rng.uniform(0, 2 * np.pi)
    for n in range(num_sinusoids):
        alpha_n = (2 * np.pi * n - np.pi + theta) / (4 * num_sinusoids)
        phi_n = rng.uniform(0, 2 * np.pi)
        phi_n2 = rng.uniform(0, 2 * np.pi)
        
        x += np.cos(2 * np.pi * fd_hz * t * np.cos(alpha_n) + phi_n)
        y += np.cos(2 * np.pi * fd_hz * t * np.sin(alpha_n) + phi_n2)
    
    # 归一化
    x = x / np.sqrt(num_sinusoids)
    y = y / np.sqrt(num_sinusoids)
    
    # 复信道系数
    h_fading = (x + 1j * y) / np.sqrt(2)
    
    return h_fading


def generate_rician_fading(
    num_samples: int,
    k_factor_linear: float,
    fd_hz: float,
    fs: float,
    num_sinusoids: int = 20,
    rng: np.random.Generator =None,
):
    """
    生成莱斯衰落
    K = 直射径功率 / 散射径功率
    K=0时退化为瑞利，K→∞时为AWGN
    rng: 随机数生成器，传入固定种子的生成器可保证结果可复现
    """
    if rng is None:
        rng=np.random.default_rng()

    # 散射分量 (瑞利)
    h_scatter = generate_rayleigh_fading(num_samples, fd_hz, fs, num_sinusoids,rng=rng)
    h_scatter = h_scatter / np.sqrt(np.mean(np.abs(h_scatter)**2))  # 归一化散射功率为1
    
    # 直射分量 (有相位变化，考虑多普勒)
    # 直射径方向的多普勒频移：假设直射径在来波方向0度
    t = np.arange(num_samples) / fs
    # 直射径相位（考虑多普勒）
    los_phase = 2 * np.pi * fd_hz * t
    h_los = np.exp(1j * los_phase)
    
    # 莱斯合成
    # h = sqrt(K/(K+1)) * h_los + sqrt(1/(K+1)) * h_scatter
    h_rician = np.sqrt(k_factor_linear / (k_factor_linear + 1)) * h_los + \
               np.sqrt(1 / (k_factor_linear + 1)) * h_scatter
    
    return h_rician


def generate_multipath_components(
    num_samples: int,
    tau_rms_s: float,
    num_paths: int,
    fd_hz: float,
    fs: float,
    k_factor_linear: float = 10.0,
    rng: np.random.Generator =None,
):
    """
    生成多径分量（抽头延迟线模型）
    返回: h_taps (num_paths x num_samples), tap_delays_s, tap_powers_db
    rng: 随机数生成器，传入固定种子的生成器可保证结果可复现
    """
    if rng is None:
        rng=np.random.default_rng()

    # 指数功率延迟分布
    tap_delays_s = np.sort(rng.exponential(tau_rms_s, num_paths))
    tap_delays_s = tap_delays_s - tap_delays_s[0]  # 第一个径在0时延处
    
    # 功率按指数衰减
    tap_powers_linear = np.exp(-tap_delays_s / tau_rms_s)
    tap_powers_linear = tap_powers_linear / np.sum(tap_powers_linear)  # 归一化
    tap_powers_db = 10 * np.log10(tap_powers_linear)
    
    # 第一个径是直射径（莱斯）
    h_taps = np.zeros((num_paths, num_samples), dtype=complex)
    
    for i in range(num_paths):
        if i == 0:
            # 直射径：莱斯K因子
            h_taps[i, :] = np.sqrt(tap_powers_linear[i]) * generate_rician_fading(
                num_samples, k_factor_linear, fd_hz * 0.8, fs,rng=rng  # 直射径多普勒略小
            )
        else:
            # 反射径：瑞利
            h_taps[i, :] = np.sqrt(tap_powers_linear[i]) * generate_rayleigh_fading(
                num_samples, fd_hz * rng.uniform(0.5, 1.0), fs, rng=rng
            )
    
    return h_taps, tap_delays_s, tap_powers_db


def calculate_fading_metrics(h_fading: np.ndarray) -> dict:
    """
    计算衰落统计指标
    """
    power = np.abs(h_fading)**2
    power_db = 10 * np.log10(power / np.mean(power))  # 归一化功率dB
    
    # 电平通过率 LCR (电平为0dB即均值电平时的通过率)
    threshold = 1.0
    crossings = np.sum(np.diff(np.abs(h_fading) > threshold))
    duration_s = len(h_fading)
    lcr = crossings / (2 * duration_s)  # 每秒穿越次数
    
    # 平均衰落持续时间 AFD (电平低于-10dB的平均持续时间)
    threshold_10db = 10**(-10/20)
    fade_intervals = np.abs(h_fading) < threshold_10db
    if np.sum(fade_intervals) > 0:
        # 找连续衰落区间
        fade_changes = np.diff(fade_intervals.astype(int))
        fade_starts = np.where(fade_changes == 1)[0]
        fade_ends = np.where(fade_changes == -1)[0]
        if len(fade_starts) > 0 and len(fade_ends) > 0:
            fade_durations = fade_ends - fade_starts[:len(fade_ends)]
            afd = np.mean(fade_durations) if len(fade_durations) > 0 else 0
        else:
            afd = 0
    else:
        afd = 0
    
    # 衰落深度 (1% CDF点)
    cdf_point = np.percentile(power_db, 1)
    
    return {
        'power_db': power_db,
        'lcr_hz': lcr,
        'afd_samples': afd,
        'fade_depth_db': cdf_point,
        'power_std_db': np.std(power_db)
    }
