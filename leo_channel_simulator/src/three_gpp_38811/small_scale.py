"""
小尺度衰落模块 - 3GPP TR 38.811 Section 6.5 & 6.7

实现：
  1. CDL (Clustered Delay Line) 抽头延迟线模型 - Section 6.7
  2. Loo 模型：阴影衰落与多径联合分布 - Section 6.5.3
  3. 莱斯 (Rician) / 瑞利 (Rayleigh) 衰落生成
  4. 多普勒频移与多普勒扩展建模

3GPP TR 38.811 卫星信道小尺度建模要点：
  - LoS 场景：直射径 (受阴影衰落调制) + 多径散射 (瑞利)
  - NLoS 场景：纯多径散射 (瑞利)
  - Loo 模型准确描述了陆地移动卫星 (LMS) 信道中直射径的遮挡效应
"""
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional
from .parameters import CDL_CONFIG_NTN, CDLConfig, LOO_MODEL_PARAMS


# ============================================================
# 基础衰落生成
# ============================================================
def generate_rayleigh_fading(
    num_samples: int,
    fd_hz: float,
    fs: float,
    num_sinusoids: int = 20,
    rng: np.random.Generator = None
) -> np.ndarray:
    """
    生成瑞利衰落 (Jakes 正弦叠加法)

    Parameters
    ----------
    num_samples : int
        序列长度
    fd_hz : float
        最大多普勒频移 (Hz)
    fs : float
        采样率 (Hz)
    num_sinusoids : int
        正弦波数量
    rng : np.random.Generator, optional

    Returns
    -------
    np.ndarray
        复信道系数 (归一化平均功率为 1)
    """
    if rng is None:
        rng = np.random.default_rng()

    t = np.arange(num_samples) / fs
    x = np.zeros(num_samples)
    y = np.zeros(num_samples)

    theta = rng.uniform(0, 2 * np.pi)
    for n in range(num_sinusoids):
        alpha_n = (2 * np.pi * n - np.pi + theta) / (4 * num_sinusoids)
        phi_n = rng.uniform(0, 2 * np.pi)
        phi_n2 = rng.uniform(0, 2 * np.pi)
        x += np.cos(2 * np.pi * fd_hz * t * np.cos(alpha_n) + phi_n)
        y += np.cos(2 * np.pi * fd_hz * t * np.sin(alpha_n) + phi_n2)

    x /= np.sqrt(num_sinusoids)
    y /= np.sqrt(num_sinusoids)

    h = (x + 1j * y) / np.sqrt(2)
    return h


def generate_rician_fading(
    num_samples: int,
    k_factor_linear: float,
    fd_hz: float,
    fs: float,
    num_sinusoids: int = 20,
    rng: np.random.Generator = None
) -> np.ndarray:
    """
    生成莱斯衰落

    h = sqrt(K/(K+1)) * h_los + sqrt(1/(K+1)) * h_scatter

    Parameters
    ----------
    num_samples : int
        序列长度
    k_factor_linear : float
        莱斯 K 因子 (线性值)
    fd_hz : float
        最大多普勒频移 (Hz)
    fs : float
        采样率 (Hz)
    rng : np.random.Generator, optional

    Returns
    -------
    np.ndarray
        复信道系数
    """
    if rng is None:
        rng = np.random.default_rng()

    # 散射分量 (瑞利)
    h_scatter = generate_rayleigh_fading(num_samples, fd_hz, fs, num_sinusoids, rng)
    # 归一化散射功率
    h_scatter /= np.sqrt(np.mean(np.abs(h_scatter) ** 2))

    # 直射分量 (恒幅，含多普勒相位)
    t = np.arange(num_samples) / fs
    los_phase = 2 * np.pi * fd_hz * t
    h_los = np.exp(1j * los_phase)

    # 莱斯合成
    h_rician = (
        np.sqrt(k_factor_linear / (k_factor_linear + 1)) * h_los +
        np.sqrt(1.0 / (k_factor_linear + 1)) * h_scatter
    )

    return h_rician


# ============================================================
# Loo 模型 - Section 6.5.3
# ============================================================
def generate_loo_fading(
    num_samples: int,
    k_factor_linear: float,
    shadow_std_db: float,
    fd_hz: float,
    fs: float,
    decorrelation_distance_m: float = 50.0,
    velocity_ms: float = 7600.0,
    rng: np.random.Generator = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成 Loo 模型衰落 (陆地移动卫星信道标准模型)

    Loo 模型描述：
      - 直射径 (LoS) 幅度受对数正态阴影衰落调制
      - 散射径服从瑞利分布
      - 接收信号复包络：
        h(t) = 10^(xi(t)/20) * sqrt(K/(K+1)) * exp(j*2*pi*fd*t)
             + sqrt(1/(K+1)) * s(t)
      其中 xi(t) 是阴影衰落 (dB)，s(t) 是瑞利散射分量

    Parameters
    ----------
    num_samples : int
        序列长度
    k_factor_linear : float
        莱斯 K 因子 (线性值)
    shadow_std_db : float
        阴影衰落标准差 (dB)
    fd_hz : float
        最大多普勒频移 (Hz)
    fs : float
        采样率 (Hz)
    decorrelation_distance_m : float
        阴影衰落去相关距离 (m)
    velocity_ms : float
        相对速度 (m/s)
    rng : np.random.Generator, optional

    Returns
    -------
    tuple (h_complex, shadow_db)
        h_complex: 复信道系数
        shadow_db: 阴影衰落序列 (dB)
    """
    if rng is None:
        rng = np.random.default_rng()

    # 1. 生成相关对数正态阴影衰落 (直射径幅度调制)
    d_sample = velocity_ms / fs
    rho = np.exp(-d_sample / decorrelation_distance_m) if decorrelation_distance_m > 0 else 0.0
    rho = np.clip(rho, 0.0, 0.9999)

    shadow_db = np.zeros(num_samples)
    shadow_db[0] = rng.normal(0, shadow_std_db)
    for i in range(1, num_samples):
        shadow_db[i] = (rho * shadow_db[i - 1] +
                        np.sqrt(1 - rho ** 2) * rng.normal(0, shadow_std_db))

    # 2. 直射径分量 (受阴影调制)
    t = np.arange(num_samples) / fs
    los_amp = 10.0 ** (shadow_db / 20.0)  # 线性幅度
    los_phase = 2 * np.pi * fd_hz * t
    h_los = los_amp * np.exp(1j * los_phase)

    # 3. 散射分量 (瑞利)
    h_scatter = generate_rayleigh_fading(num_samples, fd_hz, fs, rng=rng)
    h_scatter /= np.sqrt(np.mean(np.abs(h_scatter) ** 2))

    # 4. Loo 合成
    k_norm_los = np.sqrt(k_factor_linear / (k_factor_linear + 1))
    k_norm_scatter = np.sqrt(1.0 / (k_factor_linear + 1))

    h_loo = k_norm_los * h_los + k_norm_scatter * h_scatter

    return h_loo, shadow_db


# ============================================================
# CDL (Clustered Delay Line) 模型 - Section 6.7
# ============================================================
@dataclass
class CDLTaps:
    """CDL 抽头结果"""
    delays_s: np.ndarray      # 各簇延迟 (s)
    powers_linear: np.ndarray  # 各簇功率 (线性，归一化)
    h_taps: np.ndarray        # 复信道系数 [num_clusters, num_samples]


def generate_cdl_taps(
    num_samples: int,
    ds_ns: float,
    k_factor_db: Optional[float],
    fd_hz: float,
    fs: float,
    config: CDLConfig = CDL_CONFIG_NTN,
    rng: np.random.Generator = None
) -> CDLTaps:
    """
    生成 3GPP CDL (Clustered Delay Line) 抽头延迟线

    3GPP TR 38.811 卫星 CDL 模型：
      - 簇延迟：按指数分布，与 RMS 时延扩展成比例
      - 簇功率：指数衰减 P_n = exp(-tau_n / tau_rms)
      - 每簇内子径：独立瑞利/莱斯衰落

    Parameters
    ----------
    num_samples : int
        序列长度
    ds_ns : float
        RMS 时延扩展 (ns)
    k_factor_db : float or None
        莱斯 K 因子 (dB)，None 表示 NLoS (瑞利)
    fd_hz : float
        最大多普勒频移 (Hz)
    fs : float
        采样率 (Hz)
    config : CDLConfig
        CDL 配置
    rng : np.random.Generator, optional

    Returns
    -------
    CDLTaps
        抽头结果
    """
    if rng is None:
        rng = np.random.default_rng()

    num_clusters = config.num_clusters
    ds_s = ds_ns * 1e-9

    # 1. 生成簇延迟 (指数分布，均值为 RMS 时延扩展)
    raw_delays = rng.exponential(ds_s, num_clusters)
    delays_s = np.sort(raw_delays)
    delays_s -= delays_s[0]  # 第一个抽头在 0 时延

    # 2. 生成簇功率 (指数衰减)
    powers_linear = np.exp(-delays_s / ds_s)
    powers_linear /= np.sum(powers_linear)  # 归一化总功率

    # 3. 生成每簇的复信道系数
    h_taps = np.zeros((num_clusters, num_samples), dtype=complex)

    is_los = k_factor_db is not None
    k_linear = 10 ** (k_factor_db / 10.0) if is_los else None

    for i in range(num_clusters):
        tap_power = powers_linear[i]
        tap_fd = fd_hz * rng.uniform(0.5, 1.0)  # 各径多普勒略有差异

        if is_los and i == 0:
            # 第一个簇 (最短延迟) 为直射径，使用 Loo 模型
            # 阴影衰落标准差随仰角变化，这里用 K 因子隐含
            shadow_std = max(0.5, 4.0 - k_factor_db * 0.1)
            h_i, _ = generate_loo_fading(
                num_samples=num_samples,
                k_factor_linear=k_linear,
                shadow_std_db=shadow_std,
                fd_hz=tap_fd,
                fs=fs,
                rng=rng
            )
        else:
            # 其余簇为瑞利散射
            h_i = generate_rayleigh_fading(
                num_samples=num_samples,
                fd_hz=tap_fd,
                fs=fs,
                rng=rng
            )

        h_taps[i, :] = np.sqrt(tap_power) * h_i

    return CDLTaps(
        delays_s=delays_s,
        powers_linear=powers_linear,
        h_taps=h_taps
    )


def get_cdl_total_power_response(cdl_taps: CDLTaps, freq_hz: np.ndarray) -> np.ndarray:
    """
    计算 CDL 模型的频域总功率响应

    Parameters
    ----------
    cdl_taps : CDLTaps
        CDL 抽头
    freq_hz : np.ndarray
        频率轴 (Hz)

    Returns
    -------
    np.ndarray
        频域响应 (线性)
    """
    # 频率响应: H(f) = sum_n h_n * exp(-j*2*pi*f*tau_n)
    # 取平均功率
    h_freq = np.zeros(len(freq_hz), dtype=complex)
    for i in range(len(cdl_taps.delays_s)):
        h_avg = np.mean(cdl_taps.h_taps[i, :])
        h_freq += h_avg * np.exp(-1j * 2 * np.pi * freq_hz * cdl_taps.delays_s[i])

    return np.abs(h_freq) ** 2


# ============================================================
# 衰落统计指标
# ============================================================
def calculate_fading_metrics(h_fading: np.ndarray, fs: float) -> dict:
    """
    计算衰落统计指标

    Parameters
    ----------
    h_fading : np.ndarray
        复信道系数
    fs : float
        采样率 (Hz)

    Returns
    -------
    dict
        包含电平通过率 (LCR)、平均衰落持续时间 (AFD)、衰落深度等
    """
    power = np.abs(h_fading) ** 2
    power_db = 10 * np.log10(power / np.mean(power) + 1e-20)

    # 电平通过率 (LCR) - 在均值电平处
    threshold = 1.0
    envelope = np.abs(h_fading)
    crossings = np.sum(np.abs(np.diff((envelope > threshold).astype(int))) == 1)
    duration_s = len(h_fading) / fs
    lcr_hz = crossings / (2 * duration_s) if duration_s > 0 else 0

    # 平均衰落持续时间 (AFD) - 低于 -10dB
    threshold_10db = 10 ** (-10 / 20)
    fade_mask = envelope < threshold_10db
    afd = 0.0
    if np.sum(fade_mask) > 0:
        # 用 diff 找到状态跳变，处理首尾对齐
        padded = np.concatenate(([0], fade_mask.astype(int), [0]))
        changes = np.diff(padded)
        fade_starts = np.where(changes == 1)[0]
        fade_ends = np.where(changes == -1)[0]
        n = min(len(fade_starts), len(fade_ends))
        if n > 0:
            fade_durations = (fade_ends[:n] - fade_starts[:n]) / fs
            afd = float(np.mean(fade_durations))

    # 衰落深度 (1% CDF)
    fade_depth_db = np.percentile(power_db, 1)

    return {
        'power_db': power_db,
        'lcr_hz': lcr_hz,
        'afd_s': afd,
        'fade_depth_db': float(fade_depth_db),
        'power_std_db': float(np.std(power_db)),
    }


# ============================================================
# 示例与验证
# ============================================================
if __name__ == '__main__':
    print("=" * 70)
    print("3GPP TR 38.811 小尺度模型验证")
    print("=" * 70)

    rng = np.random.default_rng(42)
    num_samples = 5000
    fs = 10.0
    fd_hz = 500.0

    # 1. 瑞利衰落
    h_rayleigh = generate_rayleigh_fading(num_samples, fd_hz, fs, rng=rng)
    metrics = calculate_fading_metrics(h_rayleigh, fs)
    print(f"\n[Rayleigh] 衰落深度(1%): {metrics['fade_depth_db']:.1f} dB, "
          f"LCR: {metrics['lcr_hz']:.1f} Hz, AFD: {metrics['afd_s']*1000:.2f} ms")

    # 2. 莱斯衰落 (K=10dB)
    h_rician = generate_rician_fading(num_samples, k_factor_linear=10.0,
                                      fd_hz=fd_hz, fs=fs, rng=rng)
    metrics = calculate_fading_metrics(h_rician, fs)
    print(f"[Rician K=10dB] 衰落深度(1%): {metrics['fade_depth_db']:.1f} dB, "
          f"LCR: {metrics['lcr_hz']:.1f} Hz, AFD: {metrics['afd_s']*1000:.2f} ms")

    # 3. Loo 模型
    h_loo, shadow = generate_loo_fading(
        num_samples, k_factor_linear=10.0, shadow_std_db=2.0,
        fd_hz=fd_hz, fs=fs, rng=rng
    )
    metrics = calculate_fading_metrics(h_loo, fs)
    print(f"[Loo K=10dB, σ=2dB] 衰落深度(1%): {metrics['fade_depth_db']:.1f} dB, "
          f"LCR: {metrics['lcr_hz']:.1f} Hz")

    # 4. CDL 模型
    cdl = generate_cdl_taps(
        num_samples=num_samples, ds_ns=100.0, k_factor_db=9.0,
        fd_hz=fd_hz, fs=fs, rng=rng
    )
    h_total = np.sum(cdl.h_taps, axis=0)
    metrics = calculate_fading_metrics(h_total, fs)
    print(f"[CDL LoS, DS=100ns] 衰落深度(1%): {metrics['fade_depth_db']:.1f} dB, "
          f"簇数: {len(cdl.delays_s)}, 最大延迟: {cdl.delays_s[-1]*1e9:.1f} ns")
