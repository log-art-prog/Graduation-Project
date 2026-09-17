"""
大尺度参数 (LSP) 生成模块 - 3GPP TR 38.811 Section 6.6.2 & 6.6.3

功能：
  1. 从对数正态分布生成单个 LSP 样本 (DS, K, ASD, ASA, ZSD, ZSA)
  2. 生成带空间一致性的 LSP 时间序列 (一阶 AR 滤波)
  3. 生成阴影衰落 (SF) 序列

3GPP TR 38.811 中，所有 LSP 在其对应域中服从正态分布：
  - DS (RMS 时延扩展): 域为 log10(ns)
  - K (莱斯 K 因子): 域为 dB
  - ASD/ASA/ZSD/ZSA (角度扩展): 域为 log10(deg)

空间一致性 (Spatial Consistency):
  相邻位置的 LSP 通过指数相关模型关联：
    rho(d) = exp(-d / d_corr)
  其中 d_corr 为各 LSP 的去相关距离。
"""
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional
from .parameters import (
    NTNEnvironment, get_lsp_table, get_sf_std,
    DET_CORRELATION_DISTANCE
)


@dataclass
class LSPSample:
    """单个时刻的大尺度参数样本"""
    ds_ns: float            # RMS 时延扩展 (ns)
    k_db: Optional[float]   # 莱斯 K 因子 (dB)，NLoS 时为 None
    asd_deg: float          # 离开角扩展 (deg)
    asa_deg: float          # 到达角扩展 (deg)
    zsd_deg: float          # 离开天顶角扩展 (deg)
    zsa_deg: float          # 到达天顶角扩展 (deg)
    sf_db: float            # 阴影衰落 (dB)


def generate_lsp_sample(env: NTNEnvironment, is_los: bool,
                        rng: np.random.Generator = None) -> LSPSample:
    """
    生成单个 LSP 样本

    Parameters
    ----------
    env : NTNEnvironment
        环境场景
    is_los : bool
        是否 LoS
    rng : np.random.Generator, optional
        随机数生成器

    Returns
    -------
    LSPSample
        大尺度参数样本
    """
    if rng is None:
        rng = np.random.default_rng()

    lsp_table = get_lsp_table(env)
    table = lsp_table.los if is_los else lsp_table.nlos

    # 从对数正态分布采样
    def sample_log_normal(mu: float, sigma: float) -> float:
        """在对数域采样后转回线性域"""
        log_val = rng.normal(mu, sigma)
        return 10.0 ** log_val

    def sample_normal(mu: float, sigma: float) -> float:
        """在线性域 (dB) 采样"""
        return rng.normal(mu, sigma)

    # DS: log10(秒) 域正态 → 转 ns
    mu_ds, sigma_ds = table['DS']
    ds_s = sample_log_normal(mu_ds, sigma_ds)
    ds_ns = ds_s * 1e9

    # K: dB 域正态 (仅 LoS)
    k_db = None
    if is_los:
        mu_k, sigma_k = table['K']
        k_db = sample_normal(mu_k, sigma_k)

    # 角度扩展: log10(deg) 域正态
    mu_asd, sigma_asd = table['ASD']
    asd_deg = sample_log_normal(mu_asd, sigma_asd)

    mu_asa, sigma_asa = table['ASA']
    asa_deg = sample_log_normal(mu_asa, sigma_asa)

    mu_zsd, sigma_zsd = table['ZSD']
    zsd_deg = sample_log_normal(mu_zsd, sigma_zsd)

    mu_zsa, sigma_zsa = table['ZSA']
    zsa_deg = sample_log_normal(mu_zsa, sigma_zsa)

    # 阴影衰落: dB 域正态 (零均值)
    sf_std = get_sf_std(env)
    sf_db = rng.normal(0.0, sf_std)

    return LSPSample(
        ds_ns=ds_ns,
        k_db=k_db,
        asd_deg=asd_deg,
        asa_deg=asa_deg,
        zsd_deg=zsd_deg,
        zsa_deg=zsa_deg,
        sf_db=sf_db
    )


def generate_spatially_consistent_lsp(
    num_points: int,
    env: NTNEnvironment,
    is_los: bool,
    velocity_ms: float,
    fs: float,
    rng: np.random.Generator = None
) -> Dict[str, np.ndarray]:
    """
    生成带空间一致性的 LSP 时间序列

    使用一阶 AR(1) 模型实现空间相关性：
      X[n] = rho * X[n-1] + sqrt(1-rho^2) * epsilon[n]
    其中 rho = exp(-d_sample / d_corr)，d_sample = v / fs

    Parameters
    ----------
    num_points : int
        序列长度
    env : NTNEnvironment
        环境场景
    is_los : bool
        是否 LoS
    velocity_ms : float
        卫星相对地面速度 (m/s)
    fs : float
        采样率 (Hz)
    rng : np.random.Generator, optional
        随机数生成器

    Returns
    -------
    dict
        包含各 LSP 序列的字典:
        - 'ds_ns': RMS 时延扩展序列 (ns)
        - 'k_db': K 因子序列 (dB)，NLoS 时为 None
        - 'asd_deg', 'asa_deg', 'zsd_deg', 'zsa_deg': 角度扩展序列
        - 'sf_db': 阴影衰落序列 (dB)
    """
    if rng is None:
        rng = np.random.default_rng()

    lsp_table = get_lsp_table(env)
    table = lsp_table.los if is_los else lsp_table.nlos

    d_sample = velocity_ms / fs  # 每采样间隔移动的距离

    def ar1_filter(mu: float, sigma: float, d_corr: float) -> np.ndarray:
        """
        AR(1) 滤波生成空间相关序列
        在对数域进行滤波，保证物理量非负
        """
        rho = np.exp(-d_sample / d_corr) if d_corr > 0 else 0.0
        rho = np.clip(rho, 0.0, 0.9999)

        # 生成独立同分布的对数域样本
        raw = rng.normal(mu, sigma, num_points)

        # AR(1) 滤波
        filtered = np.zeros(num_points)
        filtered[0] = raw[0]
        for i in range(1, num_points):
            filtered[i] = rho * filtered[i - 1] + np.sqrt(1 - rho ** 2) * raw[i]

        return filtered

    # DS (log10(秒) 域 → ns)
    mu_ds, sigma_ds = table['DS']
    ds_log = ar1_filter(mu_ds, sigma_ds, DET_CORRELATION_DISTANCE['DS'])
    ds_ns = (10.0 ** ds_log) * 1e9

    # K (仅 LoS)
    k_db = None
    if is_los:
        mu_k, sigma_k = table['K']
        k_db = ar1_filter(mu_k, sigma_k, DET_CORRELATION_DISTANCE['K'])

    # 角度扩展
    mu_asd, sigma_asd = table['ASD']
    asd_deg = 10.0 ** ar1_filter(mu_asd, sigma_asd, DET_CORRELATION_DISTANCE['ASD'])

    mu_asa, sigma_asa = table['ASA']
    asa_deg = 10.0 ** ar1_filter(mu_asa, sigma_asa, DET_CORRELATION_DISTANCE['ASA'])

    mu_zsd, sigma_zsd = table['ZSD']
    zsd_deg = 10.0 ** ar1_filter(mu_zsd, sigma_zsd, DET_CORRELATION_DISTANCE['ZSD'])

    mu_zsa, sigma_zsa = table['ZSA']
    zsa_deg = 10.0 ** ar1_filter(mu_zsa, sigma_zsa, DET_CORRELATION_DISTANCE['ZSA'])

    # 阴影衰落 (dB 域)
    sf_std = get_sf_std(env)
    sf_db = ar1_filter(0.0, sf_std, DET_CORRELATION_DISTANCE['SF'])

    return {
        'ds_ns': ds_ns,
        'k_db': k_db,
        'asd_deg': asd_deg,
        'asa_deg': asa_deg,
        'zsd_deg': zsd_deg,
        'zsa_deg': zsa_deg,
        'sf_db': sf_db,
    }


def generate_shadow_fading_sequence(
    num_points: int,
    env: NTNEnvironment,
    velocity_ms: float,
    fs: float,
    rng: np.random.Generator = None
) -> np.ndarray:
    """
    仅生成阴影衰落序列 (带空间一致性)

    Parameters
    ----------
    num_points : int
        序列长度
    env : NTNEnvironment
        环境场景
    velocity_ms : float
        相对速度 (m/s)
    fs : float
        采样率 (Hz)
    rng : np.random.Generator, optional

    Returns
    -------
    np.ndarray
        阴影衰落序列 (dB)
    """
    if rng is None:
        rng = np.random.default_rng()

    sf_std = get_sf_std(env)
    d_sample = velocity_ms / fs
    d_corr = DET_CORRELATION_DISTANCE['SF']
    rho = np.exp(-d_sample / d_corr) if d_corr > 0 else 0.0
    rho = np.clip(rho, 0.0, 0.9999)

    raw = rng.normal(0.0, sf_std, num_points)
    sf = np.zeros(num_points)
    sf[0] = raw[0]
    for i in range(1, num_points):
        sf[i] = rho * sf[i - 1] + np.sqrt(1 - rho ** 2) * raw[i]

    return sf


# ============================================================
# 示例与验证
# ============================================================
if __name__ == '__main__':
    from .parameters import ENV_URBAN, ENV_OPEN

    print("=" * 70)
    print("3GPP TR 38.811 大尺度参数 (LSP) 验证")
    print("=" * 70)

    rng = np.random.default_rng(42)

    for env_name, env in [('Urban', ENV_URBAN), ('Open', ENV_OPEN)]:
        print(f"\n--- {env_name} 环境 ---")

        # LoS 样本
        los_sample = generate_lsp_sample(env, is_los=True, rng=rng)
        print(f"  LoS:  DS={los_sample.ds_ns:.2f} ns, K={los_sample.k_db:.1f} dB, "
              f"ASD={los_sample.asd_deg:.1f}°, SF={los_sample.sf_db:.1f} dB")

        # NLoS 样本
        nlos_sample = generate_lsp_sample(env, is_los=False, rng=rng)
        print(f"  NLoS: DS={nlos_sample.ds_ns:.2f} ns, K={nlos_sample.k_db}, "
              f"ASD={nlos_sample.asd_deg:.1f}°, SF={nlos_sample.sf_db:.1f} dB")

    # 空间一致性验证
    print("\n--- 空间一致性验证 (Urban LoS) ---")
    lsp_seq = generate_spatially_consistent_lsp(
        num_points=100, env=ENV_URBAN, is_los=True,
        velocity_ms=7600, fs=1.0, rng=rng
    )
    print(f"  DS 范围: {np.min(lsp_seq['ds_ns']):.2f} ~ {np.max(lsp_seq['ds_ns']):.2f} ns")
    print(f"  K  范围: {np.min(lsp_seq['k_db']):.1f} ~ {np.max(lsp_seq['k_db']):.1f} dB")
    print(f"  SF 范围: {np.min(lsp_seq['sf_db']):.1f} ~ {np.max(lsp_seq['sf_db']):.1f} dB")
    print(f"  DS 标准差: {np.std(lsp_seq['ds_ns']):.2f} ns (空间相关后波动减小)")
