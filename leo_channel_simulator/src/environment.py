"""
环境参数定义模块
定义开阔地、城郊、城市三种典型环境的信道参数
"""
import numpy as np
from dataclasses import dataclass

@dataclass
class EnvironmentParams:
    """环境参数数据类"""
    name: str              # 环境名称
    name_cn: str           # 中文名称
    sigma_shadow_dB: float # 阴影衰落标准差 (dB)
    max_rms_delay_ns: float # 最大RMS时延扩展 (ns) - 低仰角时
    min_rms_delay_ns: float # 最小RMS时延扩展 (ns) - 高仰角时
    k_min_dB: float        # 最小莱斯K因子 (dB) - 低仰角
    k_max_dB: float        # 最大莱斯K因子 (dB) - 高仰角
    multipath_num: int     # 多径数量
    color: str             # 绘图颜色
    linestyle: str         # 绘图线型


# 预定义三种典型环境
ENV_OPEN = EnvironmentParams(
    name='Open',
    name_cn='Open Area',
    sigma_shadow_dB=1.0,
    max_rms_delay_ns=10.0,
    min_rms_delay_ns=2.0,
    k_min_dB=10.0,
    k_max_dB=25.0,
    multipath_num=3,
    color='#2ecc71',
    linestyle='-'
)

ENV_SUBURBAN = EnvironmentParams(
    name='Suburban',
    name_cn='Suburban',
    sigma_shadow_dB=3.5,
    max_rms_delay_ns=100.0,
    min_rms_delay_ns=10.0,
    k_min_dB=3.0,
    k_max_dB=15.0,
    multipath_num=6,
    color='#3498db',
    linestyle='--'
)

ENV_URBAN = EnvironmentParams(
    name='Urban',
    name_cn='Urban',
    sigma_shadow_dB=6.0,
    max_rms_delay_ns=1000.0,
    min_rms_delay_ns=50.0,
    k_min_dB=-10.0,
    k_max_dB=5.0,
    multipath_num=12,
    color='#e74c3c',
    linestyle='-.'
)

ALL_ENVIRONMENTS = [ENV_OPEN, ENV_SUBURBAN, ENV_URBAN]


def get_rms_delay_spread(env: EnvironmentParams, elevation_deg: float) -> float:
    """
    根据仰角计算RMS时延扩展
    仰角越低，时延扩展越大
    """
    # 仰角归一化 (0~1，1对应90度)
    elev_norm = np.clip(elevation_deg / 90.0, 0, 1)
    # 时延扩展随仰角指数下降
    tau_rms = env.min_rms_delay_ns + (env.max_rms_delay_ns - env.min_rms_delay_ns) * np.exp(-3 * elev_norm)
    return tau_rms * 1e-9  # 转换为秒


def get_k_factor(env: EnvironmentParams, elevation_deg: float) -> float:
    """
    根据仰角计算莱斯K因子 (线性值)
    仰角越高，直射径越强，K因子越大
    """
    elev_norm = np.clip(elevation_deg / 90.0, 0, 1)
    # K因子随仰角线性增加
    k_dB = env.k_min_dB + (env.k_max_dB - env.k_min_dB) * elev_norm
    return 10 ** (k_dB / 10.0)


def get_coherence_bandwidth(tau_rms_s: float) -> float:
    """
    根据RMS时延扩展计算相干带宽 (近似)
    B_c ≈ 1/(5*tau_rms)  (相干带宽定义为相关系数>0.5的带宽)
    """
    return 1.0 / (5 * tau_rms_s)
