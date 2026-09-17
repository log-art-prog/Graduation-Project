"""
信道模型核心模块
整合轨道、传播、衰落模块，生成完整的信道响应
"""
import numpy as np
from dataclasses import dataclass
from typing import List, Optional

from .tle_orbit import OrbitParams, simple_orbit_model, compute_orbit_from_tle
from .propagation import calculate_total_path_loss, calculate_received_power
from .fading import generate_shadow_fading, generate_rician_fading, calculate_fading_metrics
from .environment import EnvironmentParams, get_rms_delay_spread, get_k_factor


@dataclass
class ChannelResult:
    """信道仿真结果数据类"""
    orbit: OrbitParams
    
    # 路径损耗
    fspl_db: np.ndarray
    atmosphere_db: np.ndarray
    rain_db: np.ndarray
    total_pl_db: np.ndarray
    
    # 衰落
    shadow_db: np.ndarray
    small_scale_amp: np.ndarray
    k_factor_db: np.ndarray
    tau_rms_ns: np.ndarray
    coherence_bw_mhz: np.ndarray
    
    # 接收性能
    rx_power_dbm: np.ndarray
    snr_db: np.ndarray
    
    # 衰落统计
    fade_depth_db: float
    lcr_hz: float
    environment: EnvironmentParams


def simulate_channel(
    env: EnvironmentParams,
    orbit: OrbitParams,
    fc_hz: float = 20e9,
    bandwidth_hz: float = 200e6,
    tx_eirp_dbm: float = 55.0,
    rx_gain_db: float = 30.0,
    noise_figure_db: float = 3.0,
    rain_rate_mm_h: float = 2.0,
) -> ChannelResult:
    """
    对给定环境和轨道参数进行完整信道仿真
    """
    num_points = len(orbit.t)
    c = 299792458
    wavelength = c / fc_hz
    
    # 初始化数组
    fspl_db = np.zeros(num_points)
    atmosphere_db = np.zeros(num_points)
    rain_db = np.zeros(num_points)
    total_pl_db = np.zeros(num_points)
    rx_power_dbm = np.zeros(num_points)
    snr_db = np.zeros(num_points)
    shadow_db = np.zeros(num_points)
    k_factor_db = np.zeros(num_points)
    tau_rms_ns = np.zeros(num_points)
    coherence_bw_mhz = np.zeros(num_points)
    
    # 逐点计算大尺度参数
    for i in range(num_points):
        elev = orbit.elevation_deg[i]
        range_m = orbit.range_km[i] * 1000
        
        # 路径损耗
        pl = calculate_total_path_loss(
            fc_hz=fc_hz,
            range_m=range_m,
            elevation_deg=elev,
            rain_rate_mm_h=rain_rate_mm_h
        )
        fspl_db[i] = pl['fspl_db']
        atmosphere_db[i] = pl['atmosphere_db']
        rain_db[i] = pl['rain_db']
        
        # 环境相关参数
        tau_rms = get_rms_delay_spread(env, elev)
        k_factor = get_k_factor(env, elev)
        tau_rms_ns[i] = tau_rms * 1e9
        coherence_bw_mhz[i] = (1.0 / (5 * tau_rms)) / 1e6
        k_factor_db[i] = 10 * np.log10(k_factor)
        
        # 阴影衰落预计算（后面生成时间序列）
        total_pl_db[i] = pl['total_db']
    
    # 生成相关阴影衰落序列
    mean_sat_velocity = np.mean(np.abs(orbit.velocity_kms)) * 1000  # m/s
    shadow_db = generate_shadow_fading(
        num_samples=num_points,
        sigma_db=env.sigma_shadow_dB,
        velocity_ms=mean_sat_velocity,
        fs=num_points / (orbit.t[-1] - orbit.t[0])
    )
    
    # 总路径损耗加阴影
    total_pl_with_shadow = total_pl_db + shadow_db
    
    # 接收功率和SNR
    for i in range(num_points):
        rx = calculate_received_power(
            eirp_dbm=tx_eirp_dbm,
            tx_gain_db=0,  # EIRP已包含发射增益
            rx_gain_db=rx_gain_db,
            path_loss_db=total_pl_with_shadow[i],
            noise_figure_db=noise_figure_db,
            bandwidth_hz=bandwidth_hz
        )
        rx_power_dbm[i] = rx['rx_power_dbm']
        snr_db[i] = rx['snr_db']
    
    # 生成小尺度衰落（过顶附近取典型值）
    # 取仰角45度处的平均多普勒
    mid_idx = num_points // 2
    avg_doppler = np.abs(orbit.doppler_hz[mid_idx])
    avg_k = get_k_factor(env, orbit.elevation_deg[mid_idx])
    
    small_scale = generate_rician_fading(
        num_samples=num_points,
        k_factor_linear=avg_k,
        fd_hz=avg_doppler,
        fs=num_points / (orbit.t[-1] - orbit.t[0])
    )
    small_scale_amp = np.abs(small_scale)
    
    # 衰落统计（去大尺度后）
    # 这里用接收功率的快变分量统计
    metrics = calculate_fading_metrics(small_scale)
    
    return ChannelResult(
        orbit=orbit,
        fspl_db=fspl_db,
        atmosphere_db=atmosphere_db,
        rain_db=rain_db,
        total_pl_db=total_pl_with_shadow,
        shadow_db=shadow_db,
        small_scale_amp=small_scale_amp,
        k_factor_db=k_factor_db,
        tau_rms_ns=tau_rms_ns,
        coherence_bw_mhz=coherence_bw_mhz,
        rx_power_dbm=rx_power_dbm,
        snr_db=snr_db,
        fade_depth_db=metrics['fade_depth_db'],
        lcr_hz=metrics['lcr_hz'],
        environment=env
    )


def simulate_multiple_environments(
    environments: List[EnvironmentParams],
    orbit: OrbitParams,
    **kwargs
) -> List[ChannelResult]:
    """
    对多个环境进行对比仿真
    """
    results = []
    for env in environments:
        print(f"🔄 正在仿真环境: {env.name_cn}...")
        result = simulate_channel(env, orbit, **kwargs)
        results.append(result)
        print(f"   ✅ 完成！衰落深度: {result.fade_depth_db:.1f} dB")
    return results
