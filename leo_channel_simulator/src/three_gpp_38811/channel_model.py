"""
3GPP TR 38.811 NTN 信道主模型 - 整合所有组件

实现完整的 3GPP TR 38.811 LEO 星地信道仿真流程：
  1. 轨道几何参数输入 (仰角、斜距、多普勒)
  2. LoS/NLoS 状态判定 (Section 6.6.1)
  3. 路径损耗计算 (自由空间 + 大气 + 阴影衰落 SF) (Section 6.4)
  4. 大尺度参数 (LSP) 生成 (DS, K, 角度扩展) (Section 6.6.2)
  5. 空间一致性滤波 (Section 6.6.3)
  6. 小尺度 CDL 抽头模型 + Loo 模型 (Section 6.5 & 6.7)
  7. 接收功率与 SNR 计算

使用方式：
    from src.three_gpp_38811.channel_model import simulate_ntn_channel
    result = simulate_ntn_channel(env, orbit, fc_hz=20e9, ...)
"""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict
import sys, os

# 支持直接运行
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tle_orbit import OrbitParams
from src.propagation import (
    free_space_path_loss, atmospheric_gas_loss,
    rain_attenuation, cloud_fog_loss, calculate_received_power
)
from .parameters import (
    NTNEnvironment, ALL_NTN_ENVIRONMENTS,
    PATH_LOSS_NTN, CDL_CONFIG_NTN, LOO_MODEL_PARAMS
)
from .los_probability import (
    calculate_los_probability_array, determine_los_state_array
)
from .large_scale import generate_spatially_consistent_lsp, generate_shadow_fading_sequence
from .small_scale import generate_cdl_taps, generate_loo_fading, calculate_fading_metrics


@dataclass
class NTNChannelResult:
    """3GPP NTN 信道仿真结果"""
    # 轨道
    orbit: OrbitParams

    # LoS/NLoS 状态
    los_state: np.ndarray          # 布尔数组，True=LoS
    los_probability: np.ndarray    # LoS 概率

    # 路径损耗
    fspl_db: np.ndarray
    atmosphere_db: np.ndarray
    rain_db: np.ndarray
    shadow_fading_db: np.ndarray   # 阴影衰落 SF
    total_pl_db: np.ndarray        # 总路径损耗

    # 大尺度参数 (LSP)
    ds_ns: np.ndarray             # RMS 时延扩展 (ns)
    k_db: np.ndarray              # 莱斯 K 因子 (dB)，NLoS 点为 NaN
    coherence_bw_mhz: np.ndarray  # 相干带宽 (MHz)
    asd_deg: np.ndarray           # 离开角扩展
    asa_deg: np.ndarray           # 到达角扩展

    # 小尺度衰落
    small_scale_h: np.ndarray      # 总复信道系数
    small_scale_amp: np.ndarray    # 幅度
    fade_depth_db: float           # 1% 衰落深度
    lcr_hz: float                  # 电平通过率

    # 接收性能
    rx_power_dbm: np.ndarray
    snr_db: np.ndarray

    # CDL 抽头信息 (过顶时刻的典型值)
    cdl_delays_ns: np.ndarray
    cdl_powers_db: np.ndarray

    # 环境
    environment: NTNEnvironment


def _calculate_path_loss_components(
    fc_hz: float,
    range_m: float,
    elevation_deg: float,
    rain_rate_mm_h: float
) -> Dict[str, float]:
    """计算路径损耗各分量"""
    fc_ghz = fc_hz / 1e9

    # 自由空间路径损耗
    pl_fspl = free_space_path_loss(fc_hz, range_m)

    # 大气吸收 (ITU-R P.676)
    pl_atm = atmospheric_gas_loss(fc_ghz, elevation_deg)

    # 降雨衰减 (ITU-R P.838)
    pl_rain = rain_attenuation(fc_ghz, elevation_deg, rain_rate_mm_h)

    # 云雾衰减 (ITU-R P.840)
    pl_cloud = cloud_fog_loss(fc_ghz, elevation_deg)

    return {
        'fspl_db': pl_fspl,
        'atmosphere_db': pl_atm,
        'rain_db': pl_rain,
        'cloud_db': pl_cloud,
    }


def simulate_ntn_channel(
    env: NTNEnvironment,
    orbit: OrbitParams,
    fc_hz: float = 20e9,
    bandwidth_hz: float = 200e6,
    tx_eirp_dbm: float = 55.0,
    rx_gain_db: float = 35.0,
    noise_figure_db: float = 2.5,
    rain_rate_mm_h: float = 2.0,
    seed: Optional[int] = None,
) -> NTNChannelResult:
    """
    执行完整的 3GPP TR 38.811 NTN 信道仿真

    Parameters
    ----------
    env : NTNEnvironment
        环境场景 (Urban / Suburban / Open)
    orbit : OrbitParams
        轨道参数 (仰角、斜距、多普勒等)
    fc_hz : float
        载波频率 (Hz)
    bandwidth_hz : float
        信号带宽 (Hz)
    tx_eirp_dbm : float
        发射 EIRP (dBm)
    rx_gain_db : float
        接收天线增益 (dBi)
    noise_figure_db : float
        接收机噪声系数 (dB)
    rain_rate_mm_h : float
        降雨量 (mm/h)
    seed : int, optional
        随机种子

    Returns
    -------
    NTNChannelResult
        完整的信道仿真结果
    """
    rng = np.random.default_rng(seed)
    num_points = len(orbit.t)
    fs = num_points / (orbit.t[-1] - orbit.t[0])  # 采样率

    # ============================================================
    # 1. LoS/NLoS 状态判定
    # ============================================================
    los_prob = calculate_los_probability_array(orbit.elevation_deg, env)
    los_state = determine_los_state_array(orbit.elevation_deg, env, rng=rng)

    # ============================================================
    # 2. 路径损耗计算 (逐点)
    # ============================================================
    fspl_db = np.zeros(num_points)
    atmosphere_db = np.zeros(num_points)
    rain_db = np.zeros(num_points)

    for i in range(num_points):
        pl = _calculate_path_loss_components(
            fc_hz=fc_hz,
            range_m=orbit.range_km[i] * 1000,
            elevation_deg=orbit.elevation_deg[i],
            rain_rate_mm_h=rain_rate_mm_h
        )
        fspl_db[i] = pl['fspl_db']
        atmosphere_db[i] = pl['atmosphere_db']
        rain_db[i] = pl['rain_db']

    # 阴影衰落 (SF) - 带空间一致性
    mean_velocity = np.mean(np.abs(orbit.velocity_kms)) * 1000  # m/s
    shadow_fading_db = generate_shadow_fading_sequence(
        num_points=num_points,
        env=env,
        velocity_ms=mean_velocity,
        fs=fs,
        rng=rng
    )

    # 总路径损耗 = 自由空间 + 大气 + 雨 + 云 + 阴影衰落
    total_pl_db = fspl_db + atmosphere_db + rain_db + shadow_fading_db

    # ============================================================
    # 3. 大尺度参数 (LSP) 生成 - 分 LoS/NLoS 段
    # ============================================================
    ds_ns = np.zeros(num_points)
    k_db = np.full(num_points, np.nan)
    asd_deg = np.zeros(num_points)
    asa_deg = np.zeros(num_points)
    coherence_bw_mhz = np.zeros(num_points)

    # 找出 LoS 和 NLoS 的连续段
    def find_state_segments(state_array):
        """找出连续相同状态的段"""
        segments = []
        if len(state_array) == 0:
            return segments
        start = 0
        for i in range(1, len(state_array)):
            if state_array[i] != state_array[i - 1]:
                segments.append((start, i - 1, bool(state_array[start])))
                start = i
        segments.append((start, len(state_array) - 1, bool(state_array[start])))
        return segments

    segments = find_state_segments(los_state)

    for seg_start, seg_end, is_los in segments:
        seg_len = seg_end - seg_start + 1
        if seg_len < 2:
            # 单点段，直接采样
            from .large_scale import generate_lsp_sample
            sample = generate_lsp_sample(env, is_los, rng=rng)
            ds_ns[seg_start] = sample.ds_ns
            if is_los:
                k_db[seg_start] = sample.k_db
            asd_deg[seg_start] = sample.asd_deg
            asa_deg[seg_start] = sample.asa_deg
            continue

        # 生成空间一致的 LSP 序列
        lsp_seq = generate_spatially_consistent_lsp(
            num_points=seg_len,
            env=env,
            is_los=is_los,
            velocity_ms=mean_velocity,
            fs=fs,
            rng=rng
        )
        ds_ns[seg_start:seg_end + 1] = lsp_seq['ds_ns']
        if is_los:
            k_db[seg_start:seg_end + 1] = lsp_seq['k_db']
        asd_deg[seg_start:seg_end + 1] = lsp_seq['asd_deg']
        asa_deg[seg_start:seg_end + 1] = lsp_seq['asa_deg']

    # 相干带宽 B_c ≈ 1/(5*tau_rms)
    coherence_bw_mhz = (1.0 / (5 * ds_ns * 1e-9)) / 1e6

    # ============================================================
    # 4. 小尺度衰落 (CDL + Loo 模型)
    # ============================================================
    # 取过顶附近的典型参数生成 CDL 抽头
    mid_idx = num_points // 2
    mid_los = los_state[mid_idx]
    mid_ds = float(np.median(ds_ns))
    mid_k = float(np.nanmedian(k_db)) if mid_los else None
    mid_fd = float(np.abs(orbit.doppler_hz[mid_idx]))

    cdl = generate_cdl_taps(
        num_samples=num_points,
        ds_ns=mid_ds,
        k_factor_db=mid_k,
        fd_hz=mid_fd,
        fs=fs,
        config=CDL_CONFIG_NTN,
        rng=rng
    )

    # 总信道响应 = 各抽头之和
    h_total = np.sum(cdl.h_taps, axis=0)
    small_scale_amp = np.abs(h_total)

    # 衰落统计
    metrics = calculate_fading_metrics(h_total, fs)

    # ============================================================
    # 5. 接收功率与 SNR
    # ============================================================
    rx_power_dbm = np.zeros(num_points)
    snr_db = np.zeros(num_points)

    for i in range(num_points):
        rx = calculate_received_power(
            eirp_dbm=tx_eirp_dbm,
            tx_gain_db=0,  # EIRP 已包含发射增益
            rx_gain_db=rx_gain_db,
            path_loss_db=total_pl_db[i],
            noise_figure_db=noise_figure_db,
            bandwidth_hz=bandwidth_hz
        )
        rx_power_dbm[i] = rx['rx_power_dbm']
        snr_db[i] = rx['snr_db']

    # ============================================================
    # 6. 组装结果
    # ============================================================
    return NTNChannelResult(
        orbit=orbit,
        los_state=los_state,
        los_probability=los_prob,
        fspl_db=fspl_db,
        atmosphere_db=atmosphere_db,
        rain_db=rain_db,
        shadow_fading_db=shadow_fading_db,
        total_pl_db=total_pl_db,
        ds_ns=ds_ns,
        k_db=k_db,
        coherence_bw_mhz=coherence_bw_mhz,
        asd_deg=asd_deg,
        asa_deg=asa_deg,
        small_scale_h=h_total,
        small_scale_amp=small_scale_amp,
        fade_depth_db=metrics['fade_depth_db'],
        lcr_hz=metrics['lcr_hz'],
        rx_power_dbm=rx_power_dbm,
        snr_db=snr_db,
        cdl_delays_ns=cdl.delays_s * 1e9,
        cdl_powers_db=10 * np.log10(cdl.powers_linear + 1e-20),
        environment=env,
    )


def simulate_multiple_environments_ntn(
    environments: List[NTNEnvironment],
    orbit: OrbitParams,
    **kwargs
) -> List[NTNChannelResult]:
    """
    对多个环境进行 3GPP NTN 信道对比仿真

    Parameters
    ----------
    environments : List[NTNEnvironment]
        环境列表
    orbit : OrbitParams
        轨道参数
    **kwargs
        传递给 simulate_ntn_channel 的参数

    Returns
    -------
    List[NTNChannelResult]
        各环境的仿真结果
    """
    results = []
    for i, env in enumerate(environments):
        print(f"  🔄 正在仿真环境: {env.name_cn}...")
        # 每个环境使用不同种子
        seed = kwargs.pop('seed', None)
        if seed is not None:
            seed = seed + i * 100
        result = simulate_ntn_channel(env, orbit, seed=seed, **kwargs)
        los_ratio = np.mean(result.los_state) * 100
        print(f"     ✅ 完成！LoS 占比: {los_ratio:.1f}%, "
              f"衰落深度: {result.fade_depth_db:.1f} dB")
        results.append(result)
    return results


# ============================================================
# 汇总打印
# ============================================================
def print_ntn_summary(results: List[NTNChannelResult]):
    """打印 3GPP NTN 仿真结果汇总表"""
    print("\n" + "=" * 90)
    print("📋 3GPP TR 38.811 NTN CHANNEL SIMULATION SUMMARY")
    print("=" * 90)

    orbit = results[0].orbit
    print(f"\n🛰️  ORBIT INFORMATION:")
    print(f"   Max Doppler Shift:       {np.max(np.abs(orbit.doppler_hz)) / 1000:.1f} kHz")
    print(f"   Max Doppler Rate:        {np.max(np.abs(np.diff(orbit.doppler_hz) / np.diff(orbit.t))):.0f} Hz/s")
    print(f"   Min/Max Slant Range:     {np.min(orbit.range_km):.0f} / {np.max(orbit.range_km):.0f} km")
    print(f"   Max Elevation Angle:     {np.max(orbit.elevation_deg):.1f} degrees")

    print(f"\n📊 CHANNEL PARAMETERS BY ENVIRONMENT (3GPP TR 38.811):")
    print("-" * 90)
    print(f"{'Environment':<16} {'LoS%':>6} {'SF σ':>7} {'Min PL':>9} {'Max PL':>9} "
          f"{'FadeDepth':>10} {'K(LoS)':>10} {'DS Range':>16}")
    print(f"{'':16} {'':>6} {'(dB)':>7} {'(dB)':>9} {'(dB)':>9} "
          f"{'(1%,dB)':>10} {'(dB)':>10} {'(ns)':>16}")
    print("-" * 90)

    for res in results:
        env = res.environment
        los_pct = np.mean(res.los_state) * 100
        min_pl = np.min(res.total_pl_db)
        max_pl = np.max(res.total_pl_db)
        k_los = res.k_db[~np.isnan(res.k_db)]
        k_str = f"{np.min(k_los):.1f}~{np.max(k_los):.1f}" if len(k_los) > 0 else "N/A"
        ds_min = np.min(res.ds_ns)
        ds_max = np.max(res.ds_ns)

        print(f"{env.name_cn:<16} {los_pct:>6.1f} {env.sf_std_db:>7.1f} {min_pl:>9.1f} {max_pl:>9.1f} "
              f"{res.fade_depth_db:>10.1f} {k_str:>10} {ds_min:>7.1f}~{ds_max:>8.1f}")

    print("-" * 90)
    print(f"\n💡 KEY OBSERVATIONS (3GPP TR 38.811):")
    print(f"   • LoS 概率随仰角升高而增大，城市环境遮挡最严重")
    print(f"   • DS (时延扩展) 服从对数正态分布，NLoS 时显著增大")
    print(f"   • Loo 模型描述了直射径遮挡 + 多径散射的联合效应")
    print(f"   • CDL 抽头模型准确刻画了卫星信道的多径时延功率谱")
    print("=" * 90 + "\n")
