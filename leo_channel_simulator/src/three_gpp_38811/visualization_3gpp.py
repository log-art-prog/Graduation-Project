"""
3GPP TR 38.811 NTN 信道可视化模块

绘制 3GPP 星地信道仿真结果：
  - 轨道几何 (仰角、斜距)
  - LoS 概率与状态
  - 路径损耗分解 (自由空间/大气/雨/阴影衰落)
  - 大尺度参数 (DS、K 因子、相干带宽)
  - 小尺度衰落 (Loo 模型包络、CDL 抽头)
  - 接收功率与 SNR
  - 多环境对比
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import List
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.three_gpp_38811.channel_model import NTNChannelResult

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100


def plot_ntn_single(result: NTNChannelResult, save_path: str = None):
    """
    绘制单个 3GPP NTN 环境的详细信道分析 (8 子图)
    """
    fig, axes = plt.subplots(4, 2, figsize=(16, 18))
    fig.suptitle(f'3GPP TR 38.811 NTN Channel - {result.environment.name_cn}',
                 fontsize=15, fontweight='bold')

    t = result.orbit.t / 60.0
    env = result.environment

    # 1. 仰角 & 斜距
    ax1 = axes[0, 0]
    ax1.plot(t, result.orbit.elevation_deg, 'b-', lw=2, label='Elevation')
    ax1.set_xlabel('Time (min)')
    ax1.set_ylabel('Elevation (°)', color='b')
    ax1.set_ylim(0, 95)
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Orbit Geometry')
    ax1b = ax1.twinx()
    ax1b.plot(t, result.orbit.range_km, 'r--', lw=2, label='Range')
    ax1b.set_ylabel('Slant Range (km)', color='r')

    # 2. LoS 概率 & 状态
    ax2 = axes[0, 1]
    ax2.plot(t, result.los_probability, 'g-', lw=2, label='P_LoS')
    ax2.fill_between(t, 0, result.los_probability, alpha=0.15, color='green')
    ax2.scatter(t, result.los_state.astype(float), c='red', s=3, alpha=0.5, label='LoS State')
    ax2.set_xlabel('Time (min)')
    ax2.set_ylabel('LoS Probability')
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    ax2.set_title(f'LoS Probability & State (LoS ratio: {np.mean(result.los_state)*100:.1f}%)')

    # 3. 路径损耗分解
    ax3 = axes[1, 0]
    ax3.plot(t, result.fspl_db, 'b-', lw=1.5, label='Free Space')
    ax3.plot(t, result.atmosphere_db, 'g--', lw=1.5, label='Atmosphere')
    ax3.plot(t, result.rain_db, 'r-.', lw=1.5, label='Rain')
    ax3.plot(t, result.shadow_fading_db, 'm-', lw=1, alpha=0.7, label='Shadow Fading')
    ax3.plot(t, result.total_pl_db, 'k-', lw=2, label='Total PL')
    ax3.set_xlabel('Time (min)')
    ax3.set_ylabel('Path Loss (dB)')
    ax3.legend(loc='best', fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.set_title('Path Loss Components')

    # 4. 阴影衰落直方图
    ax4 = axes[1, 1]
    ax4.hist(result.shadow_fading_db, bins=40, color='purple', alpha=0.7, edgecolor='black')
    ax4.axvline(0, color='k', linestyle='--', alpha=0.5)
    ax4.set_xlabel('Shadow Fading (dB)')
    ax4.set_ylabel('Count')
    ax4.set_title(f'Shadow Fading Distribution (σ={env.sf_std_db:.1f} dB)')
    ax4.grid(True, alpha=0.3)

    # 5. DS & 相干带宽
    ax5 = axes[2, 0]
    ax5.semilogy(t, result.ds_ns, color='#2980b9', lw=1.5, label='RMS DS')
    ax5.set_xlabel('Time (min)')
    ax5.set_ylabel('RMS Delay Spread (ns)', color='#2980b9')
    ax5.grid(True, alpha=0.3, which='both')
    ax5.set_title('Delay Spread & Coherence Bandwidth')
    ax5b = ax5.twinx()
    ax5b.semilogy(t, result.coherence_bw_mhz, color='#c0392b', lw=1.5, ls='--', label='Coherence BW')
    ax5b.set_ylabel('Coherence BW (MHz)', color='#c0392b')

    # 6. K 因子 (仅 LoS 点)
    ax6 = axes[2, 1]
    k_valid = ~np.isnan(result.k_db)
    if np.any(k_valid):
        ax6.plot(t[k_valid], result.k_db[k_valid], color='#d35400', lw=1.5, marker='o', ms=2)
    ax6.set_xlabel('Time (min)')
    ax6.set_ylabel('Rician K Factor (dB)')
    ax6.grid(True, alpha=0.3)
    ax6.set_title('Rician K Factor (LoS points only)')

    # 7. 小尺度衰落包络
    ax7 = axes[3, 0]
    fading_db = 20 * np.log10(result.small_scale_amp / np.mean(result.small_scale_amp) + 1e-10)
    ax7.plot(t, fading_db, color='#27ae60', lw=0.5, alpha=0.8)
    ax7.set_xlabel('Time (min)')
    ax7.set_ylabel('Small-Scale Fading (dB)')
    ax7.set_ylim(-40, 5)
    ax7.grid(True, alpha=0.3)
    ax7.set_title(f'Small-Scale Fading (1% fade: {result.fade_depth_db:.1f} dB)')

    # 8. 接收功率 & SNR
    ax8 = axes[3, 1]
    ax8.plot(t, result.rx_power_dbm, 'm-', lw=1.5, label='Rx Power')
    ax8.set_xlabel('Time (min)')
    ax8.set_ylabel('Rx Power (dBm)', color='m')
    ax8.grid(True, alpha=0.3)
    ax8.set_title('Received Power & SNR')
    ax8b = ax8.twinx()
    ax8b.plot(t, result.snr_db, 'c--', lw=1.5, label='SNR')
    ax8b.set_ylabel('SNR (dB)', color='c')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  📊 已保存: {save_path}")
    plt.close()


def plot_ntn_comparison(results: List[NTNChannelResult], save_path: str = None):
    """
    绘制多环境 3GPP NTN 对比图 (6 子图)
    """
    fig, axes = plt.subplots(3, 2, figsize=(16, 15))
    fig.suptitle('3GPP TR 38.811 NTN: Multi-Environment Comparison',
                 fontsize=15, fontweight='bold')

    t = results[0].orbit.t / 60.0
    elev = results[0].orbit.elevation_deg

    # 1. 仰角参考
    ax0 = axes[0, 0]
    ax0.plot(t, elev, 'k-', lw=2)
    ax0.fill_between(t, 0, elev, alpha=0.1, color='gray')
    ax0.set_xlabel('Time (min)')
    ax0.set_ylabel('Elevation (°)')
    ax0.grid(True, alpha=0.3)
    ax0.set_title('Elevation Angle (Reference)')

    # 2. LoS 概率对比
    ax1 = axes[0, 1]
    for res in results:
        ax1.plot(t, res.los_probability, color=res.environment.color,
                 ls=res.environment.linestyle, lw=2, label=res.environment.name)
    ax1.set_xlabel('Time (min)')
    ax1.set_ylabel('LoS Probability')
    ax1.set_ylim(0, 1.05)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_title('LoS Probability Comparison')

    # 3. 总路径损耗对比
    ax2 = axes[1, 0]
    for res in results:
        ax2.plot(t, res.total_pl_db, color=res.environment.color,
                 ls=res.environment.linestyle, lw=2, label=res.environment.name)
    ax2.set_xlabel('Time (min)')
    ax2.set_ylabel('Total Path Loss (dB)')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Total Path Loss Comparison')

    # 4. SNR 对比
    ax3 = axes[1, 1]
    for res in results:
        ax3.plot(t, res.snr_db, color=res.environment.color,
                 ls=res.environment.linestyle, lw=2, label=res.environment.name)
    ax3.axhline(10, color='k', ls='--', alpha=0.5, label='QPSK threshold')
    ax3.set_xlabel('Time (min)')
    ax3.set_ylabel('SNR (dB)')
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3)
    ax3.set_title('SNR Comparison')

    # 5. DS 对比
    ax4 = axes[2, 0]
    for res in results:
        ax4.semilogy(t, res.ds_ns, color=res.environment.color,
                     ls=res.environment.linestyle, lw=2, label=res.environment.name)
    ax4.set_xlabel('Time (min)')
    ax4.set_ylabel('RMS Delay Spread (ns)')
    ax4.legend(loc='best')
    ax4.grid(True, alpha=0.3, which='both')
    ax4.set_title('RMS Delay Spread Comparison')

    # 6. 接收功率对比
    ax5 = axes[2, 1]
    for res in results:
        ax5.plot(t, res.rx_power_dbm, color=res.environment.color,
                 ls=res.environment.linestyle, lw=2, label=res.environment.name)
    ax5.set_xlabel('Time (min)')
    ax5.set_ylabel('Rx Power (dBm)')
    ax5.legend(loc='best')
    ax5.grid(True, alpha=0.3)
    ax5.set_title('Received Power Comparison')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  📊 已保存: {save_path}")
    plt.close()


def plot_cdl_taps(result: NTNChannelResult, save_path: str = None):
    """绘制 CDL 抽头时延功率谱"""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.stem(result.cdl_delays_ns, result.cdl_powers_db,
            linefmt='b-', markerfmt='bo', basefmt=' ')
    ax.set_xlabel('Delay (ns)')
    ax.set_ylabel('Power (dB)')
    ax.set_title(f'3GPP CDL Delay Power Profile - {result.environment.name}')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  📊 已保存: {save_path}")
    plt.close()
