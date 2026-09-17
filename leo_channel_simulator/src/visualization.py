"""
可视化模块
绘制轨道、损耗、衰落、环境对比等各类图表
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from typing import List
from .channel_model import ChannelResult
from .environment import ALL_ENVIRONMENTS

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'WenQuanYi Zen Hei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100


def plot_single_channel_result(result: ChannelResult, save_path: str = None):
    """
    绘制单个信道的详细结果
    """
    fig, axes = plt.subplots(4, 2, figsize=(16, 18))
    fig.suptitle(f'LEO Satellite Channel Analysis - {result.environment.name_cn} Environment', fontsize=16, fontweight='bold')
    
    t = result.orbit.t / 60.0  # 转换为分钟
    elev = result.orbit.elevation_deg
    
    # 1. 仰角和斜距
    ax1 = axes[0, 0]
    ax1.plot(t, elev, 'b-', linewidth=2, label='Elevation Angle')
    ax1.set_xlabel('Time (minutes)')
    ax1.set_ylabel('Elevation Angle (degrees)', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.set_ylim(0, 95)
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Orbit Geometry: Elevation & Slant Range')
    
    ax1b = ax1.twinx()
    ax1b.plot(t, result.orbit.range_km, 'r--', linewidth=2, label='Slant Range')
    ax1b.set_ylabel('Slant Range (km)', color='r')
    ax1b.tick_params(axis='y', labelcolor='r')
    
    # 2. 多普勒频移
    ax2 = axes[0, 1]
    ax2.plot(t, result.orbit.doppler_hz / 1000, 'g-', linewidth=2)
    ax2.set_xlabel('Time (minutes)')
    ax2.set_ylabel('Doppler Shift (kHz)')
    ax2.grid(True, alpha=0.3)
    ax2.set_title(f'Doppler Shift (Max: {np.max(np.abs(result.orbit.doppler_hz))/1000:.1f} kHz)')
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    # 3. 路径损耗分解
    ax3 = axes[1, 0]
    ax3.plot(t, result.fspl_db, 'b-', linewidth=2, label='Free Space Loss')
    ax3.plot(t, result.atmosphere_db, 'g--', linewidth=2, label='Atmospheric Loss')
    ax3.plot(t, result.rain_db, 'r-.', linewidth=2, label='Rain Attenuation')
    ax3.set_xlabel('Time (minutes)')
    ax3.set_ylabel('Path Loss (dB)')
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3)
    ax3.set_title('Path Loss Components')
    
    # 4. 阴影衰落
    ax4 = axes[1, 1]
    ax4.plot(t, result.shadow_db, color='#8e44ad', linewidth=1)
    ax4.set_xlabel('Time (minutes)')
    ax4.set_ylabel('Shadow Fading (dB)')
    ax4.grid(True, alpha=0.3)
    ax4.set_title(f'Shadow Fading (σ={result.environment.sigma_shadow_dB:.1f} dB)')
    ax4.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    # 5. 接收功率和SNR
    ax5 = axes[2, 0]
    ax5.plot(t, result.rx_power_dbm, 'm-', linewidth=2, label='Received Power')
    ax5.set_xlabel('Time (minutes)')
    ax5.set_ylabel('Received Power (dBm)', color='m')
    ax5.tick_params(axis='y', labelcolor='m')
    ax5.grid(True, alpha=0.3)
    ax5.set_title('Received Power & SNR')
    
    ax5b = ax5.twinx()
    ax5b.plot(t, result.snr_db, 'c--', linewidth=2, label='SNR')
    ax5b.set_ylabel('SNR (dB)', color='c')
    ax5b.tick_params(axis='y', labelcolor='c')
    
    # 6. 莱斯K因子
    ax6 = axes[2, 1]
    ax6.plot(t, result.k_factor_db, color='#d35400', linewidth=2)
    ax6.set_xlabel('Time (minutes)')
    ax6.set_ylabel('Rician K Factor (dB)')
    ax6.grid(True, alpha=0.3)
    ax6.set_title('Rician K Factor vs Elevation')
    
    # 7. 小尺度衰落波形
    ax7 = axes[3, 0]
    ax7.plot(t, 20*np.log10(result.small_scale_amp + 1e-10), color='#27ae60', linewidth=0.8, alpha=0.8)
    ax7.set_xlabel('Time (minutes)')
    ax7.set_ylabel('Small-Scale Fading (dB)')
    ax7.grid(True, alpha=0.3)
    ax7.set_title(f'Small-Scale Fading Envelope (1% Fade Depth: {result.fade_depth_db:.1f} dB)')
    ax7.set_ylim(-30, 5)
    
    # 8. 时延扩展
    ax8 = axes[3, 1]
    ax8.plot(t, result.tau_rms_ns, color='#2980b9', linewidth=2, label='RMS Delay Spread')
    ax8.set_xlabel('Time (minutes)')
    ax8.set_ylabel('RMS Delay Spread (ns)', color='#2980b9')
    ax8.tick_params(axis='y', labelcolor='#2980b9')
    ax8.set_yscale('log')
    ax8.grid(True, alpha=0.3)
    ax8.set_title('Delay Spread & Coherence Bandwidth')
    
    ax8b = ax8.twinx()
    ax8b.plot(t, result.coherence_bw_mhz, color='#c0392b', linestyle='--', linewidth=2, label='Coherence BW')
    ax8b.set_ylabel('Coherence Bandwidth (MHz)', color='#c0392b')
    ax8b.set_yscale('log')
    ax8b.tick_params(axis='y', labelcolor='#c0392b')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 单环境结果已保存至: {save_path}")
    
    plt.close()


def plot_environment_comparison(results: List[ChannelResult], save_path: str = None):
    """
    绘制多环境对比图
    """
    fig, axes = plt.subplots(3, 2, figsize=(16, 15))
    fig.suptitle('LEO Satellite Channel: Multi-Environment Comparison', fontsize=16, fontweight='bold')
    
    t = results[0].orbit.t / 60.0
    elev = results[0].orbit.elevation_deg
    
    # 1. 仰角参考
    ax0 = axes[0, 0]
    ax0.plot(t, elev, 'k-', linewidth=2)
    ax0.set_xlabel('Time (minutes)')
    ax0.set_ylabel('Elevation Angle (degrees)')
    ax0.grid(True, alpha=0.3)
    ax0.set_title('Elevation Angle (Reference)')
    ax0.fill_between(t, 0, elev, alpha=0.1, color='gray')
    
    # 2. 总路径损耗对比
    ax1 = axes[0, 1]
    for res in results:
        ax1.plot(t, res.total_pl_db, color=res.environment.color, linestyle=res.environment.linestyle,
                 linewidth=2, label=f'{res.environment.name_cn}')
    ax1.set_xlabel('Time (minutes)')
    ax1.set_ylabel('Total Path Loss (dB)')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Total Path Loss Comparison')
    
    # 3. 接收功率对比
    ax2 = axes[1, 0]
    for res in results:
        ax2.plot(t, res.rx_power_dbm, color=res.environment.color, linestyle=res.environment.linestyle,
                 linewidth=2, label=f'{res.environment.name_cn}')
    ax2.set_xlabel('Time (minutes)')
    ax2.set_ylabel('Received Power (dBm)')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Received Power Comparison')
    
    # 4. SNR对比
    ax3 = axes[1, 1]
    for res in results:
        ax3.plot(t, res.snr_db, color=res.environment.color, linestyle=res.environment.linestyle,
                 linewidth=2, label=f'{res.environment.name_cn}')
    ax3.axhline(y=10, color='k', linestyle='--', alpha=0.5, label='QPSK threshold (10dB)')
    ax3.set_xlabel('Time (minutes)')
    ax3.set_ylabel('SNR (dB)')
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3)
    ax3.set_title('SNR Comparison')
    
    # 5. K因子对比
    ax4 = axes[2, 0]
    for res in results:
        ax4.plot(t, res.k_factor_db, color=res.environment.color, linestyle=res.environment.linestyle,
                 linewidth=2, label=f'{res.environment.name_cn}')
    ax4.set_xlabel('Time (minutes)')
    ax4.set_ylabel('Rician K Factor (dB)')
    ax4.legend(loc='best')
    ax4.grid(True, alpha=0.3)
    ax4.set_title('Rician K Factor Comparison')
    
    # 6. 时延扩展对比
    ax5 = axes[2, 1]
    for res in results:
        ax5.plot(t, res.tau_rms_ns, color=res.environment.color, linestyle=res.environment.linestyle,
                 linewidth=2, label=f'{res.environment.name_cn}')
    ax5.set_xlabel('Time (minutes)')
    ax5.set_ylabel('RMS Delay Spread (ns)')
    ax5.set_yscale('log')
    ax5.legend(loc='best')
    ax5.grid(True, alpha=0.3)
    ax5.set_title('RMS Delay Spread Comparison')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 多环境对比结果已保存至: {save_path}")
    
    plt.close()


def print_summary_table(results: List[ChannelResult]):
    """
    打印结果汇总表
    """
    print("\n" + "="*80)
    print("📋 LEO STAR-TO-GROUND CHANNEL SIMULATION SUMMARY")
    print("="*80)
    
    # 轨道信息
    orbit = results[0].orbit
    print(f"\n🛰️  ORBIT INFORMATION:")
    print(f"   Max Doppler Shift:       {np.max(np.abs(orbit.doppler_hz))/1000:.1f} kHz")
    print(f"   Max Doppler Rate:        {np.max(np.abs(np.diff(orbit.doppler_hz)/np.diff(orbit.t))):.0f} Hz/s")
    print(f"   Min Slant Range:         {np.min(orbit.range_km):.0f} km")
    print(f"   Max Slant Range:         {np.max(orbit.range_km):.0f} km")
    print(f"   Max Elevation Angle:     {np.max(orbit.elevation_deg):.1f} degrees")
    
    print(f"\n📊 CHANNEL PARAMETERS BY ENVIRONMENT:")
    print("-"*80)
    print(f"{'Environment':<12} {'Shadow σ':>10} {'Min PL':>10} {'Max PL':>10} {'Fade Depth':>12} {'K Range':>12} {'τ_rms Range':>15}")
    print(f"{'':12} {'(dB)':>10} {'(dB)':>10} {'(dB)':>10} {'(1%, dB)':>12} {'(dB)':>12} {'(ns)':>15}")
    print("-"*80)
    
    for res in results:
        env = res.environment
        min_pl = np.min(res.total_pl_db)
        max_pl = np.max(res.total_pl_db)
        k_min = np.min(res.k_factor_db)
        k_max = np.max(res.k_factor_db)
        tau_min = np.min(res.tau_rms_ns)
        tau_max = np.max(res.tau_rms_ns)
        
        print(f"{env.name_cn:<12} {env.sigma_shadow_dB:>10.1f} {min_pl:>10.1f} {max_pl:>10.1f} "
              f"{res.fade_depth_db:>12.1f} {k_min:>5.1f}~{k_max:>5.1f} {tau_min:>6.1f}~{tau_max:>7.1f}")
    
    print("-"*80)
    print(f"\n💡 KEY OBSERVATIONS:")
    print(f"   • Path loss varies by ~{np.max(results[-1].total_pl_db) - np.min(results[0].total_pl_db):.0f}dB across full pass")
    print(f"   • Urban areas suffer from deepest fades (>20dB) and largest delay spread (>100ns)")
    print(f"   • Open areas have near-AWGN conditions at high elevation (K>20dB)")
    print(f"   • Doppler shift changes sign at zenith, requiring fast reacquisition")
    print("="*80 + "\n")
