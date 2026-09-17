#!/usr/bin/env python3
"""
3GPP TR 38.811 NTN 星地信道仿真器 - 主程序入口

完整实现 3GPP TR 38.811 V17.0.0 定义的低轨卫星星地信道模型：
  - LoS/NLoS 概率模型 (Section 6.6.1)
  - 大尺度参数 LSP (DS, K, 角度扩展) (Section 6.6.2)
  - 空间一致性 (Section 6.6.3)
  - CDL 抽头延迟线 + Loo 模型 (Section 6.5, 6.7)
  - 路径损耗与阴影衰落 (Section 6.4)

使用方法:
    python main_3gpp_ntn.py
"""
import os
import sys
import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.tle_orbit import simple_orbit_model, compute_orbit_from_tle
from src.three_gpp_38811.parameters import ALL_NTN_ENVIRONMENTS
from src.three_gpp_38811.channel_model import (
    simulate_ntn_channel, simulate_multiple_environments_ntn, print_ntn_summary
)
from src.three_gpp_38811.visualization_3gpp import (
    plot_ntn_single, plot_ntn_comparison, plot_cdl_taps
)


def main():
    print("=" * 80)
    print("🚀 3GPP TR 38.811 NTN SATELLITE CHANNEL SIMULATOR")
    print("=" * 80)

    # ==================== 仿真参数配置 ====================
    print("\n⚙️  配置 3GPP NTN 仿真参数...")

    # 系统参数
    FC_HZ = 20e9               # Ka 频段 20 GHz
    BANDWIDTH_HZ = 200e6       # 带宽 200 MHz
    TX_EIRP_DBM = 55.0         # 发射 EIRP (dBm)
    RX_GAIN_DB = 35.0          # 接收增益 (dBi)
    NOISE_FIGURE_DB = 2.5      # 噪声系数 (dB)
    RAIN_RATE_MMH = 2.0        # 雨率 mm/h

    # 轨道参数
    PASS_DURATION_S = 600
    NUM_POINTS = 1000
    USE_REAL_TLE = False

    # 地面站 (北京)
    GS_LAT = 39.9042
    GS_LON = 116.4074
    GS_ALT_M = 50.0

    TLE_FILE = os.path.join(os.path.dirname(__file__), 'data', 'starlink_tle.txt')
    OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"   载波频率: {FC_HZ / 1e9:.1f} GHz")
    print(f"   信号带宽: {BANDWIDTH_HZ / 1e6:.0f} MHz")
    print(f"   过境时长: {PASS_DURATION_S} s")
    print(f"   降雨量: {RAIN_RATE_MMH} mm/h")

    # ==================== 轨道计算 ====================
    print("\n🛰️  计算卫星轨道...")

    if USE_REAL_TLE and os.path.exists(TLE_FILE):
        orbit = compute_orbit_from_tle(
            tle_file_path=TLE_FILE,
            ground_station_lat=GS_LAT, ground_station_lon=GS_LON,
            ground_station_alt_m=GS_ALT_M,
            pass_duration_s=PASS_DURATION_S, num_points=NUM_POINTS,
            fc_hz=FC_HZ
        )
    else:
        print("   使用简化过顶轨道模型...")
        orbit = simple_orbit_model(
            sat_altitude_km=550, sat_velocity_kms=7.6,
            max_elevation_deg=90, pass_duration_s=PASS_DURATION_S,
            num_points=NUM_POINTS, fc_hz=FC_HZ
        )

    print(f"   最大仰角: {np.max(orbit.elevation_deg):.1f}°")
    print(f"   最大多普勒: ±{np.max(np.abs(orbit.doppler_hz)) / 1000:.1f} kHz")

    # ==================== 3GPP NTN 多环境仿真 ====================
    print("\n📡 开始 3GPP TR 38.811 多环境信道仿真...")

    all_results = simulate_multiple_environments_ntn(
        environments=ALL_NTN_ENVIRONMENTS,
        orbit=orbit,
        fc_hz=FC_HZ,
        bandwidth_hz=BANDWIDTH_HZ,
        tx_eirp_dbm=TX_EIRP_DBM,
        rx_gain_db=RX_GAIN_DB,
        noise_figure_db=NOISE_FIGURE_DB,
        rain_rate_mm_h=RAIN_RATE_MMH,
        seed=42
    )

    # ==================== 可视化 ====================
    # 单环境详细图 (以 Urban 为例，体现 3GPP 模型特性)
    print("\n🔍 生成 Urban 环境详细分析...")
    urban_result = next(r for r in all_results if r.environment.name == 'Urban')
    plot_ntn_single(
        urban_result,
        save_path=os.path.join(OUTPUT_DIR, '3gpp_urban_detail.png')
    )

    # CDL 抽头时延功率谱
    plot_cdl_taps(
        urban_result,
        save_path=os.path.join(OUTPUT_DIR, '3gpp_cdl_taps.png')
    )

    # 多环境对比图
    print("\n📊 生成三环境对比图...")
    plot_ntn_comparison(
        all_results,
        save_path=os.path.join(OUTPUT_DIR, '3gpp_environment_comparison.png')
    )

    # ==================== 汇总报告 ====================
    print_ntn_summary(all_results)

    print("💾 所有 3GPP 仿真结果已保存至 output/ 目录:")
    print("   - 3gpp_urban_detail.png:        Urban 环境详细信道分析")
    print("   - 3gpp_cdl_taps.png:            CDL 抽头时延功率谱")
    print("   - 3gpp_environment_comparison.png: 三环境对比")
    print("\n✅ 3GPP TR 38.811 仿真完成！")

    return 0


if __name__ == '__main__':
    sys.exit(main())
