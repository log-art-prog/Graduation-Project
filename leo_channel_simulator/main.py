#!/usr/bin/env python3
"""
低轨卫星星地信道仿真器 - 主程序入口
支持真实TLE轨道驱动和多环境对比仿真
"""
import os
import sys
import numpy as np

# 添加src目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.tle_orbit import simple_orbit_model, compute_orbit_from_tle
from src.environment import ALL_ENVIRONMENTS, ENV_OPEN, ENV_SUBURBAN, ENV_URBAN
from src.channel_model import simulate_channel, simulate_multiple_environments
from src.visualization import plot_single_channel_result, plot_environment_comparison, print_summary_table


def main():
    print("=" * 80)
    print("🚀 LEO SATELLITE EARTH-TO-SPACE CHANNEL SIMULATOR")
    print("=" * 80)
    
    # ==================== 仿真参数配置 ====================
    print("\n⚙️  配置仿真参数...")
    
    # 1. 系统参数
    FC_HZ = 20e9               # 载波频率 20GHz (Ka频段)
    BANDWIDTH_HZ = 200e6       # 信号带宽 200MHz
    TX_EIRP_DBM = 55.0         # 发射EIRP (dBm)
    RX_GAIN_DB = 35.0          # 接收天线增益 (dBi)
    NOISE_FIGURE_DB = 2.5      # 接收机噪声系数 (dB)
    RAIN_RATE_MMH = 2.0        # 降雨量 mm/h (小雨=2, 中雨=10, 暴雨=50)
    
    # 2. 轨道参数
    PASS_DURATION_S = 600      # 过境时长 (秒)
    NUM_POINTS = 1000          # 仿真点数
    USE_REAL_TLE = False       # 是否使用真实TLE数据 (默认False使用标准过顶轨道，True时请替换data/starlink_tle.txt为最新TLE)
    
    # 3. 地面站位置 (默认北京)
    GS_LAT = 39.9042           # 纬度
    GS_LON = 116.4074          # 经度
    GS_ALT_M = 50.0            # 海拔 (米)
    
    # TLE文件路径
    TLE_FILE = os.path.join(os.path.dirname(__file__), 'data', 'starlink_tle.txt')
    
    # 输出目录
    OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"   载波频率: {FC_HZ/1e9:.1f} GHz")
    print(f"   信号带宽: {BANDWIDTH_HZ/1e6:.0f} MHz")
    print(f"   过境时长: {PASS_DURATION_S} 秒")
    print(f"   降雨量: {RAIN_RATE_MMH} mm/h")
    
    # ==================== 轨道计算 ====================
    print("\n🛰️  计算卫星轨道...")
    
    if USE_REAL_TLE and os.path.exists(TLE_FILE):
        orbit = compute_orbit_from_tle(
            tle_file_path=TLE_FILE,
            ground_station_lat=GS_LAT,
            ground_station_lon=GS_LON,
            ground_station_alt_m=GS_ALT_M,
            pass_duration_s=PASS_DURATION_S,
            num_points=NUM_POINTS,
            fc_hz=FC_HZ
        )
    else:
        print("   使用简化轨道模型...")
        orbit = simple_orbit_model(
            sat_altitude_km=550,
            sat_velocity_kms=7.6,
            max_elevation_deg=90,
            pass_duration_s=PASS_DURATION_S,
            num_points=NUM_POINTS,
            fc_hz=FC_HZ
        )
    
    print(f"   最大仰角: {np.max(orbit.elevation_deg):.1f}°")
    print(f"   最大多普勒频移: ±{np.max(np.abs(orbit.doppler_hz))/1000:.1f} kHz")
    print(f"   多普勒变化率峰值: ~{np.max(np.abs(np.diff(orbit.doppler_hz)/np.diff(orbit.t))):.0f} Hz/s")
    
    # ==================== 多环境信道仿真 ====================
    print("\n📡 开始多环境信道仿真...")
    
    # 同时仿真三种环境用于对比
    all_results = simulate_multiple_environments(
        environments=ALL_ENVIRONMENTS,
        orbit=orbit,
        fc_hz=FC_HZ,
        bandwidth_hz=BANDWIDTH_HZ,
        tx_eirp_dbm=TX_EIRP_DBM,
        rx_gain_db=RX_GAIN_DB,
        noise_figure_db=NOISE_FIGURE_DB,
        rain_rate_mm_h=RAIN_RATE_MMH
    )
    
    # ==================== 单环境详细仿真 (以城郊为例) ====================
    print("\n🔍 生成城郊环境详细分析...")
    suburban_result = None
    for res in all_results:
        if res.environment.name == 'Suburban':
            suburban_result = res
            break
    
    if suburban_result:
        plot_single_channel_result(
            suburban_result,
            save_path=os.path.join(OUTPUT_DIR, 'suburban_channel_detail.png')
        )
    
    # ==================== 环境对比图 ====================
    print("\n📊 生成多环境对比图...")
    plot_environment_comparison(
        all_results,
        save_path=os.path.join(OUTPUT_DIR, 'environment_comparison.png')
    )
    
    # ==================== 打印汇总报告 ====================
    print_summary_table(all_results)
    
    # ==================== 可选：单环境分析 ====================
    print("💾 所有结果已保存至 output/ 目录")
    print("   - suburban_channel_detail.png: 城郊环境详细信道参数")
    print("   - environment_comparison.png:  开阔地/城郊/城市三环境对比")
    
    print("\n✅ 仿真完成！")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
