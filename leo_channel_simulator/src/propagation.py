"""
传播损耗模型模块
实现ITU-R标准的大气、降雨、自由空间损耗计算
"""
import numpy as np

# 物理常数
C = 299792458  # 光速 m/s
BOLTZMANN = 1.380649e-23  # 玻尔兹曼常数 J/K


def free_space_path_loss(fc_hz: float, range_m: float) -> float:
    """
    自由空间路径损耗
    PL = 20*log10(4*pi*d/lambda)
    """
    wavelength = C / fc_hz
    return 20 * np.log10(4 * np.pi * range_m / wavelength)


def atmospheric_gas_loss(fc_ghz: float, elevation_deg: float) -> float:
    """
    大气气体吸收损耗 (ITU-R P.676简化模型)
    考虑氧气和水蒸气吸收
    """
    # 简化模型：海平面标准大气
    # 氧气吸收峰在60GHz，水蒸气在22GHz和183GHz
    elev_rad = np.radians(elevation_deg)
    
    if fc_ghz < 10:
        gamma_o = 0.005  # dB/km S频段以下氧气吸收很小
        gamma_w = 0.001
    elif fc_ghz < 20:
        gamma_o = 0.015
        gamma_w = 0.005 + 0.003 * (fc_ghz - 10)/10
    elif fc_ghz < 30:
        gamma_o = 0.035 + 0.02 * (fc_ghz - 20)/10
        gamma_w = 0.1 * (fc_ghz - 20)/10  # 22GHz水吸收峰
    elif fc_ghz < 50:
        gamma_o = 0.1 + 0.4 * (fc_ghz - 30)/20
        gamma_w = 0.15 + 0.1 * (fc_ghz - 30)/20
    elif fc_ghz < 70:
        gamma_o = 15 * np.exp(-((fc_ghz - 60)/5)**2)  # 60GHz氧气吸收峰
        gamma_w = 0.3 + 0.5 * (fc_ghz - 50)/20
    else:
        gamma_o = 2.0
        gamma_w = 1.0
    
    gamma = gamma_o + gamma_w  # 总比衰减 dB/km
    
    # 斜路径长度（等效高度约6km对流层）
    h_atm = 6.0  # km
    slant_range = h_atm / np.sin(elev_rad)  # km
    
    return gamma * slant_range


def rain_attenuation(fc_ghz: float, elevation_deg: float, rain_rate_mm_h: float = 2.0) -> float:
    """
    降雨衰减 (ITU-R P.838简化模型)
    rain_rate: 降雨量 mm/h，小雨0.5-2，中雨2-10，暴雨10-50
    """
    elev_rad = np.radians(elevation_deg)
    
    # 降雨比衰减系数 k 和 alpha (ITU-R P.838拟合)
    if fc_ghz < 2.5:
        k_h, k_v = 0.00003, 0.00003
        alpha_h, alpha_v = 1.0, 1.0
    elif fc_ghz < 8:
        k_h, k_v = 0.0008, 0.0007
        alpha_h, alpha_v = 1.1, 1.1
    elif fc_ghz < 15:
        k_h, k_v = 0.015, 0.013
        alpha_h, alpha_v = 1.15, 1.12
    elif fc_ghz < 30:
        k_h, k_v = 0.08, 0.07
        alpha_h, alpha_v = 1.1, 1.06
    elif fc_ghz < 50:
        k_h, k_v = 0.3, 0.25
        alpha_h, alpha_v = 0.95, 0.92
    else:
        k_h, k_v = 0.5, 0.45
        alpha_h, alpha_v = 0.85, 0.82
    
    # 圆极化平均
    k = (k_h + k_v) / 2
    alpha = (alpha_h + alpha_v) / 2
    
    # 比衰减 dB/km
    gamma_r = k * (rain_rate_mm_h ** alpha)
    
    # 雨区等效高度（根据ITU-R P.839）
    h_rain = 4.0  # km，中纬度
    
    # 斜路径长度
    slant_range = h_rain / np.sin(elev_rad)
    
    # 路径缩短因子（简化）
    r_factor = 1.0 / (1 + slant_range / 30)
    
    return gamma_r * slant_range * r_factor


def cloud_fog_loss(fc_ghz: float, elevation_deg: float, cloud_density: float = 0.1) -> float:
    """
    云雾衰减 (ITU-R P.840简化模型)
    cloud_density: 液态水密度 g/m3，薄雾0.05，浓雾0.5
    """
    if fc_ghz < 10:
        return 0.0  # 10GHz以下云雾衰减可忽略
    
    elev_rad = np.radians(elevation_deg)
    
    # 液态水比衰减系数近似
    if fc_ghz < 30:
        Kl = 0.5  # dB/(km g/m3)
    elif fc_ghz < 60:
        Kl = 1.5
    else:
        Kl = 2.5
    
    h_cloud = 1.0  # km，云层厚度
    slant_range = h_cloud / np.sin(elev_rad)
    
    return Kl * cloud_density * slant_range


def calculate_total_path_loss(
    fc_hz: float,
    range_m: float,
    elevation_deg: float,
    rain_rate_mm_h: float = 2.0,
    include_atmosphere: bool = True,
    include_rain: bool = True,
) -> dict:
    """
    计算总路径损耗及各分量
    返回字典包含各损耗分量
    """
    fc_ghz = fc_hz / 1e9
    
    pl_fspl = free_space_path_loss(fc_hz, range_m)
    
    pl_atm = atmospheric_gas_loss(fc_ghz, elevation_deg) if include_atmosphere else 0.0
    pl_rain = rain_attenuation(fc_ghz, elevation_deg, rain_rate_mm_h) if include_rain else 0.0
    pl_cloud = cloud_fog_loss(fc_ghz, elevation_deg) if fc_ghz > 10e9 else 0.0
    
    pl_total = pl_fspl + pl_atm + pl_rain + pl_cloud
    
    return {
        'fspl_db': pl_fspl,
        'atmosphere_db': pl_atm,
        'rain_db': pl_rain,
        'cloud_db': pl_cloud,
        'total_db': pl_total
    }


def calculate_received_power(
    eirp_dbm: float,
    tx_gain_db: float,
    rx_gain_db: float,
    path_loss_db: float,
    noise_figure_db: float = 3.0,
    bandwidth_hz: float = 200e6,
    temperature_k: float = 290.0,
) -> dict:
    """
    计算接收功率与信噪比
    """
    # 噪声功率
    noise_power_dbm = 10 * np.log10(BOLTZMANN * temperature_k * bandwidth_hz * 1000)  # mW -> dBm
    noise_power_dbm += noise_figure_db
    
    # 接收功率
    rx_power_dbm = eirp_dbm + rx_gain_db - path_loss_db
    
    # 信噪比
    snr_db = rx_power_dbm - noise_power_dbm
    
    return {
        'rx_power_dbm': rx_power_dbm,
        'noise_power_dbm': noise_power_dbm,
        'snr_db': snr_db
    }
