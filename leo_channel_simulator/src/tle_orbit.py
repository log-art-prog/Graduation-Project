"""
TLE轨道计算模块
使用Skyfield进行高精度轨道外推，计算卫星位置、速度、仰角、多普勒等参数
"""
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, Optional
from dataclasses import dataclass

try:
    from skyfield.api import load, wgs84, EarthSatellite
    SKYFIELD_AVAILABLE = True
except ImportError:
    SKYFIELD_AVAILABLE = False


@dataclass
class OrbitParams:
    """轨道参数数据类"""
    t: np.ndarray              # 时间轴 (秒)
    elevation_deg: np.ndarray  # 仰角 (度)
    azimuth_deg: np.ndarray    # 方位角 (度)
    range_km: np.ndarray       # 斜距 (km)
    velocity_kms: np.ndarray   # 径向速度 (km/s，正为远离，负为接近)
    doppler_hz: np.ndarray     # 多普勒频移 (Hz)
    time_utc: list             # UTC时间列表


def load_tle_from_file(tle_file_path: str) -> Tuple[str, str, str]:
    """
    从文件读取TLE数据
    返回: (sat_name, line1, line2)
    """
    with open(tle_file_path, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    
    if len(lines) >= 3:
        return lines[0], lines[1], lines[2]
    elif len(lines) == 2:
        return 'UNKNOWN', lines[0], lines[1]
    else:
        raise ValueError(f"TLE文件格式错误，需要2或3行，实际只有{len(lines)}行")


def simple_orbit_model(
    sat_altitude_km: float = 550.0,
    sat_velocity_kms: float = 7.6,
    max_elevation_deg: float = 90.0,
    pass_duration_s: float = 600.0,
    num_points: int = 1000,
    fc_hz: float = 20e9,
) -> OrbitParams:
    """
    简化轨道模型（无需Skyfield依赖时使用）
    模拟卫星过顶的经典S曲线
    """
    c = 299792458  # 光速 m/s
    R_earth = 6371.0  # 地球半径 km
    
    t = np.linspace(-pass_duration_s/2, pass_duration_s/2, num_points)
    
    # 仰角变化：-t0到+t0对称，过顶时最大
    # 使用解析几何近似
    max_elev_rad = np.radians(max_elevation_deg)
    v = sat_velocity_kms * 1000  # m/s
    h = sat_altitude_km * 1000   # m
    
    # 仰角计算
    x = v * t  # 水平距离，过顶时x=0
    d = np.sqrt(x**2 + h**2)  # 斜距
    elevation_rad = np.arcsin(h / d)
    
    # 仰角截断：当仰角低于5度时认为不可见，置为5度（避免数值问题）
    min_elev_mask = elevation_rad < np.radians(5.0)
    elevation_rad[min_elev_mask] = np.radians(5.0)
    
    # 方位角（简化：从南到北或从西到东）
    azimuth_deg = np.linspace(180, 0, num_points)  # 简单从南到北
    
    # 径向速度：vx * sin(elevation)，过顶时径向速度为0
    # 接近时速度为负（距离减小），远离时为正
    radial_velocity = v * x / d  # m/s
    # 调整符号：t<0时接近，速度应为负；t>0时远离，速度为正
    # 已经是正确的，因为x<0时径向速度负
    
    # 多普勒频移: f_d = -v_r / lambda (v_r接近时为负，f_d为正)
    wavelength = c / fc_hz
    doppler_hz = -radial_velocity / wavelength
    
    range_km = d / 1000.0
    
    time_utc = [datetime.utcnow() + timedelta(seconds=float(ts)) for ts in t]
    
    return OrbitParams(
        t=t,
        elevation_deg=np.degrees(elevation_rad),
        azimuth_deg=azimuth_deg,
        range_km=range_km,
        velocity_kms=radial_velocity / 1000.0,
        doppler_hz=doppler_hz,
        time_utc=time_utc
    )


def compute_orbit_from_tle(
    tle_file_path: str,
    ground_station_lat: float = 39.9042,   # 北京纬度
    ground_station_lon: float = 116.4074,  # 北京经度
    ground_station_alt_m: float = 50.0,
    pass_duration_s: float = 600.0,
    num_points: int = 1000,
    fc_hz: float = 20e9,
    observation_time: Optional[datetime] = None,
) -> OrbitParams:
    """
    基于真实TLE数据计算轨道参数
    如果Skyfield不可用则回退到简化模型
    """
    if not SKYFIELD_AVAILABLE:
        print("⚠️  警告：Skyfield未安装，使用简化轨道模型。可运行 pip install skyfield 安装真实轨道计算模块。")
        return simple_orbit_model(
            sat_altitude_km=550,
            fc_hz=fc_hz,
            pass_duration_s=pass_duration_s,
            num_points=num_points
        )
    
    c = 299792458  # 光速 m/s
    wavelength = c / fc_hz
    
    # 加载TLE
    sat_name, line1, line2 = load_tle_from_file(tle_file_path)
    print(f"📡 加载卫星TLE: {sat_name}")
    
    ts = load.timescale()
    satellite = EarthSatellite(line1, line2, sat_name, ts)
    ground_station = wgs84.latlon(
        ground_station_lat,
        ground_station_lon,
        elevation_m=ground_station_alt_m
    )
    
    # 确定观测时间：如果未指定则找下一次过境
    if observation_time is None:
        observation_time = datetime.utcnow()
        print(f"🔍 正在查找 {observation_time} 之后的下一次过境...")
    
    # 生成时间序列
    t0 = ts.utc(observation_time.year, observation_time.month, observation_time.day,
                observation_time.hour, observation_time.minute, observation_time.second)
    times = ts.utc(
        [observation_time.year] * num_points,
        [observation_time.month] * num_points,
        [observation_time.day] * num_points,
        [observation_time.hour] * num_points,
        [(observation_time.minute + i * pass_duration_s/(num_points*60)) % 60 for i in range(num_points)]
    )
    
    # 为简化，直接在指定时间窗口周围生成对称序列
    t_seconds = np.linspace(-pass_duration_s/2, pass_duration_s/2, num_points)
    times_list = []
    for dt in t_seconds:
        t_dt = observation_time + timedelta(seconds=float(dt))
        times_list.append(ts.utc(t_dt.year, t_dt.month, t_dt.day, t_dt.hour, t_dt.minute, t_dt.second + t_dt.microsecond/1e6))
    
    # 计算位置差分得到速度
    elevation_deg = np.zeros(num_points)
    azimuth_deg = np.zeros(num_points)
    range_km = np.zeros(num_points)
    velocity_kms = np.zeros(num_points)
    doppler_hz = np.zeros(num_points)
    time_utc = []
    
    for i, t_s in enumerate(times_list):
        # 卫星相对地面站的位置
        difference = satellite - ground_station
        topocentric = difference.at(t_s)
        alt, az, distance = topocentric.altaz()
        
        elevation_deg[i] = alt.degrees
        azimuth_deg[i] = az.degrees
        range_km[i] = distance.km
        
        # 径向速度：通过相邻点距离差分计算
        if i > 0:
            dt = t_seconds[i] - t_seconds[i-1]
            velocity_kms[i] = (range_km[i] - range_km[i-1]) / dt
        else:
            velocity_kms[i] = 0.0
        
        doppler_hz[i] = -velocity_kms[i] * 1000 / wavelength  # km/s -> m/s
        time_utc.append(t_s.utc_datetime())
    
    # 第一个点的速度用第二个点填充
    velocity_kms[0] = velocity_kms[1]
    doppler_hz[0] = doppler_hz[1]
    
    # 仰角<0的点设为0（不可见）
    elevation_deg = np.maximum(elevation_deg, 0.1)
    
    print(f"✅ 轨道计算完成，最大仰角: {np.max(elevation_deg):.1f}°")
    
    return OrbitParams(
        t=t_seconds,
        elevation_deg=elevation_deg,
        azimuth_deg=azimuth_deg,
        range_km=range_km,
        velocity_kms=velocity_kms,
        doppler_hz=doppler_hz,
        time_utc=time_utc
    )
