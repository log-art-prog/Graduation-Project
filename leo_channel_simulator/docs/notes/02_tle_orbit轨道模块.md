# 02 — tle_orbit.py 轨道几何模块

> 作用：生成卫星过顶期间每个时刻的仰角、方位角、斜距、径向速度、多普勒频移。
> 提供两个模型：简化解析模型（默认）和真实 TLE + Skyfield 模型。

---

## 1. 输出数据结构 OrbitParams

```python
@dataclass
class OrbitParams:
    t: np.ndarray              # 时间轴(秒)，以过顶时刻为0，左右对称
    elevation_deg: np.ndarray  # 仰角(度)
    azimuth_deg: np.ndarray    # 方位角(度)
    range_km: np.ndarray       # 斜距(km)
    velocity_kms: np.ndarray   # 径向速度(km/s)，负=接近，正=远离
    doppler_hz: np.ndarray     # 多普勒频移(Hz)
    time_utc: list             # UTC时间
```

这就是后面所有信道计算的"输入源"。

## 2. 简化轨道模型 simple_orbit_model 的几何推导

卫星在高度 h 以速度 v 水平飞过头顶，地面站位于正下方点。设 t=0 为过顶时刻：

```python
x = v * t                       # 卫星相对正上方的水平距离
d = sqrt(x² + h²)               # 地面站到卫星的斜距（直角三角形斜边）
elevation = arcsin(h / d)       # 仰角：对边h / 斜边d
radial_velocity = v * x / d     # 径向速度（速度在斜距方向的投影）
doppler = -radial_velocity / λ  # 多普勒频移，λ=c/fc
```

### 三个关键现象（务必理解）

| 时刻 | x | 仰角 | 斜距 | 径向速度 | 多普勒 |
|------|---|------|------|----------|--------|
| 接近时(t<0) | 负 | 小 | 大 | 负（距离减小） | 正（频率升高） |
| 过顶(t=0) | 0 | 最大≈90° | 最小=h | **0** | **0（变号点）** |
| 远离时(t>0) | 正 | 小 | 大 | 正 | 负（频率降低） |

这就是 LEO 信道最典型的 **S 形多普勒曲线**，过顶时必须快速重新捕获频率。

### 细节处理
- 仰角低于 5° 时截断为 5°，避免 `arcsin` 和大气路径公式出现数值问题。
- 方位角用 `linspace(180,0)` 简单模拟从南到北，不影响损耗计算。

## 3. 真实 TLE 模型 compute_orbit_from_tle

TLE（两行轨道根数）是描述卫星轨道的标准文本，`data/starlink_tle.txt` 是示例：

```
STARLINK-1007
1 44713U 19074A ...
2 44713  53.0 ...
```

流程：
1. 用 `skyfield` 库的 `EarthSatellite(line1, line2)` 加载轨道。
2. 用 `wgs84.latlon(纬度,经度,海拔)` 建地面站。
3. 在观测时间窗内逐时刻求卫星相对地面站的 `altaz()` → 仰角/方位角/距离。
4. **径向速度用相邻点距离差分**得到：`v[i]=(d[i]-d[i-1])/dt`，再算多普勒。
5. Skyfield 未安装时自动回退到简化模型（try/except 开关 `SKYFIELD_AVAILABLE`）。

> 注：当前真实 TLE 的时间窗生成逻辑较简化，学习阶段以简化模型为主即可。

## 4. 典型数值（550km 高度，20GHz）

- 最大仰角 ≈ 89.8°
- 斜距：550（过顶）~ 2345 km（低仰角）
- 最大多普勒 ≈ ±493 kHz
- 多普勒变化率峰值 ≈ 7000 Hz/s

## 5. 自测问题

- 为什么过顶时多普勒为 0？（径向速度=0，运动方向与视线垂直）
- 为什么多普勒在过顶前后变号？（距离先减小后增大）
- 仰角数组将作为 #5 LoS 概率模型的输入，它的取值范围是多少？（约 5°~90°）
