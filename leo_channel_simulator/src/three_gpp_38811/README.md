# 3GPP TR 38.811 NTN 星地信道模型 — 使用说明

本模块完整实现了 **3GPP TR 38.811 V17.0.0** 中定义的低轨卫星 (LEO) 星地信道模型，独立于项目原有的 ITU-R 经验模型，放置在 `src/three_gpp_38811/` 目录下。

---

## 1. 快速开始

### 1.1 运行完整仿真

```bash
cd leo_channel_simulator
python main_3gpp_ntn.py
```

运行后在 `output/` 目录生成以下图表：

| 文件 | 说明 |
|------|------|
| `3gpp_urban_detail.png` | Urban 环境 8 子图详细信道分析 |
| `3gpp_cdl_taps.png` | CDL 抽头时延功率谱 |
| `3gpp_environment_comparison.png` | Urban / Suburban / Open 三环境对比 |

### 1.2 在代码中调用

```python
from src.tle_orbit import simple_orbit_model
from src.three_gpp_38811.parameters import ALL_NTN_ENVIRONMENTS, ENV_URBAN
from src.three_gpp_38811.channel_model import simulate_ntn_channel, simulate_multiple_environments_ntn

# 1. 生成轨道
orbit = simple_orbit_model(
    sat_altitude_km=550, sat_velocity_kms=7.6,
    max_elevation_deg=90, pass_duration_s=600,
    num_points=1000, fc_hz=20e9
)

# 2. 单环境仿真
result = simulate_ntn_channel(
    env=ENV_URBAN,
    orbit=orbit,
    fc_hz=20e9,
    bandwidth_hz=200e6,
    tx_eirp_dbm=55.0,
    rx_gain_db=35.0,
    noise_figure_db=2.5,
    rain_rate_mm_h=2.0,
    seed=42
)

# 3. 多环境对比仿真
results = simulate_multiple_environments_ntn(
    environments=ALL_NTN_ENVIRONMENTS,
    orbit=orbit,
    fc_hz=20e9, ...
)
```

---

## 2. 文件结构

```
src/three_gpp_38811/
├── __init__.py              # 包入口
├── parameters.py            # 3GPP 参数表 (Table 6.6.2-1 等)
├── los_probability.py       # LoS/NLoS 概率 (Section 6.6.1)
├── large_scale.py           # 大尺度参数 + 空间一致性 (Section 6.6.2/6.6.3)
├── small_scale.py           # CDL 抽头 + Loo 模型 (Section 6.5/6.7)
├── channel_model.py         # 主模型整合
└── visualization_3gpp.py    # 3GPP 专用可视化
```

项目根目录下还有：

```
main_3gpp_ntn.py             # 3GPP 模型独立运行入口
```

---

## 3. 3GPP 模型实现对照

| 3GPP TR 38.811 章节 | 实现内容 | 所在模块 |
|---------------------|----------|----------|
| **Section 6.4** | 路径损耗：自由空间 + 大气吸收(ITU-R P.676) + 降雨衰减(P.838) + 阴影衰落(SF) | `channel_model.py` |
| **Section 6.6.1** | LoS 概率 Sigmoid 模型 + Markov 状态转移 | `los_probability.py` |
| **Section 6.6.2** | 大尺度参数 (LSP) 对数正态生成 | `large_scale.py` + `parameters.py` |
| **Section 6.6.3** | 空间一致性 AR(1) 滤波 | `large_scale.py` |
| **Section 6.5.3** | **Loo 模型**：直射径对数正态阴影 + 瑞利散射径 | `small_scale.py` |
| **Section 6.7** | **CDL 抽头延迟线**：指数功率延迟谱 + 每簇独立衰落 | `small_scale.py` |

---

## 4. 模块 API 参考

### 4.1 `parameters.py` — 参数定义

**环境场景** (`NTNEnvironment`)：

| 环境 | `name` | 阴影 σ (dB) | LoS 模型 (a, b, c) |
|------|--------|-------------|---------------------|
| Urban | `ENV_URBAN` | 10.0 | (12.0, 0.12, 15.0) |
| Suburban | `ENV_SUBURBAN` | 6.0 | (6.0, 0.10, 10.0) |
| Open | `ENV_OPEN` | 2.0 | (0.5, 0.08, 5.0) |

**大尺度参数表** (LSPTable，单位说明)：

| 参数 | 分布域 | 说明 |
|------|--------|------|
| `DS` | log₁₀(秒) | RMS 时延扩展，使用时 ×10⁹ 转 ns |
| `K` | dB | 莱斯 K 因子，仅 LoS |
| `ASD` / `ASA` | log₁₀(度) | 离开/到达角扩展 |
| `ZSD` / `ZSA` | log₁₀(度) | 离开/到达天顶角扩展 |

Urban LoS 示例值：DS μ=-7.44 (≈36 ns), K μ=9 dB, ASD μ=1.0 (≈10°)

**空间一致性去相关距离**：DS=50m, K=40m, SF=100m 等。

### 4.2 `los_probability.py` — LoS 概率

```python
from src.three_gpp_38811.los_probability import (
    calculate_los_probability,
    determine_los_state_array
)

# 单点概率
p = calculate_los_probability(elevation_deg=30, env=ENV_URBAN)

# 批量状态序列 (带 Markov 时间相关性)
los_states = determine_los_state_array(elevation_array, ENV_URBAN, rng)
```

LoS 概率公式：

```
P_LoS(θ) = 1 / (1 + a · exp(-b · (θ - c)))
```

仰角 >80° 时强制 P_LoS = 1。

### 4.3 `large_scale.py` — 大尺度参数

```python
from src.three_gpp_38811.large_scale import (
    generate_lsp_sample,
    generate_spatially_consistent_lsp,
    generate_shadow_fading_sequence
)

# 单个 LSP 样本
sample = generate_lsp_sample(ENV_URBAN, is_los=True, rng=rng)
# sample.ds_ns, sample.k_db, sample.asd_deg, sample.sf_db ...

# 空间一致序列 (AR(1) 滤波)
lsp_seq = generate_spatially_consistent_lsp(
    num_points=1000, env=ENV_URBAN, is_los=True,
    velocity_ms=7600, fs=1.67, rng=rng
)
# lsp_seq['ds_ns'], lsp_seq['k_db'], lsp_seq['sf_db'] ...
```

空间一致性滤波：`X[n] = ρ·X[n-1] + √(1-ρ²)·ε[n]`，其中 `ρ = exp(-d_sample / d_corr)`。

### 4.4 `small_scale.py` — 小尺度衰落

```python
from src.three_gpp_38811.small_scale import (
    generate_rayleigh_fading,
    generate_rician_fading,
    generate_loo_fading,
    generate_cdl_taps,
    calculate_fading_metrics
)

# 瑞利衰落 (Jakes 正弦叠加法)
h = generate_rayleigh_fading(num_samples, fd_hz=500, fs=10)

# 莱斯衰落
h = generate_rician_fading(num_samples, k_factor_linear=10, fd_hz=500, fs=10)

# Loo 模型 (直射径阴影 + 散射径瑞利)
h_loo, shadow_db = generate_loo_fading(
    num_samples, k_factor_linear=10, shadow_std_db=2.0,
    fd_hz=500, fs=10
)

# CDL 抽头延迟线
cdl = generate_cdl_taps(
    num_samples=5000, ds_ns=100, k_factor_db=9.0,
    fd_hz=500, fs=10
)
# cdl.delays_s, cdl.powers_linear, cdl.h_taps [num_clusters, num_samples]

# 衰落统计: LCR, AFD, 1% 衰落深度
metrics = calculate_fading_metrics(h, fs=10)
```

**Loo 模型公式**：

```
h(t) = √(K/(K+1)) · 10^(ξ(t)/20) · e^(j2πf_d t)   ← 直射径(受阴影ξ调制)
     + √(1/(K+1)) · s(t)                            ← 瑞利散射径
```

**CDL 模型**：8 个簇，每簇 20 子径，功率指数衰减 `P_n ∝ exp(-τ_n / τ_rms)`。

### 4.5 `channel_model.py` — 主模型

```python
from src.three_gpp_38811.channel_model import simulate_ntn_channel

result = simulate_ntn_channel(env, orbit, fc_hz=20e9, ...)
```

**返回 `NTNChannelResult` 字段**：

| 字段 | 类型 | 说明 |
|------|------|------|
| `los_state` | np.ndarray[bool] | LoS/NLoS 状态序列 |
| `los_probability` | np.ndarray | LoS 概率序列 |
| `fspl_db` / `atmosphere_db` / `rain_db` | np.ndarray | 路径损耗各分量 |
| `shadow_fading_db` | np.ndarray | 阴影衰落 (SF) |
| `total_pl_db` | np.ndarray | 总路径损耗 |
| `ds_ns` | np.ndarray | RMS 时延扩展 (ns) |
| `k_db` | np.ndarray | 莱斯 K 因子 (dB)，NLoS 为 NaN |
| `coherence_bw_mhz` | np.ndarray | 相干带宽 (MHz) |
| `small_scale_h` | np.ndarray[complex] | 总复信道系数 |
| `small_scale_amp` | np.ndarray | 信道幅度 |
| `fade_depth_db` | float | 1% 衰落深度 |
| `rx_power_dbm` / `snr_db` | np.ndarray | 接收功率 / SNR |
| `cdl_delays_ns` / `cdl_powers_db` | np.ndarray | CDL 抽头信息 |

### 4.6 `visualization_3gpp.py` — 可视化

```python
from src.three_gpp_38811.visualization_3gpp import (
    plot_ntn_single, plot_ntn_comparison, plot_cdl_taps
)

plot_ntn_single(result, save_path='output/3gpp_urban_detail.png')
plot_ntn_comparison(results, save_path='output/3gpp_comparison.png')
plot_cdl_taps(result, save_path='output/3gpp_cdl.png')
```

---

## 5. 仿真结果示例

典型输出汇总表：

```
Environment        LoS%    SF σ    Min PL    Max PL  FadeDepth     K(LoS)         DS Range
Urban (城市)       42.1    10.0     146.7     215.9      -19.6  -0.4~17.6     1.7~  6291.1
Suburban (城郊)    54.6     6.0     160.0     203.4      -20.0   1.5~19.1     2.6~  2436.0
Open (开阔地)       95.7     2.0     168.4     193.3      -20.9   0.0~19.0     0.8~   409.9
```

**关键观察**：
- Urban 环境 LoS 占比仅 ~42%，大量时间处于 NLoS，DS 可达 μs 级
- Open 环境几乎始终 LoS (95%+)，DS 较小
- NLoS 时段 DS 远大于 LoS 时段（对数正态分布）
- 1% 衰落深度约 -20 dB，符合 Loo 模型预期

---

## 6. 参数配置指南

### 6.1 系统参数 (main_3gpp_ntn.py)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `FC_HZ` | 20e9 | 载波频率，支持 S/C/Ku/Ka/Q/V 频段 |
| `BANDWIDTH_HZ` | 200e6 | 信号带宽 |
| `TX_EIRP_DBM` | 55.0 | 发射 EIRP (dBm) |
| `RX_GAIN_DB` | 35.0 | 接收天线增益 (dBi) |
| `NOISE_FIGURE_DB` | 2.5 | 噪声系数 (dB) |
| `RAIN_RATE_MMH` | 2.0 | 雨率 (mm/h)，小雨=2, 中雨=10, 暴雨=50 |

### 6.2 轨道参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `PASS_DURATION_S` | 600 | 过境时长 (秒) |
| `NUM_POINTS` | 1000 | 仿真点数 |
| `USE_REAL_TLE` | False | True 时使用 `data/starlink_tle.txt` 真实 TLE |
| `GS_LAT/LON/ALT` | 北京 | 地面站经纬度海拔 |

### 6.3 切换真实 TLE 数据

```python
USE_REAL_TLE = True
# 将最新 TLE 数据替换 data/starlink_tle.txt
```

TLE 数据可从 [CelesTrak](https://celestrak.org/) 获取。

---

## 7. 与原 ITU-R 模型的区别

| 特性 | 原 ITU-R 模型 (`src/`) | 3GPP TR 38.811 模型 (`src/three_gpp_38811/`) |
|------|------------------------|----------------------------------------------|
| LoS 处理 | 始终假设 LoS | 基于仰角的概率判定 + Markov 状态 |
| 阴影衰落 | AR(1) 加在总 PL | 3GPP SF 标准差按环境 (Urban=10dB, Open=2dB) |
| K 因子 | 随仰角线性变化 | 对数正态随机变量 (μ=9~11 dB, σ=3) |
| 时延扩展 | 指数衰减经验公式 | 对数正态分布 (Table 6.6.2-1) |
| 小尺度 | 独立莱斯/瑞利 | **Loo 模型** + **CDL 抽头** |
| 角度扩展 | 未建模 | ASD/ASA/ZSD/ZSA 全部实现 |
| 空间一致性 | 无 | AR(1) 空间相关滤波 |

两个模型可独立运行：
- 原模型：`python main.py`
- 3GPP 模型：`python main_3gpp_ntn.py`

---

## 8. 参考标准

- **3GPP TR 38.811**: Study on New Radio (NR) to support non-terrestrial networks
- **3GPP TR 38.901**: Study on channel model for frequencies from 0.5 to 100 GHz
- **ITU-R P.676**: 大气气体衰减
- **ITU-R P.838**: 降雨衰减
- **ITU-R P.840**: 云雾衰减
