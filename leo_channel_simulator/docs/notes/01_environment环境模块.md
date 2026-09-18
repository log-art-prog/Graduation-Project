# 01 — environment.py 环境参数模块

> 作用：定义三种环境（开阔地/城郊/城市）的信道参数，并提供「仰角 → 参数」的换算函数。

---

## 1. 参数数据类 EnvironmentParams

```python
@dataclass
class EnvironmentParams:
    name: str               # 英文名 'Urban'
    name_cn: str            # 显示名
    sigma_shadow_dB: float  # 阴影衰落标准差(dB)
    max_rms_delay_ns: float # 最大RMS时延扩展(ns)——低仰角
    min_rms_delay_ns: float # 最小RMS时延扩展(ns)——高仰角
    k_min_dB: float         # 最小K因子(dB)——低仰角
    k_max_dB: float         # 最大K因子(dB)——高仰角
    multipath_num: int      # 多径数量
    color / linestyle       # 画图属性
```

要点：
- `@dataclass` = 纯装数据的类，自动生成 `__init__`，直接写字段即可。
- 参数成对出现（max/min），因为时延扩展和 K 因子都随仰角变化。
- 规律：**低仰角 → 时延大、K 小；高仰角 → 时延小、K 大**。

## 2. 三种环境参数表

| 环境 | σ阴影(dB) | 时延(ns) | K(dB) | 多径数 |
|------|-----------|----------|-------|--------|
| Open 开阔地 | 1.0 | 2~10 | 10~25 | 3 |
| Suburban 城郊 | 3.5 | 10~100 | 3~15 | 6 |
| Urban 城市 | 6.0 | 50~1000 | -10~5 | 12 |

物理含义：越开阔 → 遮挡越少 → 阴影越小、直射径越强(K大)、多径越少。

## 3. 仰角 → RMS 时延扩展（指数衰减插值）

```python
elev_norm = np.clip(elevation_deg / 90.0, 0, 1)   # 仰角归一化到 0~1
tau_rms = min + (max - min) * np.exp(-3 * elev_norm)
return tau_rms * 1e-9                              # ns → s
```

- `np.clip(x,0,1)`：把异常仰角限制在 [0,1]，防止越界。
- 仰角 0°：`exp(0)=1` → τ 取 **max**（低仰角时延最大）。
- 仰角 90°：`exp(-3)≈0.05` → τ 接近 **min**。
- 系数 3：控制随时仰角下降的陡峭程度。

## 4. 仰角 → K 因子（线性插值）

```python
k_dB = k_min_dB + (k_max_dB - k_min_dB) * elev_norm
return 10 ** (k_dB / 10.0)      # dB → 线性值
```

- 0° 取 k_min，90° 取 k_max，中间线性变化。
- `10**(k_dB/10)` 是 dB 转线性值（莱斯合成公式要用线性 K）。

## 5. 相干带宽

```python
B_c = 1.0 / (5 * tau_rms_s)
```

工程近似（相关系数>0.5 的定义）。时延扩展越大 → 相干带宽越窄，信号越容易产生频率选择性衰落。

## 6. 与后续 #5 LoS/NLoS 任务的关系

将来会给 EnvironmentParams 增加 3 个 LoS 概率参数字段：

```python
los_a, los_b, los_c   # Sigmoid: P_LoS = 1/(1+a·exp(-b(θ-c)))
```

## 7. 自测问题

- 为什么时延用指数插值、K 用线性插值？（经验上：遮挡随时仰角变化非线性更明显，K 用线性是简化）
- dB 和线性值为什么要来回转？（dB 便于加减/图表，线性值用于功率合成公式）
