# 04 — fading.py 衰落模型模块

> 作用：生成两类衰落：
> ① 阴影衰落（大尺度、慢、对数正态、空间相关）
> ② 小尺度衰落（快、瑞利/莱斯、多径干涉）
> 并计算 LCR / AFD / 衰落深度等统计量。

---

## 1. 阴影衰落 generate_shadow_fading（AR(1) 模型）

阴影由建筑物/地形遮挡造成，在空间上是**渐变**的，不能逐点独立乱跳。

```python
d_sample = velocity_ms / fs                       # 每个采样点卫星移动距离
rho = exp(-d_sample / decorrelation_distance_m)   # 相关系数
shadow[0] = rng.standard_normal() * sigma_db
shadow[i] = rho·shadow[i-1] + sqrt(1-rho²)·ε[i]   # ε~N(0,σ)
```

- `rho∈[0,1]`：相邻点相关程度。距离越近 rho 越接近 1，序列越平滑。
- 这是**一阶自回归 AR(1)**：当前值 = 上一点的惯性 ＋ 一点新随机扰动。
- `sqrt(1-rho²)` 的作用：保证合成后序列的方差始终等于 σ²（能量守恒）。
- 输出单位 dB，均值 0，最后叠加到路径损耗上。

## 2. 瑞利衰落 generate_rayleigh_fading（Jakes 正弦叠加法）

无直射径时（NLoS），大量反射波叠加，复包络服从瑞利分布。

```python
for n in range(num_sinusoids):           # 默认20个正弦波
    alpha_n = ...到达角（均匀分布）
    x += cos(2π·fd·t·cos(α) + φ)         # 同相支路
    y += cos(2π·fd·t·sin(α) + φ')        # 正交支路
h = (x + 1j·y) / sqrt(2)
```

- 每个正弦波有随机到达角和随机相位，模拟四面八方来的散射波。
- `fd` 是最大多普勒频移，决定衰落起伏的快慢。
- 复系数 h 的模 `|h|` 服从瑞利分布，平均功率归一化为 1。

## 3. 莱斯衰落 generate_rician_fading（有直射径 LoS）

在瑞利散射基础上，加一条强直射径：

```
h = sqrt(K/(K+1))·h_LoS + sqrt(1/(K+1))·h_scatter
```

- `h_LoS = exp(j·2π·fd·t)`：恒幅、带多普勒相位的直射分量。
- K = 直射功率/散射功率（线性值）。
- 退化关系：**K=0 → 纯瑞利**；**K→∞ → 理想 AWGN（无衰落）**。
- 内部调用瑞利函数时必须 `rng=rng` 透传随机源。

## 4. 多径抽头延迟线 generate_multipath_components

```python
tap_delays  = sort(exponential(τ_rms))     # 各径延迟，指数分布
tap_powers  = exp(-tap_delays/τ_rms)       # 功率随时延指数衰减
# 归一化后：第1径用莱斯，其余径用瑞利
```

- 越晚到的反射径功率越小（功率延迟谱 PDP）。
- 第 1 径（0 时延）当直射径 → 莱斯；其余 → 瑞利，多普勒各乘 0.5~1 随机系数。
- 当前是随机抽头（不可复现结构），改进项 #3 计划换成标准 CDL 抽头表。

## 5. 衰落统计 calculate_fading_metrics

| 指标 | 含义 | 代码做法 |
|------|------|----------|
| LCR 电平通过率 | 每秒向下穿越均值电平的次数 | 对 `|h|>阈值` 的布尔序列求差分跳变数 |
| AFD 平均衰落持续时间 | 信号低于 -10dB 的平均持续长度 | 找连续 True 区间求平均长度 |
| 衰落深度 | 1% 时间会低于的电平（CDF 的 1% 分位） | `np.percentile(power_db, 1)` |

衰落越深（如城市），1% 分位数值越负。

## 6. 两类衰落对比（重点记忆）

| | 阴影衰落 | 小尺度衰落 |
|---|----------|------------|
| 尺度 | 大（几十~几百米） | 小（波长量级） |
| 变化 | 慢 | 快 |
| 分布 | 对数正态（dB域正态） | 瑞利/莱斯 |
| 模型 | AR(1) 空间相关 | Jakes 正弦叠加 |
| 叠加位置 | 加到总路径损耗 | 乘到信号复包络 |

## 7. 随机种子改造记录（任务 #9）

所有函数已增加 `rng: np.random.Generator = None` 参数：
- 函数开头：`if rng is None: rng = np.random.default_rng()`（兜底，保证可单独调用）
- 内部 `np.random.randn/uniform` 全部换成 `rng.standard_normal/uniform`
- 函数间调用用 `rng=rng` **逐级透传**
- docstring 必须是函数体第一条语句（否则 `__doc__` 为 None）

## 8. 自测问题

- AR(1) 里 `sqrt(1-rho²)` 为什么不能省？（保证输出方差=σ²）
- K 因子大小如何影响衰落？（K 越大直射越强、起伏越小）
- 阴影和小尺度分别作用在接收机的哪个环节？（损耗相加 vs 信号相乘）
