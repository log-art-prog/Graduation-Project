# 05 — channel_model.py 信道核心模块

> 作用：整合轨道 + 传播 + 环境 + 衰落，完成一次完整信道仿真。
> 这是理解整个程序数据流的关键文件。

---

## 1. 输出数据结构 ChannelResult

集中保存一次仿真的全部结果数组：

```
轨道相关   : orbit
路径损耗   : fspl_db, atmosphere_db, rain_db, total_pl_db
衰落       : shadow_db, small_scale_amp, k_factor_db,
             tau_rms_ns, coherence_bw_mhz
接收性能   : rx_power_dbm, snr_db
统计量     : fade_depth_db, lcr_hz
环境       : environment
```

所有数组长度相同（= 仿真点数 num_points），可按时间轴对齐画图。

## 2. simulate_channel 执行流程（逐步）

### 步骤 A：准备随机源（#9 已改造）

```python
if rng is None:
    rng = np.random.default_rng(seed)
```

### 步骤 B：逐点计算大尺度参数（第一个 for 循环）

对每个时刻 i：
1. 取该点仰角 `elev` 和斜距 `range_m`。
2. 调 `calculate_total_path_loss` → 存 fspl/atm/rain。
3. 调 `get_rms_delay_spread(env, elev)`、`get_k_factor(env, elev)` → 存时延、K、相干带宽。
4. 暂存不含阴影的总损耗 `total_pl_db[i]`。

> 仰角就是在这个循环里驱动所有大尺度参数的。

### 步骤 C：生成整条阴影衰落序列

```python
mean_velocity = mean(|orbit.velocity_kms|) * 1000
shadow_db = generate_shadow_fading(num_points, σ, velocity_ms, fs, rng=rng)
total_pl_with_shadow = total_pl_db + shadow_db
```

阴影是一次性生成的**整条空间相关序列**，然后逐点"加"到损耗上（dB 相加）。

### 步骤 D：逐点算接收功率 / SNR（第二个 for 循环）

```python
rx = calculate_received_power(EIRP, 0, rx_gain, total_pl_with_shadow[i], NF, B)
```

### 步骤 E：生成小尺度衰落（当前是简化处理）

```python
mid_idx = num_points // 2                 # 取过顶中点
avg_doppler = |doppler[mid_idx]|          # 只用中点一个典型值
avg_k = get_k_factor(env, elev[mid_idx])
small_scale = generate_rician_fading(..., avg_k, avg_doppler, fs, rng=rng)
```

⚠️ **已知简化点（改进项 #2）**：小尺度衰落整条序列只用过顶处的一个 K 和一个多普勒，
没有随仰角变化。未来要按仰角分段生成并做相位平滑拼接。

### 步骤 F：统计 + 封装

```python
metrics = calculate_fading_metrics(small_scale)
return ChannelResult(...)
```

## 3. 多环境调度 simulate_multiple_environments

```python
parent_rng = np.random.default_rng(seed)
child_rngs = parent_rng.spawn(len(environments))   # 分裂独立子流
for env, env_rng in zip(environments, child_rngs):
    result = simulate_channel(env, orbit, rng=env_rng, **kwargs)
```

为什么用 spawn：
- Urban/Suburban/Open 各得一条**独立但确定**的随机流，互不雷同；
- 某环境在列表中处于相同位置时，单独跑和一起跑结果一致；
- 同一个 seed 永远可复现。

## 4. 全流程图（背下来）

```
orbit(仰角/斜距/速度)
   │
   ├─ 逐点 ─→ 路径损耗(fspl+atm+rain) + K因子 + 时延扩展
   │
   ├─ 阴影衰落(AR(1)整条) ─→ 与损耗相加 ─→ 接收功率/SNR
   │
   └─ 小尺度莱斯衰落(整条) ─→ 衰落统计(LCR/AFD/深度)
   │
   ▼
ChannelResult → visualization 画图
```

## 5. 后续 #5 LoS/NLoS 将改在哪里

1. 在步骤 B 循环中根据仰角判定每个点的 LoS/NLoS 状态（新增 los_state 数组）。
2. NLoS 点：K=0（纯瑞利）、时延取上限、附加更重阴影。
3. 阴影序列按状态分段重建，接缝处平滑过渡。
4. ChannelResult 增加 los_state、los_probability 字段。

## 6. 自测问题

- 阴影衰落为什么一次性生成整条，而路径损耗要逐点算？
  （阴影需要空间相关性，必须整条 AR 生成；损耗只取决于当点几何参数）
- 小尺度衰落当前的简化在哪里？（#2：只用中点单一 K 和多普勒）
