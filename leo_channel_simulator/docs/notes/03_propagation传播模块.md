# 03 — propagation.py 传播损耗模块

> 作用：计算信号从卫星到地面站的大尺度路径损耗，以及接收功率、SNR。
> 参考 ITU-R 系列标准（P.676 大气、P.838 雨、P.840 云雾）。

---

## 1. 物理常数

```python
C = 299792458          # 光速 m/s
BOLTZMANN = 1.38e-23   # 玻尔兹曼常数 J/K
```

## 2. 自由空间路径损耗 free_space_path_loss

Friis 公式（无任何遮挡的理想损耗）：

```
PL_fspl(dB) = 20·log10(4π·d / λ)，  λ = c/fc
```

- 距离 d 越远、频率 f 越高 → 损耗越大。
- 距离翻倍 → 损耗 +6 dB；频率翻倍 → 损耗 +6 dB。

## 3. 大气气体吸收 atmospheric_gas_loss（P.676 简化）

思路：**比衰减 γ(dB/km) × 斜穿大气层的路径长度**

```python
# 按频段分段给 γ（氧气+水蒸气），例如 60GHz 是氧气吸收峰
slant_range = h_atm / sin(仰角)     # 对流层等效高度 6km
PL_atm = gamma * slant_range
```

关键：`斜路径 = 6/sin(θ)`。**仰角越低，信号斜穿大气层越厚，损耗越大**（θ→0 时趋于无穷）。

## 4. 降雨衰减 rain_attenuation（P.838 简化）

```python
gamma_r = k · R^α            # R=雨率mm/h，k/α 随频段查表
slant_range = 4km / sin(θ)   # 雨区等效高度
r_factor = 1/(1+slant/30)    # 路径缩短因子
PL_rain = gamma_r * slant_range * r_factor
```

- k、α 按频率分段（圆极化取水平/垂直极化平均）。
- 雨率参考：小雨 2、中雨 10、暴雨 50 mm/h。
- Ka 及以上频段雨衰显著，S/C 频段可忽略。

## 5. 云雾衰减 cloud_fog_loss（P.840）

- 10 GHz 以下直接返回 0（可忽略）。
- 以上按液态水比衰减系数 × 云厚斜路径估算。

## 6. 总损耗组装 calculate_total_path_loss

```python
PL_total = PL_fspl + PL_atm + PL_rain + PL_cloud
```

返回一个字典，分别给出各分量和总量——方便画"损耗分解图"。

## 7. 接收功率与 SNR calculate_received_power

```
噪声功率(dBm) = 10·log10(k·T·B·1000) + 噪声系数
接收功率(dBm) = EIRP + 接收天线增益 - 路径损耗
SNR(dB)      = 接收功率 - 噪声功率
```

要点：
- EIRP 已含发射增益，所以调用时 `tx_gain_db=0`（该参数实际冗余，属待清理项 #10）。
- 噪声 = 热噪声 kTB ＋ 接收机噪声系数，带宽越大噪声越大。

## 8. dB 运算的核心心法

- 功率的**乘除**在 dB 域变成**加减**：总损耗=各分量直接相加。
- dBm = 相对 1mW 的分贝：`dBm = 10·log10(P / 1mW)`。
- 代码里 `×1000` 是把 W 换成 mW 再取对数。

## 9. 自测问题

- 为什么所有大气类损耗都有 `1/sin(仰角)` 因子？（低仰角斜穿路径更长）
- 总损耗为什么可以直接相加？（各分量都已换算为 dB）
- SNR 为什么随带宽增大而下降？（热噪声功率 kTB 随带宽线性增大）
