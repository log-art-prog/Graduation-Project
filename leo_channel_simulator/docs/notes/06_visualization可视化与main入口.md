# 06 — visualization.py 可视化 与 main.py 入口

---

## 第一部分：main.py 程序入口

### 执行顺序

1. 配置参数（集中在 main() 开头，方便修改）：

| 参数 | 默认值 | 含义 |
|------|--------|------|
| FC_HZ | 20e9 | 载波频率 20GHz（Ka） |
| BANDWIDTH_HZ | 200e6 | 带宽 |
| TX_EIRP_DBM | 55 | 发射 EIRP |
| RX_GAIN_DB | 35 | 接收天线增益 |
| NOISE_FIGURE_DB | 2.5 | 噪声系数 |
| RAIN_RATE_MMH | 2 | 雨率 |
| PASS_DURATION_S | 600 | 过境时长 |
| NUM_POINTS | 1000 | 仿真点数 |
| USE_REAL_TLE | False | 是否用真实 TLE |
| SEED | 42 | 随机种子（#9 新增） |

2. 计算轨道：`simple_orbit_model(...)` 或 `compute_orbit_from_tle(...)`。
3. 三环境仿真：`simulate_multiple_environments(..., seed=SEED)`。
4. 画图：单环境详细图 + 三环境对比图。
5. 打印汇总表 `print_summary_table`。

### 路径处理
TLE 文件和 output 目录都用
`os.path.join(os.path.dirname(__file__), ...)`
基于脚本自身位置定位，保证从任意工作目录运行都能找到。

---

## 第二部分：visualization.py 可视化

中文字体配置：
```python
plt.rcParams['font.sans-serif'] = ['DejaVu Sans','SimHei',...]
plt.rcParams['axes.unicode_minus'] = False
```
（当前环境无中文字体时图中中文会显示方块，英文标签正常。）

### 单环境 8 子图 plot_single_channel_result

| 子图 | 内容 |
|------|------|
| 1 | 仰角（左轴）+ 斜距（右轴双轴） |
| 2 | 多普勒频移曲线，过顶过零 |
| 3 | 路径损耗三分量分解 |
| 4 | 阴影衰落时间序列 |
| 5 | 接收功率 + SNR（双轴） |
| 6 | K 因子随时间（仰角）变化 |
| 7 | 小尺度衰落包络(dB) |
| 8 | 时延扩展 + 相干带宽（对数纵轴双轴） |

### 多环境 6 子图 plot_environment_comparison
仰角参考、总损耗、接收功率、SNR（含 QPSK 10dB 门限线）、K 因子、时延扩展。
用各环境自己的 `color` 和 `linestyle` 区分。

### 汇总表 print_summary_table
在终端打印轨道信息 + 三环境参数对比 + 关键结论。

### 常用 matplotlib 技巧
- `twinx()`：同一图两个纵轴。
- `set_yscale('log')`：对数纵轴（时延/带宽跨数量级时用）。
- `savefig(dpi=150, bbox_inches='tight')`：高清保存并自动裁边。
- 画完 `plt.close()`：批量出图时释放内存，避免多图叠加。

## 输出文件
```
output/suburban_channel_detail.png     # 城郊详细
output/environment_comparison.png      # 三环境对比
```
