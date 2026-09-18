"""Day 1 验证：表3-1 三参数随仰角曲线，对照论文图 3-1。

在项目根目录运行：
    D:\\anaconda\\envs\\leo_gen\\python.exe scripts\\day1_plot_params.py
"""
import sys
from pathlib import Path

# 让脚本无论从哪个目录启动，都能找到项目根
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")  # 无窗口环境也能存图
import matplotlib.pyplot as plt

from generative.data.corazza import corazza_params

ALPHA = np.linspace(20.0, 80.0, 300)
K, MU, SIGMA = corazza_params(ALPHA)

# 1) 打印关键仰角数值，和笔记 08 表格核对
print(f"{'alpha':>6} {'K':>8} {'mu':>8} {'sigma':>8}")
for a in (20.0, 35.0, 50.0, 65.0, 80.0):
    k, mu, sig = corazza_params(a)
    print(f"{a:6.0f} {k:8.3f} {mu:8.3f} {sig:8.3f}")

# 2) 三联子图（标题用英文，避免 matplotlib 中文乱码）
fig, axes = plt.subplots(3, 1, figsize=(7, 9), sharex=True)
axes[0].plot(ALPHA, K, color="C0")
axes[0].set_ylabel("K (raw table value, unit TBD)")
axes[1].plot(ALPHA, MU, color="C2")
axes[1].set_ylabel("mu (raw table value, unit TBD)")
axes[2].plot(ALPHA, SIGMA, color="C3")
axes[2].set_ylabel("sigma (raw table value, unit TBD)")
axes[2].set_xlabel("Elevation alpha (deg)")
for ax in axes:
    ax.grid(alpha=0.3)
fig.suptitle("Corazza rural parameters vs elevation (paper Table 3-1)")
fig.tight_layout()

out_path = ROOT / "figs" / "p1_fig3-1_params.png"
fig.savefig(out_path, dpi=150)
print(f"figure saved: {out_path}")