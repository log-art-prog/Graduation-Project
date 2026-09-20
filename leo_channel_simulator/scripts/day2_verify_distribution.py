"""Day 2 验证：i.i.d. 复合包络分布对拍理论曲线，复现论文图 3-2/3-3。

运行：D:\\anaconda\\envs\\leo_gen\\python.exe scripts\\day2_verify_distribution.py
产出：figs/p1_fig3-2_pdf.png、figs/p1_fig3-3_cdf.png，控制台打印 KS 与单调性表。
"""
import sys
from pathlib import Path

# 让脚本无论从哪个目录启动，都能找到项目根（与 day1 脚本同套路）
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")          # 后台渲染，不需要弹窗；远程/无窗口环境也能存图
import matplotlib.pyplot as plt

from generative.utils.seed import make_rng
from generative.data.corazza import (corazza_params, sample_corazza_iid,
                                     corazza_cdf_theory)

# ========== 全局参数 ==========
ALPHAS = [20, 40, 60, 80]      # 论文图 3-2/3-3 的四个仰角
N = 1_000_000                  # 每仰角样本数（快速冒烟可改 1e5）
FIGS_DIR = ROOT / "figs"


def main():
    # ===== 任务 2.7：生成四个仰角的 i.i.d. 样本 =====
    samples = {}                                  # key=仰角, value=长度N的数组
    params = {}                                   # 顺便存对应的 (K, mu, sigma)
    for a in ALPHAS:
        samples[a] = sample_corazza_iid(float(a), N, make_rng(a))
        params[a] = corazza_params(float(a))

    # ===== 任务 2.8：PDF 对比图（复现论文图 3-2）=====
    # plt.subplots(2, 2) 一次创建 2行2列 共4个子图，返回 (fig, axes)
    # fig 是整张画布，axes 是形状 (2,2) 的数组，axes[i][j] 取第 i 行第 j 列
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)

    for ax, a in zip(axes.flat, ALPHAS):
        # ax 是当前子图对象；zip 把 axes.flat（4个子图按行展开）和 ALPHAS 配对
        K, mu, sig = params[a]
        r = samples[a]

        # --- 经验 PDF：直方图 ---
        # bins=80 分80个柱子；density=True 把柱子总面积归一到1，才能和理论PDF直接比
        # alpha=0.6 透明度，让理论曲线能透出来；label 给图例用
        ax.hist(r, bins=80, density=True, alpha=0.5, color="C0",
                label="empirical (1e6)")

        # --- 理论 PDF：由理论 CDF 数值差分得到 ---
        # 网格上限取 99.9% 分位，避免 lognormal 长尾把横坐标拉扁
        r_max = np.quantile(r, 0.999)
        r_grid = np.linspace(1e-4, r_max, 400)

        # 对网格上每个 r 算理论 CDF（标量循环，因为 corazza_cdf_theory 不支持向量）
        th_cdf = np.array([corazza_cdf_theory(float(x), K, mu, sig)
                           for x in r_grid])
        # np.gradient(y, x) 数值求导 dy/dx；CDF 的导数就是 PDF
        th_pdf = np.gradient(th_cdf, r_grid)

        ax.plot(r_grid, th_pdf, "r-", linewidth=2, label="theory (eq.7)")

        ax.set_title(f"alpha = {a} deg")
        ax.set_xlabel("envelope r")
        ax.set_ylabel("PDF")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle("Corazza composite PDF  (r = R * S, i.i.d.)")
    fig.tight_layout()              # 自动调整子图间距，防止标题重叠
    fig.savefig(FIGS_DIR / "p1_fig3-2_pdf.png", dpi=150)
    plt.close(fig)                  # 释放内存（画完一张关一张）
    print(f"saved: {FIGS_DIR / 'p1_fig3-2_pdf.png'}")

    # ===== 任务 2.9：CDF 对比图（复现论文图 3-3）=====
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)

    for ax, a in zip(axes.flat, ALPHAS):
        K, mu, sig = params[a]
        r = samples[a]
        r_sorted = np.sort(r)                         # 从小到大排
        # 经验 CDF：第 k 小的值对应的累积概率是 k/N
        emp_cdf = np.arange(1, N + 1) / N

        # 每 1000 个点取一个画，100万点全画会卡且没必要
        ax.plot(r_sorted[::1000], emp_cdf[::1000], color="C0", alpha=0.6,
                linewidth=2, label="empirical")

        # 理论 CDF：直接复用上面的 r_grid / th_cdf 逻辑
        r_max = np.quantile(r, 0.999)
        r_grid = np.linspace(1e-4, r_max, 400)
        th_cdf = np.array([corazza_cdf_theory(float(x), K, mu, sig)
                           for x in r_grid])
        ax.plot(r_grid, th_cdf, "r--", linewidth=2, label="theory (eq.7)")

        ax.set_title(f"alpha = {a} deg")
        ax.set_xlabel("envelope r")
        ax.set_ylabel("CDF")
        ax.set_ylim(0, 1.02)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle("Corazza composite CDF  (r = R * S, i.i.d.)")
    fig.tight_layout()
    fig.savefig(FIGS_DIR / "p1_fig3-3_cdf.png", dpi=150)
    plt.close(fig)
    print(f"saved: {FIGS_DIR / 'p1_fig3-3_cdf.png'}")

    # ===== 任务 2.10：KS 统计量（验收关卡 G2 硬指标 < 0.01）=====
    print("\n" + "=" * 60)
    print("KS 检验（经验 CDF vs 理论 CDF，要求 < 0.01）")
    print("-" * 60)
    all_pass = True
    for a in ALPHAS:
        K, mu, sig = params[a]
        r_sorted = np.sort(samples[a])
        # 在 [min, max] 均匀取 200 个电平做对比
        r_levels = np.linspace(r_sorted[0], r_sorted[-1], 200)
        # searchsorted：每个 r_level 在排好序的样本里排第几位，除以 N 就是经验 CDF
        emp = np.searchsorted(r_sorted, r_levels) / N
        th = np.array([corazza_cdf_theory(float(x), K, mu, sig)
                       for x in r_levels])
        ks = np.max(np.abs(emp - th))
        flag = "PASS" if ks < 0.01 else "FAIL"
        if ks >= 0.01:
            all_pass = False
        print(f"  alpha = {a:2d} deg   KS = {ks:.5f}   [{flag}]")
    print("-" * 60)
    print(f"G2 KS 关卡：{'全部通过' if all_pass else '未通过'}")

    # ===== 任务 2.11：物理单调性检查表 =====
    print("\n物理单调性（仰角升高 -> 包络均值增大、深衰落概率减小）")
    print("-" * 60)
    print(f"  {'alpha':>6}  {'mean(r)':>10}  {'P(r<0.5)':>10}")
    for a in ALPHAS:
        r = samples[a]
        print(f"  {a:6d}  {r.mean():10.4f}  {np.mean(r < 0.5):10.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
