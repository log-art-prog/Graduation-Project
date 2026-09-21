"""Day3 预学习：贝塞尔函数与 Jakes 多普勒谱的关系可视化。

本脚本不产生仿真数据，仅用于理解：
  1. 第一类零阶贝塞尔函数 J0(x) 长什么样
  2. Jakes 多普勒谱 S(f) 长什么样
  3. 为什么 Jakes 谱的逆傅里叶变换 = J0(2π f_d τ)
  4. 不同 f_d/f_s 下 ACF 的形状差异

运行：python scripts/day3_bessel_explainer.py
输出：figs/day3_bessel_*.png（共 4 张）
"""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 非交互后端，避免 GUI 崩溃
import matplotlib.pyplot as plt
from scipy.special import j0  # 第一类零阶贝塞尔函数
from pathlib import Path

FIG_DIR = Path(__file__).resolve().parent.parent / "figs"
FIG_DIR.mkdir(exist_ok=True)

# 中文字体（Windows 默认自带）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False  # 负号正常显示

# ========== 图 1：贝塞尔函数 J0(x) 本身长什么样 ==========
def fig1_bessel_shape():
    x = np.linspace(0, 20, 1000)
    y = j0(x)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(x, y, "b-", lw=2, label=r"$J_0(x)$")
    ax.axhline(0, color="k", lw=0.5)
    ax.axhline(1.0, color="r", ls="--", lw=0.8, label=r"$J_0(0)=1$")

    # 标注前几个零点
    zeros = [2.4048, 5.5201, 8.6537, 11.7915, 14.9309]
    for z in zeros:
        ax.axvline(z, color="gray", ls=":", lw=0.6)
        ax.text(z, -0.9, f"{z:.1f}", ha="center", fontsize=8, color="gray")

    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$J_0(x)$")
    ax.set_title("图1：第一类零阶贝塞尔函数 J0(x) —— 像一个「衰减的余弦波」")
    ax.legend()
    ax.set_ylim(-1.1, 1.1)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "day3_fig1_bessel_shape.png", dpi=120)
    plt.close()
    print("saved: figs/day3_fig1_bessel_shape.png")


# ========== 图 2：Jakes 多普勒谱 S(f) ==========
def fig2_jakes_spectrum():
    fd = 50.0  # 最大多普勒频移 Hz
    f = np.linspace(-fd * 1.3, fd * 1.3, 2000)

    # Jakes 谱：S(f) ∝ 1 / sqrt(1 - (f/fd)^2),  |f| < fd
    # 加小正则 epsilon 防止端点奇异
    eps = 1e-6
    ratio = f / fd
    S = np.where(np.abs(ratio) < 1.0,
                 1.0 / np.sqrt(np.maximum(1.0 - ratio ** 2, eps)),
                 0.0)
    # 归一化：让积分 = 1（功率谱密度下总功率=1）
    S = S / np.trapezoid(S, f)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(f, S, "b-", lw=2)
    ax.fill_between(f, S, alpha=0.15)
    ax.axvline(-fd, color="r", ls="--", lw=0.8, label=f"±f_d = ±{fd:.0f} Hz")
    ax.axvline(fd, color="r", ls="--", lw=0.8)
    ax.set_xlabel("频率 f (Hz)")
    ax.set_ylabel("功率谱密度 S(f)")
    ax.set_title("图2：Jakes 多普勒谱 —— 两端发散的「U型碗」")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "day3_fig2_jakes_spectrum.png", dpi=120)
    plt.close()
    print("saved: figs/day3_fig2_jakes_spectrum.png")


# ========== 图 3：数值验证 —— Jakes 谱的 IFFT 确实是 J0 ==========
def fig3_ifft_matches_j0():
    """
    严谨验证：对 Jakes 谱做数值 IFFT，看结果是否等于 J0(2π f_d τ)。

    数学上：
      S(f) = 1/(π f_d) * 1/sqrt(1-(f/fd)^2),  |f|<fd
      R(τ) = ∫ S(f) e^{j2πfτ} df  =  J0(2π f_d τ)
    """
    fd = 50.0
    N = 2 ** 16  # 足够大的 FFT 点数
    fs = 10000.0  # 采样率，远大于 fd 以避免混叠
    df = fs / N

    f = np.fft.fftfreq(N, d=1.0 / fs)
    # 构造 Jakes 谱
    eps = 1e-10
    ratio = f / fd
    S = np.where(np.abs(ratio) < 1.0,
                 1.0 / np.sqrt(np.maximum(1.0 - ratio ** 2, eps)),
                 0.0)
    # 归一化使连续积分 ∫S(f)df = 1，即离散和 ΣS·Δf = 1
    # 等价于 ΣS = 1/Δf = N/fs
    S = S * (N / fs) / S.sum()

    # IFFT 得到自相关：R(n/fs) = fs · ifft(S)[n]
    # （连续自相关 R(τ)=∫S(f)e^{j2πfτ}df 的离散近似）
    R_numeric = (fs * np.fft.ifft(S)).real
    # 取前半段（正滞后）
    R_numeric = R_numeric[:N // 2]
    tau = np.arange(N // 2) / fs

    # 理论值 J0(2π f_d τ)
    R_theory = j0(2 * np.pi * fd * tau)

    # 只画前 0.1 秒（约 5 个振荡周期），再后面 J0 衰减到接近 0
    cutoff = 0.1
    mask = tau <= cutoff

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})
    axes[0].plot(tau[mask] * 1000, R_theory[mask], "r-", lw=2,
                 label=r"理论 $J_0(2\pi f_d \tau)$")
    axes[0].plot(tau[mask] * 1000, R_numeric[mask], "b--", lw=1.5,
                 label="数值 IFFT(Jakes 谱)")
    axes[0].axhline(0, color="k", lw=0.5)
    axes[0].set_ylabel("自相关 R(τ)")
    axes[0].set_title("图3：Jakes 谱 与 J0 自相关 —— 傅里叶变换对")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    err = np.abs(R_numeric[mask] - R_theory[mask])
    axes[1].plot(tau[mask] * 1000, err, "g-", lw=1)
    axes[1].set_ylabel("|数值 - 理论|")
    axes[1].set_xlabel("滞后 τ (ms)")
    axes[1].set_title(f"最大误差 = {err.max():.2e}")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "day3_fig3_ifft_matches_j0.png", dpi=120)
    plt.close()
    print("saved: figs/day3_fig3_ifft_matches_j0.png")
    print(f"  最大绝对误差 = {err.max():.2e}")


# ========== 图 4：不同 f_d/f_s 下 ACF 的形状 ==========
def fig4_fd_ratio_effect():
    """
    关键无量纲量 f_d/f_s 决定归一化 ACF 形状。
    f_d 越大（卫星飞得越快），相关时间越短（信号起伏越快）。
    """
    fs = 1000.0
    ratios = [0.01, 0.05, 0.10]
    colors = ["#2ca02c", "#1f77b4", "#d62728"]

    # 滞后用「采样点」为单位，最多画 200 个点
    n_lag = 200
    lag_samples = np.arange(n_lag)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for ratio, color in zip(ratios, colors):
        fd = ratio * fs
        tau = lag_samples / fs
        R = j0(2 * np.pi * fd * tau)
        ax.plot(lag_samples, R, color=color, lw=2,
                label=f"f_d/f_s = {ratio}  (f_d = {fd:.0f} Hz)")

    ax.axhline(0, color="k", lw=0.5)
    ax.axhline(np.exp(-1), color="gray", ls=":", lw=0.8,
               label=r"$e^{-1} \approx 0.368$（相关时间参考线）")
    ax.set_xlabel("滞后（采样点数，f_s=1000Hz 即 ms）")
    ax.set_ylabel("归一化自相关 R(τ)")
    ax.set_title("图4：多普勒越大，信号起伏越快，相关时间越短")
    ax.legend()
    ax.set_xlim(0, n_lag)
    ax.set_ylim(-0.5, 1.05)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "day3_fig4_fd_ratio_effect.png", dpi=120)
    plt.close()
    print("saved: figs/day3_fig4_fd_ratio_effect.png")


if __name__ == "__main__":
    fig1_bessel_shape()
    fig2_jakes_spectrum()
    fig3_ifft_matches_j0()
    fig4_fd_ratio_effect()
    print("\n全部画图完成。请查看 figs/day3_fig*.png")
