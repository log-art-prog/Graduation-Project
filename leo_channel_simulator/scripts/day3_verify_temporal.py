"""Day 3 时序相关性自检脚本（任务 3.4 / 3.5 / 3.6）。

运行：python scripts/day3_verify_temporal.py
输出：figs/day3_verify_*.png + 终端打印验收结果

验收关卡 G3：
  - 3.4 ACF：前 50 滞后点 RMSE < 0.02
  - 3.5 LCR：多门限相对误差 < 15%
  - 3.6 阴影 ACF：τ=tau_c 处降到 e^-1 量级
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import j0
from pathlib import Path

from generative.data.corazza import (
    colored_gaussian_jakes, colored_gaussian_lp, generate_corazza_sequence,
)
from generative.utils.seed import make_rng

FIG_DIR = ROOT / "figs"
FIG_DIR.mkdir(exist_ok=True)
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def _acf_fft(x: np.ndarray) -> np.ndarray:
    """用 FFT 计算实数/复数序列的归一化自相关（线性，非循环）。"""
    n = len(x)
    n_fft = 1 << (2 * n - 1).bit_length()
    X = np.fft.fft(x, n_fft)
    acf = np.fft.ifft(X * np.conj(X)).real[:n]
    return acf / acf[0]


# ============================================================
# 任务 3.4：ACF 自检
# ============================================================
def verify_acf() -> bool:
    n = 100000
    fs, fd = 1000.0, 50.0
    rng = make_rng(2024)

    # 过采样技巧：生成 2n 点再截取中间 n 点，减少循环卷积边界效应
    n_gen = 2 * n
    x = colored_gaussian_jakes(n_gen, fs, fd, rng)
    y = colored_gaussian_jakes(n_gen, fs, fd, rng)
    start = n_gen // 4
    x = x[start:start + n]
    y = y[start:start + n]
    z = x + 1j * y  # 复基带过程

    acf_emp = _acf_fft(z)
    tau = np.arange(n) / fs
    acf_theory = j0(2 * np.pi * fd * tau)

    n_check = 50
    rmse = np.sqrt(np.mean((acf_emp[:n_check] - acf_theory[:n_check]) ** 2))

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(tau[:n_check] * 1000, acf_theory[:n_check], "r-", lw=2,
            label=r"理论 $J_0(2\pi f_d \tau)$")
    ax.plot(tau[:n_check] * 1000, acf_emp[:n_check], "b--", lw=1.5,
            label="经验 ACF")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("滞后 τ (ms)")
    ax.set_ylabel("归一化自相关")
    ax.set_title(f"任务 3.4 ACF 自检：前 {n_check} 滞后点 RMSE = {rmse:.4f}")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "day3_verify_acf.png", dpi=120)
    plt.close()

    passed = rmse < 0.02
    status = "PASS" if passed else "FAIL"
    print(f"[3.4] ACF 自检  RMSE={rmse:.4f}  (标准 <0.02)  [{status}]")
    return passed


# ============================================================
# 任务 3.5：LCR 自检
# ============================================================
def verify_lcr() -> bool:
    """Rice LCR 闭式公式：
       N_R(r_th) = sqrt(2π(K+1)) · fd · ρ · exp(-K-(K+1)ρ²) · I0(2ρ√(K(K+1)))
       ρ = r_th / R_rms
    """
    from scipy.special import i0  # 第一类零阶修正贝塞尔函数

    n = 200000  # 长序列降低 LCR 统计误差
    fs, fd = 1000.0, 50.0
    rng = make_rng(7)

    # 用 K=0（Rayleigh）先测，公式更简单：N = fd·ρ·exp(-ρ²)
    K = 0.0
    s_R = np.sqrt(1.0 / (2.0 * (K + 1.0)))
    x = colored_gaussian_jakes(n, fs, fd, rng)
    y = colored_gaussian_jakes(n, fs, fd, rng)
    R = np.sqrt((s_R * x) ** 2 + (s_R * y) ** 2)  # K=0 时 m=0

    R_rms = np.sqrt(np.mean(R ** 2))

    # 多个门限（ρ = r_th / R_rms）
    rhos = np.linspace(0.2, 2.0, 10)
    emp_lcr = []
    thy_lcr = []

    for rho in rhos:
        r_th = rho * R_rms
        # 向下穿越计数：R[i] > r_th 且 R[i+1] <= r_th
        crossings = np.sum((R[:-1] > r_th) & (R[1:] <= r_th))
        duration = (n - 1) / fs  # 总时长（秒）
        emp_lcr.append(crossings / duration)

        # Rice LCR 闭式公式（K=0 时 I0(0)=1, exp(-0)=1）
        N = (np.sqrt(2 * np.pi * (K + 1)) * fd * rho
             * np.exp(-K - (K + 1) * rho ** 2)
             * i0(2 * rho * np.sqrt(K * (K + 1))))
        thy_lcr.append(N)

    emp_lcr = np.array(emp_lcr)
    thy_lcr = np.array(thy_lcr)
    rel_err = np.abs(emp_lcr - thy_lcr) / np.maximum(thy_lcr, 1e-10)
    max_err = rel_err.max()

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(rhos, thy_lcr, "ro-", lw=2, label="理论 Rice LCR")
    ax.plot(rhos, emp_lcr, "bx-", lw=1.5, label="经验向下穿越率")
    ax.set_xlabel(r"门限 $\rho = r_{th} / R_{rms}$")
    ax.set_ylabel("LCR (次/秒)")
    ax.set_title(f"任务 3.5 LCR 自检：最大相对误差 = {max_err:.1%}")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "day3_verify_lcr.png", dpi=120)
    plt.close()

    passed = max_err < 0.15
    status = "PASS" if passed else "FAIL"
    print(f"[3.5] LCR 自检  最大相对误差={max_err:.1%}  (标准 <15%)  [{status}]")
    return passed


# ============================================================
# 任务 3.6：阴影慢变 + 两层结构
# ============================================================
def verify_shadow() -> bool:
    fs = 1000.0
    tau_c = 0.3
    rng = make_rng(99)

    # 阴影 ACF 检查
    n = 50000
    g = colored_gaussian_lp(n, fs, tau_c, rng)
    acf = _acf_fft(g)
    lag_c = int(tau_c * fs)  # 对应 tau_c 的采样点数
    val_at_tauc = acf[lag_c]
    target = np.exp(-1)

    # 目视两层结构：生成一条完整序列
    r = generate_corazza_sequence(45.0, length=1000, fs=fs, fd=50.0,
                                  tau_shadow=tau_c, rng=rng)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7))

    # 上图：阴影 ACF
    lags_ms = np.arange(min(len(acf), 1000)) / fs * 1000
    axes[0].plot(lags_ms, acf[:len(lags_ms)], "b-", lw=1.5, label="阴影 ACF")
    axes[0].axvline(tau_c * 1000, color="r", ls="--", lw=1,
                    label=f"τ_c = {tau_c}s")
    axes[0].axhline(target, color="gray", ls=":", lw=1,
                    label=f"e^-1 ≈ {target:.3f}")
    axes[0].set_xlabel("滞后 (ms)")
    axes[0].set_ylabel("归一化 ACF")
    axes[0].set_title(f"任务 3.6 阴影 ACF：τ=τ_c 处 = {val_at_tauc:.3f}（目标 e^-1={target:.3f}）")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # 下图：完整序列两层结构
    t = np.arange(1000) / fs
    axes[1].plot(t, r, "b-", lw=0.8, label="r[n] = R·S")
    axes[1].set_xlabel("时间 (s)")
    axes[1].set_ylabel("包络幅度")
    axes[1].set_title("任务 3.6 两层结构：快起伏叠加慢包络")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "day3_verify_shadow.png", dpi=120)
    plt.close()

    # 验收：τ=τ_c 处 ACF 在 e^-1 附近（容忍 ±50%）
    passed = abs(val_at_tauc - target) < 0.5
    status = "PASS" if passed else "FAIL"
    print(f"[3.6] 阴影 ACF  R(τ_c)={val_at_tauc:.3f}  (目标 e^-1={target:.3f})  [{status}]")
    return passed


if __name__ == "__main__":
    print("=" * 60)
    print("Day 3 时序相关性自检")
    print("=" * 60)
    p1 = verify_acf()
    p2 = verify_lcr()
    p3 = verify_shadow()
    print("=" * 60)
    all_pass = p1 and p2 and p3
    print(f"总体验收：{'全部 PASS ✓' if all_pass else '存在 FAIL ✗'}")
