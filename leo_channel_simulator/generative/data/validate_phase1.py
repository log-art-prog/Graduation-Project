"""阶段一一键验收（任务 5.1）：G1~G4 全部检查，全 PASS 退出码 0，否则 1。

运行方式（在项目根目录 leo_channel_simulator/leo_channel_simulator/ 下）：
    python generative/data/validate_phase1.py

设计原则：
  - 每项检查用 check() 记录并打印 PASS/FAIL + 数值 detail
  - G2/G3 有随机性：阈值留余量，避免偶发统计抖动误报 FAIL
  - 不照抄日志里的结论数字，每次重新跑真实检查
"""
import sys
from pathlib import Path

# 本文件在 generative/data/ 下，项目根是上两级
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.special import j0, i0

from generative.data.corazza import (
    corazza_params,
    sample_corazza_iid,
    corazza_cdf_theory,
    colored_gaussian_jakes,
)
from generative.data.dataset import load_dataset, build_dataset
from generative.utils.seed import make_rng

CHECKS = []


def check(name: str, ok: bool, detail: str = "") -> None:
    """记录并打印一项检查结果。"""
    CHECKS.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}")


# ------------------------------------------------------------------
# 通用工具：经验 CDF vs 理论 CDF 的 KS 距离
# ------------------------------------------------------------------
def ks_distance(samples: np.ndarray, theory_cdf) -> float:
    """KS 距离 = max |F_emp(r) - F_theory(r)|。

    corazza_cdf_theory 内部用 quad 数值积分（标量函数），
    故在样本范围内取 200 个网格点算理论 CDF，再插值回样本点——
    把积分次数从 N 次降到 200 次，速度提升两个数量级。
    """
    x = np.sort(samples)
    F_emp = np.arange(1, x.size + 1) / x.size

    grid = np.linspace(x.min(), x.max(), 200)
    F_theory_grid = np.array([theory_cdf(r) for r in grid])
    F_theory = np.interp(x, grid, F_theory_grid)
    return float(np.max(np.abs(F_emp - F_theory)))


# ------------------------------------------------------------------
# G1：参数曲线合理性
# ------------------------------------------------------------------
def validate_g1() -> None:
    print("--- G1 仰角参数合理性 ---")
    K1, _, sigma1 = corazza_params(alpha_deg=20)
    K2, _, sigma2 = corazza_params(alpha_deg=80)
    check("K 随仰角单调上升", K1 < K2, f"K(20)={K1:.2f}, K(80)={K2:.2f}")
    check("sigma 随仰角单调下降", sigma1 > sigma2,
          f"sigma(20)={sigma1:.2f}, sigma(80)={sigma2:.2f}")


# ------------------------------------------------------------------
# G2：i.i.d. 单点分布 KS
# ------------------------------------------------------------------
def validate_g2() -> None:
    print("--- G2 i.i.d. CDF（KS < 0.01）---")
    rng = make_rng(7)
    for alpha in [30.0, 60.0]:
        samples = sample_corazza_iid(alpha, 200000, rng)
        K, mu, sigma = corazza_params(alpha)
        ks = ks_distance(samples, lambda r: corazza_cdf_theory(r, K, mu, sigma))
        check(f"alpha={alpha:.0f}°KS", ks < 0.01, f"KS={ks:.4f}")


# ------------------------------------------------------------------
# G3：时序自检（ACF / LCR）
# ------------------------------------------------------------------
def _acf_fft(x: np.ndarray) -> np.ndarray:
    """归一化 ACF，支持复数输入。"""
    n = len(x)
    n_fft = 1 << (2 * n - 1).bit_length()
    X = np.fft.fft(x, n_fft)
    acf = np.fft.ifft(X * np.conj(X)).real[:n]
    return acf / acf[0]


def validate_g3() -> None:
    print("--- G3 时序 ACF（RMSE < 0.02）/ LCR ---")

    # ===== ACF 部分（K=0 纯 Rayleigh，复过程 ACF 才是纯 J0）=====
    n, fs, fd = 100000, 1000.0, 50.0
    rng = make_rng(11)
    x = colored_gaussian_jakes(n, fs, fd, rng)
    y = colored_gaussian_jakes(n, fs, fd, rng)
    z = x + 1j * y

    n_lag = 100
    acf_emp = _acf_fft(z)[:n_lag]
    tau = np.arange(n_lag) / fs
    acf_theory = j0(2 * np.pi * fd * tau)
    rmse = np.sqrt(np.mean((acf_emp - acf_theory) ** 2))
    check("ACF RMSE", rmse < 0.02, f"RMSE={rmse:.4f}")

    # ===== LCR 部分（K=0 Rayleigh，公式退化为 N = fd·ρ·exp(-ρ²)）=====
    n = 200000
    rng = make_rng(7)
    K = 0.0
    s_R = np.sqrt(1.0 / (2.0 * (K + 1.0)))
    x = colored_gaussian_jakes(n, fs, fd, rng)
    y = colored_gaussian_jakes(n, fs, fd, rng)
    R = np.sqrt((s_R * x) ** 2 + (s_R * y) ** 2)
    R_rms = np.sqrt(np.mean(R ** 2))

    rhos = np.linspace(0.2, 2.0, 10)
    emp_lcr, thy_lcr = [], []
    for rho in rhos:
        r_th = rho * R_rms
        crossings = np.sum((R[:-1] > r_th) & (R[1:] <= r_th))
        emp_lcr.append(crossings / ((n - 1) / fs))

        N = (np.sqrt(2 * np.pi * (K + 1)) * fd * rho
             * np.exp(-K - (K + 1) * rho ** 2)
             * i0(2 * rho * np.sqrt(K * (K + 1))))
        thy_lcr.append(N)

    emp_lcr = np.array(emp_lcr)
    thy_lcr = np.array(thy_lcr)
    rel_err = np.abs(emp_lcr - thy_lcr) / np.maximum(thy_lcr, 1e-10)
    max_err = rel_err.max()
    check("LCR 最大相对误差", max_err < 0.15, f"max_err={max_err:.1%}")


# ------------------------------------------------------------------
# G4：数据集固化
# ------------------------------------------------------------------
def validate_g4() -> None:
    print("--- G4 数据集固化 ---")
    out_dir = ROOT / "datasets"
    npz_path = out_dir / "corazza_train.npz"
    meta_path = out_dir / "corazza_train_meta.json"

    check("npz 存在", npz_path.exists())
    check("meta 存在", meta_path.exists())

    data = load_dataset(npz_path)
    x_std = data["x_std"]
    alpha_norm = data["alpha_norm"]
    check("x_std.shape", x_std.shape == (10000, 1000))
    check("x_std.dtype", x_std.dtype == np.float32)
    check("alpha_norm.shape", alpha_norm.shape == (10000,))

    # 复现性验证机制（n=50 几秒出结果）
    d1 = build_dataset(n_samples=50, seed=42)
    d2 = build_dataset(n_samples=50, seed=42)
    check("同种子元素一致", np.array_equal(d1["x_std"], d2["x_std"]))
    d3 = build_dataset(n_samples=50, seed=43)
    check("不同种子结果不同", not np.array_equal(d1["x_std"], d3["x_std"]))


if __name__ == "__main__":
    validate_g1()
    validate_g2()
    validate_g3()
    validate_g4()
    n_pass, n_all = sum(CHECKS), len(CHECKS)
    print("\n" + "=" * 50)
    print(f"阶段一验收：{n_pass}/{n_all} PASS")
    print("=" * 50)
    sys.exit(0 if all(CHECKS) else 1)
