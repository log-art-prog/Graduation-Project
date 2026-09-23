"""Day 5：5.2 仰角分箱统计 + Spearman 单调性；5.3 75° 参考样本。

运行方式（在项目根目录下）：
    python scripts/day5_binned_stats.py

产出：
    figs/p1_mean_std_vs_alpha.png     分箱 mean±std vs 仰角
    datasets/alpha75_ref.npz          75° 参考样本（泛化评估预留）
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Day 3 踩过的坑：中文字体
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

from scipy.stats import spearmanr

from generative.data.dataset import load_dataset
from generative.data.transforms import inverse_standardize, standardize
from generative.data.corazza import generate_corazza_sequence
from generative.utils.seed import make_rng

DATA_DIR = ROOT / "datasets"


# =====================================================================
# 5.2 分箱统计
# =====================================================================
def binned_stats(data: dict, bin_width: float = 10.0) -> dict:
    """按仰角分箱统计还原后的包络 mean/std 与深衰落概率。

    返回 dict：centers / means / stds / p_deep / spearman
    """
    alpha_deg = data["alpha_deg"]
    x_std = data["x_std"]
    mu_log = float(data["mu_log"])
    sigma_log = float(data["sigma_log"])

    # 1. 还原全库物理包络（用数据集里存的统一常量，不要重新 fit）
    r_all = inverse_standardize(x_std, mu_log, sigma_log)

    # 2. 深衰落门限：全库 5% 分位数
    r0 = np.quantile(r_all, 0.05)

    # 3. 分箱（np.digitize 返回箱号，越界用 clip 兜底）
    bins = np.arange(20.0, 80.0 + bin_width, bin_width)
    idx = np.digitize(alpha_deg, bins)
    idx = np.clip(idx, 1, len(bins) - 1)
    centers = (bins[:-1] + bins[1:]) / 2

    # 4. 每箱统计：mean / std / P(r < r0)
    means, stds, p_deep = [], [], []
    for b in range(1, len(bins)):
        mask = (idx == b)
        if mask.sum() == 0:
            continue
        means.append(r_all[mask].mean())
        stds.append(r_all[mask].std())
        p_deep.append((r_all[mask] < r0).mean())

    # 5. Spearman 单调性（rho=1 表示完美单调）
    rho, _ = spearmanr(centers, means)
    print(f"中心角: {centers}")
    print(f"平均包络: {means}")
    print(f"标准差:   {stds}")
    print(f"深衰落概率: {p_deep}")
    print(f"Spearman(仰角, 平均包络) = {rho:.4f}")
    assert rho > 0.95, f"Spearman {rho:.4f} 不达标（目标 > 0.95）"
    print(f"最低箱深衰落 {p_deep[0]:.4f} > 最高箱 {p_deep[-1]:.4f}: "
          f"{p_deep[0] > p_deep[-1]}")

    return {"centers": centers, "means": means, "stds": stds,
            "p_deep": p_deep, "spearman": rho}


def plot_binned(result: dict) -> None:
    """画 mean±std vs 仰角图。"""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(result["centers"], result["means"], yerr=result["stds"],
                fmt="o-", capsize=3, color="#1f77b4")
    ax.set_xlabel("仰角 (°)")
    ax.set_ylabel("平均包络 ± 标准差")
    ax.set_title(f"分箱 mean±std vs 仰角  Spearman={result['spearman']:.4f}")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(ROOT / "figs" / "p1_mean_std_vs_alpha.png", dpi=120)


# =====================================================================
# 5.3 75° 参考样本
# =====================================================================
def build_alpha75_ref(n_ref: int = 50) -> None:
    """生成 50 条 75° 参考序列存 datasets/alpha75_ref.npz。

    用途：训练集仰角是连续均匀采样，几乎不可能正好含 75.0°，
    这批样本是模型没见过的"标准答案"，供阶段四/五做泛化评估。
    """
    rng = make_rng(2026)  # 与训练集 seed=42 不同 -> 随机数流独立

    # 用训练集的统一常量标准化（不要重算）
    d = load_dataset(DATA_DIR / "corazza_train.npz")
    mu_log, sigma_log = float(d["mu_log"]), float(d["sigma_log"])

    raw = np.empty((n_ref, 1000))
    for i in range(n_ref):
        raw[i] = generate_corazza_sequence(75.0, length=1000, rng=rng)

    x_std = standardize(raw, mu_log, sigma_log).astype(np.float32)
    np.savez(DATA_DIR / "alpha75_ref.npz",
             x_std=x_std,
             alpha_deg=np.full(n_ref, 75.0),
             mu_log=mu_log, sigma_log=sigma_log)

    # 自检
    d1 = load_dataset(DATA_DIR / "alpha75_ref.npz")
    assert d1["x_std"].shape == (n_ref, 1000), "alpha75_ref shape 错误"
    print("alpha75_ref 自检 PASS")


if __name__ == "__main__":
    data = load_dataset(DATA_DIR / "corazza_train.npz")
    result = binned_stats(data)
    plot_binned(result)
    build_alpha75_ref()
    print("\nDay 5 脚本全部完成")
