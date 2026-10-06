"""阶段三验收：CGAN 生成质量评估（7 个测试仰角 × 三层指标 + 泛化）。

运行：python scripts/phase3_eval_cgan.py [--ckpt ...] [--n 500]

流程：
  1. 加载 CGAN（EMA 权重），对 20°~80° 步长 10° 各生成 n 条包络
  2. 每个仰角生成同数量的 Corazza 参考（首次生成后缓存 results/phase3/）
  3. 调用阶段二冻结的指标库逐项打分，汇总成表
  4. 验收（IMPROVEMENT_PLAN 阶段3）：KS 全部 < 0.07 量级；
     仰角-均值 Spearman ≈ 0.99 以上；训练曲线收敛（见 train_curves.png）

产物：results/phase3/cgan_metrics.csv + figs/phase3_cgan_pdf_cdf_45deg.png
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from generative.data.corazza import generate_corazza_sequence
from generative.metrics.distribution import evaluate_distribution
from generative.metrics.temporal import evaluate_temporal
from generative.metrics.ber import ber_rmse, DEFAULT_EBN0_DB
from generative.metrics.generalization import (
    kl_js_divergence, spearman_correlation,
)
from generative.sample import load_generator, generate_envelopes
from generative.utils.seed import make_rng

FS, FD, TAU_SHADOW, LENGTH = 1000.0, 50.0, 0.3, 1000
TEST_ALPHAS = [20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]
N_BITS_EVAL = 20_000  # 评估用 2 万比特/条/点（floor 5e-5，两模型同口径）
CACHE_DIR = ROOT / "results" / "phase3"

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def get_reference(alpha: float, n: int) -> np.ndarray:
    """生成（或读缓存的）指定仰角 Corazza 参考序列，固定种子保证可复现。"""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"ref_alpha{int(alpha)}_n{n}.npz"
    if cache.exists():
        return np.load(cache)["r"]
    rng = make_rng(9000 + int(alpha))
    r = np.empty((n, LENGTH), dtype=np.float64)
    for i in range(n):
        r[i] = generate_corazza_sequence(alpha, length=LENGTH, fs=FS, fd=FD,
                                         tau_shadow=TAU_SHADOW, rng=rng)
    np.savez(cache, r=r)
    return r


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str,
                   default=str(ROOT / "checkpoints" / "cgan" / "cgan_final.pt"))
    p.add_argument("--n", type=int, default=500, help="每仰角生成/参考条数")
    args = p.parse_args()

    G, meta = load_generator(args.ckpt)
    rows = []
    gen_means = []

    for alpha in TEST_ALPHAS:
        t0 = time.time()
        r_gen = generate_envelopes(G, meta, alpha, args.n,
                                   seed=1000 + int(alpha))
        r_ref = get_reference(alpha, args.n)

        d = evaluate_distribution(r_ref, r_gen)
        t = evaluate_temporal(r_ref, r_gen, fs=FS)
        kj = kl_js_divergence(r_ref, r_gen)
        b = ber_rmse(r_ref, r_gen, DEFAULT_EBN0_DB,
                     n_bits=N_BITS_EVAL, fs=FS, rng=make_rng(5))

        row = {
            "alpha": alpha,
            "pdf_l1": d["pdf_l1"]["value"],
            "ks": d["ks"]["value"],
            "wasserstein": d["wasserstein"]["value"],
            "mean_rel": d["moments"]["value"]["mean"],
            "std_rel": d["moments"]["value"]["std"],
            "acf_rmse": t["acf"]["value"],
            "lcr_rel": t["lcr"]["value"],
            "afd_rel": t["afd"]["value"],
            "kl": kj["value"]["kl"],
            "js": kj["value"]["js"],
            "ber_rmse": b["value"],
        }
        rows.append(row)
        gen_means.append(float(r_gen.mean()))
        print(f"α={alpha:4.0f}°  KS={row['ks']:.4f}  L1={row['pdf_l1']:.4f}  "
              f"W={row['wasserstein']:.4f}  ACF={row['acf_rmse']:.4f}  "
              f"LCR={row['lcr_rel']:.3f}  AFD={row['afd_rel']:.3f}  "
              f"BER={row['ber_rmse']:.4f}  ({time.time()-t0:.0f}s)")

    # ---- 汇总表 ----
    keys = ["pdf_l1", "ks", "wasserstein", "mean_rel", "std_rel",
            "acf_rmse", "lcr_rel", "afd_rel", "kl", "js", "ber_rmse"]
    means = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    print("\n===== 均值汇总 =====")
    for k in keys:
        print(f"  {k:12s} = {means[k]:.4f}")

    sp = spearman_correlation(np.array(TEST_ALPHAS), np.array(gen_means))
    print(f"\n仰角-生成均值 Spearman = {sp['value']:.4f}（学长 0.9980）")

    # ---- 验收 ----
    ks_max = max(r["ks"] for r in rows)
    ok_ks = ks_max < 0.07
    ok_sp = sp["value"] > 0.99
    print(f"\n验收：KS 全部<0.07（最大 {ks_max:.4f}）→ {'PASS' if ok_ks else 'FAIL'}；"
          f"Spearman>0.99（{sp['value']:.4f}）→ {'PASS' if ok_sp else 'FAIL'}")

    # ---- 存 CSV ----
    import csv
    with open(CACHE_DIR / "cgan_metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["alpha"] + keys)
        w.writeheader()
        w.writerows(rows)

    # ---- 45° PDF/CDF 对比图（对标论文图 3-6）----
    alpha45 = 45.0
    r_gen45 = generate_envelopes(G, meta, alpha45, args.n, seed=1045)
    r_ref45 = get_reference(alpha45, args.n)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    lo = min(r_ref45.min(), r_gen45.min())
    hi = max(r_ref45.max(), r_gen45.max())
    bins = np.linspace(lo, hi, 101)
    axes[0].hist(r_ref45.flatten(), bins=bins, density=True, alpha=0.6,
                 label="Corazza 参考")
    axes[0].hist(r_gen45.flatten(), bins=bins, density=True, alpha=0.6,
                 label="CGAN 生成")
    axes[0].set_title("45°仰角 包络 PDF")
    axes[0].legend()
    for r, lbl, ls in [(r_ref45, "Corazza 参考", "-"), (r_gen45, "CGAN 生成", "--")]:
        xs = np.sort(r.flatten())
        cdf = np.arange(1, len(xs) + 1) / len(xs)
        axes[1].plot(xs, cdf, ls, lw=1.2, label=lbl)
    axes[1].set_title("45°仰角 包络 CDF")
    axes[1].legend()
    for ax in axes:
        ax.grid(alpha=0.3)
    plt.tight_layout()
    fig_dir = ROOT / "figs"
    fig_dir.mkdir(exist_ok=True)
    plt.savefig(fig_dir / "phase3_cgan_pdf_cdf_45deg.png", dpi=120)
    plt.close()
    print(f"\n图与表已存：{CACHE_DIR / 'cgan_metrics.csv'}、"
          f"{fig_dir / 'phase3_cgan_pdf_cdf_45deg.png'}")


if __name__ == "__main__":
    main()
