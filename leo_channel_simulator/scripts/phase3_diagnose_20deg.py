"""阶段三诊断：20° 仰角验收失败原因定位。

运行：python scripts/phase3_diagnose_20deg.py

背景：phase3_eval_cgan.py 中 30°~80° 六个仰角 KS 全部 <0.055 达标，
唯独 20° KS=0.2263（其他仰角的 5 倍），Spearman 0.9643 < 0.99。

诊断设计（三个假设）：
  H1 边界效应：20° → c=(20-50)/30=-1 压在条件区间边界，
     若 21°/22°/25° 也差 → 是边界区域问题；若只有 20° 差 → 单点问题
  H2 整体偏移：生成均值/方差系统性偏离（log 域 mean/std 对比）
  H3 尾部失配：低仰角深衰落尾部更重，分布形状（skew/kurt/分位数）失配

产物：results/phase3/diag_20deg.csv
      figs/phase3_diag_20deg_dist.png   （20° 线性 PDF / log10 PDF / CDF）
      figs/phase3_diag_20deg_meancurve.png（均值-仰角曲线：单调性与 Spearman 归因）
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from generative.data.corazza import generate_corazza_sequence
from generative.metrics.distribution import evaluate_distribution
from generative.sample import load_generator, generate_envelopes
from generative.utils.seed import make_rng

FS, FD, TAU_SHADOW, LENGTH = 1000.0, 50.0, 0.3, 1000
N = 500
CACHE_DIR = ROOT / "results" / "phase3"
FIG_DIR = ROOT / "figs"
SCAN_ALPHAS = [20.0, 21.0, 22.0, 25.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]
QUANTILES = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def get_reference(alpha: float, n: int) -> np.ndarray:
    """与 phase3_eval_cgan.py 完全一致的参考生成协议（含缓存）。"""
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


def log_moments(r: np.ndarray) -> dict:
    """log 域矩（训练在标准化 log 域进行，矩对比应在同域做）。"""
    lr = np.log(r + 1e-6).flatten()
    return {"mean": float(lr.mean()), "std": float(lr.std()),
            "skew": float(stats.skew(lr)), "kurt": float(stats.kurtosis(lr))}


def main() -> None:
    G, meta = load_generator(ROOT / "checkpoints" / "cgan" / "cgan_final.pt")
    rows = []
    ref_means, gen_means = [], []

    for alpha in SCAN_ALPHAS:
        r_gen = generate_envelopes(G, meta, alpha, N, seed=1000 + int(alpha))
        r_ref = get_reference(alpha, N)

        d = evaluate_distribution(r_ref, r_gen)
        m_ref, m_gen = log_moments(r_ref), log_moments(r_gen)
        q_ref = np.quantile(r_ref, QUANTILES)
        q_gen = np.quantile(r_gen, QUANTILES)

        rows.append({
            "alpha": alpha,
            "ks": d["ks"]["value"], "pdf_l1": d["pdf_l1"]["value"],
            "wasserstein": d["wasserstein"]["value"],
            "mean_rel": d["moments"]["value"]["mean"],
            "std_rel": d["moments"]["value"]["std"],
            "logmean_ref": m_ref["mean"], "logmean_gen": m_gen["mean"],
            "logstd_ref": m_ref["std"], "logstd_gen": m_gen["std"],
            "skew_ref": m_ref["skew"], "skew_gen": m_gen["skew"],
            "kurt_ref": m_ref["kurt"], "kurt_gen": m_gen["kurt"],
            **{f"q{int(q*100)}_ref": qr for q, qr in zip(QUANTILES, q_ref)},
            **{f"q{int(q*100)}_gen": qg for q, qg in zip(QUANTILES, q_gen)},
        })
        ref_means.append(float(r_ref.mean()))
        gen_means.append(float(r_gen.mean()))
        print(f"α={alpha:4.0f}°  KS={d['ks']['value']:.4f}  "
              f"L1={d['pdf_l1']['value']:.4f}  "
              f"log均值 ref/gen = {m_ref['mean']:+.4f}/{m_gen['mean']:+.4f}  "
              f"logstd ref/gen = {m_ref['std']:.4f}/{m_gen['std']:.4f}")

    # ---- 汇总表 ----
    keys = list(rows[0].keys())
    with open(CACHE_DIR / "diag_20deg.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    # ---- 图 A：20° 分布形状对比（线性 PDF / log10 PDF / CDF）----
    r_gen20 = generate_envelopes(G, meta, 20.0, N, seed=1020)
    r_ref20 = get_reference(20.0, N)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    lo, hi = min(r_ref20.min(), r_gen20.min()), max(r_ref20.max(), r_gen20.max())
    bins = np.linspace(lo, hi, 101)
    axes[0].hist(r_ref20.flatten(), bins=bins, density=True, alpha=0.6, label="Corazza 参考")
    axes[0].hist(r_gen20.flatten(), bins=bins, density=True, alpha=0.6, label="CGAN 生成")
    axes[0].set_title("20° 包络 PDF（线性域）")
    axes[0].legend()
    llo, lhi = np.log10(lo + 1e-6), np.log10(hi + 1e-6)
    lbins = np.linspace(llo, lhi, 101)
    axes[1].hist(np.log10(r_ref20 + 1e-6).flatten(), bins=lbins, density=True,
                 alpha=0.6, label="Corazza 参考")
    axes[1].hist(np.log10(r_gen20 + 1e-6).flatten(), bins=lbins, density=True,
                 alpha=0.6, label="CGAN 生成")
    axes[1].set_title("20° log10 包络 PDF")
    axes[1].legend()
    for r, lbl, ls in [(r_ref20, "Corazza 参考", "-"), (r_gen20, "CGAN 生成", "--")]:
        xs = np.sort(r.flatten())
        axes[2].plot(xs, np.arange(1, len(xs) + 1) / len(xs), ls, lw=1.2, label=lbl)
    axes[2].set_title("20° 包络 CDF")
    axes[2].legend()
    for ax in axes:
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "phase3_diag_20deg_dist.png", dpi=120)
    plt.close()

    # ---- 图 B：均值-仰角曲线（单调性 / Spearman 归因）----
    al = [r["alpha"] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(al, ref_means, "o-", lw=1.5, label="Corazza 参考均值")
    ax.plot(al, gen_means, "s--", lw=1.5, label="CGAN 生成均值")
    ax.set_xlabel("仰角 (deg)")
    ax.set_ylabel("包络均值")
    ax.set_title("均值-仰角单调性检查（Spearman 归因）")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "phase3_diag_20deg_meancurve.png", dpi=120)
    plt.close()

    # ---- 自动诊断结论 ----
    r20 = rows[0]
    near = rows[1:4]  # 21/22/25
    ks_near = max(r["ks"] for r in near)
    print("\n===== 诊断结论 =====")
    print(f"[H1 边界效应] 20° KS={r20['ks']:.4f}；21/22/25° KS 最大={ks_near:.4f}"
          f" → {'边界区域整体劣化' if ks_near > 0.10 else '仅 20° 单点异常'}")
    print(f"[H2 整体偏移] 20° log均值偏差={r20['logmean_gen']-r20['logmean_ref']:+.4f}  "
          f"logstd偏差={r20['logstd_gen']-r20['logstd_ref']:+.4f}  "
          f"(对比 45°: log均值偏差={rows[5]['logmean_gen']-rows[5]['logmean_ref']:+.4f})")
    print(f"[H3 尾部失配] 20° skew ref/gen={r20['skew_ref']:+.3f}/{r20['skew_gen']:+.3f}  "
          f"kurt ref/gen={r20['kurt_ref']:+.3f}/{r20['kurt_gen']:+.3f}")
    print("20° 分位数 ref vs gen：")
    for q, qr, qg in zip(QUANTILES,
                         [r20[f"q{int(q*100)}_ref"] for q in QUANTILES],
                         [r20[f"q{int(q*100)}_gen"] for q in QUANTILES]):
        print(f"   q{int(q*100):02d}: {qr:.4f} vs {qg:.4f}  (相对偏差 {qg/qr-1:+.1%})")
    sp_x = np.array(al)
    sp_full = stats.spearmanr(sp_x, gen_means).statistic
    sp_no20 = stats.spearmanr(sp_x[1:], gen_means[1:]).statistic
    print(f"Spearman 全仰角={sp_full:.4f}；剔除 20° 后={sp_no20:.4f}")
    print(f"\n表已存 {CACHE_DIR / 'diag_20deg.csv'}；图已存 {FIG_DIR}")


if __name__ == "__main__":
    main()
