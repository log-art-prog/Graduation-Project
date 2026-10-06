"""Day 9：全库双向自测（阶段二收官验收）。

运行：python scripts/validate_metrics.py

覆盖四类指标（L1 分布 / L2 时序 / L3 BER / 泛化），三组检验：
  A. 阴性对照：同分布对半切，全部指标 ≈ 0
  B. 阳性对照：加偏移（L1↑）、打乱时序（L2↑）、换信道衰落强度（L3↑）
  C. 泛化与复现：Spearman（仰角→均值单调性）、同种子逐位一致

对应 PHASE2_TASKS 的 DoD："双向自测脚本一键通过，输出自测报告"。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from generative.data.corazza import generate_corazza_sequence
from generative.data.dataset import load_dataset
from generative.data.transforms import inverse_standardize
from generative.metrics.distribution import evaluate_distribution
from generative.metrics.temporal import evaluate_temporal
from generative.metrics.ber import ber_rmse, DEFAULT_EBN0_DB
from generative.metrics.generalization import (
    kl_js_divergence, spearman_correlation,
)
from generative.utils.seed import make_rng

FS, FD, TAU_SHADOW, LENGTH = 1000.0, 50.0, 0.3, 1000
N_BITS = 50_000  # 汇总自测降比特数提速（day8 专项用 10 万）


def _gen_seqs(alpha: float, n: int, rng) -> np.ndarray:
    out = np.empty((n, LENGTH), dtype=np.float64)
    for i in range(n):
        out[i] = generate_corazza_sequence(
            alpha, length=LENGTH, fs=FS, fd=FD,
            tau_shadow=TAU_SHADOW, rng=rng)
    return out


def _check(name: str, value: float, op: str, th: float) -> bool:
    ok = (value < th) if op == "<" else (value > th)
    print(f"    {name:28s} = {v_fmt(value):>12s}  ({op}{th})  [{'PASS' if ok else 'FAIL'}]")
    return ok


def v_fmt(v: float) -> str:
    return f"{v:.4e}" if (abs(v) < 0.01 and v != 0) else f"{v:.4f}"


def main() -> bool:
    print("=" * 64)
    print("Day 9 全库双向自测")
    print("=" * 64)

    # 数据集：直接用阶段一冻结的 corazza_train.npz（10000 条混合仰角），
    # 逆标准化回物理包络。用全量样本的理由：Day 7 专项（5e6 点/半）LCR/AFD
    # 对半切噪声 5.7%，若只用 2e5 点/半，统计噪声放大 √25≈5 倍（~27%），
    # 会超过 15% 关卡——LCR/AFD 的噪声地板由样本量决定，必须用大样本。
    d_npz = load_dataset(ROOT / "datasets" / "corazza_train.npz")
    x_all = inverse_standardize(d_npz["x_std"].astype(np.float64),
                                float(d_npz["mu_log"]),
                                float(d_npz["sigma_log"]),
                                float(d_npz["eps"]))
    rng_split = make_rng(42)
    idx = rng_split.permutation(x_all.shape[0])
    half = len(idx) // 2
    half1, half2 = x_all[idx[:half]], x_all[idx[half:]]  # 各 5000 条
    print(f"  参考数据：corazza_train.npz 对半切（各 {half} 条 × {x_all.shape[1]} 点）")

    results: list[bool] = []

    # ------------------------------------------------ A 阴性对照
    print("\n[A] 阴性对照：同分布对半切（全部指标应 ≈ 0）")
    d = evaluate_distribution(half1, half2)
    # 阈值说明：单仰角分布比 Day6 的全仰角混合更"尖"，同样本量下
    # pdf_l1 噪声更大（Day6 混合仰角 0.0032，此处量级 ~0.01~0.03）
    results.append(_check("L1 pdf_l1", d["pdf_l1"]["value"], "<", 0.04))
    results.append(_check("L1 ks", d["ks"]["value"], "<", 0.02))
    results.append(_check("L1 wasserstein", d["wasserstein"]["value"], "<", 0.01))
    for m, e in d["moments"]["value"].items():
        # 高阶矩（偏度/峰度）统计噪声天然大（Day6 踩坑记录），放宽到 15%
        th = 0.05 if m in ("mean", "std") else 0.15
        results.append(_check(f"L1 moment[{m}]", e, "<", th))

    t = evaluate_temporal(half1, half2, fs=FS)
    results.append(_check("L2 acf_rmse", t["acf"]["value"], "<", 0.02))
    results.append(_check("L2 lcr_rel_rmse", t["lcr"]["value"], "<", 0.15))
    results.append(_check("L2 afd_rel_rmse", t["afd"]["value"], "<", 0.15))

    # BER 链路是逐符号蒙特卡洛，5000 条全量太慢，抽 300 条/半（统计足够）
    b = ber_rmse(half1[:300], half2[:300], DEFAULT_EBN0_DB,
                 n_bits=N_BITS, fs=FS, rng=make_rng(3))
    results.append(_check("L3 ber_rmse(log10)", b["value"], "<", 0.10))

    g = kl_js_divergence(half1, half2)
    results.append(_check("泛化 KL", g["value"]["kl"], "<", 0.01))
    results.append(_check("泛化 JS", g["value"]["js"], "<", 0.005))

    # ------------------------------------------------ B 阳性对照
    print("\n[B] 阳性对照：注入已知失真（对应指标应显著变大）")
    # B1 幅度偏移 → L1 与泛化变大
    shifted = half2 + 0.5
    d2 = evaluate_distribution(half1, shifted)
    g2 = kl_js_divergence(half1, shifted)
    results.append(_check("B1 偏移: pdf_l1", d2["pdf_l1"]["value"], ">", 0.5))
    results.append(_check("B1 偏移: wasserstein", d2["wasserstein"]["value"], ">", 0.4))
    # KL 的具体数值受 eps 平滑影响，只要求"远大于基线"（基线 2.4e-3，失真后 ~0.8）
    results.append(_check("B1 偏移: KL", g2["value"]["kl"], ">", 0.3))

    # B2 打乱时序 → L2 变大而 L1 几乎不变（L2 存在的意义）
    rng_sh = make_rng(11)
    shuffled = half2.copy()
    for i in range(len(shuffled)):
        rng_sh.shuffle(shuffled[i])
    d3 = evaluate_distribution(half1, shuffled)
    t3 = evaluate_temporal(half1, shuffled, fs=FS)
    results.append(_check("B2 打乱: ks 不变", d3["ks"]["value"], "<", 0.02))
    results.append(_check("B2 打乱: acf_rmse 变大", t3["acf"]["value"], ">", 0.1))
    results.append(_check("B2 打乱: lcr 变大", t3["lcr"]["value"], ">", 1.0))

    # B3 衰落强度差异（45° vs 20°）→ L3 BER 显著变大
    r_45 = _gen_seqs(45.0, 100, make_rng(76))
    r_lowk = _gen_seqs(20.0, 100, make_rng(77))
    b3 = ber_rmse(r_45, r_lowk, DEFAULT_EBN0_DB, n_bits=N_BITS, fs=FS, rng=make_rng(4))
    results.append(_check("B3 45°vs20°: ber_rmse", b3["value"], ">", 0.5))

    # ------------------------------------------------ C 泛化与复现
    print("\n[C] 泛化与可复现性")
    # C1 Spearman：Corazza 数据层仰角→包络均值应 ≈ 1（阶段一已验证的物理单调性）
    alphas = np.arange(20.0, 81.0, 10.0)
    means = []
    rng_c = make_rng(555)
    for a in alphas:
        seqs = _gen_seqs(a, 30, rng_c)
        means.append(float(seqs.mean()))
    sp = spearman_correlation(alphas, np.array(means))
    results.append(_check("C1 Spearman(仰角,均值)", sp["value"], ">", 0.99))

    # C2 同种子逐位一致（抽查 L1+L2 汇总入口）
    e1 = evaluate_temporal(half1, half2, fs=FS)
    e2 = evaluate_temporal(half1, half2, fs=FS)
    same = (e1["acf"]["value"] == e2["acf"]["value"]
            and e1["lcr"]["value"] == e2["lcr"]["value"])
    print(f"    {'C2 时序指标确定性':28s} = {str(same):>12s}  [{'PASS' if same else 'FAIL'}]")
    results.append(same)

    # ------------------------------------------------ 汇总
    print("\n" + "=" * 64)
    n_pass = sum(results)
    print(f"总计：{n_pass}/{len(results)} 项通过",
          "— 全部 PASS ✓" if n_pass == len(results) else "— 存在 FAIL ✗")
    return n_pass == len(results)


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
