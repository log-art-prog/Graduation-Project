"""Day 9：泛化指标库 — KL / JS / Spearman。

数学规格：docs/notes/09_评价指标定义.md 第 5 节。
接口冻结：同阶段二统一规则。
"""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def kl_js_divergence(r_ref, r_gen, n_bins: int = 100,
                     eps: float = 1e-10) -> dict:
    """KL 散度与 JS 散度（直方图估计）。

    KL(p_gen ‖ p_ref) = Σ p_gen log(p_gen / p_ref)
    JS = ½ KL(p ‖ m) + ½ KL(q ‖ m)，其中 m = (p+q)/2

    对直方图做 +ε 平滑防止零概率 bin 导致 log(0)。
    bins 边界由参考样本 [min, max] 定，与 pdf_l1_distance 一致。

    Args:
        r_ref:   参考样本
        r_gen:   生成样本
        n_bins:  直方图 bin 数
        eps:     零概率平滑常数

    Returns:
        {"value": {"kl": float, "js": float},
         "detail": {"n_bins", "eps", "p_ref", "p_gen"}}
    """
    r_ref = np.asarray(r_ref).flatten()
    r_gen = np.asarray(r_gen).flatten()

    lo, hi = float(r_ref.min()), float(r_ref.max())
    bins = np.linspace(lo, hi, n_bins + 1)

    p_ref, _ = np.histogram(r_ref, bins=bins, density=True)
    p_gen, _ = np.histogram(r_gen, bins=bins, density=True)

    # 加 ε 平滑并重新归一化（保持和为 1）
    p_ref = p_ref + eps
    p_gen = p_gen + eps
    p_ref /= p_ref.sum()
    p_gen /= p_gen.sum()

    m = 0.5 * (p_ref + p_gen)

    # KL 注意：0·log(0) 在极限下为 0，但这里已加 eps，安全
    kl_ref_gen = float(np.sum(p_gen * np.log(p_gen / p_ref)))
    kl_ref_m = float(np.sum(p_ref * np.log(p_ref / m)))
    kl_gen_m = float(np.sum(p_gen * np.log(p_gen / m)))
    js = 0.5 * (kl_ref_m + kl_gen_m)

    return {
        "value": {"kl": kl_ref_gen, "js": js},
        "detail": {
            "n_bins": n_bins,
            "eps": eps,
            "p_ref": p_ref.tolist(),
            "p_gen": p_gen.tolist(),
        },
    }


def spearman_correlation(alpha_deg: np.ndarray,
                         mean_envelope: np.ndarray) -> dict:
    """Spearman 秩相关：检验"仰角 → 包络均值"的单调映射是否学到。

    学长报告值 0.9980。输入为同一批样本的仰角和对应包络均值。

    Args:
        alpha_deg:      (N,) 仰角序列（度）
        mean_envelope:  (N,) 对应包络均值序列

    Returns:
        {"value": float,   # Spearman ρ
         "detail": {"pvalue": float}}
    """
    alpha_deg = np.asarray(alpha_deg).flatten()
    mean_envelope = np.asarray(mean_envelope).flatten()

    rho, pval = spearmanr(alpha_deg, mean_envelope)
    return {
        "value": float(rho),
        "detail": {"pvalue": float(pval)},
    }


def evaluate_generalization(r_ref, r_gen,
                            alpha_deg: np.ndarray | None = None,
                            mean_envelope: np.ndarray | None = None,
                            n_bins: int = 100) -> dict:
        """一键计算泛化指标。

        KL/JS 必算；Spearman 在提供 alpha_deg 和 mean_envelope 时计算。

        Returns:
            {"kl_js": {...}, "spearman": {...} | None}
        """
        out = {
            "kl_js": kl_js_divergence(r_ref, r_gen, n_bins),
        }
        if alpha_deg is not None and mean_envelope is not None:
            out["spearman"] = spearman_correlation(alpha_deg, mean_envelope)
        else:
            out["spearman"] = None
        return out
