"""Day 6：L1 静态分布指标库（阶段二 Day 6）。

本模块实现比较两组样本"某一瞬间幅值分布"的全部指标：
  - pdf_l1_distance    归一化直方图逐 bin 差绝对值积分
  - ks_distance        CDF 的 KS 距离（scipy.stats.ks_2samp）
  - wasserstein_distance  Wasserstein-1 距离
  - moment_stats       mean/std/skewness/kurtosis 相对误差
  - evaluate_distribution  一键调用以上四个，返回汇总 dict

接口冻结规则（阶段二全程遵守）：
  - 输入 r_ref / r_gen 可以是任意形状 ndarray，函数内部 .flatten()
  - 返回 dict 必含 "value" 键（主指标值或子指标 dict）
  - bins / 门限显式传入不硬编码，参考与生成用相同参数
"""
from __future__ import annotations

import numpy as np
from scipy.stats import ks_2samp, wasserstein_distance as _wasserstein
from scipy.stats import skew, kurtosis


# =====================================================================
# 6.2 PDF L1 距离
# =====================================================================
def pdf_l1_distance(r_ref, r_gen, n_bins: int = 100) -> dict:
    """PDF L1 距离 = ∫|p_gen(r) - p_ref(r)| dr。

    用归一化直方图（density=True）近似 PDF，bins 边界以参考样本的
    [min, max] 定，生成样本超出范围的部分由 np.histogram 自动归到
    边缘 bin（这是希望的行为：超出参考范围的部分会被算作分布差）。

    Args:
        r_ref:   参考样本，任意形状 ndarray
        r_gen:   生成样本，任意形状 ndarray
        n_bins:  直方图 bin 数，默认 100

    Returns:
        {"value": float, "detail": {"n_bins": int, "bin_width": float}}
    """
    r_ref = np.asarray(r_ref).flatten()
    r_gen = np.asarray(r_gen).flatten()

    # 边界用参考样本——参考是"标准答案"，生成样本被映射到参考坐标系评估
    lo, hi = float(r_ref.min()), float(r_ref.max())
    bins = np.linspace(lo, hi, n_bins + 1)

    # density=True 让直方图面积为 1，得到的是概率密度而非计数
    p_ref, _ = np.histogram(r_ref, bins=bins, density=True)
    p_gen, _ = np.histogram(r_gen, bins=bins, density=True)

    bin_width = float(bins[1] - bins[0])
    l1 = float(np.sum(np.abs(p_gen - p_ref)) * bin_width)

    return {"value": l1, "detail": {"n_bins": n_bins, "bin_width": bin_width}}


# =====================================================================
# 6.3 KS 距离
# =====================================================================
def ks_distance(r_ref, r_gen) -> dict:
    """CDF 的 KS 距离 = max|F_gen(x) - F_ref(x)|。

    直接调用 scipy.stats.ks_2samp 的 statistic 字段。KS 对样本量敏感：
    统计噪声 ~ 1/√n，大样本下同分布对半切 KS 通常 < 0.01。

    Args:
        r_ref: 参考样本
        r_gen: 生成样本

    Returns:
        {"value": float, "detail": {}}
    """
    r_ref = np.asarray(r_ref).flatten()
    r_gen = np.asarray(r_gen).flatten()
    stat = float(ks_2samp(r_ref, r_gen).statistic)
    return {"value": stat, "detail": {}}


# =====================================================================
# 6.4 Wasserstein-1 距离
# =====================================================================
def wasserstein_distance(r_ref, r_gen) -> dict:
    """Wasserstein-1 距离（推土机距离）。

    物理含义：把 ref 分布"推"成 gen 分布所需的最小"搬运量"。
    比 KS 更敏感于尾部偏移，比 L1 更敏感于整体平移。

    Args:
        r_ref: 参考样本
        r_gen: 生成样本

    Returns:
        {"value": float, "detail": {}}
    """
    r_ref = np.asarray(r_ref).flatten()
    r_gen = np.asarray(r_gen).flatten()
    w = float(_wasserstein(r_ref, r_gen))
    return {"value": w, "detail": {}}


# =====================================================================
# 6.5 矩统计
# =====================================================================
def moment_stats(r_ref, r_gen) -> dict:
    """四阶矩统计：mean / std / skewness / kurtosis，报告相对误差。

    kurtosis 用 scipy.stats.kurtosis 的默认值（Fisher 定义，正态分布
    返回 0，不是 Pearson 的 3）。skewness 同理，正态分布返回 0。

    Args:
        r_ref: 参考样本
        r_gen: 生成样本

    Returns:
        {
          "value": {  # 相对误差（生成 vs 参考），键名与矩名一致
            "mean":     float,   # |gen-mean − ref-mean| / |ref-mean|
            "std":      float,
            "skewness": float,
            "kurtosis": float,
          },
          "detail": {"ref": {...}, "gen": {...}}  # 两组原始四矩值
        }
    """
    r_ref = np.asarray(r_ref).flatten()
    r_gen = np.asarray(r_gen).flatten()

    def _four_moments(x):
        return {
            "mean": float(x.mean()),
            "std": float(x.std()),
            "skewness": float(skew(x)),
            "kurtosis": float(kurtosis(x)),
        }

    ref = _four_moments(r_ref)
    gen = _four_moments(r_gen)

    # 分母加 1e-12 防除零（mean 可能接近 0，skewness/kurtosis 对称分布也可能近 0）
    rel_err = {k: abs(gen[k] - ref[k]) / (abs(ref[k]) + 1e-12) for k in ref}

    return {"value": rel_err, "detail": {"ref": ref, "gen": gen}}


# =====================================================================
# 6.6 一键汇总
# =====================================================================
def evaluate_distribution(r_ref, r_gen, n_bins: int = 100) -> dict:
    """一键调用以上四个指标，返回汇总 dict。

    Args:
        r_ref:   参考样本
        r_gen:   生成样本
        n_bins:  PDF L1 用的 bin 数

    Returns:
        {
          "pdf_l1":      pdf_l1_distance(...),
          "ks":          ks_distance(...),
          "wasserstein": wasserstein_distance(...),
          "moments":     moment_stats(...),
        }
    """
    return {
        "pdf_l1": pdf_l1_distance(r_ref, r_gen, n_bins),
        "ks": ks_distance(r_ref, r_gen),
        "wasserstein": wasserstein_distance(r_ref, r_gen),
        "moments": moment_stats(r_ref, r_gen),
    }
