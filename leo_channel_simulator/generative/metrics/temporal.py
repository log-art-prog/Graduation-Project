"""Day 7：L2 时序特性指标库（阶段二 Day 7）。

本模块比较两组包络序列"随时间起伏的节奏"：
  - envelope_acf      包络归一化自相关（FFT 快速相关，多序列集成平均）
  - acf_rmse          前 n_lags 个滞后点上两条 ACF 的 RMSE
  - threshold_stats   多门限向下穿越统计：P(r<th) / LCR / AFD
  - lcr_distance      多门限 LCR 曲线的相对 RMSE
  - afd_distance      多门限 AFD 曲线的相对 RMSE
  - evaluate_temporal 一键汇总

与 distribution.py 的关键差异：
  - 输入形状 (n_samples, length) 或 (length,)，沿时间轴 axis=1 计算，
    【绝不跨序列边界 flatten】——拼接边界会制造假穿越；
  - 门限用归一化电平 ρ(dB)：r_th = 10^(dB/20) · r_rms，每条序列用
    自己的 RMS，聚焦"起伏形状"而非整体功率；
  - LCR/AFD 是多门限曲线，主指标用逐点相对误差的 RMSE。

数学规格见 docs/notes/09_评价指标定义.md 第 3 节（论文式 2-36~2-39）。
"""
from __future__ import annotations

import numpy as np


# 标准门限集（dB，幅度比），与学长论文设置对齐
DEFAULT_THRESHOLDS_DB: tuple[float, ...] = (-10.0, -5.0, 0.0, 5.0, 10.0)


def _as_2d(r) -> np.ndarray:
    """统一成 (n_samples, length) 的 float64 二维数组。

    1D 输入视为单条序列；不做跨序列展平。
    """
    r = np.asarray(r, dtype=np.float64)
    if r.ndim == 1:
        r = r[None, :]
    if r.ndim != 2:
        raise ValueError(f"时序指标只接受 1D/2D 输入，收到 ndim={r.ndim}")
    return r


# =====================================================================
# 7.1 包络 ACF
# =====================================================================
def envelope_acf(r, n_lags: int = 50) -> np.ndarray:
    """包络的归一化自相关（多序列集成平均）。

    对每条序列独立计算（notes/09 式 2-36 口径）：
                     Σ_t (r[t]−r̄)(r[t+τ]−r̄)
        R_rr(τ) = ─────────────────────────────
                       Σ_t (r[t]−r̄)²
    分母固定为零滞后能量，故 R(0)=1；FFT 实现线性（非循环）相关，
    再对所有序列等权平均。

    Args:
        r:       (n_samples, length) 或 (length,) 的包络序列
        n_lags:  返回的滞后点数（含滞后 0）

    Returns:
        acf: shape (n_lags,)，acf[0]=1
    """
    r = _as_2d(r)
    n_seq, length = r.shape
    if n_lags > length:
        raise ValueError(f"n_lags={n_lags} 不能超过序列长度 {length}")

    # 逐序列去均值（零均值化）
    x = r - r.mean(axis=1, keepdims=True)

    # FFT 线性相关：补零到 ≥ 2L-1 的 2 的幂，避免循环卷积
    n_fft = 1 << (2 * length - 1).bit_length()
    X = np.fft.fft(x, n_fft, axis=1)
    acf = np.fft.ifft(X * np.conj(X), axis=1).real[:, :length]
    acf /= acf[:, [0]]  # 每条用自己的零滞后能量归一化 → R(0)=1

    # 多序列等权集成（抑制单条 ACF 的统计抖动）
    return acf[:, :n_lags].mean(axis=0)


def acf_rmse(r_ref, r_gen, n_lags: int = 50, fs: float = 1000.0) -> dict:
    """前 n_lags 个滞后点上，gen 与 ref 集成 ACF 之差的 RMSE。

    Args:
        r_ref:   参考包络 (n_samples, length) 或 (length,)
        r_gen:   生成包络
        n_lags:  参与打分的滞后点数（含滞后 0；0 点恒等，实际区分度在 1:）
        fs:      采样率（Hz），仅用于 detail 里的物理时延轴

    Returns:
        {"value": float( RMSE ),
         "detail": {"n_lags", "tau_s", "acf_ref", "acf_gen"}}
    """
    acf_ref = envelope_acf(r_ref, n_lags)
    acf_gen = envelope_acf(r_gen, n_lags)
    rmse = float(np.sqrt(np.mean((acf_gen - acf_ref) ** 2)))
    return {
        "value": rmse,
        "detail": {
            "n_lags": n_lags,
            "tau_s": (np.arange(n_lags) / fs).tolist(),
            "acf_ref": acf_ref.tolist(),
            "acf_gen": acf_gen.tolist(),
        },
    }


# =====================================================================
# 7.2 多门限穿越统计（LCR / AFD 共用一次计数）
# =====================================================================
def threshold_stats(r,
                    thresholds_db=DEFAULT_THRESHOLDS_DB,
                    fs: float = 1000.0) -> dict:
    """逐门限统计 P(r<th)、LCR、AFD（多序列聚合）。

    门限（幅度域）：r_th = 10^(dB/20) · r_rms，r_rms 为每条序列自身 RMS。

    聚合口径（等长序列下等价于先逐序列再平均，物理含义更直接）：
      - LCR = 总向下穿越次数 / 总观测时长        （次/秒）
      - P   = 门限以下采样点总数 / 总采样点数
      - AFD = 门限以下总时长 / 总穿越次数 = P/fs / (LCR·...) （秒）
              直接用 below_count / (fs · crossings)，定义自洽
    向下穿越：r[t] > r_th 且 r[t+1] ≤ r_th。

    Args:
        r:             (n_samples, length) 或 (length,)
        thresholds_db: 门限集合（dB，幅度比 10^(dB/20)）
        fs:            采样率（Hz）

    Returns:
        {"thresholds_db": list[float], "rho_lin": (n_th,),
         "p_below": (n_th,), "lcr": (n_th,), "afd_s": (n_th,)}
        某门限全程无穿越时 AFD 记为 nan（detail 中保留原始计数）。
    """
    r = _as_2d(r)
    n_seq, length = r.shape
    db = np.asarray(tuple(thresholds_db), dtype=np.float64)
    rho_lin = 10.0 ** (db / 20.0)  # dB 幅度比 → 线性

    # 每条序列自己的 RMS：(n_seq,)，门限广播成 (n_th, n_seq)
    r_rms = np.sqrt(np.mean(r ** 2, axis=1))
    r_th = rho_lin[:, None] * r_rms[None, :]

    # --- 向下穿越：(n_th, n_seq, length-1) 布尔计数 ---
    above_before = r[None, :, :-1] > r_th[:, :, None]
    below_now = r[None, :, 1:] <= r_th[:, :, None]
    crossings = np.sum(above_before & below_now, axis=2)  # (n_th, n_seq)

    # --- 门限以下占比：(n_th, n_seq, length) ---
    n_below = np.sum(r[None] <= r_th[:, :, None], axis=2)  # (n_th, n_seq)

    total_cross = crossings.sum(axis=1).astype(np.float64)   # (n_th,)
    total_below = n_below.sum(axis=1).astype(np.float64)
    total_intervals = n_seq * (length - 1)
    total_samples = n_seq * length

    duration_s = total_intervals / fs
    lcr = total_cross / duration_s
    p_below = total_below / total_samples
    # AFD = 低于门限总时长 / 穿越次数；无穿越 → nan
    afd = np.divide(total_below, fs * total_cross,
                    out=np.full_like(total_below, np.nan),
                    where=total_cross > 0)

    return {
        "thresholds_db": db.tolist(),
        "rho_lin": rho_lin,
        "p_below": p_below,
        "lcr": lcr,
        "afd_s": afd,
        "_total_crossings": total_cross,  # 内部排查用（含 0 计数信息）
    }


def _relative_rmse(gen: np.ndarray, ref: np.ndarray) -> float:
    """逐点相对误差的 RMSE：sqrt(mean(((gen-ref)/ref)^2))。

    ref 各点预期均 > 0（LCR/AFD 物理量），仍加 1e-12 防数值意外。
    """
    rel = (gen - ref) / (np.abs(ref) + 1e-12)
    return float(np.sqrt(np.mean(rel ** 2)))


def lcr_distance(r_ref, r_gen,
                 thresholds_db=DEFAULT_THRESHOLDS_DB,
                 fs: float = 1000.0) -> dict:
    """多门限 LCR 曲线的相对 RMSE（主指标）。

    Returns:
        {"value": float,
         "detail": {"thresholds_db", "lcr_ref", "lcr_gen",
                    "per_threshold_rel_err"}}
    """
    s_ref = threshold_stats(r_ref, thresholds_db, fs)
    s_gen = threshold_stats(r_gen, thresholds_db, fs)
    per = np.abs(s_gen["lcr"] - s_ref["lcr"]) / (np.abs(s_ref["lcr"]) + 1e-12)
    return {
        "value": _relative_rmse(s_gen["lcr"], s_ref["lcr"]),
        "detail": {
            "thresholds_db": s_ref["thresholds_db"],
            "lcr_ref": s_ref["lcr"].tolist(),
            "lcr_gen": s_gen["lcr"].tolist(),
            "per_threshold_rel_err": per.tolist(),
        },
    }


def afd_distance(r_ref, r_gen,
                 thresholds_db=DEFAULT_THRESHOLDS_DB,
                 fs: float = 1000.0) -> dict:
    """多门限 AFD 曲线的相对 RMSE（主指标，单位：秒）。

    Returns:
        {"value": float,
         "detail": {"thresholds_db", "afd_ref_s", "afd_gen_s",
                    "per_threshold_rel_err"}}
    """
    s_ref = threshold_stats(r_ref, thresholds_db, fs)
    s_gen = threshold_stats(r_gen, thresholds_db, fs)
    per = np.abs(s_gen["afd_s"] - s_ref["afd_s"]) / (np.abs(s_ref["afd_s"]) + 1e-12)
    return {
        "value": _relative_rmse(s_gen["afd_s"], s_ref["afd_s"]),
        "detail": {
            "thresholds_db": s_ref["thresholds_db"],
            "afd_ref_s": s_ref["afd_s"].tolist(),
            "afd_gen_s": s_gen["afd_s"].tolist(),
            "per_threshold_rel_err": per.tolist(),
        },
    }


# =====================================================================
# 7.3 一键汇总
# =====================================================================
def evaluate_temporal(r_ref, r_gen,
                      fs: float = 1000.0,
                      thresholds_db=DEFAULT_THRESHOLDS_DB,
                      n_lags: int = 50) -> dict:
    """一键计算 ACF / LCR / AFD 三类时序指标。

    Args:
        r_ref:         参考包络 (n_samples, length) 或 (length,)
        r_gen:         生成包络
        fs:            采样率（Hz）
        thresholds_db: LCR/AFD 门限集（dB）
        n_lags:        ACF 打分滞后点数

    Returns:
        {"acf": acf_rmse(...), "lcr": lcr_distance(...),
         "afd": afd_distance(...)}
    """
    return {
        "acf": acf_rmse(r_ref, r_gen, n_lags=n_lags, fs=fs),
        "lcr": lcr_distance(r_ref, r_gen, thresholds_db=thresholds_db, fs=fs),
        "afd": afd_distance(r_ref, r_gen, thresholds_db=thresholds_db, fs=fs),
    }
