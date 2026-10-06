"""Day 8：L3 系统性能指标库 — QPSK BER 链路。

本模块把包络序列过一遍真实调制解调链路，计算 Eb/N0 扫描下的 BER 曲线，
并在 log10 域与参考曲线比较 RMSE。

数学规格：docs/notes/09_评价指标定义.md 第 4 节（论文式 2-40）。
接口冻结：同阶段二统一规则 (r_ref, r_gen, ...) → dict 含 "value"。
"""
from __future__ import annotations

import numpy as np
from scipy.special import erfc


#: 论文一致的 Eb/N0 扫描网格（dB）
DEFAULT_EBN0_DB = np.arange(0, 36, 2, dtype=np.float64)


def _qpsk_ber_awgn(ebn0_db: np.ndarray) -> np.ndarray:
    """理论 AWGN 下 QPSK 误比特率 P_b = ½ erfc(√γ_b)。

    对 QPSK，每个符号 2 比特，Eb/N0 = γ_b。
    """
    gamma = 10.0 ** (ebn0_db / 10.0)
    return 0.5 * erfc(np.sqrt(gamma))


def _channel_gain(r: np.ndarray) -> np.ndarray:
    """把包络 r 归一化为单位平均功率的信道增益 g = r / √(E[r²])。

    输入: (n_samples, length) 或 (length,)
    输出: 与输入同形状
    """
    r = np.asarray(r, dtype=np.float64)
    if r.ndim == 1:
        return r / np.sqrt(np.mean(r ** 2))
    # 2D: 逐序列归一化
    rms = np.sqrt(np.mean(r ** 2, axis=1, keepdims=True))
    return r / rms


def _run_ber_single(gain: np.ndarray,
                    ebn0_db: np.ndarray,
                    n_bits: int,
                    rng: np.random.Generator) -> np.ndarray:
    """对单条增益序列跑完整 QPSK 链路，返回各 Eb/N0 点的 BER。

    Args:
        gain:    1D 数组，长度 L，已归一化的信道增益 g[t]
        ebn0_db: Eb/N0 网格（dB）
        n_bits:  总比特数（会被向上取整到 2 的倍数）
        rng:     numpy Generator

    Returns:
        ber: shape (len(ebn0_db),)，各点误比特率
    """
    n_bits = (n_bits + 1) // 2 * 2  # 确保偶数（QPSK 每符号 2 比特）
    n_symbols = n_bits // 2

    # 1) 随机比特 → QPSK 符号 s ∈ {(±1±j)/√2}
    bits = rng.integers(0, 2, size=n_bits)
    i_bits = bits[0::2]
    q_bits = bits[1::2]
    s = (2 * i_bits - 1 + 1j * (2 * q_bits - 1)) / np.sqrt(2)

    # 2) 信道作用：y = s * g + n
    #    增益序列循环重复覆盖所有符号
    L = len(gain)
    g = np.tile(gain, (n_symbols + L - 1) // L)[:n_symbols]

    # 噪声方差：复基带，σ² = 1 / (2 · 10^(EbN0/10))
    # 这里每个符号能量 Es = 1（因 s 模为 1），Eb = Es/2 = 0.5
    # 所以 SNR = Es / σ² = 1/σ²，而 Eb/N0 = Es/(2σ²)，故 σ² = 1/(2·γ)
    gamma_lin = 10.0 ** (ebn0_db / 10.0)
    sigma2 = 1.0 / (2.0 * gamma_lin)  # (n_ebn0,)

    ber = np.empty(len(ebn0_db), dtype=np.float64)
    for idx, s2 in enumerate(sigma2):
        n = rng.normal(0, np.sqrt(s2 / 2), size=n_symbols) + \
            1j * rng.normal(0, np.sqrt(s2 / 2), size=n_symbols)
        y = s * g + n

        # 3) 判决
        decoded_i = (y.real > 0).astype(int)
        decoded_q = (y.imag > 0).astype(int)

        err = np.sum(decoded_i != i_bits) + np.sum(decoded_q != q_bits)
        ber[idx] = err / n_bits

    return ber


def ber_curve(r,
              ebn0_db: np.ndarray = DEFAULT_EBN0_DB,
              n_bits: int = 200_000,
              fs: float = 1000.0,
              rng: np.random.Generator | None = None) -> dict:
    """计算多序列包络在 Eb/N0 网格上的 BER 曲线（均值）。

    Args:
        r:        (n_samples, length) 或 (length,) 包络序列
        ebn0_db:  Eb/N0 网格（dB），默认 0~34 步长 2
        n_bits:   每条序列的仿真比特数，默认 20 万
        fs:       采样率（Hz），仅用于 detail 标注
        rng:      随机数生成器；None 时新建

    Returns:
        {
          "ebn0_db": list[float],
          "ber":     ndarray,      # 各点平均 BER
          "ber_log10": ndarray,    # log10(BER)，已做平滑下限
          "n_bits":  int,
          "n_seqs":  int,
        }
    """
    r = np.asarray(r, dtype=np.float64)
    if r.ndim == 1:
        r = r[None, :]
    n_seqs, length = r.shape
    if rng is None:
        rng = np.random.default_rng(42)

    gains = _channel_gain(r)  # (n_seqs, length)

    all_ber = []
    for i in range(n_seqs):
        ber = _run_ber_single(gains[i], ebn0_db, n_bits, rng)
        all_ber.append(ber)
    ber_mean = np.mean(all_ber, axis=0)

    # 平滑下限：BER=0 时 log 无意义
    floor = 1.0 / n_bits
    ber_smooth = np.maximum(ber_mean, floor)
    ber_log10 = np.log10(ber_smooth)

    return {
        "ebn0_db": ebn0_db.tolist(),
        "ber": ber_mean,
        "ber_log10": ber_log10,
        "n_bits": n_bits,
        "n_seqs": n_seqs,
        "fs": fs,
    }


def ber_rmse(r_ref, r_gen,
             ebn0_db: np.ndarray = DEFAULT_EBN0_DB,
             n_bits: int = 200_000,
             fs: float = 1000.0,
             rng: np.random.Generator | None = None) -> dict:
    """log10 域 BER 曲线的 RMSE（论文式 3-4 口径）。

    Args:
        r_ref: 参考包络
        r_gen: 生成包络
        ebn0_db, n_bits, fs, rng: 同 ber_curve

    Returns:
        {"value": float,
         "detail": {"ebn0_db", "ber_log10_ref", "ber_log10_gen",
                    "rmse_per_point"}}
    """
    if rng is None:
        rng = np.random.default_rng(42)
    c_ref = ber_curve(r_ref, ebn0_db, n_bits, fs, rng)
    # 注意：对生成样本要用独立 rng 实例避免状态耦合；这里直接新建
    rng2 = np.random.default_rng(rng.integers(0, 2**31))
    c_gen = ber_curve(r_gen, ebn0_db, n_bits, fs, rng2)

    diff = c_gen["ber_log10"] - c_ref["ber_log10"]
    rmse = float(np.sqrt(np.mean(diff ** 2)))

    return {
        "value": rmse,
        "detail": {
            "ebn0_db": c_ref["ebn0_db"],
            "ber_log10_ref": c_ref["ber_log10"].tolist(),
            "ber_log10_gen": c_gen["ber_log10"].tolist(),
            "rmse_per_point": diff.tolist(),
        },
    }


def evaluate_ber(r_ref, r_gen,
                 ebn0_db: np.ndarray = DEFAULT_EBN0_DB,
                 n_bits: int = 200_000,
                 fs: float = 1000.0,
                 rng: np.random.Generator | None = None) -> dict:
    """一键计算 BER 指标（当前只有 ber_rmse，预留扩展）。"""
    return {
        "ber_rmse": ber_rmse(r_ref, r_gen, ebn0_db, n_bits, fs, rng),
    }
