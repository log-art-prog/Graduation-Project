"""Day 8 自测：L3 QPSK BER 指标库（generative/metrics/ber.py）。

运行：python scripts/day8_verify_ber.py

思考题：见 docs/notes/11_自测题库_Day1to9.md（Day 8 章）

四场景：
  1. 恒幅信道 g=1：BER 应与理论 ½erfc(√γ) 重合（低 SNR 段精确比较）
  2. 阴性对照：Corazza 45° 参考对半切，BER RMSE 应 ≈ 0
  3. 阳性对照：Rayleigh(K=0) vs 高仰角大 K Rice，BER RMSE 应显著大
  4. 纯函数：同种子两次逐位一致
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from generative.data.corazza import generate_corazza_sequence
from generative.metrics.ber import (
    DEFAULT_EBN0_DB, _qpsk_ber_awgn, ber_curve, ber_rmse,
)
from generative.utils.seed import make_rng

FS = 1000.0
FD = 50.0
TAU_SHADOW = 0.3
LENGTH = 1000
N_BITS = 100_000  # 自测用 10 万比特（floor=1e-5），兼顾速度与分辨率


def _gen_seqs(alpha: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """生成 n 条指定仰角的 Corazza 包络序列。"""
    out = np.empty((n, LENGTH), dtype=np.float64)
    for i in range(n):
        out[i] = generate_corazza_sequence(
            alpha, length=LENGTH, fs=FS, fd=FD,
            tau_shadow=TAU_SHADOW, rng=rng)
    return out


# ============================================================
# 场景 1：恒幅信道对理论曲线 ½erfc(√γ)
# ============================================================
def verify_constant_channel() -> bool:
    rng = make_rng(8)
    r_const = np.ones((1, LENGTH))

    curve = ber_curve(r_const, DEFAULT_EBN0_DB, n_bits=N_BITS, fs=FS, rng=rng)
    ber_sim = curve["ber"]
    ber_theory = _qpsk_ber_awgn(DEFAULT_EBN0_DB)

    # 只在"期望误码数 ≥ 100"的点上精确比较（统计上可信）
    # 期望误码数 = theory × n_bits ≥ 100 → theory ≥ 1e-3（n_bits=1e5）
    reliable = ber_theory >= 100.0 / N_BITS
    # 高 SNR 段：仿真值不允许离谱（≤ 50 × floor = 5e-4）
    unreliable = ~reliable
    floor = 1.0 / N_BITS

    log_diff = np.abs(np.log10(np.maximum(ber_sim[reliable], floor))
                      - np.log10(ber_theory[reliable]))
    max_log_err = float(log_diff.max())
    high_snr_ok = bool(np.all(ber_sim[unreliable] <= 50 * floor))

    n_rel = int(reliable.sum())
    p1 = max_log_err < 0.1   # log10 域误差 < 0.1（约 26% 线性误差以内）
    p2 = high_snr_ok
    status = "PASS" if (p1 and p2) else "FAIL"
    print(f"[8.1] 恒幅信道对理论  低SNR段({n_rel}点) log10最大误差={max_log_err:.4f} (<0.1)  "
          f"高SNR段未爆表={high_snr_ok}  [{status}]")
    print(f"       理论 BER@0/4/8dB = {ber_theory[0]:.2e}/{ber_theory[2]:.2e}/{ber_theory[4]:.2e}")
    print(f"       仿真 BER@0/4/8dB = {ber_sim[0]:.2e}/{ber_sim[2]:.2e}/{ber_sim[4]:.2e}")
    return p1 and p2


# ============================================================
# 场景 2：阴性对照 — 对半切 RMSE ≈ 0
# ============================================================
def verify_split_half() -> bool:
    rng = make_rng(45)
    ref = _gen_seqs(45.0, 100, rng)
    half1, half2 = ref[:50], ref[50:]

    res = ber_rmse(half1, half2, DEFAULT_EBN0_DB, n_bits=N_BITS, fs=FS, rng=make_rng(1))
    v = res["value"]
    passed = v < 0.10  # 对半切应远小于学长模型级误差 0.1536
    status = "PASS" if passed else "FAIL"
    print(f"[8.2] 对半切（阴性对照）  log10域 BER RMSE={v:.4f}  (<0.10)  [{status}]")
    return passed


# ============================================================
# 场景 3：阳性对照 — Rayleigh vs 大K Rice，RMSE 显著大
# ============================================================
def verify_positive_control() -> bool:
    rng = make_rng(7)
    # 20° 低仰角 K 很小（接近 Rayleigh），80° 高仰角 K 大
    r_lowk = _gen_seqs(20.0, 50, rng)
    r_highk = _gen_seqs(80.0, 50, rng)

    res = ber_rmse(r_lowk, r_highk, DEFAULT_EBN0_DB, n_bits=N_BITS, fs=FS, rng=make_rng(2))
    v = res["value"]
    passed = v > 0.5  # 深衰落信道 BER 恶化数量级，log 域差异必须显著
    status = "PASS" if passed else "FAIL"
    print(f"[8.3] 阳性对照(20° vs 80°)  log10域 BER RMSE={v:.4f}  (>0.5)  [{status}]")
    return passed


# ============================================================
# 场景 4：同种子逐位复现
# ============================================================
def verify_reproducible() -> bool:
    rng = make_rng(30)
    ref = _gen_seqs(45.0, 20, rng)

    r1 = ber_curve(ref, DEFAULT_EBN0_DB, n_bits=N_BITS, fs=FS, rng=make_rng(99))["ber"]
    r2 = ber_curve(ref, DEFAULT_EBN0_DB, n_bits=N_BITS, fs=FS, rng=make_rng(99))["ber"]
    identical = bool(np.array_equal(r1, r2))
    status = "PASS" if identical else "FAIL"
    print(f"[8.4] 同种子逐位一致  {identical}  [{status}]")
    return identical


if __name__ == "__main__":
    print("=" * 60)
    print("Day 8 QPSK BER 指标库自测")
    print("=" * 60)
    results = [
        verify_constant_channel(),
        verify_split_half(),
        verify_positive_control(),
        verify_reproducible(),
    ]
    print("=" * 60)
    n_pass = sum(results)
    print(f"总计：{n_pass}/{len(results)} 通过",
          "— 全部 PASS ✓" if n_pass == len(results) else "— 存在 FAIL ✗")
