"""Day 7 自测脚本：L2 时序指标（ACF / LCR / AFD）双向自测。

五个测试场景：
  1. ACF 形状健全性：白噪声 ACF 近似 δ、平滑序列 ACF 展宽（验证算法本身）
  2. Rayleigh(K=0) LCR 绝对锚点：对 Rice 闭式公式最大相对误差 < 15%
  3. 参考对半切（按【序列条数】对半，不砍时间轴）：ACF/LCR/AFD 应 ≈ 0
  4. 打乱时序：PDF 不变但 ACF 必须被毁、LCR/AFD 显著变化（抓"只学边际分布"的假模型）
  5. 同输入两次结果完全一致（纯函数确定性）

验收关卡（PHASE2_TASKS.md Day 7 + notes/09 第 8 节）：
  - 白噪声 max|ACF[1:50]| < 0.02；平滑序列 ACF(τ=0.3s) > 0.2
  - Rayleigh LCR 对理论 < 15%
  - 对半切 ACF RMSE < 0.02；LCR/AFD 相对 RMSE < 15%
  - 打乱后 ACF/LCR/AFD 误差 > 对半切基线的 5 倍

======================================================================
思考题：见 docs/notes/11_自测题库_Day1to7.md 的 Day 7 章
        （Q7.1~Q7.12，题目 + 参考答案 A7.1~A7.12）
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.special import i0

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from generative.data.corazza import colored_gaussian_jakes, colored_gaussian_lp
from generative.data.dataset import load_dataset
from generative.data.transforms import inverse_standardize
from generative.metrics.temporal import (
    envelope_acf, threshold_stats, evaluate_temporal,
)
from generative.utils.seed import make_rng


CHECKS: list[bool] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    CHECKS.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}")


# ----------------------------------------------------------------------
# 场景 1：ACF 形状健全性（白噪声 δ / 平滑展宽）
# ----------------------------------------------------------------------
def scenario_acf_shape() -> None:
    print("\n=== 场景 1：ACF 形状健全性（白噪声 → δ，慢阴影 → 展宽）===")
    fs = 1000.0
    n_lags = 50
    rng = make_rng(2024)
    n = 100_000

    # 白噪声：理论 ACF 为 δ，lag≥1 抖动量级 ~1/√n ≈ 0.0032
    white = rng.standard_normal(n)
    acf_w = envelope_acf(white, n_lags)
    tail_max = float(np.max(np.abs(acf_w[1:])))
    print(f"  白噪声   max|ACF[1:50]| = {tail_max:.5f}")
    check("白噪声 ACF 近似 δ（lag≥1 < 0.02）", tail_max < 0.02,
          f"max={tail_max:.5f}")

    # 一阶低通噪声，τ_c=0.3s：理论 ACF=exp(-τ/τ_c)，τ=0.3s 处应≈e^-1=0.368
    smooth = colored_gaussian_lp(n, fs, tau_c=0.3, rng=rng)
    acf_s = envelope_acf(smooth, n)  # 取全长，定位 lag=300
    val_at_tauc = float(acf_s[int(0.3 * fs)])
    print(f"  平滑序列 ACF(τ=0.3s) = {val_at_tauc:.4f}（理论 e^-1≈0.368）")
    check("平滑序列 ACF 展宽（ACF(0.3s) > 0.2）", val_at_tauc > 0.2,
          f"val={val_at_tauc:.4f}")


# ----------------------------------------------------------------------
# 场景 2：Rayleigh(K=0) LCR 对 Rice 闭式公式的绝对锚点
# N(ρ)=√(2π(K+1))·fd·ρ·exp(-K-(K+1)ρ²)·I0(2ρ√(K(K+1)))
# ----------------------------------------------------------------------
def scenario_rayleigh_anchor() -> None:
    print("\n=== 场景 2：Rayleigh(K=0) LCR 理论锚点（< 15%）===")
    n = 200_000
    fs, fd = 1000.0, 50.0
    K = 0.0
    rng = make_rng(7)

    s_R = np.sqrt(1.0 / (2.0 * (K + 1.0)))
    x = colored_gaussian_jakes(n, fs, fd, rng)
    y = colored_gaussian_jakes(n, fs, fd, rng)
    R = np.sqrt((s_R * x) ** 2 + (s_R * y) ** 2)

    # Day 3 已验证的门限网格 ρ∈[0.2,2.0]（dB 与线性互换传参）
    rhos = np.linspace(0.2, 2.0, 10)
    db = 20.0 * np.log10(rhos)
    stats = threshold_stats(R, thresholds_db=db, fs=fs)
    emp = stats["lcr"]
    thy = (np.sqrt(2 * np.pi * (K + 1)) * fd * rhos
           * np.exp(-K - (K + 1) * rhos ** 2)
           * i0(2 * rhos * np.sqrt(K * (K + 1))))
    rel_err = np.abs(emp - thy) / np.maximum(thy, 1e-10)
    max_err = float(rel_err.max())
    print(f"  各门限相对误差: {np.round(rel_err, 4)}")
    print(f"  最大相对误差 = {max_err:.2%}")
    check("Rayleigh LCR 对闭式公式 < 15%", max_err < 0.15,
          f"max_err={max_err:.2%}")


# ----------------------------------------------------------------------
# 加载物理包络（与 Day 6 同一口径：inverse_standardize 后评估）
# ----------------------------------------------------------------------
def load_physical_envelope() -> np.ndarray:
    d = load_dataset(str(ROOT / "datasets" / "corazza_train.npz"))
    r = inverse_standardize(d["x_std"], float(d["mu_log"]),
                            float(d["sigma_log"]), eps=float(d["eps"]))
    return r.astype(np.float64)  # (10000, 1000)


# ----------------------------------------------------------------------
# 场景 3：按序列条数对半切，时序指标应 ≈ 0
# ----------------------------------------------------------------------
def scenario_split_half(r: np.ndarray) -> dict:
    print("\n=== 场景 3：参考对半切（按序列条数，ACF/LCR/AFD 应 ≈ 0）===")
    rng = make_rng(42)
    idx = rng.permutation(r.shape[0])
    half = len(idx) // 2
    r_ref = r[idx[:half]]
    r_gen = r[idx[half:]]

    res = evaluate_temporal(r_ref, r_gen, fs=1000.0)
    acf_v = res["acf"]["value"]
    lcr_v = res["lcr"]["value"]
    afd_v = res["afd"]["value"]
    print(f"  ACF RMSE      = {acf_v:.6f}")
    print(f"  LCR 相对 RMSE = {lcr_v:.6f}")
    print(f"  AFD 相对 RMSE = {afd_v:.6f}")
    print(f"  LCR 逐门限误差: {np.round(res['lcr']['detail']['per_threshold_rel_err'], 4)}")
    print(f"  AFD 逐门限误差: {np.round(res['afd']['detail']['per_threshold_rel_err'], 4)}")

    check("对半切 ACF RMSE < 0.02", acf_v < 0.02, f"acf={acf_v:.6f}")
    check("对半切 LCR 相对 RMSE < 15%", lcr_v < 0.15, f"lcr={lcr_v:.6f}")
    check("对半切 AFD 相对 RMSE < 15%", afd_v < 0.15, f"afd={afd_v:.6f}")
    return res


# ----------------------------------------------------------------------
# 场景 4：每条序列独立打乱时间顺序 → ACF 必须被毁，LCR/AFD 显著变化
# ----------------------------------------------------------------------
def scenario_shuffle(r: np.ndarray, baseline: dict) -> None:
    print("\n=== 场景 4：打乱时序（边际分布不变，时间结构被毁）===")
    rng = make_rng(123)
    r_ref = r[: r.shape[0] // 2]
    r_shuffled = np.empty_like(r_ref)
    for i in range(r_ref.shape[0]):
        r_shuffled[i] = r_ref[i, rng.permutation(r_ref.shape[1])]

    res = evaluate_temporal(r_ref, r_shuffled, fs=1000.0)
    acf_v, lcr_v, afd_v = res["acf"]["value"], res["lcr"]["value"], res["afd"]["value"]
    print(f"  打乱后 ACF RMSE      = {acf_v:.6f}")
    print(f"  打乱后 LCR 相对 RMSE = {lcr_v:.6f}")
    print(f"  打乱后 AFD 相对 RMSE = {afd_v:.6f}")

    check("打乱后 ACF > 5×基线", acf_v > 5 * baseline["acf"]["value"],
          f"shuf={acf_v:.4f}, base={baseline['acf']['value']:.6f}")
    check("打乱后 LCR > 5×基线", lcr_v > 5 * baseline["lcr"]["value"],
          f"shuf={lcr_v:.4f}, base={baseline['lcr']['value']:.6f}")
    check("打乱后 AFD > 5×基线", afd_v > 5 * baseline["afd"]["value"],
          f"shuf={afd_v:.4f}, base={baseline['afd']['value']:.6f}")


# ----------------------------------------------------------------------
# 场景 5：纯函数确定性（同输入两次逐位一致）
# ----------------------------------------------------------------------
def scenario_determinism(r: np.ndarray) -> None:
    print("\n=== 场景 5：同输入两次结果逐位一致 ===")
    r_a, r_b = r[:1000], r[1000:2000]
    res1 = evaluate_temporal(r_a, r_b)
    res2 = evaluate_temporal(r_a, r_b)
    ok = (res1["acf"]["value"] == res2["acf"]["value"]
          and res1["lcr"]["value"] == res2["lcr"]["value"]
          and res1["afd"]["value"] == res2["afd"]["value"])
    check("时序指标纯函数确定性", ok, "")


def main() -> int:
    print("加载物理包络数据集...")
    r_all = load_physical_envelope()
    print(f"  shape={r_all.shape}, dtype={r_all.dtype}")

    scenario_acf_shape()
    scenario_rayleigh_anchor()
    baseline = scenario_split_half(r_all)
    scenario_shuffle(r_all, baseline)
    scenario_determinism(r_all)

    n_pass, n_total = sum(CHECKS), len(CHECKS)
    print(f"\n==== Day 7 自测：{n_pass}/{n_total} PASS ====")
    return 0 if n_pass == n_total else 1


if __name__ == "__main__":
    sys.exit(main())
