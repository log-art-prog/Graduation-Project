"""Day 6 自测脚本：L1 静态分布指标双向自测。

三个测试场景：
  1. 参考对半切：r_ref 和 r_gen 来自同一分布，各指标应 ≈ 0
  2. 加偏置：r_gen = r_ref + 偏移 / 缩放，各指标应显著变大（> 5 倍）
  3. 同种子复现：固定 rng 跑两遍，结果应完全一致

验收关卡（Day6_TASKS.md 第 2 节）：
  - 参考对半 L1 < 0.01 / KS < 0.01 / Wasserstein < 0.01
  - 矩相对误差 < 1%
  - 加偏移后 L1/KS/W 均 > 对半切的 5 倍
  - 同种子两次结果完全一致

======================================================================
思考题：见 docs/notes/11_自测题库_Day1to9.md 的 Day 6 章
        （Q6.1~Q6.12，题目 + 参考答案 A6.1~A6.12）
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# 让脚本可以从 scripts/ 直接跑，找到 generative 包
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from generative.data.dataset import load_dataset
from generative.data.transforms import inverse_standardize
from generative.metrics.distribution import evaluate_distribution
from generative.utils.seed import make_rng


CHECKS: list[bool] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    CHECKS.append(ok)
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag}] {name}  {detail}")


# ----------------------------------------------------------------------
# 加载参考数据：用 inverse_standardize 还原成物理包络
# 为什么用物理包络而非 x_std：
#   阶段2的指标都在"原始物理量"上定义（BER/LCR 门限都是 dB），
#   分布指标也保持一致，避免在 log 域算出来的 L1 数值意义不直观。
# ----------------------------------------------------------------------
def load_physical_envelope(npz_path: str) -> np.ndarray:
    """读 npz，把 x_std 还原成原始包络 r，展平返回 1D。"""
    d = load_dataset(npz_path)
    x_std = d["x_std"]
    mu_log = float(d["mu_log"])
    sigma_log = float(d["sigma_log"])
    eps = float(d["eps"])
    r = inverse_standardize(x_std, mu_log, sigma_log, eps=eps)
    return r.flatten()


# ----------------------------------------------------------------------
# 场景 1：参考对半切，各指标应 ≈ 0
# ----------------------------------------------------------------------
def scenario_split_half(r_all: np.ndarray) -> dict:
    print("\n=== 场景 1：参考对半切（同分布，各指标应 ≈ 0）===")
    rng = make_rng(42)
    idx = rng.permutation(len(r_all))
    half = len(idx) // 2
    r_ref = r_all[idx[:half]]
    r_gen = r_all[idx[half:]]

    res = evaluate_distribution(r_ref, r_gen)

    # 主指标数值
    l1 = res["pdf_l1"]["value"]
    ks = res["ks"]["value"]
    w = res["wasserstein"]["value"]
    print(f"  L1         = {l1:.6f}")
    print(f"  KS         = {ks:.6f}")
    print(f"  Wasserstein= {w:.6f}")
    print(f"  矩相对误差: {res['moments']['value']}")

    check("对半切 L1 < 0.01", l1 < 0.01, f"L1={l1:.6f}")
    check("对半切 KS < 0.01", ks < 0.01, f"KS={ks:.6f}")
    check("对半切 Wasserstein < 0.01", w < 0.01, f"W={w:.6f}")

    # 矩相对误差 < 1%
    moments_err = res["moments"]["value"]
    for k, v in moments_err.items():
        check(f"对半切 {k} < 1%", v < 0.01, f"{k}={v:.6f}")

    return res


# ----------------------------------------------------------------------
# 场景 2：加偏移 / 缩放，指标应显著变大
# ----------------------------------------------------------------------
def scenario_bias(r_ref: np.ndarray, baseline: dict) -> None:
    print("\n=== 场景 2：加偏移 +0.5 / 缩放 ×1.2，指标应显著变大 ===")

    # 偏移 0.5（线性量纲，相对 r 典型 O(1)）
    r_gen_bias = r_ref + 0.5
    res_bias = evaluate_distribution(r_ref, r_gen_bias)

    # 缩放 1.2
    r_gen_scale = r_ref * 1.2
    res_scale = evaluate_distribution(r_ref, r_gen_scale)

    b_l1, b_ks, b_w = res_bias["pdf_l1"]["value"], res_bias["ks"]["value"], res_bias["wasserstein"]["value"]
    s_l1, s_ks, s_w = res_scale["pdf_l1"]["value"], res_scale["ks"]["value"], res_scale["wasserstein"]["value"]
    print(f"  偏移 +0.5: L1={b_l1:.6f}  KS={b_ks:.6f}  W={b_w:.6f}")
    print(f"  缩放 ×1.2: L1={s_l1:.6f}  KS={s_ks:.6f}  W={s_w:.6f}")

    # 验收：偏移后 L1/KS/W 都 > 对半切基线的 5 倍
    base_l1, base_ks, base_w = baseline["pdf_l1"]["value"], baseline["ks"]["value"], baseline["wasserstein"]["value"]

    # 取偏移和缩放中更大的那个作为"被放大"的代表
    worse_l1 = max(b_l1, s_l1)
    worse_ks = max(b_ks, s_ks)
    worse_w = max(b_w, s_w)

    check("偏置后 L1 > 5×基线", worse_l1 > 5 * base_l1,
          f"worst={worse_l1:.6f}, baseline={base_l1:.6f}")
    check("偏置后 KS > 5×基线", worse_ks > 5 * base_ks,
          f"worst={worse_ks:.6f}, baseline={base_ks:.6f}")
    check("偏置后 W > 5×基线", worse_w > 5 * base_w,
          f"worst={worse_w:.6f}, baseline={base_w:.6f}")

    # 偏移后 mean 应该明显变化（矩相对误差应显著）
    # moment_stats 返回的键是 mean/std/skewness/kurtosis（不带 _rel_err 后缀）
    mean_rel_err_bias = res_bias["moments"]["value"]["mean"]
    check("偏移 0.5 后 mean 相对误差 > 5%", mean_rel_err_bias > 0.05,
          f"mean_rel_err={mean_rel_err_bias:.6f}")

    std_rel_err_scale = res_scale["moments"]["value"]["std"]
    check("缩放 1.2 后 std 相对误差 > 5%", std_rel_err_scale > 0.05,
          f"std_rel_err={std_rel_err_scale:.6f}")


# ----------------------------------------------------------------------
# 场景 3：同种子复现，两次结果应完全一致
# ----------------------------------------------------------------------
def scenario_reproducibility(r_all: np.ndarray) -> None:
    print("\n=== 场景 3：同种子两次结果应完全一致 ===")

    def _one_run(seed: int):
        rng = make_rng(seed)
        idx = rng.permutation(len(r_all))
        half = len(idx) // 2
        r_ref = r_all[idx[:half]]
        r_gen = r_all[idx[half:]]
        return evaluate_distribution(r_ref, r_gen)

    res1 = _one_run(2026)
    res2 = _one_run(2026)

    # 比较所有主指标值
    all_match = True
    for k in ["pdf_l1", "ks", "wasserstein"]:
        v1, v2 = res1[k]["value"], res2[k]["value"]
        if v1 != v2:
            all_match = False
            print(f"  {k} 不一致: {v1} vs {v2}")

    # 矩统计子项也要一致
    for k, v in res1["moments"]["value"].items():
        if v != res2["moments"]["value"][k]:
            all_match = False
            print(f"  moments.{k} 不一致: {v} vs {res2['moments']['value'][k]}")

    check("同种子两次结果完全一致", all_match, "")


# ----------------------------------------------------------------------
# 主入口
# ----------------------------------------------------------------------
def main() -> int:
    npz_path = ROOT / "datasets" / "corazza_train.npz"
    print(f"加载参考数据: {npz_path}")
    r_all = load_physical_envelope(str(npz_path))
    print(f"  总样本点数: {len(r_all):,}")

    baseline = scenario_split_half(r_all)
    scenario_bias(r_all[:len(r_all) // 2], baseline)  # 用前一半当 ref，节省内存
    scenario_reproducibility(r_all)

    n_pass = sum(CHECKS)
    n_total = len(CHECKS)
    print(f"\n==== Day 6 自测：{n_pass}/{n_total} PASS ====")
    return 0 if n_pass == n_total else 1


if __name__ == "__main__":
    sys.exit(main())
