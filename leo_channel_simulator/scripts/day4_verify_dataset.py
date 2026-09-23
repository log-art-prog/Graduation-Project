"""Day 4 验证脚本：复现性（4.4）、体量检查（4.5）、抽检可视化（4.6）。

运行方式（在 leo_channel_simulator/leo_channel_simulator/ 目录下）：
    python scripts/day4_verify_dataset.py
"""
import sys
import os
from pathlib import Path

# ---- 让脚本能 import 到项目包（和 day3 脚本同样的套路）----
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Day 3 踩过的坑：中文字体
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

from generative.data.dataset import (
    build_dataset,
    save_dataset,
    load_dataset,
    CorazzaDataset,
)
from generative.data.transforms import inverse_standardize


# =====================================================================
# 4.4 复现性验证（用小数据 n=100，跑得快）
# =====================================================================
def verify_reproducibility():
    print("=" * 60)
    print("4.4 复现性验证")
    print("=" * 60)

    d1 = build_dataset(n_samples=100, seed=42)
    d2 = build_dataset(n_samples=100, seed=42)
    d3 = build_dataset(n_samples=100, seed=43)

    same = np.array_equal(d1["x_std"], d2["x_std"])
    print(f"同种子(42 vs 42)完全一致: {same}")
    assert same, "同种子结果不一致！"

    diff = not np.array_equal(d1["x_std"], d3["x_std"])
    print(f"异种子(42 vs 43)结果不同:   {diff}")
    assert diff, "异种子结果竟然相同！"

    print("4.4 PASS\n")


# =====================================================================
# 4.5 体量检查（全量 10000 条：构建 -> 保存 -> 加载 -> 打印信息）
# =====================================================================
def verify_full_dataset():
    print("=" * 60)
    print("4.5 全量数据集体量检查（10000 条，需要几分钟）")
    print("=" * 60)

    out_dir = ROOT / "datasets"

    # 如果已经生成过就不重复生成，直接加载（调试时省时间）
    npz_path = out_dir / "corazza_train.npz"
    if npz_path.exists():
        print("检测到已有数据集，直接加载（删除 datasets/ 可强制重新生成）")
        data = load_dataset(npz_path)
    else:
        data = build_dataset()
        save_dataset(data, out_dir)

    # meta.json 必须存在（G4：元数据齐全）——
    # 上面的"直接加载"分支会绕过 save_dataset，任何产出物缺失都要在这里兜住
    meta_path = out_dir / "corazza_train_meta.json"
    assert meta_path.exists(), "meta.json 缺失！G4 元数据不齐全"
    print(f"meta.json 存在: {meta_path.name}")

    # ---- 打印每个 key 的 shape / dtype ----
    print("\n数据集内容：")
    for k, v in data.items():
        print(f"  {k:12s} shape={v.shape}, dtype={v.dtype}")

    # ---- 文件大小 ----
    size_mb = os.path.getsize(npz_path) / 1024 / 1024
    print(f"\nnpz 文件大小: {size_mb:.1f} MB（预期约 40 MB）")

    # ---- shape / dtype 断言 ----
    assert data["x_std"].shape == (10000, 1000)
    assert data["alpha_norm"].shape == (10000,)
    assert data["x_std"].dtype == np.float32

    # ---- 顺便验证 CorazzaDataset 封装 ----
    ds = CorazzaDataset(npz_path)
    print(f"CorazzaDataset 长度: {len(ds)}（预期 10000）")
    x_t, a_t = ds[0]
    print(f"  __getitem__ -> x: {x_t.shape} {x_t.dtype}, alpha: {a_t}")

    print("4.5 PASS\n")
    return data


# =====================================================================
# 4.6 抽检可视化：低/中/高仰角各取一条，还原后画图
# =====================================================================
def verify_visual(data: dict):
    print("=" * 60)
    print("4.6 抽检可视化")
    print("=" * 60)

    alpha_deg = data["alpha_deg"]
    x_std = data["x_std"]
    mu_log = float(data["mu_log"])
    sigma_log = float(data["sigma_log"])

    # 在随机采样的仰角里，找最接近 25 / 50 / 75 度的样本索引
    # 用 argmin 而不是假设正好命中（均匀随机采样几乎不可能命中整数角度）
    targets = [25.0, 50.0, 75.0]
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    t_ms = np.arange(x_std.shape[1])   # fs=1000Hz -> 每个采样点恰好 1 ms

    for ax, target in zip(axes, targets):
        idx = int(np.argmin(np.abs(alpha_deg - target)))
        # 还原成物理包络（用数据集里存的全库统一常量，不要重算）
        r = inverse_standardize(x_std[idx], mu_log, sigma_log)
        ax.plot(t_ms, r, lw=0.7)
        ax.set_title(f"实际仰角 {alpha_deg[idx]:.1f}°")
        ax.set_ylabel("包络")
        ax.grid(True, alpha=0.3)

    ax.set_xlabel("时间 (ms)")
    fig.suptitle("Day 4 数据集抽检：不同仰角的 Corazza 包络序列")
    plt.tight_layout()
    fig.savefig(ROOT / "figs" / "day4_dataset_samples.png", dpi=120)
    print("图已保存到 figs/day4_dataset_samples.png")
    print("请目视确认：每条都有【毫秒级快起伏】叠加在【~300ms 慢包络】上")
    print("4.6 PASS\n")


if __name__ == "__main__":
    verify_reproducibility()
    data = verify_full_dataset()
    verify_visual(data)
    print("=" * 60)
    print("Day 4 全部验证通过！可以更新任务表并写日志")
    print("=" * 60)
