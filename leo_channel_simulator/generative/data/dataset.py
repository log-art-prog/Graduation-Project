"""Day 4：Corazza 信道数据集的构建、保存、加载。

本文件包含 4 个部分：
  1. build_dataset   —— 一次性生成 N 条时序，统一标准化
  2. save_dataset    —— 存成 .npz + .json 元数据
  3. load_dataset    —— 读回 npz
  4. CorazzaDataset  —— PyTorch Dataset 薄封装（Day 5 训练用）
"""
from __future__ import annotations

import datetime
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from generative.data.corazza import generate_corazza_sequence
from generative.data.transforms import (
    EPS,
    fit_standardizer,
    standardize,
    normalize_alpha,
)
from generative.utils.seed import make_rng


# =====================================================================
# 任务 4.1：构建数据集
# =====================================================================
def build_dataset(n_samples: int = 10000,
                  length: int = 1000,
                  seed: int = 42,
                  fs: float = 1000.0,
                  fd: float = 50.0,
                  tau_shadow: float = 0.3,
                  alphas: np.ndarray | None = None) -> dict:
    """生成数据集并统一标准化。

    流程：仰角均匀采样 -> 逐条生成原始包络(float64) ->
    全部生成完后一次性估计 mu_log/sigma_log -> 统一标准化 -> 转 float32。

    Args:
        n_samples: 样本条数（默认 10000，学长论文设定）
        length:    每条序列长度（默认 1000）
        seed:      随机种子，保证同种子完全可复现
        fs:        采样率（Hz），Day 3 复现设定
        fd:        最大多普勒频移（Hz），Day 3 复现设定
        tau_shadow:阴影相关时间（s），Day 3 复现设定
        alphas:    可选，外部指定的仰角数组；None 时在 [20,80] 均匀采样

    Returns:
        dict，包含 6 个 key：
          x_std      : (N, length) float32，标准化后的序列（训练输入）
          alpha_norm : (N,)        float32，仰角归一化到 [-1,1]（条件标签）
          alpha_deg  : (N,)        float64，仰角原始角度（画图/检查用）
          mu_log     : 标量，全库统一的 log 域均值
          sigma_log  : 标量，全库统一的 log 域标准差
          eps        : 标量，log 防零常数
    """
    rng = make_rng(seed)

    # 仰角：必须先抽仰角再抽序列——rng 的消耗顺序固定，同 seed 才能复现
    if alphas is None:
        alphas = rng.uniform(20.0, 80.0, size=n_samples)
    else:
        assert len(alphas) == n_samples, "传入 alphas 长度必须等于 n_samples"

    # 预分配 float64 保证生成与统计精度，存盘前才转 float32
    raw = np.empty((n_samples, length), dtype=np.float64)
    for i in range(n_samples):
        raw[i] = generate_corazza_sequence(
            alphas[i], length=length, fs=fs, fd=fd,
            tau_shadow=tau_shadow, rng=rng)
        if (i + 1) % 1000 == 0:
            print(f"  已生成 {i + 1}/{n_samples} 条")

    # 标准化常量必须由全库一次性估计（不能边生成边算，见 Day4_TASKS 0.2 节）
    mu_log, sigma_log = fit_standardizer(raw)
    x_std = standardize(raw, mu_log, sigma_log).astype(np.float32)
    alpha_norm = normalize_alpha(alphas).astype(np.float32)

    return {
        "x_std": x_std,
        "alpha_norm": alpha_norm,
        "alpha_deg": alphas.astype(np.float64),
        "mu_log": mu_log,
        "sigma_log": sigma_log,
        "eps": EPS,
    }


# =====================================================================
# 任务 4.2：保存到磁盘
# =====================================================================
def save_dataset(data: dict, out_dir: str | Path,
                 npz_name: str = "corazza_train.npz",
                 meta_name: str = "corazza_train_meta.json",
                 seed: int = 42,
                 fs: float = 1000.0,
                 fd: float = 50.0,
                 tau_shadow: float = 0.3) -> None:
    """把数据集存成 .npz，构建参数存成 meta.json。

    meta.json 记录复现所需的全部信息（种子/时序参数/日期/标准化常量），
    是 G4 "元数据齐全" 验收项的载体。

    Args:
        data:    build_dataset 返回的 dict
        out_dir: 输出目录（不存在会自动创建）
        seed/fs/fd/tau_shadow: 构建时的参数，如实写进 meta（默认值
            与 build_dataset 的默认参数一致）
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 数组部分：**data 把 dict 解包成关键字参数，key 名即 npz 内部文件名
    np.savez(out_dir / npz_name, **data)

    # 元数据：JSON 只认 Python 原生类型，numpy 标量必须 float()/int() 包一层
    meta = {
        "created_at": datetime.date.today().isoformat(),
        "seed": seed,
        "n_samples": int(data["x_std"].shape[0]),
        "length": int(data["x_std"].shape[1]),
        "fs": fs,
        "fd": fd,
        "tau_shadow": tau_shadow,
        "mu_log": float(data["mu_log"]),
        "sigma_log": float(data["sigma_log"]),
        "eps": float(data["eps"]),
        "dtype": "float32",
    }
    with open(out_dir / meta_name, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"数据集已保存: {out_dir / npz_name} + {meta_name}")


# =====================================================================
# 任务 4.3：加载
# =====================================================================
def load_dataset(npz_path: str | Path) -> dict:
    """读回 npz，返回 dict（key 与 build_dataset 返回的一致）。

    注：npz 是 zip 容器，无法直接内存映射；40MB 全量读进内存即可。
    """
    z = np.load(npz_path)
    return {k: z[k] for k in z.files}


class CorazzaDataset(Dataset):
    """PyTorch Dataset：Day 5 训练 CGAN/DDPM 时用。

    用法：
        ds = CorazzaDataset("datasets/corazza_train.npz")
        x, alpha = ds[0]      # x: shape(1000,) 的 float32 张量
        len(ds)               # 10000
    """

    def __init__(self, npz_path: str | Path):
        d = load_dataset(npz_path)
        self.x = d["x_std"]           # (N, length) float32
        self.alpha = d["alpha_norm"]  # (N,) float32

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        """返回第 idx 条样本：(序列张量 shape=(length,), 仰角标量张量)。"""
        x_tensor = torch.from_numpy(self.x[idx]).float()
        # alpha[idx] 是 numpy 标量（0 维），from_numpy 不接受标量，
        # 必须先 float() 再用 torch.tensor() 包装
        alpha_tensor = torch.tensor(float(self.alpha[idx])).float()
        return x_tensor, alpha_tensor
