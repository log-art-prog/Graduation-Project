"""阶段四：扩散调度表与前向过程 q(x_t|x_0)。

两种 β 调度（IMPROVEMENT_PLAN 阶段4 消融项 1）：
  - linear：β 从 1e-4 线性增到 0.02（DDPM 原文 Ho et al. 2020 设定）；
  - cosine：Nichol & Dhariwal 2021《Improved DDPM》闭式，
        f(t)=cos²((t/T+s)/(1+s)·π/2)，β_t=1−f(t)/f(t−1)，s=0.008，
    信息破坏更慢、末端噪声占比平滑，对幅度范围不大的包络数据通常更友好。

索引约定（所有调度表长度 T，索引 i=0..T−1）：
    alpha_bar[i] = ∏_{j=0}^{i} (1−β_j)        累积信号留存
    q_sample: x_t = √ᾱ_t·x_0 + √(1−ᾱ_t)·ε     （ε~N(0,I)）
训练时 t 均匀采自 {0..T−1}；反向采样从 i=T−1 走到 0。
"""
from __future__ import annotations

import torch
import torch.nn as nn


class DiffusionSchedule(nn.Module):
    """持有 β/α/ᾱ 全套调度表（注册为 buffer，随 .to(device) 迁移）。"""

    def __init__(self, T: int = 1000, kind: str = "linear",
                 beta_start: float = 1e-4, beta_end: float = 0.02,
                 s: float = 0.008):
        super().__init__()
        if kind not in ("linear", "cosine"):
            raise ValueError(f"未知调度类型: {kind!r}（支持 linear / cosine）")
        self.T = T
        self.kind = kind

        if kind == "linear":
            betas = torch.linspace(beta_start, beta_end, T, dtype=torch.float64)
        else:
            # T+1 个网格点 0..T 上的 cosine 曲线，差分得到 T 个 β
            x = torch.linspace(0, T, T + 1, dtype=torch.float64)
            f = torch.cos(((x / T) + s) / (1 + s) * torch.pi / 2.0) ** 2
            betas = torch.clamp(1.0 - f[1:] / f[:-1], min=1e-8, max=0.999)

        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)

        # float32 注册（与网络计算精度一致）
        self.register_buffer("betas", betas.float())
        self.register_buffer("alphas", alphas.float())
        self.register_buffer("alpha_bar", alpha_bar.float())
        self.register_buffer("sqrt_ab", alpha_bar.sqrt().float())
        self.register_buffer("sqrt_1mab", (1.0 - alpha_bar).sqrt().float())

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor,
                 noise: torch.Tensor | None = None) -> torch.Tensor:
        """前向扩散闭式（无需逐步加噪）。

        Args:
            x0:    (B, 1, L) 干净标准化包络
            t:     (B,) 时间步索引，整数 ∈ [0, T−1]（各样本可不同）
            noise: (B, 1, L) 可选外部噪声（测试/DDIM 复现用）；None 则采样
        Returns:
            (B, 1, L) 带噪样本 x_t
        """
        if noise is None:
            noise = torch.randn_like(x0)
        ab = self.alpha_bar[t].view(-1, 1, 1)
        return ab.sqrt() * x0 + (1.0 - ab).sqrt() * noise
