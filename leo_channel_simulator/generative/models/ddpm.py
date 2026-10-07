"""阶段四：DDPM 训练目标与反向采样（Ho et al. 2020）。

训练（ε-prediction，最简单的 MSE，无对抗博弈）：
    t ~ U{0..T−1}，ε ~ N(0,I)
    L = ‖ε − ε_θ(√ᾱ_t x_0 + √(1−ᾱ_t) ε, t, α)‖²

反向采样（DDPM 祖先采样，每步注入新噪声）：
    x_{i−1} = 1/√a_i · (x_i − β_i/√(1−ᾱ_i) · ε̂) + √β_i · z,  i>0
    x_{−1 步不加噪}
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianDiffusion(nn.Module):
    """组合去噪网络与调度表，提供训练损失和 DDPM 采样。"""

    def __init__(self, model: nn.Module, schedule: nn.Module):
        super().__init__()
        self.model = model
        self.schedule = schedule

    @property
    def T(self) -> int:
        return self.schedule.T

    def training_loss(self, x0: torch.Tensor, c: torch.Tensor,
                      rng_t: torch.Tensor | None = None) -> torch.Tensor:
        """单步 ε 预测 MSE。x0:(B,1,L) 标准化包络；c:(B,) 归一化仰角。"""
        B = x0.shape[0]
        if rng_t is not None:
            t = rng_t
        else:
            t = torch.randint(0, self.T, (B,), device=x0.device)
        noise = torch.randn_like(x0)
        xt = self.schedule.q_sample(x0, t, noise)
        eps_pred = self.model(xt, t, c)
        return F.mse_loss(eps_pred, noise)

    @torch.no_grad()
    def p_step(self, x: torch.Tensor, i: int,
               c: torch.Tensor,
               clip_x0: float | None = 30.0) -> torch.Tensor:
        """DDPM 反向一步：从时刻 i 到 i−1（i=0 时输出干净样本）。

        clip_x0 非 None 时先由 ε̂ 反算 x̂₀ 并裁剪，再按前向闭式反推一致
        的 ε̂，防止欠训练时 β/√(1−ᾱ) 把失准噪声逐级放大（同 DDIM 兜底）。
        """
        B = x.shape[0]
        t = torch.full((B,), i, dtype=torch.long, device=x.device)
        eps = self.model(x, t, c)
        beta = self.schedule.betas[i]
        alpha = self.schedule.alphas[i]
        ab = self.schedule.alpha_bar[i]
        if clip_x0 is not None:
            x0_pred = (x - (1.0 - ab).sqrt() * eps) / ab.sqrt()
            x0_pred = x0_pred.clamp(-clip_x0, clip_x0)
            eps = (x - ab.sqrt() * x0_pred) / (1.0 - ab).sqrt()
        mean = (x - beta / (1.0 - ab).sqrt() * eps) / alpha.sqrt()
        if i > 0:
            mean = mean + beta.sqrt() * torch.randn_like(x)
        return mean

    @torch.no_grad()
    def sample(self, c: torch.Tensor, n: int,
               seq_len: int = 1000,
               x_init: torch.Tensor | None = None,
               clip_x0: float | None = 30.0) -> torch.Tensor:
        """DDPM 完整 T 步祖先采样，返回标准化 log 域样本 (n,1,seq_len)。"""
        if x_init is None:
            x = torch.randn(n, 1, seq_len, device=c.device)
        else:
            x = x_init
        for i in range(self.T - 1, -1, -1):
            x = self.p_step(x, i, c, clip_x0=clip_x0)
        return x
