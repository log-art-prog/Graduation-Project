"""阶段四：DDIM 确定性采样（Song et al. 2021），训练后无需重训即可加速。

取时间索引子序列（linspace 均匀跳步），η=0 时链路完全确定：
    x̂_0  = (x_i − √(1−ᾱ_i) ε̂) / √ᾱ_i
    x_{prev} = √ᾱ_prev · x̂_0 + √(1−ᾱ_prev) · ε̂
最后一跳 ᾱ_prev 取 1.0（干净时刻）。
同一套训练权重可直接走 20/50/100 步，用于速度-质量 Pareto 对比。
"""
from __future__ import annotations

import torch
import torch.nn as nn


@torch.no_grad()
def ddim_sample(model: nn.Module, schedule: nn.Module, c: torch.Tensor,
                xT: torch.Tensor, num_steps: int = 50,
                eta: float = 0.0,
                clip_x0: float | None = 30.0) -> torch.Tensor:
    """DDIM 采样。

    Args:
        model:      ε_θ 网络（调用方负责 eval / EMA 权重 / device）
        schedule:   DiffusionSchedule
        c:          (B,) 归一化仰角
        xT:         (B,1,L) 初始纯噪声（外部给定以保证可复现）
        num_steps:  跳步步数（≤ schedule.T），规划取 20/50/100
        eta:        0.0 为确定性 DDIM；1.0 为随机（与 DDPM 同族）
        clip_x0:    x̂₀ 裁剪门限（标准化 log 域）。欠训练模型在大 t 处 ε̂
                    失准，除以小 √ᾱ 会把 x̂₀ 放大上万倍，逆标准化 exp 溢出；
                    裁剪后按前向闭式反推一致的 ε（guided-diffusion 做法）。
                    真实数据标准化范围仅 [-16.7, 3.5]，默认 30.0 保留全部
                    真实尾部；None 表示不裁剪。
    Returns:
        (B,1,L) 标准化 log 域样本
    """
    if num_steps > schedule.T:
        raise ValueError(f"num_steps({num_steps}) 不能大于 T({schedule.T})")
    if not (0.0 <= eta <= 1.0):
        raise ValueError(f"eta 须在 [0,1]，收到 {eta}")

    n = xT.shape[0]
    device = xT.device
    # 降序时间索引（含 T−1 与 0），round 后在 num_steps≤T 时无重复
    ts = torch.linspace(schedule.T - 1, 0, num_steps).round().long()
    x = xT

    for k in range(num_steps):
        i = int(ts[k].item())
        i_prev = int(ts[k + 1].item()) if k + 1 < num_steps else -1
        t = torch.full((n,), i, dtype=torch.long, device=device)
        eps = model(x, t, c)

        ab = schedule.alpha_bar[i]
        ab_prev = (schedule.alpha_bar[i_prev] if i_prev >= 0
                   else torch.tensor(1.0, device=device))
        x0_pred = (x - (1.0 - ab).sqrt() * eps) / ab.sqrt()

        if clip_x0 is not None:
            # 裁剪后由前向闭式 x = √ᾱ·x̂₀ + √(1−ᾱ)·ε 反推一致的 ε，
            # 保证后续组合仍沿该前向关系，不会在一步内塞进矛盾信息
            x0_pred = x0_pred.clamp(-clip_x0, clip_x0)
            eps = (x - ab.sqrt() * x0_pred) / (1.0 - ab).sqrt()

        if eta == 0.0:
            x = ab_prev.sqrt() * x0_pred + (1.0 - ab_prev).sqrt() * eps
        else:
            # 一般 DDIM（保留随机项，当前研究默认走 η=0）
            sigma = eta * torch.sqrt(
                (1.0 - ab_prev) / (1.0 - ab)
                * (1.0 - ab / ab_prev.clamp_min(1e-8))
            ).clamp_min(0.0)
            dir_coef = (1.0 - ab_prev - sigma ** 2).clamp_min(0.0).sqrt()
            x = (ab_prev.sqrt() * x0_pred + dir_coef * eps
                 + sigma * torch.randn_like(x))
    return x
