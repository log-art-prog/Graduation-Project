"""阶段四：条件 1D U-Net 去噪网络 ε_θ(x_t, t, α)。

设计（IMPROVEMENT_PLAN 阶段4）：
  - 3 级下采样 1000→500→250→125，通道默认 64→128→256，瓶颈 125 点；
    上采样用 F.interpolate 对齐跳跃连接的精确长度（1000 非 2 的幂，
    不用 ConvTranspose 以避免长度错配）。
  - kernel=15：与 CGAN 生成器同款大核，捕捉信道自相关的时间相关结构。
  - 时间步 t：正弦位置编码 → 2 层 MLP → 逐残差块相加注入。
  - 仰角条件 α：标量 → 嵌入 MLP，每个残差块一组独立 FiLM (γ,β)
        h ← h · (1+γ) + β
    逐通道仿射调制（与 CGAN 的"单次常数通道拼接"形成机制对比）。

稳定性：FiLM 投影层与最终输出头 zero-init —— 训练第 0 步网络等价
无条件模型且 ε̂≈0，随后逐步学入条件与结构。
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

SEQ_LEN = 1000
GROUPS = 8  # GroupNorm 组数（本项目所有通道配置均能被 8 整除）


class SinusoidalTimeEmbedding(nn.Module):
    """正弦时间步编码（与 Transformer 位置编码同款）。"""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) 整数时间步
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device) / half
        )
        args = t.float()[:, None] * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class ResidualBlock(nn.Module):
    """Conv1d 残差块 + 时间相加注入 + 仰角 FiLM 仿射调制。"""

    def __init__(self, in_ch: int, out_ch: int, time_dim: int,
                 kernel: int = 15):
        super().__init__()
        pad = kernel // 2
        self.norm1 = nn.GroupNorm(GROUPS, in_ch)
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel, padding=pad)
        self.time_proj = nn.Linear(time_dim, out_ch)
        # FiLM：从条件嵌入产出逐通道 γ、β（zero-init）
        self.film_proj = nn.Linear(time_dim, 2 * out_ch)
        nn.init.zeros_(self.film_proj.weight)
        nn.init.zeros_(self.film_proj.bias)
        self.norm2 = nn.GroupNorm(GROUPS, out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel, padding=pad)
        self.skip = (nn.Conv1d(in_ch, out_ch, 1)
                     if in_ch != out_ch else nn.Identity())

    def forward(self, h: torch.Tensor, t_emb: torch.Tensor,
                c_emb: torch.Tensor) -> torch.Tensor:
        residual = self.skip(h)
        h = self.conv1(F.silu(self.norm1(h)))
        h = h + self.time_proj(t_emb)[:, :, None]          # 时间：逐通道相加
        gamma, beta = self.film_proj(c_emb).chunk(2, dim=1)  # FiLM：逐通道仿射
        h = h * (1.0 + gamma[:, :, None]) + beta[:, :, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + residual


class Downsample(nn.Module):
    """stride=2 卷积下采样（1000→500→250→125）。"""

    def __init__(self, ch: int):
        super().__init__()
        self.op = nn.Conv1d(ch, ch, 3, stride=2, padding=1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.op(h)


class ConditionalUNet1D(nn.Module):
    """条件 1D U-Net：输入 (x_t, t, α)，预测噪声 ε̂，形状与 x_t 相同。"""

    def __init__(self, channels: tuple = (64, 128, 256), kernel: int = 15,
                 time_dim: int = 128, cond_dim: int = 16,
                 seq_len: int = SEQ_LEN):
        super().__init__()
        self.seq_len = seq_len
        c0, c1, c2 = channels

        # 时间：正弦编码 → 2 层 MLP
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim), nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        # 仰角：标量 → cond_dim → time_dim（FiLM 与时间共享嵌入维度）
        self.cond_embed = nn.Sequential(
            nn.Linear(1, cond_dim), nn.SiLU(),
            nn.Linear(cond_dim, time_dim), nn.SiLU(),
        )

        self.init_conv = nn.Conv1d(1, c0, kernel, padding=kernel // 2)

        # 编码器（每个残差块的输出同时作为跳跃连接）
        self.down0 = ResidualBlock(c0, c0, time_dim, kernel)
        self.down0_pool = Downsample(c0)
        self.down1 = ResidualBlock(c0, c1, time_dim, kernel)
        self.down1_pool = Downsample(c1)
        self.down2 = ResidualBlock(c1, c2, time_dim, kernel)
        self.down2_pool = Downsample(c2)

        # 瓶颈
        self.mid1 = ResidualBlock(c2, c2, time_dim, kernel)
        self.mid2 = ResidualBlock(c2, c2, time_dim, kernel)

        # 解码器（上采样后与同尺度跳跃拼接）
        self.up2 = ResidualBlock(2 * c2, c1, time_dim, kernel)
        self.up1 = ResidualBlock(2 * c1, c0, time_dim, kernel)
        self.up0 = ResidualBlock(2 * c0, c0, time_dim, kernel)

        # 输出头：zero-init（训练起点 ε̂=0）
        self.head = nn.Sequential(
            nn.GroupNorm(GROUPS, c0), nn.SiLU(),
            nn.Conv1d(c0, 1, kernel, padding=kernel // 2),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor,
                c: torch.Tensor) -> torch.Tensor:
        """x: (B,1,L) 带噪序列；t: (B,) 时间步；c: (B,) 归一化仰角 [-1,1]。"""
        t_emb = self.time_embed(t)
        c_emb = self.cond_embed(c[:, None])

        h0 = self.down0(self.init_conv(x), t_emb, c_emb)   # (B,c0,L)
        h1 = self.down1(self.down0_pool(h0), t_emb, c_emb)  # (B,c1,L/2)
        h2 = self.down2(self.down1_pool(h1), t_emb, c_emb)  # (B,c2,L/4)
        h = self.down2_pool(h2)                              # (B,c2,L/8)

        h = self.mid1(h, t_emb, c_emb)
        h = self.mid2(h, t_emb, c_emb)

        h = F.interpolate(h, size=h2.shape[-1], mode="nearest")
        h = self.up2(torch.cat([h, h2], dim=1), t_emb, c_emb)
        h = F.interpolate(h, size=h1.shape[-1], mode="nearest")
        h = self.up1(torch.cat([h, h1], dim=1), t_emb, c_emb)
        h = F.interpolate(h, size=h0.shape[-1], mode="nearest")
        h = self.up0(torch.cat([h, h0], dim=1), t_emb, c_emb)
        return self.head(h)
