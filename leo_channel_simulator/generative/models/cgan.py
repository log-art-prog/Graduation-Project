"""阶段三：CGAN 复现（对照组）— 严格按学长论文表 3-2。

架构（论文只给了结构描述与超参，层宽等细节为合理自选，论文写作时如实说明）：
  - 生成器 SequenceNoiseFilterGenerator：
      输入 z ~ (B, 32, 1000) 逐时刻高斯噪声 + 仰角条件通道（常数通道拼接），
      全程 stride=1 的 Conv1d 堆叠（"时序噪声滤波"：把白噪声滤波成
      具有时间相关性的包络序列），输出 (B, 1, 1000) 标准化 log 包络。
  - 判别器 LargeReceptiveFieldDiscriminator：
      5 层 Conv1d stride=2（1000→500→250→125→63→32），
      感受野 = 14·(1+2+4+8+16)+1 = 435 点，末端时间维全局平均池化 + Linear。

训练框架：WGAN-GP（λ_gp=10）+ TTUR（G 1e-4 / D 3e-4）+ Adam β=(0.0,0.9)
          + n_critic=5 + 生成器 EMA（decay=0.999）。
"""
from __future__ import annotations

import torch
import torch.nn as nn

NOISE_DIM = 32
SEQ_LEN = 1000


def _cond_channel(c: torch.Tensor, length: int) -> torch.Tensor:
    """把标量条件 c (B,) 展成常数通道 (B, 1, length) 与序列拼接。"""
    return c.view(-1, 1, 1).expand(-1, 1, length)


class SequenceNoiseFilterGenerator(nn.Module):
    """Conv1d 时序噪声滤波生成器（不改变序列长度）。"""

    def __init__(self, noise_dim: int = NOISE_DIM, seq_len: int = SEQ_LEN,
                 channels: tuple = (64, 128, 128, 64), kernel: int = 15):
        super().__init__()
        self.noise_dim = noise_dim
        self.seq_len = seq_len
        pad = kernel // 2
        layers: list[nn.Module] = []
        in_ch = noise_dim + 1  # +1 为仰角条件通道
        for ch in channels:
            layers += [nn.Conv1d(in_ch, ch, kernel, padding=pad),
                       nn.LeakyReLU(0.2, inplace=True)]
            in_ch = ch
        self.body = nn.Sequential(*layers)
        self.head = nn.Conv1d(in_ch, 1, kernel, padding=pad)  # 输出无激活（WGAN）

    def forward(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """z: (B, noise_dim, L)；c: (B,) 归一化仰角。返回 (B, 1, L)。"""
        h = torch.cat([z, _cond_channel(c, z.shape[-1])], dim=1)
        return self.head(self.body(h))


class LargeReceptiveFieldDiscriminator(nn.Module):
    """5 层 stride=2 Conv1d 大感受野判别器（WGAN-GP 中不用 BatchNorm）。"""

    def __init__(self, seq_len: int = SEQ_LEN,
                 channels: tuple = (32, 64, 128, 256, 512), kernel: int = 15):
        super().__init__()
        pad = kernel // 2
        layers: list[nn.Module] = []
        in_ch = 1 + 1  # 包络通道 + 仰角条件通道
        for ch in channels:
            layers += [nn.Conv1d(in_ch, ch, kernel, stride=2, padding=pad),
                       nn.LeakyReLU(0.2, inplace=True)]
            in_ch = ch
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(channels[-1], 1)  # 时间维全局平均池化后打分

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, L)；c: (B,)。返回 (B,)  critic 分数（非概率）。"""
        h = torch.cat([x, _cond_channel(c, x.shape[-1])], dim=1)
        h = self.body(h).mean(dim=2)  # (B, C) 全局平均池化
        return self.head(h).squeeze(-1)


class EMA:
    """生成器参数指数滑动平均（采样/评估用 EMA 权重）。"""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone()
                       for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:  # 非浮点缓存（如 num_batches_tracked）直接拷贝
                self.shadow[k].copy_(v)

    def state_dict(self) -> dict:
        return {k: v.clone() for k, v in self.shadow.items()}


def gradient_penalty(disc: nn.Module, real: torch.Tensor, fake: torch.Tensor,
                     c: torch.Tensor, lambda_gp: float = 10.0) -> torch.Tensor:
    """WGAN-GP 梯度惩罚：在真假样本连线上约束 ‖∇D‖ ≈ 1。"""
    B = real.shape[0]
    eps = torch.rand(B, 1, 1, device=real.device, dtype=real.dtype)
    interp = (eps * real + (1 - eps) * fake.detach()).requires_grad_(True)
    out = disc(interp, c)
    grad = torch.autograd.grad(out, interp,
                               grad_outputs=torch.ones_like(out),
                               create_graph=True)[0]
    gp = ((grad.view(B, -1).norm(2, dim=1) - 1.0) ** 2).mean()
    return lambda_gp * gp
