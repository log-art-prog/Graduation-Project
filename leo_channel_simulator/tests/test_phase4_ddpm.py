"""阶段四单测（3/4）：DDPM ε-MSE 训练目标与随机反向采样。

运行：python -m unittest tests.test_phase4_ddpm -v
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from generative.models.schedule import DiffusionSchedule
from generative.models.unet1d import ConditionalUNet1D
from generative.models.ddpm import GaussianDiffusion

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T_SMALL = 50
B, L = 4, 256


class _HugeNet(nn.Module):
    """桩网络：恒预测巨大 ε̂（模拟欠训练时大 t 处失准，引爆反向均值）。"""

    def forward(self, x, t, c):
        return torch.full_like(x, 1e4)


class TestDDPM(unittest.TestCase):

    def _diff(self):
        net = ConditionalUNet1D(channels=(16, 32, 64), kernel=15,
                                time_dim=64).to(DEVICE)
        sch = DiffusionSchedule(T=T_SMALL, kind="linear").to(DEVICE)
        return GaussianDiffusion(net, sch)

    def test_loss_scalar_finite_backward(self):
        diff = self._diff()
        x0 = torch.randn(B, 1, L, device=DEVICE)
        c = torch.linspace(-1, 1, B, device=DEVICE)
        loss = diff.training_loss(x0, c)
        self.assertEqual(loss.shape, ())
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(any(p.grad is not None for p in diff.parameters()))

    def test_zero_network_loss_is_noise_variance(self):
        # ε̂≡0（head zero-init 的初始网络）时 MSE = E‖ε‖² = 1
        diff = self._diff()
        torch.manual_seed(0)
        x0 = torch.randn(512, 1, 64, device=DEVICE)
        c = torch.zeros(512, device=DEVICE)
        loss = diff.training_loss(x0, c)
        self.assertAlmostEqual(loss.item(), 1.0, delta=0.05)

    def test_sample_shape_and_randomness(self):
        diff = self._diff().eval()
        c = torch.linspace(-1, 1, B, device=DEVICE)
        torch.manual_seed(7)
        s1 = diff.sample(c, B, seq_len=L)
        self.assertEqual(s1.shape, (B, 1, L))
        self.assertTrue(torch.isfinite(s1).all())
        s2 = diff.sample(c, B, seq_len=L)  # 未固定种子 → 不同
        self.assertFalse(torch.allclose(s1, s2))
        torch.manual_seed(7)               # 同种子 → 可复现
        s1b = diff.sample(c, B, seq_len=L)
        self.assertTrue(torch.allclose(s1, s1b, atol=1e-5))

    def test_clipping_tames_exploding_epsilon(self):
        sch = DiffusionSchedule(T=T_SMALL, kind="linear").to(DEVICE)
        c = torch.zeros(B, device=DEVICE)
        torch.manual_seed(0)
        x_init = torch.randn(B, 1, 64, device=DEVICE)
        # 不裁剪：荒谬 ε̂ 经 β/√(1−ᾱ) 逐级放大，输出爆炸
        raw = GaussianDiffusion(_HugeNet().to(DEVICE), sch).sample(
            c, B, seq_len=64, x_init=x_init.clone(), clip_x0=None)
        self.assertGreater(raw.abs().max().item(), 1e3)
        # 安全兜底裁剪（+ε 一致性反推）：输出有限且量级受控
        torch.manual_seed(0)
        safe = GaussianDiffusion(_HugeNet().to(DEVICE), sch).sample(
            c, B, seq_len=64, x_init=x_init.clone(), clip_x0=30.0)
        self.assertTrue(torch.isfinite(safe).all())
        self.assertLess(safe.abs().max().item(), 100.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
