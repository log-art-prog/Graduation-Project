"""阶段四单测（4/4）：DDIM η=0 确定性采样与多档跳步。

运行：python -m unittest tests.test_phase4_ddim -v
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
from generative.models.ddim import ddim_sample

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T_SMALL = 50
B, L = 4, 256


class _ZeroNet(nn.Module):
    """桩网络：恒预测 ε̂=0。"""

    def forward(self, x, t, c):
        return torch.zeros_like(x)


class _HugeNet(nn.Module):
    """桩网络：恒预测巨大 ε̂（模拟欠训练时大 t 处的失准预测，引爆 √ᾱ 除法）。"""

    def forward(self, x, t, c):
        return torch.full_like(x, 1e4)


class TestDDIM(unittest.TestCase):

    def _net(self):
        return ConditionalUNet1D(channels=(16, 32, 64), kernel=15,
                                 time_dim=64).to(DEVICE).eval()

    def test_deterministic_given_same_start(self):
        net = self._net()
        sch = DiffusionSchedule(T=T_SMALL, kind="linear").to(DEVICE)
        c = torch.zeros(B, device=DEVICE)
        xT = torch.randn(B, 1, L, device=DEVICE)
        with torch.no_grad():
            a = ddim_sample(net, sch, c, xT=xT, num_steps=10, eta=0.0)
            b = ddim_sample(net, sch, c, xT=xT, num_steps=10, eta=0.0)
        self.assertTrue(torch.allclose(a, b, atol=1e-6),
                        "η=0 的 DDIM 必须确定性可复现")

    def test_step_counts_and_bounds(self):
        net = self._net()
        sch = DiffusionSchedule(T=T_SMALL, kind="cosine").to(DEVICE)
        c = torch.linspace(-1, 1, B, device=DEVICE)
        for s in (5, 10, 25, 50):
            with self.subTest(steps=s):
                torch.manual_seed(3)
                xT = torch.randn(B, 1, L, device=DEVICE)
                with torch.no_grad():
                    out = ddim_sample(net, sch, c, xT=xT, num_steps=s, eta=0.0)
                self.assertEqual(out.shape, (B, 1, L))
                self.assertTrue(torch.isfinite(out).all())

    def test_rejects_steps_exceeding_T(self):
        net = self._net()
        sch = DiffusionSchedule(T=T_SMALL).to(DEVICE)
        with self.assertRaises(ValueError):
            ddim_sample(net, sch, torch.zeros(B, device=DEVICE),
                        xT=torch.randn(B, 1, L, device=DEVICE),
                        num_steps=T_SMALL + 1)

    def test_one_shot_zero_net_closed_form(self):
        # 单步 + ε̂=0：x = x0̂ = x_T/√ᾱ_T（终点 ab_prev 取 1.0）
        sch = DiffusionSchedule(T=T_SMALL, kind="linear").to(DEVICE)
        torch.manual_seed(0)
        xT = torch.randn(B, 1, L, device=DEVICE)
        with torch.no_grad():
            out = ddim_sample(_ZeroNet().to(DEVICE), sch,
                              torch.zeros(B, device=DEVICE),
                              xT=xT, num_steps=1, eta=0.0)
        expect = xT / sch.alpha_bar[T_SMALL - 1].sqrt()
        self.assertTrue(torch.allclose(out, expect, atol=1e-5))

    def test_full_steps_eta0_zero_net_matches_one_shot(self):
        # η=0 零网络：多步链的总系数累乘 = 1/√ᾱ_{T-1}，与单步直达一致
        sch = DiffusionSchedule(T=T_SMALL, kind="linear").to(DEVICE)
        torch.manual_seed(0)
        xT = torch.randn(B, 1, L, device=DEVICE)
        with torch.no_grad():
            full = ddim_sample(_ZeroNet().to(DEVICE), sch,
                               torch.zeros(B, device=DEVICE),
                               xT=xT.clone(), num_steps=T_SMALL, eta=0.0)
            one = ddim_sample(_ZeroNet().to(DEVICE), sch,
                              torch.zeros(B, device=DEVICE),
                              xT=xT.clone(), num_steps=1, eta=0.0)
        expect = xT / sch.alpha_bar[T_SMALL - 1].sqrt()
        self.assertTrue(torch.allclose(full, expect, atol=1e-4))
        self.assertTrue(torch.allclose(one, expect, atol=1e-5))


    def test_clipping_tames_exploding_epsilon(self):
        sch = DiffusionSchedule(T=T_SMALL, kind="cosine").to(DEVICE)
        c = torch.zeros(B, device=DEVICE)
        torch.manual_seed(0)
        xT = torch.randn(B, 1, L, device=DEVICE)
        net = _HugeNet().to(DEVICE)
        # 不裁剪：荒谬 ε̂ 经 √ᾱ 除法放大，输出爆炸
        with torch.no_grad():
            raw = ddim_sample(net, sch, c, xT=xT.clone(), num_steps=10,
                              eta=0.0, clip_x0=None)
        self.assertGreater(raw.abs().max().item(), 1e3)
        # 安全兜底裁剪（且裁剪后 ε 一致性反推）：输出有限且量级受控
        with torch.no_grad():
            safe = ddim_sample(net, sch, c, xT=xT.clone(), num_steps=10,
                               eta=0.0, clip_x0=30.0)
        self.assertTrue(torch.isfinite(safe).all())
        self.assertLess(safe.abs().max().item(), 100.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
