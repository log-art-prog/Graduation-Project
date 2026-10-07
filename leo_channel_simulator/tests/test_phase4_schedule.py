"""阶段四单测（1/4）：扩散调度表与前向扩散 q_sample。

运行：python -m unittest tests.test_phase4_schedule -v
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from generative.models.schedule import DiffusionSchedule

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T_SMALL = 50
B, L = 4, 256


class TestSchedule(unittest.TestCase):

    def test_linear_endpoints_and_length(self):
        sch = DiffusionSchedule(T=T_SMALL, kind="linear")
        self.assertEqual(sch.betas.shape, (T_SMALL,))
        self.assertAlmostEqual(sch.betas[0].item(), 1e-4, places=7)
        self.assertAlmostEqual(sch.betas[-1].item(), 0.02, places=7)

    def test_alpha_bar_monotone_and_bounds(self):
        for kind in ("linear", "cosine"):
            with self.subTest(kind=kind):
                sch = DiffusionSchedule(T=T_SMALL, kind=kind)
                ab = sch.alpha_bar
                self.assertEqual(ab.shape, (T_SMALL,))
                self.assertTrue(torch.all(ab[1:] < ab[:-1]),
                                "alpha_bar 必须严格单调递减")
                self.assertTrue(torch.all(ab > 0) and torch.all(ab <= 1))
                self.assertGreater(ab[-1].item(), 1e-8)
        # 正式 T=1000 时两种调度末端信号都应接近消失（ᾱ ≲ 1e-3）
        for kind in ("linear", "cosine"):
            with self.subTest(kind=kind, T=1000):
                self.assertLess(
                    DiffusionSchedule(T=1000, kind=kind).alpha_bar[-1].item(),
                    1e-3)

    def test_cosine_matches_reference_formula(self):
        T, s = T_SMALL, 0.008
        sch = DiffusionSchedule(T=T, kind="cosine", s=s)
        x = torch.linspace(0, T, T + 1, dtype=torch.float64)
        f = torch.cos(((x / T) + s) / (1 + s) * torch.pi / 2) ** 2
        ref_betas = torch.clamp(1 - f[1:] / f[:-1], max=0.999)
        self.assertTrue(torch.allclose(sch.betas.double(), ref_betas, atol=1e-6))

    def test_q_sample_closed_form_mixed_t(self):
        sch = DiffusionSchedule(T=T_SMALL, kind="linear").to(DEVICE)
        torch.manual_seed(0)
        x0 = torch.randn(B, 1, L, device=DEVICE)
        noise = torch.randn_like(x0)
        t = torch.tensor([0, 7, 23, T_SMALL - 1], device=DEVICE)
        xt = sch.q_sample(x0, t, noise=noise)
        ab = sch.alpha_bar[t].view(B, 1, 1)
        expect = ab.sqrt() * x0 + (1 - ab).sqrt() * noise
        self.assertTrue(torch.allclose(xt, expect, atol=1e-6))

    def test_q_sample_extreme_t(self):
        # cosine 在 T=50 末端 ᾱ≈1.5e-4：t=0 无噪声≈原信号，t=T−1 无噪声≈零
        sch = DiffusionSchedule(T=T_SMALL, kind="cosine").to(DEVICE)
        torch.manual_seed(1)
        x0 = torch.randn(B, 1, L, device=DEVICE)
        zero_noise = torch.zeros_like(x0)
        near_x0 = sch.q_sample(x0, torch.zeros(B, dtype=torch.long, device=DEVICE),
                               noise=zero_noise)
        self.assertTrue(torch.allclose(near_x0, x0, rtol=2e-3))
        near_zero = sch.q_sample(x0,
                                 torch.full((B,), T_SMALL - 1, dtype=torch.long,
                                            device=DEVICE),
                                 noise=zero_noise)
        self.assertLess(near_zero.abs().mean().item(), 0.02)
        # 无噪声时能量必须随 t 单调衰减（linear T=50 末端仍有较强信号）
        sch_lin = DiffusionSchedule(T=T_SMALL, kind="linear").to(DEVICE)
        e_start = sch_lin.q_sample(
            x0, torch.zeros(B, dtype=torch.long, device=DEVICE),
            noise=zero_noise).square().mean()
        e_end = sch_lin.q_sample(
            x0, torch.full((B,), T_SMALL - 1, dtype=torch.long, device=DEVICE),
            noise=zero_noise).square().mean()
        self.assertLess(e_end, e_start)

    def test_buffers_registered_and_move(self):
        sch = DiffusionSchedule(T=T_SMALL, kind="linear").to(DEVICE)
        self.assertEqual(sch.betas.device.type, DEVICE.type)
        self.assertEqual(sch.T, T_SMALL)


if __name__ == "__main__":
    unittest.main(verbosity=2)
