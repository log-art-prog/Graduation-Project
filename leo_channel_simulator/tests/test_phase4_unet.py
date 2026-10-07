"""阶段四单测（2/4）：条件 1D U-Net — 形状保持 / FiLM / 时间注入 / 反传。

运行：python -m unittest tests.test_phase4_unet -v
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from generative.models.unet1d import ConditionalUNet1D

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T_SMALL = 50
B, L = 4, 256


def _wake_zero_inits(net: ConditionalUNet1D) -> None:
    """解除 zero-init（head + 各 FiLM 投影），用于检验条件通路本身是否连通。"""
    for name, mod in net.named_modules():
        if isinstance(mod, nn.Linear) and "film" in name:
            nn.init.normal_(mod.weight, std=0.1)
            nn.init.zeros_(mod.bias)
    nn.init.kaiming_uniform_(net.head[-1].weight)
    nn.init.zeros_(net.head[-1].bias)


class TestUNet(unittest.TestCase):

    def _net(self):
        return ConditionalUNet1D(channels=(16, 32, 64), kernel=15,
                                 time_dim=64).to(DEVICE)

    def test_output_shape_preserved(self):
        net = self._net()
        x = torch.randn(B, 1, L, device=DEVICE)
        t = torch.randint(0, T_SMALL, (B,), device=DEVICE)
        c = torch.linspace(-1, 1, B, device=DEVICE)
        out = net(x, t, c)
        self.assertEqual(out.shape, (B, 1, L))
        self.assertTrue(torch.isfinite(out).all())

    def test_nominal_length_1000(self):
        # 正式序列 1000（奇数瓶颈 125），上采样必须精确还原
        net = self._net()
        x = torch.randn(2, 1, 1000, device=DEVICE)
        t = torch.randint(0, T_SMALL, (2,), device=DEVICE)
        c = torch.tensor([-1.0, 1.0], device=DEVICE)
        self.assertEqual(net(x, t, c).shape, (2, 1, 1000))

    def test_film_changes_output_with_alpha(self):
        # 输出头 zero-init 会让初始输出恒为 0，这里先把 head 重新随机化，
        # 专门检验 FiLM 条件通路：同 x/t、不同仰角必须给出不同结果
        net = self._net().eval()
        _wake_zero_inits(net)
        x = torch.randn(B, 1, L, device=DEVICE)
        t = torch.full((B,), 10, device=DEVICE)
        with torch.no_grad():
            o_lo = net(x, t, torch.full((B,), -1.0, device=DEVICE))
            o_hi = net(x, t, torch.full((B,), 1.0, device=DEVICE))
        self.assertGreater(o_lo.abs().max().item(), 1e-4, "head 随机化失败")
        self.assertFalse(torch.allclose(o_lo, o_hi, atol=1e-5),
                         "FiLM 失效：不同仰角输出完全相同")

    def test_time_embedding_changes_output(self):
        net = self._net().eval()
        _wake_zero_inits(net)
        x = torch.randn(B, 1, L, device=DEVICE)
        c = torch.zeros(B, device=DEVICE)
        with torch.no_grad():
            o0 = net(x, torch.zeros(B, dtype=torch.long, device=DEVICE), c)
            o1 = net(x, torch.full((B,), T_SMALL - 1, dtype=torch.long,
                                   device=DEVICE), c)
        self.assertFalse(torch.allclose(o0, o1, atol=1e-5),
                         "时间注入失效：不同 t 输出完全相同")

    def test_zero_initialized_head_starts_near_zero(self):
        # 输出头 zero-init：训练第 0 步 ε̂≈0，稳定起步
        net = self._net().eval()
        x = torch.randn(B, 1, L, device=DEVICE)
        with torch.no_grad():
            out = net(x, torch.zeros(B, dtype=torch.long, device=DEVICE),
                      torch.zeros(B, device=DEVICE))
        self.assertLess(out.abs().max().item(), 1e-5)

    def test_backward_runs(self):
        net = self._net()
        x = torch.randn(B, 1, L, device=DEVICE)
        t = torch.randint(0, T_SMALL, (B,), device=DEVICE)
        c = torch.linspace(-1, 1, B, device=DEVICE)
        loss = net(x, t, c).square().mean()
        loss.backward()
        grads = [p.grad for p in net.parameters() if p.grad is not None]
        self.assertTrue(len(grads) > 0)
        self.assertTrue(all(torch.isfinite(g).all() for g in grads))


if __name__ == "__main__":
    unittest.main(verbosity=2)
