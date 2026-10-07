"""阶段三修复单测：条件单调一致性正则（方案 R）。

物理先验：同一噪声场 z 下，仰角越高阴影越弱，生成包络的样本均值
应随归一化仰角 c 非降。monotone_condition_penalty 对所有逆序条件对
施加铰链惩罚 ReLU(margin - (m_hi - m_lo))，修复 WGAN-GP 跨条件
均值漂移（seed42/seed1234 双种子实证：相邻仰角均值互相拖拽）。

运行：python -m unittest tests.test_phase3_mono -v
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from generative.models.cgan import (
    SequenceNoiseFilterGenerator, NOISE_DIM, SEQ_LEN,
    monotone_condition_penalty,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class _MonoStub(nn.Module):
    """输出均值严格随 c 递增：x = c（广播到全序列），无惩罚。"""

    def forward(self, z, c):
        return c.view(-1, 1, 1).expand(-1, 1, z.shape[-1])


class _AntiMonoStub(nn.Module):
    """输出均值严格随 c 递减：x = -c，每个条件对都逆序。"""

    def forward(self, z, c):
        return -c.view(-1, 1, 1).expand(-1, 1, z.shape[-1])


class TestMonotonePenalty(unittest.TestCase):

    def test_scalar_finite(self):
        z = torch.randn(8, NOISE_DIM, SEQ_LEN, device=DEVICE)
        pen = monotone_condition_penalty(_MonoStub().to(DEVICE), z, n_grid=4)
        self.assertEqual(pen.shape, ())
        self.assertTrue(torch.isfinite(pen))

    def test_monotone_generator_zero_penalty(self):
        z = torch.randn(8, NOISE_DIM, SEQ_LEN, device=DEVICE)
        pen = monotone_condition_penalty(_MonoStub().to(DEVICE), z, n_grid=4)
        self.assertEqual(pen.item(), 0.0)

    def test_antimonotone_generator_penalized(self):
        # K=4 共 6 对，全部逆序：m_hi-m_lo = -(c_hi-c_lo)
        # 相邻对差 -2/3，惩罚值应等于各对平均缺口（正且与手算一致）
        z = torch.randn(8, NOISE_DIM, SEQ_LEN, device=DEVICE)
        pen = monotone_condition_penalty(_AntiMonoStub().to(DEVICE), z,
                                         n_grid=4, margin=0.0)
        conds = torch.linspace(-1.0, 1.0, 4)
        gaps = []
        for i in range(4):
            for j in range(i + 1, 4):
                gaps.append(float(conds[j] - conds[i]))
        self.assertAlmostEqual(pen.item(), sum(gaps) / len(gaps), places=5)
        self.assertGreater(pen.item(), 0.5)

    def test_margin_enlarges_penalty(self):
        z = torch.randn(8, NOISE_DIM, SEQ_LEN, device=DEVICE)
        p0 = monotone_condition_penalty(_AntiMonoStub().to(DEVICE), z,
                                        n_grid=4, margin=0.0)
        pm = monotone_condition_penalty(_AntiMonoStub().to(DEVICE), z,
                                        n_grid=4, margin=0.2)
        self.assertGreater(pm.item(), p0.item())

    def test_grad_flows_to_generator(self):
        G = SequenceNoiseFilterGenerator().to(DEVICE)
        z = torch.randn(2, NOISE_DIM, SEQ_LEN, device=DEVICE)
        # 大 margin 强制铰链激活（随机初始化的 G 可能恰好不逆序），
        # 这里只验证计算图：激活时梯度必须回传到 G
        pen = monotone_condition_penalty(G, z, n_grid=3, margin=100.0)
        self.assertGreater(pen.item(), 0.0)
        pen.backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                            for p in G.parameters()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
