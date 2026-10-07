"""阶段四单测（5/5）：采样接口 generate_envelopes 的物理域契约。

逆标准化 r = exp(x·σ+μ) − eps 在 x 极负（数值裁剪尾部）时会给出
-eps 量级的伪负值；物理包络按定义必须非负，采样接口负责把这点
数值伪影投影回 0，避免下游 LCR/AFD/BER 指标吃到负包络。

运行：python -m unittest tests.test_phase4_sample -v
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from generative.data.dataset import load_dataset
from generative.sample_ddpm import generate_envelopes


class _StubDiffusion:
    """只实现 generate_envelopes 在 ddpm 分支所需的 sample，回放固定样本。"""

    def __init__(self, x_std: torch.Tensor):
        self._x = x_std

    def sample(self, c, n, seq_len=1000, x_init=None, clip_x0=30.0):
        return self._x


class TestPhysicalProjection(unittest.TestCase):

    def _meta(self):
        d = load_dataset(ROOT / "datasets" / "corazza_train.npz")
        return {
            "mu_log": float(d["mu_log"]),
            "sigma_log": float(d["sigma_log"]),
            "eps": float(d["eps"]),
            "device": torch.device("cpu"),
            "seq_len": 4,
        }

    def test_envelopes_are_nonnegative(self):
        meta = self._meta()
        # -30 经逆标准化必为负（约 -8e-7 量级伪影）；-16.75 是真实数据最小值
        x = torch.tensor([[[-30.0, -16.75, 0.0, 3.45]]])
        r = generate_envelopes(_StubDiffusion(x), meta, 45.0, 1,
                               sampler="ddpm", seed=0)
        self.assertEqual(r.shape, (1, 4))
        self.assertTrue((r >= 0.0).all(), "物理包络必须处处非负")
        self.assertEqual(r[0, 0], 0.0, "伪负值应投影为 0")

    def test_in_range_points_keep_strict_inverse(self):
        # 真实范围内的点不得被非负投影改动（投影只作用于伪负值）
        meta = self._meta()
        x_val = -2.0
        expect = np.exp(x_val * meta["sigma_log"] + meta["mu_log"]) - meta["eps"]
        self.assertGreater(expect, 0.0)
        x = torch.tensor([[[x_val]]])
        r = generate_envelopes(_StubDiffusion(x), meta, 45.0, 1,
                               sampler="ddpm", seed=0)
        self.assertAlmostEqual(float(r[0, 0]), expect, places=8)


if __name__ == "__main__":
    unittest.main(verbosity=2)
