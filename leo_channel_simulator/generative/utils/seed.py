"""统一随机种子管理：阶段1数据集复现、阶段5多种子实验共用。"""
from __future__ import annotations

import random

import numpy as np
import torch

def make_rng(seed: int | None = 42) -> np.random.Generator:
    """创建 numpy 新版随机数生成器（数据生成阶段用）。

    沿用旧物理仿真器改造时的约定：优先用 np.random.Generator，
    而不是已过时的 np.random.seed 全局接口。
    """
    return np.random.default_rng(seed)

def seed_everything(seed: int = 42) -> np.random.Generator:
    """一次性固定 numpy / Python random / torch（含 CUDA），训练阶段用。

    cudnn.deterministic + benchmark=False 保证 GPU 卷积结果逐次可复现，
    代价是略慢——做科学对比时这个代价值得。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return np.random.default_rng(seed)