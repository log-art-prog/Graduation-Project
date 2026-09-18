"""Corazza 参考模型（论文 2.4.5 节、3.2.1 节）。

Day 1：只实现表 3-1 的仰角参数化 corazza_params()。
注意：K / mu / sigma 的单位（dB 还是 ln 奈培）论文未写明，
单位核对见 docs/notes/08 第 6 节。本函数只返回表 3-1 的原始数值，
单位换算留到 Day 2 的采样函数里处理。
"""
from __future__ import annotations

import numpy as np

# --- 表 3-1 系数（农村环境，L 频段；alpha 单位为度，拟合范围 20~80）---
_K_COEFF = (2.371, -1.074e-1, 2.774e-3)          # K(alpha)   二次
_MU_COEFF = (-2.331, 1.142e-1, -1.939e-3, 1.094e-5)  # mu(alpha) 三次
_SIGMA_COEFF = (4.5, -0.05)                       # sigma(alpha) 一次

ALPHA_MIN_DEG = 20.0
ALPHA_MAX_DEG = 80.0

def corazza_params(alpha_deg, check_range: bool = True):
    """按论文表 3-1 返回给定仰角下的 (K, mu, sigma) 原始表值。

    参数
    ----
    alpha_deg : float 或 array_like
        仰角，单位度。标量或 numpy 数组均可（向量化计算）。
    check_range : bool
        超出 20~80° 时是否报错。多项式是经验拟合，禁止外推。

    返回
    ----
    (K, mu, sigma)：float 或 np.ndarray，形状与输入一致。
    """
    alpha = np.asarray(alpha_deg, dtype=float)

    if check_range and (np.any(alpha < ALPHA_MIN_DEG)
                        or np.any(alpha > ALPHA_MAX_DEG)):
        raise ValueError(
            f"仰角超出表3-1拟合范围 [{ALPHA_MIN_DEG}, {ALPHA_MAX_DEG}] 度，禁止外推"
        )

    K = _K_COEFF[0] + _K_COEFF[1] * alpha + _K_COEFF[2] * alpha ** 2
    mu = (_MU_COEFF[0] + _MU_COEFF[1] * alpha
          + _MU_COEFF[2] * alpha ** 2 + _MU_COEFF[3] * alpha ** 3)
    sigma = _SIGMA_COEFF[0] + _SIGMA_COEFF[1] * alpha
    return K, mu, sigma