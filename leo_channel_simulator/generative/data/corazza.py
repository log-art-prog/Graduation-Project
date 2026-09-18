"""Corazza 参考模型（学长论文 2.4.5 节、3.2.1 节；原始出处 Corazza &
Vatalaro, IEEE TVT 43(3), 1994, DOI:10.1109/25.312773）。

Day 1：实现仰角参数化 corazza_params()。

单位约定（已据 1994 原文式(3) 裁决，详见 docs/notes/08）：
  - K     ：线性莱斯因子（功率比，非 dB）
  - mu    ：奈培(Np)，即 ln S 的均值
  - sigma ：dB，即 20log10(S) 的标准差（原文称 "dB spread"）
  原文式(3) 用 h = ln(10)/20 把两者放进同一个 ln 域分布：
      ln S ~ N(mu, (h*sigma)^2)
  即采样阴影时 S = exp(mu + h*sigma*z), z~N(0,1)。
注意：学长表 3-1 把 K0 印作 2.371，系抄录笔误；原文 Table I 为
2.731，且学长图 3-1 实际绘制曲线也对应 2.731。本实现以原文为准。
"""
from __future__ import annotations

import numpy as np

#: dB -> 奈培换算因子（幅度/电压域，20log10）。ln S 的标准差 = H * sigma_dB
H_DB_TO_NEPER = np.log(10.0) / 20.0  # ≈ 0.1151292546；逆换算 1 Np = 8.6858896 dB

# --- 原文 Table I 系数（rural tree-shadowed，L 频段；alpha 单位为度，20~80）---
_K_COEFF = (2.731, -1.074e-1, 2.774e-3)          # K(alpha)   二次，K 线性
_MU_COEFF = (-2.331, 1.142e-1, -1.939e-3, 1.094e-5)  # mu(alpha) 三次，单位 Np
_SIGMA_COEFF = (4.5, -0.05)                       # sigma(alpha) 一次，单位 dB

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
        K 为线性莱斯因子；mu 单位为奈培(Np)；sigma 单位为 dB。
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