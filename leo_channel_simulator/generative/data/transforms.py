"""Corazza 数据预处理：包络 log 标准化 + 仰角归一化。

对应学长论文式(3-7)、式(3-8)。所有变换逐元素（向量化），可接受
标量、一维数组或多维数组。

用途：
  - 训练前把原始包络 r 压到零均值单位方差，便于神经网络学习；
  - 生成后用 inverse_standardize 还原成物理包络做评价指标。
  - 注意 mu_log / sigma_log 必须由【全训练集】一次性估计并固定，
    所有样本共用，不能每个 batch 重算（Day 4 固化进数据集）。
"""
from __future__ import annotations

import numpy as np

#: 防 log(0) 的小常数（学长论文表 3-2）
EPS = 1e-6


def fit_standardizer(samples: np.ndarray, eps: float = EPS):
    """从一批样本估计 log 域的均值和标准差。

    参数
    ----
    samples ：原始包络数组（任意形状，通常是 N×length 的训练集）
    eps     ：防 log(0) 的小常数

    返回
    ----
    (mu_log, sigma_log)：log(samples + eps) 的均值和标准差（两个标量）。
    这两个量是【全训练集统一】的，所有样本共用，不能每批重算。
    """
    log_vals = np.log(samples + eps)
    return log_vals.mean(), log_vals.std()


def standardize(r, mu_log: float, sigma_log: float, eps: float = EPS):
    """把原始包络 r 标准化到零均值单位方差（学长式 3-8）。

    x = ( log(r + eps) - mu_log ) / sigma_log
    """
    return (np.log(r + eps) - mu_log) / sigma_log


def inverse_standardize(x, mu_log: float, sigma_log: float, eps: float = EPS):
    """standardize 的严格逆变换：把标准化结果 x 还原成原始包络 r。

    r = exp( x * sigma_log + mu_log ) - eps
    """
    return np.exp(x * sigma_log + mu_log) - eps


def normalize_alpha(alpha_deg):
    """仰角归一化到 [-1, 1]：(alpha - 50) / 30（学长式 3-7）。

    仰角范围 20~80 deg，50 是中点、30 是半宽，映射后正好 [-1, 1]。
    """
    return (alpha_deg - 50.0) / 30.0


def denormalize_alpha(alpha_norm):
    """normalize_alpha 的逆变换，把 [-1,1] 还原成角度。"""
    return alpha_norm * 30.0 + 50.0
