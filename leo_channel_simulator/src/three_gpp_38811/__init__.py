"""
3GPP TR 38.811 NTN (Non-Terrestrial Network) 星地信道模型包

完整实现 3GPP TR 38.811 V17.0.0 中定义的低轨卫星 (LEO) 星地信道模型，包括：
  - 基于仰角的 LoS/NLoS 概率 (Section 6.6.1)
  - 大尺度参数 (LSP) 生成：时延扩展、莱斯K因子、角度扩展 (Section 6.6.2)
  - 空间一致性相关模型 (Spatial Consistency, Section 6.6.3)
  - CDL 抽头延迟线小尺度模型 (Section 6.7)
  - Loo 模型：阴影衰落与多径联合分布
  - 路径损耗与阴影衰落 (Section 6.4)

参考标准: 3GPP TR 38.811 "Study on New Radio (NR) to support non-terrestrial networks"
"""
from . import parameters, los_probability, large_scale, small_scale, channel_model

__all__ = [
    'parameters',
    'los_probability',
    'large_scale',
    'small_scale',
    'channel_model',
]
