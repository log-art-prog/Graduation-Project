"""
3GPP TR 38.811 参数表模块

包含 3GPP TR 38.811 V17.0.0 中定义的 LEO 卫星星地信道全部参数：
  - 环境场景定义 (Urban / Suburban / Open)
  - 大尺度参数 (LSP) 均值与标准差 (Table 6.6.2-1)
  - 阴影衰落标准差 (Table 6.4.2-1)
  - LoS 概率模型参数 (Section 6.6.1)
  - CDL 抽头模型配置 (Section 6.7)
  - 空间一致性相关距离参数 (Section 6.6.3)

所有角度扩展类参数在对数域 (log10) 中服从正态分布。
时延扩展 DS 在 log10(ns) 域服从正态分布。
K 因子在 dB 域服从正态分布。
"""
from dataclasses import dataclass, field
from typing import Dict, Tuple
import numpy as np


# ============================================================
# 环境场景定义
# ============================================================
@dataclass
class NTNEnvironment:
    """3GPP TR 38.811 NTN 环境场景"""
    name: str               # 英文名
    name_cn: str            # 中文名
    # 阴影衰落标准差 (dB) - 用于路径损耗的 SF
    sf_std_db: float
    # LoS 概率模型参数 (Sigmoid: P = 1 / (1 + a*exp(-b*(theta - c))))
    los_a: float
    los_b: float
    los_c: float
    # 绘图属性
    color: str
    linestyle: str


# 三种典型 NTN 场景
ENV_URBAN = NTNEnvironment(
    name='Urban',
    name_cn='Urban (城市)',
    sf_std_db=10.0,
    los_a=12.0,
    los_b=0.12,
    los_c=15.0,
    color='#e74c3c',
    linestyle='-.'
)

ENV_SUBURBAN = NTNEnvironment(
    name='Suburban',
    name_cn='Suburban (城郊)',
    sf_std_db=6.0,
    los_a=6.0,
    los_b=0.10,
    los_c=10.0,
    color='#3498db',
    linestyle='--'
)

ENV_OPEN = NTNEnvironment(
    name='Open',
    name_cn='Open (开阔地)',
    sf_std_db=2.0,
    los_a=0.5,
    los_b=0.08,
    los_c=5.0,
    color='#2ecc71',
    linestyle='-'
)

ALL_NTN_ENVIRONMENTS = [ENV_URBAN, ENV_SUBURBAN, ENV_OPEN]


# ============================================================
# 大尺度参数 (LSP) 表 - Table 6.6.2-1 (LEO 星地链路)
# ============================================================
# 参数说明 (与 3GPP TR 38.901/38.811 一致):
#   DS   : RMS 时延扩展，域为 log10(秒)，使用时转 ns (×1e9)
#   K    : 莱斯 K 因子，域为 dB (仅 LoS)
#   ASD  : 离开角扩展，域为 log10(度)
#   ASA  : 到达角扩展，域为 log10(度)
#   ZSD  : 离开天顶角扩展，域为 log10(度)
#   ZSA  : 到达天顶角扩展，域为 log10(度)
# 每个参数存储 (mu, sigma) 对，单位为对应域

@dataclass
class LSPTable:
    """大尺度参数表 (LoS 和 NLoS 分开)"""
    # LoS 场景
    los: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    # NLoS 场景
    nlos: Dict[str, Tuple[float, float]] = field(default_factory=dict)


# ---- Urban ----
URBAN_LSP = LSPTable(
    los={
        'DS':  (-7.44, 0.43),
        'K':   (9.0, 3.0),
        'ASD': (1.0, 0.40),
        'ASA': (1.0, 0.40),
        'ZSD': (0.5, 0.40),
        'ZSA': (0.5, 0.40),
    },
    nlos={
        'DS':  (-6.42, 0.37),
        'ASD': (1.5, 0.40),
        'ASA': (1.6, 0.40),
        'ZSD': (0.8, 0.40),
        'ZSA': (0.9, 0.40),
    }
)

# ---- Suburban ----
SUBURBAN_LSP = LSPTable(
    los={
        'DS':  (-7.44, 0.43),
        'K':   (9.0, 3.0),
        'ASD': (1.0, 0.40),
        'ASA': (0.9, 0.40),
        'ZSD': (0.5, 0.40),
        'ZSA': (0.5, 0.40),
    },
    nlos={
        'DS':  (-6.70, 0.37),
        'ASD': (1.3, 0.40),
        'ASA': (1.4, 0.40),
        'ZSD': (0.7, 0.40),
        'ZSA': (0.7, 0.40),
    }
)

# ---- Open ----
OPEN_LSP = LSPTable(
    los={
        'DS':  (-7.63, 0.43),
        'K':   (11.0, 3.0),
        'ASD': (0.8, 0.40),
        'ASA': (0.7, 0.40),
        'ZSD': (0.3, 0.40),
        'ZSA': (0.3, 0.40),
    },
    nlos={
        'DS':  (-7.20, 0.37),
        'ASD': (1.0, 0.40),
        'ASA': (1.1, 0.40),
        'ZSD': (0.5, 0.40),
        'ZSA': (0.5, 0.40),
    }
)

# 环境到 LSP 表的映射
ENV_LSP_MAP = {
    ENV_URBAN.name: URBAN_LSP,
    ENV_SUBURBAN.name: SUBURBAN_LSP,
    ENV_OPEN.name: OPEN_LSP,
}


# ============================================================
# 空间一致性相关距离参数 - Section 6.6.3
# ============================================================
# 各 LSP 的去相关距离 (m)，用于空间一致性滤波
# 卫星场景下，相关距离远大于地面场景 (卫星移动快，地面用户慢)
DET_CORRELATION_DISTANCE = {
    'DS':  50.0,    # m
    'K':   40.0,    # m
    'ASD': 30.0,    # m
    'ASA': 30.0,    # m
    'ZSD': 20.0,    # m
    'ZSA': 20.0,    # m
    'SF':  100.0,   # m (阴影衰落去相关距离)
}


# ============================================================
# CDL 抽头模型参数 - Section 6.7
# ============================================================
@dataclass
class CDLConfig:
    """CDL (Clustered Delay Line) 模型配置"""
    num_clusters: int           # 簇数量
    num_rays_per_cluster: int   # 每簇子径数
    # 簇间延迟比例 (相对于 RMS 时延扩展)
    delay_scaling: float = 1.0
    # 簇功率衰减常数 (指数分布)
    power_decay: float = 1.0


# 3GPP TR 38.811 卫星场景 CDL 配置
CDL_CONFIG_NTN = CDLConfig(
    num_clusters=8,
    num_rays_per_cluster=20,
    delay_scaling=1.0,
    power_decay=1.0
)


# ============================================================
# Loo 模型参数 - Section 6.5.3
# ============================================================
# Loo 模型: 直射径受对数正态阴影衰落，散射径为瑞利
# r^2 = (10^(xi/20) * sqrt(K/(K+1)))^2 + (sqrt(1/(K+1)) * s)^2
# 其中 xi 是阴影衰落 (dB), s 是瑞利散射分量
LOO_MODEL_PARAMS = {
    # 直射径阴影衰落标准差 (dB)，随仰角变化
    'shadow_std_min': 0.5,    # 高仰角
    'shadow_std_max': 4.0,    # 低仰角
}


# ============================================================
# 路径损耗参数 - Section 6.4
# ============================================================
@dataclass
class PathLossParams:
    """3GPP 路径损耗参数"""
    # 自由空间路径损耗系数: PL = A + 10*alpha*log10(d) + 20*log10(f)
    a_coeff: float = 32.45    # dB
    alpha: float = 2.0        # 路径损耗指数


PATH_LOSS_NTN = PathLossParams(a_coeff=32.45, alpha=2.0)


# ============================================================
# 辅助函数
# ============================================================
def get_lsp_table(env: NTNEnvironment) -> LSPTable:
    """根据环境获取对应的 LSP 表"""
    return ENV_LSP_MAP[env.name]


def get_los_params(env: NTNEnvironment) -> Tuple[float, float, float]:
    """获取 LoS 概率模型参数 (a, b, c)"""
    return env.los_a, env.los_b, env.los_c


def get_sf_std(env: NTNEnvironment) -> float:
    """获取阴影衰落标准差 (dB)"""
    return env.sf_std_db
