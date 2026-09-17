"""
LoS/NLoS 概率模块 - 3GPP TR 38.811 Section 6.6.1

基于卫星仰角计算视距 (Line-of-Sight) 概率。
不同环境场景采用 Sigmoid 模型：
    P_LoS(theta) = 1 / (1 + a * exp(-b * (theta - c)))

其中 theta 为仰角 (度)，a/b/c 为环境相关参数。

特性：
  - 仰角越低，障碍物遮挡概率越大，P_LoS 越低
  - 仰角越高，P_LoS 趋近于 1
  - 城市环境遮挡最严重，开阔地几乎始终 LoS
"""
import numpy as np
from .parameters import (
    NTNEnvironment, get_los_params,
    ENV_URBAN, ENV_SUBURBAN, ENV_OPEN,
    ALL_NTN_ENVIRONMENTS
)


def calculate_los_probability(elevation_deg: float, env: NTNEnvironment) -> float:
    """
    计算指定仰角和环境下的 LoS 概率

    Parameters
    ----------
    elevation_deg : float
        卫星仰角 (度)，范围 0~90
    env : NTNEnvironment
        环境场景

    Returns
    -------
    float
        LoS 概率 (0~1)
    """
    a, b, c = get_los_params(env)
    theta = np.clip(elevation_deg, 0, 90)

    # Sigmoid 模型
    p_los = 1.0 / (1.0 + a * np.exp(-b * (theta - c)))

    # 高仰角 (>80°) 时强制为 1 (过顶附近几乎无遮挡)
    if theta > 80:
        p_los = 1.0

    return float(np.clip(p_los, 0.0, 1.0))


def determine_los_state(elevation_deg: float, env: NTNEnvironment,
                        rng: np.random.Generator = None) -> bool:
    """
    根据 LoS 概率随机判断当前是否为 LoS 状态

    Parameters
    ----------
    elevation_deg : float
        卫星仰角 (度)
    env : NTNEnvironment
        环境场景
    rng : np.random.Generator, optional
        随机数生成器

    Returns
    -------
    bool
        True 表示 LoS, False 表示 NLoS
    """
    if rng is None:
        rng = np.random.default_rng()

    p_los = calculate_los_probability(elevation_deg, env)
    return rng.random() < p_los


def calculate_los_probability_array(elevation_deg_array: np.ndarray,
                                    env: NTNEnvironment) -> np.ndarray:
    """
    批量计算仰角序列对应的 LoS 概率

    Parameters
    ----------
    elevation_deg_array : np.ndarray
        仰角序列 (度)
    env : NTNEnvironment
        环境场景

    Returns
    -------
    np.ndarray
        LoS 概率序列
    """
    a, b, c = get_los_params(env)
    theta = np.clip(elevation_deg_array, 0, 90)

    p_los = 1.0 / (1.0 + a * np.exp(-b * (theta - c)))
    p_los = np.where(theta > 80, 1.0, p_los)

    return np.clip(p_los, 0.0, 1.0)


def determine_los_state_array(elevation_deg_array: np.ndarray,
                              env: NTNEnvironment,
                              rng: np.random.Generator = None) -> np.ndarray:
    """
    批量生成 LoS/NLoS 状态序列 (带时间相关性)

    使用一阶 Markov 链保持状态的时间连续性，避免频繁跳变。

    Parameters
    ----------
    elevation_deg_array : np.ndarray
        仰角序列
    env : NTNEnvironment
        环境场景
    rng : np.random.Generator, optional
        随机数生成器

    Returns
    -------
    np.ndarray
        布尔数组，True=LoS, False=NLoS
    """
    if rng is None:
        rng = np.random.default_rng()

    p_los_array = calculate_los_probability_array(elevation_deg_array, env)
    num_points = len(p_los_array)

    los_state = np.zeros(num_points, dtype=bool)

    # 初始化第一个状态
    los_state[0] = rng.random() < p_los_array[0]

    # 状态转移概率 (保持当前状态的概率)
    # 高相关：0.95 概率保持 LoS，0.9 概率保持 NLoS
    p_stay_los = 0.95
    p_stay_nlos = 0.90

    for i in range(1, num_points):
        if los_state[i - 1]:
            # 当前是 LoS
            if rng.random() < p_stay_los:
                los_state[i] = True
            else:
                los_state[i] = rng.random() < p_los_array[i]
        else:
            # 当前是 NLoS
            if rng.random() < p_stay_nlos:
                los_state[i] = False
            else:
                los_state[i] = rng.random() < p_los_array[i]

    return los_state


# ============================================================
# 示例与验证
# ============================================================
if __name__ == '__main__':
    print("=" * 70)
    print("3GPP TR 38.811 LoS 概率模型验证")
    print("=" * 70)

    elevations = [0, 10, 20, 30, 45, 60, 80, 90]

    print(f"\n{'仰角(°)':>10}", end='')
    for env in ALL_NTN_ENVIRONMENTS:
        print(f"{env.name_cn:>18}", end='')
    print()

    for elev in elevations:
        print(f"{elev:>10}", end='')
        for env in ALL_NTN_ENVIRONMENTS:
            p = calculate_los_probability(elev, env)
            print(f"{p:>18.4f}", end='')
        print()

    print("\n说明：仰角越高，LoS 概率越大；城市遮挡最严重。")
