"""Corazza 参考模型（学长论文 2.4.5 节、3.2.1 节；原始出处 Corazza &
Vatalaro, IEEE TVT 43(3), 1994, DOI:10.1109/25.312773）。

Day 1：仰角参数化 corazza_params()。
Day 2：i.i.d. 采样器 sample_rice_iid / sample_shadow_iid / sample_corazza_iid。

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
from scipy.stats import ncx2, norm
from scipy.integrate import quad

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


def sample_rice_iid(n: int, K_lin: float,
                    rng: np.random.Generator) -> np.ndarray:
    """采样 n 个归一化 Rice 包络 R（E[R^2]=1，Corazza 原文式(5)）。

    物理图景：复基带信号 Z = (m + s_R*z1) + j*(s_R*z2)，
    实部 = 直射分量 m + 随机多径抖动；虚部只有多径抖动；
    包络 R = |Z| 就是复平面上该点到原点的距离。

    参数
    ----
    n      ：样本个数
    K_lin  ：线性莱斯因子（不是 dB）= 直射功率/散射功率
    rng    ：numpy 随机数生成器（由 make_rng 产生）

    返回
    ----
    长度 n 的一维数组，每个元素是一个 Rice 分布的包络值（>=0）。

    备注
    ----
    m、s_R 由两个条件联立解出：
      (1) K = m^2/(2*s_R^2)        莱斯因子定义
      (2) m^2 + 2*s_R^2 = 1        归一化 E[R^2]=1
    K=0 时 m=0，自动退化为 Rayleigh 分布，无需特判。
    """
    # 散射分量 I/Q 每路的标准差（两路散射功率合计 2*s_R^2）
    s_R = np.sqrt(1.0 / (2.0 * (K_lin + 1.0)))
    # 直射分量幅度（沿复平面实轴的固定偏移量）
    m = np.sqrt(K_lin / (K_lin + 1.0))

    # 一次生成 2 行 n 列标准正态数，按行拆成相互独立的 I/Q 两路
    z = rng.standard_normal((2, n))
    z1, z2 = z

    # 逐元素计算复包络模长；**2 是乘方（注意不是 ^，^ 在 Python 里是位异或）
    R = np.sqrt((m + s_R * z1) ** 2 + (s_R * z2) ** 2)
    return R


def sample_shadow_iid(n: int, mu_Np: float, sigma_dB: float,
                      rng: np.random.Generator) -> np.ndarray:
    """采样 n 个 Lognormal 阴影分量 S（Corazza 原文式(3)）。

    ln S ~ N(mu_Np, (h*sigma_dB)^2)，h = ln(10)/20。
    mu 给的是奈培、sigma 给的是 dB，进 exp 前必须用 h 桥接。

    参数
    ----
    n         ：样本个数
    mu_Np     ：ln S 的均值，单位奈培(Np)
    sigma_dB  ：20*log10(S) 的标准差，单位 dB（原文称 dB spread）
    rng       ：numpy 随机数生成器

    返回
    ----
    长度 n 的一维数组，阴影增益 S（>0；S<1 表示遮挡使信号变小）。
    """
    z = rng.standard_normal(n)                      # 一路标准正态
    S = np.exp(mu_Np + H_DB_TO_NEPER * sigma_dB * z)  # h*sigma 把 dB 换成 Np
    return S


def sample_corazza_iid(alpha_deg: float, n: int,
                       rng: np.random.Generator) -> np.ndarray:
    """给定仰角，采样 n 个 i.i.d. 复合包络 r = R * S（Day 2 静态分布）。

    参数
    ----
    alpha_deg ：卫星仰角（度），必须在 20~80 拟合范围内
    n         ：样本个数
    rng       ：numpy 随机数生成器

    返回
    ----
    长度 n 的一维数组复合包络。

    注意
    ----
    R、S 共用同一个 rng 串行取数：随机流前后两段天然独立，
    符合模型 "Rice 与 Lognormal 两过程统计独立" 的前提。
    切勿在内部另建 make_rng，否则两路随机数会相关/重复。
    """
    # Day 1：仰角 -> (线性 K, 奈培 mu, dB sigma)
    K, mu, sigma = corazza_params(alpha_deg)
    # 快衰落（直射+多径）与慢阴影分别采样，再逐元素相乘
    R = sample_rice_iid(n, K, rng)
    S = sample_shadow_iid(n, mu, sigma, rng)
    return R * S


def marcum_q1(a: float, b: float) -> float:
    """一阶 Marcum Q 函数 Q1(a, b)。

    物理含义：参数为 (a, b) 的 Rice 分布中，包络超过 b 的概率，即
    P(R > b)。本模型用它写 Rice 分量的 CDF：
        F_R(r0) = 1 - Q1(sqrt(2K), r0 * sqrt(2(K+1)))

    数学上用非中心卡方分布的生存函数等价计算：
        Q1(a, b) = ncx2.sf(b^2, df=2, nc=a^2)
    （自由度 2、非中心参数 a^2 的非中心卡方变量大于 b^2 的概率）

    参数
    ----
    a, b ：非负实数

    返回
    ----
    Q1(a, b) ∈ [0, 1]。
    """
    return ncx2.sf(b ** 2, df=2, nc=a ** 2)


def corazza_cdf_theory(r: float, K_lin: float, mu_Np: float,
                       sigma_dB: float) -> float:
    """复合包络 r = R * S 的理论 CDF（Corazza 原文式(7)，数值积分）。

    阴影 S 使 Rice 电平 r/S 随机化；对所有可能的 S 按其 lognormal 概率
    加权平均，即得复合分布的累积概率 P(R*S <= r)。

    为避免 lognormal 在 S 域的长尾难收敛，做换元 x = ln S：
        dS = S * dx  ->  被积函数乘 S 后正好抵消，剩下正态密度权重。

    参数
    ----
    r        ：待求概率的包络门限（>=0）
    K_lin    ：线性莱斯因子
    mu_Np    ：ln S 的均值（奈培）
    sigma_dB ：20*log10(S) 的标准差（dB）

    返回
    ----
    F(r) = P(R*S <= r) ∈ [0, 1]。
    """
    # ln S 域的标准差（h 把 dB 换成 Np）
    s_x = H_DB_TO_NEPER * sigma_dB

    def integrand(x: float) -> float:
        """被积函数：正态权重 * 条件 Rice CDF。"""
        # x = ln S 的正态密度（换元后的权重 p_S(S)*S）
        w = norm.pdf(x, loc=mu_Np, scale=s_x)
        # 给定 S=exp(x)，r/S 是 Rice 包络电平；其 CDF = 1 - Q1(...)
        rice_cdf = 1.0 - marcum_q1(
            np.sqrt(2.0 * K_lin),
            (r / np.exp(x)) * np.sqrt(2.0 * (K_lin + 1.0)),
        )
        return w * rice_cdf

    # 积分区间取 mu +/- 6 个标准差，截断概率 < 1e-9，可忽略
    return quad(integrand, mu_Np - 6.0 * s_x, mu_Np + 6.0 * s_x)[0]


# ============================================================
# Day 3：时序相关性
# ============================================================

def colored_gaussian_jakes(n: int, fs: float, fd: float,
                           rng: np.random.Generator) -> np.ndarray:
    """生成自相关服从 Jakes 谱的实数高斯色噪声（Day 3 任务 3.1）。

    时域 ACF 理论值：R(τ) = J0(2π fd τ)
    实现：白噪声 → FFT → 乘 √S_jakes(f) → IFFT → 归一化单位方差。

    参数
    ----
    n   ：输出序列长度
    fs  ：采样率 (Hz)
    fd  ：最大多普勒频移 (Hz)，须满足 fd < fs/2
    rng ：numpy 随机数生成器

    返回
    ----
    长度 n 的一维数组，零均值、单位方差、近似 Jakes 自相关。
    """
    # 1. 白噪声
    w = rng.standard_normal(n)

    # 2. FFT 到频域
    W = np.fft.fft(w)
    f = np.fft.fftfreq(n, d=1.0 / fs)  # 与 W 一一对应的频率（FFT 原始顺序）

    # 3. Jakes 功率谱 S(f) ∝ 1/√(1-(f/fd)²)，|f|<fd
    eps = 1e-10
    ratio = f / fd
    S = np.where(np.abs(ratio) < 1.0,
                 1.0 / np.sqrt(np.maximum(1.0 - ratio ** 2, eps)),
                 0.0)

    # 4. 频域滤波：幅度乘 √S（功率乘 S）
    W_filtered = W * np.sqrt(S)

    # 5. IFFT 回时域，取实部（数值误差残留的虚部丢弃）
    x = np.fft.ifft(W_filtered).real

    # 6. 强制零均值单位方差
    x = (x - x.mean()) / x.std(ddof=0)
    return x


def colored_gaussian_lp(n: int, fs: float, tau_c: float,
                        rng: np.random.Generator) -> np.ndarray:
    """一阶 IIR 低通高斯色噪声（Day 3 任务 3.2）。

    等效相关时间 tau_c（秒）：ACF 在 τ=tau_c 处降到 e^-1。
    实现：g[n] = (1-a)*w[n] + a*g[n-1]，a = exp(-1/(fs*tau_c))。

    参数
    ----
    n     ：输出序列长度
    fs    ：采样率 (Hz)
    tau_c ：目标相关时间 (秒)
    rng   ：numpy 随机数生成器

    返回
    ----
    长度 n 的一维数组，零均值、单位方差、指数衰减自相关。
    """
    # 由目标相关时间反推 IIR 系数 a
    a = np.exp(-1.0 / (fs * tau_c))

    # 白噪声输入
    w = rng.standard_normal(n)

    # 递推滤波
    g = np.empty(n)
    g[0] = (1.0 - a) * w[0]
    for i in range(1, n):
        g[i] = (1.0 - a) * w[i] + a * g[i - 1]

    # 归一化零均值单位方差
    g = (g - g.mean()) / g.std(ddof=0)
    return g


def generate_corazza_sequence(alpha_deg: float, length: int = 1000,
                              fs: float = 1000.0, fd: float = 50.0,
                              tau_shadow: float = 0.3,
                              rng: np.random.Generator | None = None
                              ) -> np.ndarray:
    """生成一条 Corazza 复合包络时序序列（Day 3 任务 3.3）。

    r[n] = R[n] * S[n]
      - R[n]：Rice 快衰落，I/Q 两路独立 Jakes 色噪声构成
      - S[n]：Lognormal 阴影，一阶 IIR 低通慢变

    参数
    ----
    alpha_deg  ：仰角（度），20~80
    length     ：序列长度，默认 1000（学长论文）
    fs         ：采样率 (Hz)，默认 1000
    fd         ：最大多普勒 (Hz)，默认 50
    tau_shadow ：阴影相关时间 (秒)，默认 0.3
    rng        ：numpy 随机数生成器；None 时用默认种子

    返回
    ----
    长度 length 的一维数组 r[n]，单位与 Rice 包络一致（线性幅度）。
    """
    from generative.utils.seed import make_rng
    if rng is None:
        rng = make_rng()

    # 1. 仰角 -> Corazza 参数
    K, mu, sigma = corazza_params(alpha_deg)

    # 2. Rice 分量参数（与 sample_rice_iid 一致）
    s_R = np.sqrt(1.0 / (2.0 * (K + 1.0)))
    m = np.sqrt(K / (K + 1.0))

    # 3. 两路独立 Jakes 色噪声（I / Q）
    x = colored_gaussian_jakes(length, fs, fd, rng)
    y = colored_gaussian_jakes(length, fs, fd, rng)

    # 4. Rice 包络
    R = np.sqrt((m + s_R * x) ** 2 + (s_R * y) ** 2)

    # 5. 阴影慢变
    g = colored_gaussian_lp(length, fs, tau_shadow, rng)
    S = np.exp(mu + H_DB_TO_NEPER * sigma * g)

    # 6. 复合包络
    r = R * S
    return r

