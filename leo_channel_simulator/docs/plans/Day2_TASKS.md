# Day 2 任务表 — i.i.d. 复合包络 + log 标准化

> 父计划：[PHASE1_TASKS.md](PHASE1_TASKS.md) 阶段一 Day 2 节
> 理论依据：Corazza & Vatalaro 1994 原文式(3)(5)(7)(8)（`docs/references/corazza1994_land_mobile_satellite_channels.pdf`）+ 学长论文 2.4.5 节、3.2.2 节
> 前置条件：Day 1 G1 已通过（单位裁决 K 线性/μ Np/σ dB、K0=2.731 勘误），`corazza_params()` 可用
> 目标：**静态分布做对**——固定仰角下生成大样本 i.i.d. 包络 r=R·S，经验 CDF 与原文式(7) 理论 CDF 的 KS < 0.01；复现论文图 3-2/3-3；完成 log 标准化器
> 本日**不做时序相关**（点与点独立），时序留到 Day 3

---

## 0. 本日核心公式（动手前先看懂）

### 0.1 模型结构 r = R · S

接收包络 r 由两个独立随机变量相乘：
- **R**：Rice 快衰落分量（直射 + 多径散射），E[R²]=1（原文式(5) 归一化约定）
- **S**：Lognormal 阴影分量（慢变遮挡），由 ln S ~ N(μ, (hσ)²) 给出（原文式(3)）

### 0.2 Rice 采样（原文式(5)，E[R²]=1）

```
m   = sqrt( K / (K+1) )               # 直射分量幅度（归一化后）
s_R = sqrt( 1 / (2*(K+1)) )           # 散射分量 I/Q 每路标准差
z1, z2 ~ N(0,1)  独立
R = sqrt( (m + s_R*z1)^2 + (s_R*z2)^2 )
```
验证：E[R²] = m² + 2·s_R² = K/(K+1) + 1/(K+1) = 1 ✓。K=0 退化为 Rayleigh。

### 0.3 阴影采样（原文式(3)，μ 单位 Np、σ 单位 dB）

```
h = ln(10)/20 ≈ 0.115129              # dB → Np 桥接常数
z ~ N(0,1)
S = exp( μ + h*σ*z )
```
等价写法：X_dB ~ N(8.686·μ, σ²)，S = 10^(X_dB/20)。两种实现任选，数值应一致。

### 0.4 复合 CDF（原文式(7)，无闭式，数值积分）

Rice 条件 CDF 用 Marcum Q 函数 Q₁(a,b) 表示（R~Rice 时 P(R≤r₀)=1−Q₁(√(2K), r₀√(2(K+1)))）：

```
F(r) = ∫₀^∞ [ 1 − Q₁( sqrt(2K),  (r/S)*sqrt(2(K+1)) ) ] · p_S(S) dS
```
其中 p_S(S) 是 lognormal 密度。积分对 S 做（变量替换到 ln S 域用高斯权重更稳）。

### 0.5 数据预处理（学长论文式 3-8、表 3-2）

```
ε = 1e-6
x = ( log(r + ε) − μ_log ) / σ_log        # μ_log, σ_log 由全训练集统计
α_norm = ( α − 50 ) / 30                   # ∈ [−1, 1]，Day 4 才用
```
注意：ε 是加在 r 上防 log(0)，不是加在 S 上。

---

## 1. 任务清单

### 1.1 采样器（`generative/data/corazza.py` 追加）

| ☐ | 编号 | 操作 | 预期签名 |
|---|------|------|----------|
| ☐ | 2.1 | `sample_rice_iid(n, K_lin, rng)`：按 0.2 生成 n 个 Rice 包络，断言返回数组均值平方≈1（大样本下 |mean(R²)−1|<0.02） | `(n,) np.ndarray` |
| ☐ | 2.2 | `sample_shadow_iid(n, mu_Np, sigma_dB, rng)`：内部用 `H_DB_TO_NEPER`，按 0.3 生成 S | `(n,) np.ndarray` |
| ☐ | 2.3 | `sample_corazza_iid(alpha_deg, n, rng)`：调 `corazza_params` 取参，分别采样 R、S，返回 r=R*S | `(n,) np.ndarray` |

**自检**：K=0 时 `sample_rice_iid` 应退化为 Rayleigh，理论均值 = sqrt(π/2)·s_R·... 实测 |E[R²]−1|<0.02。

### 1.2 理论裁判曲线（`generative/data/corazza.py` 追加）

| ☐ | 编号 | 操作 | 说明 |
|---|------|------|------|
| ☐ | 2.4 | `marcum_q1(a, b)`：Marcum Q 一阶函数。优先 `scipy.stats.ncx2.sf(b**2, 2, a**2)`（等价于 Q₁(a,b)）；若版本无则自写级数 | 标量或向量 |
| ☐ | 2.5 | `corazza_cdf_theory(r, K_lin, mu_Np, sigma_dB)`：按 0.4 对 S 数值积分（`scipy.integrate.quad`，积分变量用 x=ln S 避免长尾），返回 F(r) | float |
| ☐ | 2.6 | （可选）`corazza_pdf_theory`：在加密 r 网格上对 CDF 做中心差分，或独立积分 p_R(r/S)·p_S(S)/S | float |

**关键坑**：S 的积分上限要取到 μ+5hσ 以上（lognormal 长尾）；用 `quad` 时给 `points`/`limit` 防收敛告警。

### 1.3 分布验证（新建 `scripts/day2_verify_distribution.py`）

| ☐ | 编号 | 操作 | 预期 |
|---|------|------|------|
| ☐ | 2.7 | 在 α=20/40/60/80° 各采 n=10⁶ 个 i.i.d. 样本，存内存（不持久化） | 4 组样本 |
| ☐ | 2.8 | 画经验 PDF（直方图归一化）与理论 PDF 叠图，4 仰角 2×2 子图，存 `figs/p1_fig3-2_pdf.png` | 目视贴合 |
| ☐ | 2.9 | 画经验 CDF（`np.sort` + 秩）与理论 CDF 叠图，存 `figs/p1_fig3-3_cdf.png` | 目视贴合 |
| ☐ | 2.10 | 计算 4 仰角 KS 统计量（`scipy.stats.ks_2samp` 或自算 max|F_emp−F_theory|），打印并断言全部 < 0.01 | PASS |
| ☐ | 2.11 | 物理单调性检查表：打印每仰角 mean(r)、P(r<0.5)；确认 mean 随仰角↑、深衰落概率随仰角↓ | 趋势正确 |

### 1.4 标准化器（新建 `generative/data/transforms.py`）

| ☐ | 编号 | 操作 | 说明 |
|---|------|------|------|
| ☐ | 2.12 | `fit_standardizer(samples, eps=1e-6)`：返回 `(mu_log, sigma_log)`，即 log(r+ε) 的均值和标准差 | tuple |
| ☐ | 2.13 | `standardize(r, mu_log, sigma_log, eps=1e-6)` / `inverse_standardize(x, ...)`：往返变换 | array |
| ☐ | 2.14 | `normalize_alpha(alpha_deg)` / `denormalize_alpha(alpha_norm)`：(α−50)/30 | array |
| ☐ | 2.15 | 自测：随机 10⁵ 个 r，`inverse_standardize(standardize(r))` 与 r 最大误差 < 1e-10（float64） | PASS |

### 1.5 收尾

| ☐ | 编号 | 操作 |
|---|------|------|
| ☐ | 2.16 | 在 [08_corazza实现细节.md](../notes/08_corazza实现细节.md) 第 9 节把"计划函数"更新为实际签名（若与计划有出入） |
| ☐ | 2.17 | 跑通 `scripts/day2_verify_distribution.py`，把 KS 数值与两张图路径记到当日工作日志 |

---

## 2. 验收关卡 G2

全部满足才进入 Day 3：

- [ ] 4 仰角 KS 全部 < 0.01
- [ ] `figs/p1_fig3-2_pdf.png`、`figs/p1_fig3-3_cdf.png` 目视与论文图 3-2/3-3 趋势一致（高仰角更尖、低仰角更扁）
- [ ] 单调性：mean(r) 随仰角严格递增，P(r<0.5) 随仰角递减
- [ ] `sample_rice_iid` K=0 退化 Rayleigh、E[R²]≈1 自检通过
- [ ] 标准化器往返误差 < 1e-10

---

## 3. 风险与对策

| 风险 | 现象 | 对策 |
|------|------|------|
| Marcum Q 调 scipy 接口不对 | CDF 曲线异常或 nan | 用 `scipy.stats.ncx2.sf(b**2, 2, a**2)` 等价关系先画 K=0（Rayleigh）验证，再上 K>0 |
| lognormal 积分长尾 | quad 收敛慢/告警 | 积分变量换 x=ln S，被积函数乘 S；限幅到 μ±6hσ |
| KS 卡在 0.01~0.05 | 理论曲线偏 | 先查单位（μ 是否漏乘 h），再查 Rice 归一化（s_R 是否为 sqrt(1/[2(K+1)])） |
| 10⁶ 样本内存爆 | OOM | 用 float32（约 4 MB/仰角），或分块统计 ECDF |

---

## 4. 交付物

1. `generative/data/corazza.py`：追加采样器 + 理论 CDF
2. `generative/data/transforms.py`：标准化器 + 仰角归一化
3. `scripts/day2_verify_distribution.py`：一键跑 2.7~2.11
4. `figs/p1_fig3-2_pdf.png`、`figs/p1_fig3-3_cdf.png`
5. 当日工作日志记录 KS 数值
