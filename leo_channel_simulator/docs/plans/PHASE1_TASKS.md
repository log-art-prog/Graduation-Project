# 阶段一细化任务表 — Corazza 数据层（五天计划）

> 来源：[IMPROVEMENT_PLAN.md](IMPROVEMENT_PLAN.md) 阶段 0 收尾 + 阶段 1 细化（2026-09-18 重写）
> 目标：**产出一个分布正确、时序正确、可复现的 Corazza 信道包络数据集**，作为后续 CGAN 与扩散模型公平对比的唯一共同数据源。
> 编号规则：每日任务编号与 Day 对齐（Day 1 的任务是 1.x，Day 3 的任务是 3.x），完成一项就在对应 ☐ 打勾。
> 范围：本阶段全部为 `generative/` 下的新增文件，**不改动 `src/` 物理仿真器任何代码**。

---

## 这一阶段在毕设中的位置

```
阶段0 环境+精读 → 【阶段1 数据层】→ 阶段2 指标库 → 阶段3 CGAN → 阶段4 DDPM → 阶段5 对比
                       ↑
                 数据错了，后面两个模型全错
                 所以本阶段设"论文原图 + 理论公式"双重验收关卡
```

学长论文第 3 章只给了**单时间点分布公式**（r = R·S），没有写 1000 点带时间相关性的序列如何生成（无采样率、无多普勒、无阴影相关时间）。因此本阶段分两步走：**先把静态分布做对（Day 2 可对论文原图），再把时序相关性做对（Day 3 只能对经典理论，无法对学长原图，须注明出处）**。

---

## 日程总览

| 时间 | 任务 | 难度 | 产出 | 关卡 |
|------|------|------|------|------|
| 前置半天 | 阶段0 收尾：GPU 验证 + 两篇精读笔记 | ★☆☆ | torch 可用 GPU；docs/notes 08/09 | 打印出 RTX 5060 |
| Day 1 | 目录骨架 + Corazza 三参数仰角多项式 | ★★☆ | `corazza.py` 参数部分；论文图 3-1 | G1：曲线趋势与论文一致 |
| Day 2 | i.i.d. 复合包络（先不做时序相关） | ★★☆ | Rice/Lognormal 采样 + log 标准化；图 3-2/3-3 | G2：静态 PDF/CDF 与理论积分吻合 |
| Day 3 | **时序相关性（本阶段第一难点）** | ★★★ | Jakes 谱整形高斯 + 慢变阴影；ACF/LCR 自检图 | G3：ACF 对 J₀、LCR 对 Rice 理论 |
| Day 4 | 数据集生成与固化 | ★★☆ | `datasets/corazza_train.npz`（10000×1000） | G4：同种子字节级复现 |
| Day 5 | 全量验收、复现图、阶段笔记 | ★★☆ | `figs/` 复现图 + docs/notes/10 | DoD 全部打勾 |

---

## 前置收尾（阶段0，约半天）

| ☐ | 编号 | 操作 | 备注 |
|---|------|------|------|
| ☑ | P.1 | 等 `leo_gen` 环境安装完成，运行验证：`python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"` | 期望 `2.x+cu128 True ...RTX 5060` |
| ☑ | P.2 | 整理 [notes/08_corazza实现细节.md](../notes/08_corazza实现细节.md)：r=R·S 模型、表3-1 系数、复合 PDF 积分式（式2-23）、Marcum Q CDF（式2-26） | 内容已在口头精读中讲过，落成文字 |
| ☑ | P.3 | 整理 [notes/09_评价指标定义.md](../notes/09_评价指标定义.md)：KS/Wasserstein、ACF/LCR/AFD 公式（式2-35~39）、QPSK BER（式2-40）；注明"本文 ACF 统一定义为**包络**归一化自相关"，不沿用学长复信道/包络的模糊表述 | 阶段2直接照此实现 |

---

## Day 1：目录骨架 + Corazza 仰角参数

**目标**：表 3-1 的三个多项式能画出论文图 3-1。

**学习点（先讲懂再动手）**
- K（Rice 因子）= 直射功率 / 散射功率；低仰角小、高仰角大
- μ、σ 是阴影分量 ln S（或 log S，**单位待核对**）的均值、标准差
- 多项式只是对农村 L 频段实测数据的经验拟合，超出 20°~80° 不得外推

**任务**

| ☐ | 编号 | 操作 | 文件/位置 |
|---|------|------|-----------|
| ☑ | 1.1 | 建目录：`generative/{data,models,metrics,experiments,utils}` 及各级 `__init__.py`；根目录建 `datasets/ checkpoints/ results/ figs/` | 空目录 |
| ☑ | 1.2 | 写 `generative/utils/seed.py`：`make_rng(seed) -> np.random.Generator`、`seed_everything(seed)`（numpy + torch + cudnn 确定性） | 新文件 |
| ☑ | 1.3 | 写 `corazza_params(alpha_deg)`，按表 3-1 实现：<br>K(α) = K0+K1·α+K2·α²，K0=2.371, K1=−1.074×10⁻¹, K2=2.774×10⁻³<br>μ(α) = μ0+μ1·α+μ2·α²+μ3·α³，μ0=−2.331, μ1=1.142×10⁻¹, μ2=−1.939×10⁻³, μ3=1.094×10⁻⁵<br>σ(α) = σ0+σ1·α，σ0=4.5, σ1=−0.05 | `generative/data/corazza.py` |
| ☑ | 1.4 | α 取 20~80° 画 K、μ、σ 三联子图，存 `figs/p1_fig3-1_params.png`，与论文图 3-1 目视对照 | 绘图脚本 |

**⚠️ Day 1 必须核对的两个疑点（OCR 风险，禁止跳过）**

1. **K 的单位**：代入系数后 K(20°)≈1.33、K(80°)≈11.5，这组数值更像 **dB**（Corazza 原始模型农村 K 约 3~14 dB）。代码中先输出 `K_dB`，采样时用 `K_lin = 10**(K_dB/10)`；若图 3-1 趋势/量纲与论文不符，改按线性值处理并在笔记中记录。
2. **μ、σ 的底与系数准确性**：PDF 提取的系数可能有 OCR 误差，且 σ(20°)=4.5 若按 ln 域约 15 dB、按 log10 域约 35 dB，都偏大。**核对来源：Corazza & Vatalaro, "A statistical model for land mobile satellite channels and its application to nongeostationary orbit systems", IEEE Trans. VT, 1994**（以及该文给出的农村参数表）。核对结果写进 docs/notes/08，代码系数旁注明出处。

**验收关卡 G1**：K 随仰角单调上升、σ 单调下降；两个疑点有书面结论（无论是否改值）。

---

## Day 2：i.i.d. 复合包络 + 标准化（静态分布做对）

**目标**：固定仰角下生成大样本，分布与 r=R·S 理论分布吻合，复现论文图 3-2/3-3。此日**点与点之间相互独立**，时序相关 Day 3 才加。

**学习点**
- Rice 采样：两个零均值高斯 x,y（方差 σ₀²）+ 实轴直射分量 m = σ₀√(2K)，R = √((x+m)²+y²)
- Lognormal 采样：S = exp(μ + σ·z)，z~N(0,1)（若核对后 μ/σ 为 log10 域，则 S = 10^(μ+σz)）
- 复合分布无闭式解：理论 PDF = ∫ p_R(r/S)·p_S(S)/S dS（式2-23），用数值积分做裁判曲线
- 数据预处理：`x = (log(r+ε) − μ_log)/σ_log`，ε=1e-6（论文表3-2）；仰角 `α_norm = (α−50)/30` ∈ [−1,1]

**任务**

| ☐ | 编号 | 操作 | 文件/位置 |
|---|------|------|-----------|
| ☐ | 2.1 | 实现 `sample_rice_iid(n, K_lin, rng)`、`sample_lognormal_iid(n, mu, sigma, rng, base='ln')`、`sample_corazza_iid(alpha_deg, n, rng)` | `corazza.py` |
| ☐ | 2.2 | 实现理论裁判：`corazza_pdf_theory(r, K_lin, mu, sigma)` 数值积分（scipy `quad`/向量化积分）；CDF 用 PDF 积分或 Marcum Q（scipy.stats 无内置时用式2-26数值积分） | `corazza.py` |
| ☐ | 2.3 | 在 α = 20/40/60/80° 各生成 10⁶ 个 i.i.d. 样本：经验 PDF/CDF 与理论曲线叠图，计算 KS | 验证脚本 |
| ☐ | 2.4 | 画多仰角 PDF/CDF 对比图，存 `figs/p1_fig3-2_pdf.png`、`figs/p1_fig3-3_cdf.png` | 绘图脚本 |
| ☐ | 2.5 | 实现 `fit_standardizer(samples)` / `standardize()` / `inverse_standardize()`、`normalize_alpha()`；自测：反变换后与原序列最大误差 < 1e-10 | `data/transforms.py` |
| ☐ | 2.6 | 确认物理单调性：随仰角升高，包络均值增大、低包络区 CDF 概率减小 | 打印统计表 |

**验收关卡 G2**：4 个仰角经验 CDF 与理论 CDF 的 KS < 0.01（i.i.d. 大样本下必须很好，否则公式/系数仍有问题）；论文图 3-2/3-3 目视复现。

---

## Day 3：时间相关性（本阶段第一技术难点）

**目标**：1000 点序列是一条连续起伏的真实信道，而不是雪花噪声。Rice 分量的多普勒相关结构和阴影慢变都要对。

**学习点**
- 复高斯过程实部/虚部按 **Jakes 多普勒谱**整形：S(f) ∝ 1/√(1−(f/f_d)²)，|f|<f_d；其时域自相关理论上为 J₀(2πf_dτ)
- 关键无量纲量是 **f_d/f_s**（多普勒频率/采样率），决定归一化 ACF 形状；绝对秒数只影响 LCR 的物理单位
- 阴影在 ln 域是低通高斯过程（相关时间数百 ms 量级），实现：白噪声过一阶 IIR 或高斯窗卷积
- Rice 理论 LCR 有闭式公式：N_R(r_th) = √(2π(K+1))·f_d·ρ·exp(−K−(K+1)ρ²)·I₀(2ρ√(K(K+1)))，ρ=r_th/R_rms —— 作为自检裁判

**任务**

| ☐ | 编号 | 操作 | 文件/位置 |
|---|------|------|-----------|
| ☐ | 3.1 | 实现 `colored_gaussian_jakes(n, fs, fd, rng)`：频域构造双边 Jakes 谱（带小正则防端点奇异），白噪声 FFT → 乘 √S → IFFT，输出实数零均值单位方差序列 | `corazza.py` |
| ☐ | 3.2 | 实现 `colored_gaussian_lp(n, fs, tau_c, rng)`：ln 域阴影相关序列（一阶 IIR，等效相关时间 tau_c） | `corazza.py` |
| ☐ | 3.3 | 实现 `generate_corazza_sequence(alpha_deg, length=1000, fs=1000.0, fd=50.0, tau_shadow=0.3, rng=None)`：<br>① x,y = 两路 Jakes 高斯；② m=σ₀√(2K_lin)；③ R=\|(x+m)+jy\|；④ g=低通高斯，S=exp(μ+σ·g)；⑤ r=R·S | `corazza.py` |
| ☐ | 3.4 | **ACF 自检**：用纯散射 K=0 生成长序列，复过程 ACF 与 J₀(2πf_dτ) 叠图，前 50 个滞后点 RMSE 记录 | `tests/test_acf.py` |
| ☐ | 3.5 | **LCR 自检**：多个门限下统计向下穿越率（次/秒），与 Rice LCR 闭式公式对比，相对误差 < 15% | `tests/test_lcr.py` |
| ☐ | 3.6 | 阴影慢变自检：S 序列 ACF 在 τ=tau_c 附近降到 e⁻¹ 量级；目视 r 曲线呈现"快起伏叠加慢包络"两层结构 | 绘图 |
| ☐ | 3.7 | 参数敏感性记录（小加分）：f_d/f_s ∈ {0.01, 0.05, 0.10} 各画一次 ACF，确认多普勒越大相关时间越短，写进 docs/notes/10 | 记录 |

**默认参数与出处策略**
- 默认 f_s = 1000 Hz、f_d = 50 Hz（f_d/f_s=0.05）、阴影相关时间 0.3 s。**这些论文未给，属于"复现设定"而非学长参数**，代码顶部集中定义、论文中明确写出并引用 Jakes/Corazza 原始文献；不以复现学长 LCR 绝对数值为目标。

**验收关卡 G3**：ACF/LCR 两条理论自检同时通过；生成曲线图上快慢两层分明。过不了这关不许进入 Day 4。

---

## Day 4：数据集生成与固化

**目标**：产出后续两个模型共用的唯一训练数据文件。

**任务**

| ☐ | 编号 | 操作 | 文件/位置 |
|---|------|------|-----------|
| ☐ | 4.1 | 实现 `build_dataset(n_samples=10000, length=1000, seed=42, out_path=...)`：仰角在 [20°,80°] 均匀采样（每样本固定一个仰角），调 Day 3 序列生成器，逐样本 log 标准化 | `data/dataset.py` |
| ☐ | 4.2 | **标准化常量全库统一**：μ_log/σ_log 用全训练集估计一次，所有样本共用，常量随文件保存 | 同上 |
| ☐ | 4.3 | 保存 `datasets/corazza_train.npz`：`x_std`(N×1000)、`alpha_norm`(N,)、`alpha_deg`(N,)、`mu_log`、`sigma_log`、`eps`；另存 `corazza_train_meta.json`（代码版本/git commit、参数、日期、种子） | npz+json |
| ☐ | 4.4 | 实现 numpy 内存映射读取 + （供阶段3/4用的）`torch.utils.data.Dataset` 薄封装，本阶段不引入训练逻辑 | `dataset.py` |
| ☐ | 4.5 | 复现性验证：seed=42 连生成两次，`np.array_equal` 逐元素一致；再用 seed=43 确认结果不同 | 测试 |
| ☐ | 4.6 | 体量检查：10000×1000 float32 ≈ 40 MB，确认文件大小与 shape 正确 | — |

**验收关卡 G4**：数据集文件可被独立加载；同种子字节级一致；元数据齐全。

---

## Day 5：全量验收与阶段文档

**任务**

| ☐ | 编号 | 操作 | 产出 |
|---|------|------|------|
| ☐ | 5.1 | 一键验收脚本 `generative/data/validate_phase1.py`：依次跑 G1~G4 全部检查并打印 PASS/FAIL | 脚本+日志 |
| ☐ | 5.2 | 从固化数据集中按仰角分箱，统计 mean/std，验证随仰角单调性，计算 Spearman（目标 > 0.95，学长为 0.998） | `figs/p1_mean_std_vs_alpha.png` |
| ☐ | 5.3 | 取未参与训练采样的中间仰角（如 75°）单独生成参考样本，为阶段4/5的"连续仰角泛化"预留接口 | 说明记录 |
| ☐ | 5.4 | 写 `docs/notes/10_阶段1_数据层实现说明.md`：公式→代码对应关系、单位与 OCR 核对结论、时序参数出处、三张复现图、自检数值、已知局限 | 笔记 |
| ☐ | 5.5 | 在 [IMPROVEMENT_PLAN.md](IMPROVEMENT_PLAN.md) 里程碑 M1 处勾选完成、填写实际日期 | 文档更新 |

---

## 完成定义（DoD）

- [ ] G1：论文图 3-1 复现，K 单位 / 系数疑点有书面核对结论
- [ ] G2：i.i.d. 样本经验 CDF 与式2-23/2-26 理论曲线 KS < 0.01；图 3-2/3-3 目视复现
- [ ] G3：包络 ACF 对 J₀(2πf_dτ)、LCR 对 Rice 闭式公式双自检通过
- [ ] G4：`corazza_train.npz`（10000×1000）固化，同种子字节级可复现，元数据齐全
- [ ] 均值随仰角 Spearman > 0.95；低仰角深衰落概率高于高仰角
- [ ] log 标准化/反变换往返误差 < 1e-10
- [ ] 全程只新增 `generative/` 与数据目录文件，`src/` 零改动
- [ ] docs/notes 08/09/10 完成，所有"论文未给参数"均标注为复现设定并给出经典文献出处

---

## 风险与回退

| 风险 | 现象 | 对策 |
|------|------|------|
| 表3-1 系数 OCR 错误 | 图3-1 形状怪异或 σ 大得离谱 | Day 1 查 Corazza & Vatalaro 1994 原表修正，笔记记录差异 |
| Jakes 谱端点奇异 | IFFT 后序列出现尖峰/方差不对 | 谱在 ±f_d 端点截断时加 ε 正则，事后强制单位方差归一化 |
| 阴影与快起伏时间尺度没拉开 | 曲线看不出两层结构 | 调 tau_shadow（0.1~1 s 扫描），以 ACF 双尺度形态为准 |
| 理论自检对不上 | 先怀疑实现，不怀疑公式 | K=0 退化为 Rayleigh 先过一遍；固定长序列（10⁵ 点）降低统计误差 |
| 阶段超时 | Day 3 卡住 | 先冻结一版"频域滤波+固定默认参数"数据集保证主线，敏感性扫描（3.7）可后补 |

---

## 本阶段结束时的交接物（给阶段2/3/4）

1. `datasets/corazza_train.npz` + meta json —— 两个模型唯一共同训练源
2. `generative/data/corazza.py` —— 阶段2"参考vs参考零误差"自测和阶段5大样本参考数据都靠它
3. `figs/p1_fig3-1/3-2/3-3` + ACF/LCR 自检图 —— 论文"数据平台"章节素材
4. docs/notes/08/09/10 —— 论文理论基础章草稿素材
