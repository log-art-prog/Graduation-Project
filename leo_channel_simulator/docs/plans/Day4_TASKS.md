# Day 4 任务表 — 数据集生成与固化

> 父计划：[PHASE1_TASKS.md](PHASE1_TASKS.md) 阶段一 Day 4 节
> 前置条件：Day 3 G3 已通过（ACF RMSE=0.0048、LCR 误差=9.9%、阴影 ACF PASS）
> 目标：**产出一个分布正确、时序正确、可复现的 Corazza 信道包络数据集**（10000×1000），作为后续 CGAN 与 DDPM 的唯一共同训练数据源。
> 关键约束：**标准化常量全库统一**——mu_log/sigma_log 由全训练集一次性估计，所有样本共用，不能每个 batch 重算。

---

## 0. 本日核心概念（动手前先看懂）

### 0.1 为什么需要"固化"数据集

Day 1~3 我们实现了 `generate_corazza_sequence`，但它每次调用都重新生成随机数。
后续训练 CGAN/DDPM 时，每次 epoch 都要加载同一批数据，不能每次重新生成（速度慢 + 不可复现）。
所以 Day 4 要做的是：**一次性生成 10000 条序列，标准化后存成 .npz 文件，后续直接加载**。

### 0.2 为什么标准化常量必须全库统一

Day 2 实现了 `fit_standardizer` / `standardize` / `inverse_standardize`。
如果每条序列各自算自己的 mu_log/sigma_log，模型学到的不是信道分布而是"每条各自的形状"。
正确做法：**10000 条全部生成后，一次性算 mu_log/sigma_log，再用同一组常量标准化所有样本**。

### 0.3 仰角采样策略

学长论文对仰角的处理是"条件生成"——给定仰角，生成对应的信道序列。
数据集中每条样本带一个仰角标签，仰角在 [20°, 80°] 均匀采样。
训练时模型以 (标准化序列, 归一化仰角) 对为输入。

---

## 1. 任务表

| ☐ | 编号 | 操作 | 文件/位置 |
|---|------|------|-----------|
| ☐ | 4.1 | 实现 `build_dataset(n_samples=10000, length=1000, seed=42, alphas=None)`：<br>① 仰角在 [20,80] 均匀采样（或传入指定数组）；② 逐条调 `generate_corazza_sequence`；③ 全部生成后用 `fit_standardizer` 算 mu_log/sigma_log；④ 用 `standardize` 标准化所有样本；⑤ 返回 `x_std(N×length)`, `alpha_norm(N,)`, `alpha_deg(N,)`, `mu_log`, `sigma_log`, `eps` | `generative/data/dataset.py` |
| ☐ | 4.2 | 保存 `datasets/corazza_train.npz`：上述全部数组 + 标量常量；另存 `datasets/corazza_train_meta.json`（种子、参数、日期、代码版本） | `dataset.py` |
| ☐ | 4.3 | 实现加载函数 `load_dataset(path)`：返回 npz 内容的 dict；实现 `CorazzaDataset(torch.utils.data.Dataset)` 薄封装：内存映射读取 npz，`__getitem__` 返回 (x_tensor, alpha_tensor) | `dataset.py` |
| ☐ | 4.4 | 复现性验证：seed=42 连生成两次，`np.array_equal` 逐元素一致；seed=43 确认结果不同 | `scripts/day4_verify_dataset.py` |
| ☐ | 4.5 | 体量检查：10000×1000 float32 ≈ 40 MB，打印文件大小与 shape | 同上 |
| ☐ | 4.6 | 抽检可视化：从数据集中随机取 3 条不同仰角的序列，inverse_standardize 还原后画图，确认两层结构（快起伏+慢包络） | 同上 |

---

## 2. 默认参数

| 参数 | 默认值 | 出处 | 说明 |
|------|--------|------|------|
| n_samples | 10000 | 学长论文 | 训练集大小 |
| length | 1000 | 学长论文 | 序列长度 |
| seed | 42 | 复现设定 | 固定种子 |
| alpha_range | [20, 80] | 原文 Table I | 仰角采样范围 |
| fs | 1000 Hz | 复现设定（Day 3） | 采样率 |
| fd | 50 Hz | 复现设定（Day 3） | 最大多普勒 |
| tau_shadow | 0.3 s | 复现设定（Day 3） | 阴影相关时间 |
| eps | 1e-6 | 学长论文表 3-2 | log 防零 |

---

## 3. 验收关卡 G4

| 检查项 | 标准 | 方法 |
|--------|------|------|
| 文件可加载 | npz 能被 np.load 读出所有 key | `load_dataset()` |
| 同种子复现 | seed=42 两次生成逐元素一致 | `np.array_equal` |
| 异种子不同 | seed=43 与 seed=42 结果不同 | `not np.array_equal` |
| 元数据齐全 | meta.json 含种子、参数、日期 | 人工检查 |
| 文件大小 | 10000×1000 float32 ≈ 40 MB | os.path.getsize |
| shape 正确 | x_std: (10000, 1000), alpha: (10000,) | 打印 .shape |
| 抽检图两层结构 | 快起伏叠加慢包络分明 | 目视 |

过不了这关不许进入 Day 5。

---

## 4. 风险与对策

| 风险 | 现象 | 对策 |
|------|------|------|
| 生成慢 | 10000 条逐条调，单线程可能几分钟 | 可接受；如需优化用 np.vectorize 或多进程，但优先保证正确 |
| 标准化常量不稳定 | 不同种子算出的 mu_log/sigma_log 差异大 | 10000 条足够稳定，差异应在 1% 以内 |
| 文件太大 | npz 超过预期 | 确认 dtype=float32（不是 float64），40MB 是预期值 |
| 仰角分布不均 | 均匀采样后某些角度样本少 | 10000 条 × [20,80] 均匀，每度约 167 条，足够 |

---

## 5. 依赖的已有函数

| 函数 | 文件 | Day |
|------|------|-----|
| `generate_corazza_sequence` | `corazza.py` | Day 3 |
| `fit_standardizer` / `standardize` / `inverse_standardize` | `transforms.py` | Day 2 |
| `normalize_alpha` / `denormalize_alpha` | `transforms.py` | Day 2 |
| `make_rng` | `utils/seed.py` | Day 1 |
