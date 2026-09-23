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
| ☑ | 4.1 | 实现 `build_dataset(n_samples=10000, length=1000, seed=42, alphas=None)`：<br>① 仰角在 [20,80] 均匀采样（或传入指定数组）；② 逐条调 `generate_corazza_sequence`；③ 全部生成后用 `fit_standardizer` 算 mu_log/sigma_log；④ 用 `standardize` 标准化所有样本；⑤ 返回 `x_std(N×length)`, `alpha_norm(N,)`, `alpha_deg(N,)`, `mu_log`, `sigma_log`, `eps` | `generative/data/dataset.py` |
| ☑ | 4.2 | 保存 `datasets/corazza_train.npz`：上述全部数组 + 标量常量；另存 `datasets/corazza_train_meta.json`（种子、参数、日期、代码版本） | `dataset.py` |
| ☑ | 4.3 | 实现加载函数 `load_dataset(path)`：返回 npz 内容的 dict；实现 `CorazzaDataset(torch.utils.data.Dataset)` 薄封装：内存映射读取 npz，`__getitem__` 返回 (x_tensor, alpha_tensor) | `dataset.py` |
| ☑ | 4.4 | 复现性验证：seed=42 连生成两次，`np.array_equal` 逐元素一致；seed=43 确认结果不同 | `scripts/day4_verify_dataset.py` |
| ☑ | 4.5 | 体量检查：10000×1000 float32 ≈ 40 MB，打印文件大小与 shape | 同上 |
| ☑ | 4.6 | 抽检可视化：从数据集中随机取 3 条不同仰角的序列，inverse_standardize 还原后画图，确认两层结构（快起伏+慢包络） | 同上 |

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

**✅ G4 已通过（2026-09-22）**：npz 40.1 MB、shape (10000, 1000) float32；同种子逐元素一致、异种子不同（n=100 验证机制）；meta.json 齐全（seed、fs/fd/tau_shadow、mu_log/sigma_log、eps、日期等字段）；抽检图 25°/50°/75° 两层结构清晰。过程记录见 [Day4 工作日志](../logs/2026-09-22_Day4工作日志.md)。

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

---

## 6. 手把手实现步骤（自己写，不抄答案）

### 6.1 文件结构（本日只新增 2 个文件）

```
leo_channel_simulator/
├── generative/data/
│   └── dataset.py          ← 新建：build_dataset / save / load / Dataset 类
├── scripts/
│   └── day4_verify_dataset.py  ← 新建：生成 + 全部自检
└── datasets/               ← 新建目录：存 npz 和 meta.json
    ├── corazza_train.npz
    └── corazza_train_meta.json
```

### 6.2 任务 4.1：build_dataset（在 dataset.py 里）

**步骤**：
1. 文件顶部 import：`json`、`datetime`（后两个 4.2 才用，先写也行）、`numpy as np`、`torch`（4.3 才用），再从 `generative.data.corazza` 导入 `generate_corazza_sequence`，从 `generative.data.transforms` 导入 4 个变换函数，从 `generative.utils.seed` 导入 `make_rng`。
2. 写函数签名：
   ```python
   def build_dataset(n_samples=10000, length=1000, seed=42,
                     fs=1000.0, fd=50.0, tau_shadow=0.3,
                     alphas=None):
   ```
3. `rng = make_rng(seed)`
4. 仰角数组：如果 `alphas is None`，用 `rng.uniform(20.0, 80.0, size=n_samples)`；否则校验它的长度等于 n_samples。
5. 预分配 `raw = np.empty((n_samples, length), dtype=np.float64)`（先 float64 保证精度）。
6. for 循环逐条生成：
   ```python
   for i in range(n_samples):
       raw[i] = generate_corazza_sequence(alphas[i], length=length,
                                          fs=fs, fd=fd,
                                          tau_shadow=tau_shadow, rng=rng)
   ```
   加个进度打印：每 1000 条 print 一次（不然几分钟没输出以为卡死了）。
7. **全部生成后**才调 `fit_standardizer(raw)` 得到 `mu_log, sigma_log`（关键：不能边生成边算）。
8. `x_std = standardize(raw, mu_log, sigma_log).astype(np.float32)`（存盘用 float32 省空间）。
9. `alpha_norm = normalize_alpha(alphas)`，再 `.astype(np.float32)`。
10. 返回一个 dict，key 为：`x_std, alpha_norm, alpha_deg, mu_log, sigma_log, eps`（eps 从 transforms.EPS 取）。

**坑点**：
- 同一个 `rng` 要同时用于仰角采样和序列生成——先抽仰角再抽序列，顺序固定，复现才稳定。
- 不要在循环内反复 fit_standardizer。
- raw 用 float64，存盘前转 float32。

### 6.3 任务 4.2：保存

**步骤**：
1. 写 `save_dataset(data: dict, out_dir)`：
   - `Path(out_dir).mkdir(parents=True, exist_ok=True)`
   - 用 `np.savez(out_dir / "corazza_train.npz", **data)` 一次存所有数组。
2. meta.json 单独组织（JSON 只能放纯 Python 类型）：
   ```python
   meta = {
       "created_at": datetime.date.today().isoformat(),
       "seed": 42,
       "n_samples": ..., "length": ...,
       "fs": ..., "fd": ..., "tau_shadow": ...,
       "mu_log": float(data["mu_log"]),
       "sigma_log": float(data["sigma_log"]),
       "eps": float(data["eps"]),
       "dtype": "float32",
   }
   ```
   用 `json.dump(meta, f, indent=2, ensure_ascii=False)`。

**坑点**：
- numpy 标量（np.float64）不能直接 json.dump，必须 `float(...)` 转 Python 原生类型。
- npz 里的 key 名字要和 load 时预期的一致，建议在文件顶部定义常量或直接照抄上面的 6 个名字。

### 6.4 任务 4.3：加载 + Dataset 封装

**步骤**：
1. `load_dataset(path)`：
   ```python
   def load_dataset(npz_path):
       z = np.load(npz_path)
       return {k: z[k] for k in z.files}
   ```
2. Dataset 类（Day 5 训练时才真正用到，今天先把骨架立住）：
   ```python
   from torch.utils.data import Dataset

   class CorazzaDataset(Dataset):
       def __init__(self, npz_path):
           d = load_dataset(npz_path)
           self.x = d["x_std"]            # 已经是 float32
           self.alpha = d["alpha_norm"]

       def __len__(self):
           return self.x.shape[0]

       def __getitem__(self, idx):
           return (
               torch.from_numpy(self.x[idx]).float(),      # shape (1000,)
               torch.tensor(self.alpha[idx]).float(),      # 标量
           )
   ```

**坑点**：
- 今天可以先用小数据（n_samples=200）跑通流程，再换 10000 全量，避免调试时反复等几分钟。
- `np.load` 默认不支持 mmap 参数对 npz 内文件直接 mmap；任务书说的"内存映射读取"在 npz（压缩包）里做不到，直接全量读内存即可（40MB 完全没问题）。如果以后改 .npy 裸格式才能用 `mmap_mode='r'`。

### 6.5 任务 4.4~4.6：验证脚本 day4_verify_dataset.py

**脚本结构**（和 day3_verify_temporal.py 同样的 sys.path 注入套路）：

1. **4.5 体量检查**（先做这个，最快）：build → save → `os.path.getsize` 打印 MB 数 → load → 打印每个 key 的 shape/dtype。
2. **4.4 复现性**：
   ```python
   d1 = build_dataset(n_samples=100, seed=42)
   d2 = build_dataset(n_samples=100, seed=42)
   d3 = build_dataset(n_samples=100, seed=43)
   assert np.array_equal(d1["x_std"], d2["x_std"])   # 同种子完全一致
   assert not np.array_equal(d1["x_std"], d3["x_std"])  # 异种子不同
   ```
   注意：复现性验证用 n_samples=100 即可（快）；全量 10000 的数据集只生成一次存盘。
3. **4.6 抽检可视化**：从**已加载的全量数据集**里挑 3 条（建议挑低/中/高仰角各一条，可以用 `np.argmin(|alpha_deg - 目标角度|)` 找索引），用 `inverse_standardize` 还原成物理包络，画 3 行子图（共享 x 轴），横轴时间 ms。目视确认：快起伏（毫秒级）叠加在慢包络（~300ms 级）上。
4. 保存 `figs/day4_dataset_samples.png`。

**坑点**：
- 画图记得加中文字体设置（Day 3 踩过的坑）：
  ```python
  plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
  plt.rcParams["axes.unicode_minus"] = False
  ```
- 选"最接近某个角度"的索引用 argmin，不要假设数据里正好有 20.0/50.0/80.0（均匀随机采样几乎不可能正好命中整数）。

### 6.6 推荐实施顺序

1. 先写 dataset.py 的 4.1 build_dataset → 用 n_samples=200 在验证脚本里跑通。
2. 写 4.2 save + 4.3 load/Dataset → 验证脚本能 load 回来、shape 对。
3. 写 4.4 复现性 assert → 跑过。
4. 小数据全部跑通后，把 n_samples 改成 10000 全量生成一次（约几分钟），存盘。
5. 对全量数据做 4.5（体量）+ 4.6（画图）。
6. 全绿后更新本任务表勾选，写 Day 4 日志。
