# Day 6 任务表 — L1 静态分布指标库

> 父计划：[PHASE2_TASKS.md](PHASE2_TASKS.md) 阶段二 Day 6 节
> 前置条件：阶段一 M1 达成，`datasets/corazza_train.npz` 可用
> 目标：实现 `generative/metrics/distribution.py`，包含 PDF L1、KS、Wasserstein、矩统计四类指标，并通过双向自测。
> 本日只写指标 + 自测，不涉及模型。

---

## 0. 本日核心概念（动手前先看懂）

### 0.1 为什么 L1 是最基础的一层

L1 回答"某一瞬间幅值的概率分布对不对"。如果连单个时间点的分布都不对，时序再对也没用——L1 是必要条件（但不是充分条件，打乱点序 PDF 不变但 ACF 全毁）。所以 L1 是三层指标里最先做、最容易做对的一层。

### 0.2 两组样本怎么对齐

参考样本 `r_ref` 来自 Corazza 大样本（阶段一产物），生成样本 `r_gen` 来自模型采样。两者**样本数可以不同**（如参考 10000 条、生成 500 条），但统计量是无偏的，所以直接比较即可。所有指标函数内部自动展平为 1D。

### 0.3 为什么不用 KDE

PDF L1 用**归一化直方图**而非核密度估计（KDE）。KDE 需要选带宽，带宽不同结果不同，引入主观参数。直方图只要 bins 相同，结果唯一确定。bins 边界以参考样本的 min/max 定，生成样本超出范围的归到边缘 bin。

---

## 1. 任务表

| ☐ | 编号 | 操作 | 文件/位置 |
|---|------|------|-----------|
| ☐ | 6.1 | 建 `generative/metrics/` 目录及 `__init__.py` | 目录 |
| ☐ | 6.2 | 实现 `pdf_l1_distance(r_ref, r_gen, n_bins=100) -> dict`：相同 bins 归一化直方图，逐 bin 差绝对值乘 bin 宽求和 | `metrics/distribution.py` |
| ☐ | 6.3 | 实现 `ks_distance(r_ref, r_gen) -> dict`：直接 `scipy.stats.ks_2samp`，取 statistic | `metrics/distribution.py` |
| ☐ | 6.4 | 实现 `wasserstein_distance(r_ref, r_gen) -> dict`：`scipy.stats.wasserstein_distance` | `metrics/distribution.py` |
| ☐ | 6.5 | 实现 `moment_stats(r_ref, r_gen) -> dict`：mean/std/skewness/kurtosis，报告参考值、生成值、相对误差 | `metrics/distribution.py` |
| ☐ | 6.6 | 实现 `evaluate_distribution(r_ref, r_gen, n_bins=100) -> dict`：一键调用以上四个，返回汇总 dict | `metrics/distribution.py` |
| ☐ | 6.7 | 写自测脚本 `scripts/day6_verify_distribution.py`：① 参考对半切，各指标 ≈ 0；② 加偏移/缩放，指标变大；③ 同种子可复现 | 自测脚本 |

---

## 2. 验收关卡

| 检查项 | 标准 | 方法 |
|--------|------|------|
| 参考对半 L1 ≈ 0 | L1 < 0.01 | 同一 Corazza 样本随机分两半互比 |
| 参考对半 KS ≈ 0 | KS < 0.01 | 同上 |
| 参考对半 Wasserstein ≈ 0 | W < 0.01 | 同上 |
| 矩相对误差 | < 1% | 对半切的 mean/std/skew/kurt |
| 加偏移后指标变大 | L1/KS/W 均 > 对半切的 5 倍 | r_gen = r_ref + 0.5 |
| 同种子复现 | 两次结果完全一致 | 固定 rng 跑两遍 |

---

## 3. 函数签名规范（阶段2接口冻结）

```python
def pdf_l1_distance(r_ref, r_gen, n_bins: int = 100) -> dict:
    """PDF L1 距离 = ∫|p_gen - p_ref|dr。
    返回 {"value": float, "detail": {"n_bins": int, "bin_width": float}}
    """

def ks_distance(r_ref, r_gen) -> dict:
    """CDF KS 距离 = max|F_gen - F_ref|。
    返回 {"value": float, "detail": {}}
    """

def wasserstein_distance(r_ref, r_gen) -> dict:
    """Wasserstein-1 距离。
    返回 {"value": float, "detail": {}}
    """

def moment_stats(r_ref, r_gen) -> dict:
    """矩统计：mean/std/skewness/kurtosis。
    返回 {"value": {"mean_rel_err": float, "std_rel_err": float, ...},
          "detail": {"ref": {...}, "gen": {...}}}
    """

def evaluate_distribution(r_ref, r_gen, n_bins: int = 100) -> dict:
    """一键调用以上四个，返回汇总 dict。"""
```

**统一规则**：
- `r_ref` / `r_gen` 可以是任意形状的 ndarray，函数内部 `.flatten()`
- 返回 dict 必须含 `"value"` 键（主指标值或子指标 dict）
- 不打印，只返回；调用方决定是否打印

---

## 4. 手把手实现步骤

### 4.1 pdf_l1_distance

```python
def pdf_l1_distance(r_ref, r_gen, n_bins=100):
    r_ref = np.asarray(r_ref).flatten()
    r_gen = np.asarray(r_gen).flatten()
    # bins 边界以参考样本 min/max 定
    lo, hi = r_ref.min(), r_ref.max()
    bins = np.linspace(lo, hi, n_bins + 1)
    # 两组用相同 bins
    p_ref, _ = np.histogram(r_ref, bins=bins, density=True)
    p_gen, _ = np.histogram(r_gen, bins=bins, density=True)
    bin_width = bins[1] - bins[0]
    l1 = np.sum(np.abs(p_gen - p_ref)) * bin_width
    return {"value": float(l1), "detail": {"n_bins": n_bins, "bin_width": float(bin_width)}}
```

**坑点**：
- 生成样本可能有值超出 [lo, hi]，`np.histogram` 会自动归到边缘 bin，不影响 L1（因为 p_ref 在边缘 bin 外是 0，超出部分的 |p_gen - 0| 会被算进去）
- `density=True` 保证直方图面积=1，L1 才是概率密度的差

### 4.2 ks_distance

```python
from scipy.stats import ks_2samp

def ks_distance(r_ref, r_gen):
    r_ref = np.asarray(r_ref).flatten()
    r_gen = np.asarray(r_gen).flatten()
    stat = ks_2samp(r_ref, r_gen).statistic
    return {"value": float(stat), "detail": {}}
```

**坑点**：`ks_2samp` 对样本量敏感，大样本 KS 会偏大（统计噪声 1/√n），这是正常的。

### 4.3 wasserstein_distance

```python
from scipy.stats import wasserstein_distance as wd

def wasserstein_distance(r_ref, r_gen):
    r_ref = np.asarray(r_ref).flatten()
    r_gen = np.asarray(r_gen).flatten()
    return {"value": float(wd(r_ref, r_gen)), "detail": {}}
```

### 4.4 moment_stats

```python
from scipy.stats import skew, kurtosis

def moment_stats(r_ref, r_gen):
    r_ref = np.asarray(r_ref).flatten()
    r_gen = np.asarray(r_gen).flatten()
    ref = {"mean": r_ref.mean(), "std": r_ref.std(),
           "skewness": skew(r_ref), "kurtosis": kurtosis(r_ref)}
    gen = {"mean": r_gen.mean(), "std": r_gen.std(),
           "skewness": skew(r_gen), "kurtosis": kurtosis(r_gen)}
    rel_err = {k: abs(gen[k] - ref[k]) / (abs(ref[k]) + 1e-12) for k in ref}
    return {"value": rel_err, "detail": {"ref": ref, "gen": gen}}
```

**坑点**：相对误差分母加 1e-12 防除零（mean 可能接近 0）。

### 4.5 evaluate_distribution

```python
def evaluate_distribution(r_ref, r_gen, n_bins=100):
    return {
        "pdf_l1": pdf_l1_distance(r_ref, r_gen, n_bins),
        "ks": ks_distance(r_ref, r_gen),
        "wasserstein": wasserstein_distance(r_ref, r_gen),
        "moments": moment_stats(r_ref, r_gen),
    }
```

### 4.6 自测脚本骨架

```python
# scripts/day6_verify_distribution.py
from generative.metrics.distribution import evaluate_distribution
from generative.data.dataset import load_dataset
from generative.utils.seed import make_rng

# 加载参考数据，对半切
data = load_dataset("datasets/corazza_train.npz")
r_all = data["x_std"]  # 或 inverse_standardize 还原
rng = make_rng(42)
idx = rng.permutation(len(r_all))
half = len(idx) // 2
r_ref = r_all[idx[:half]]
r_gen = r_all[idx[half:]]

result = evaluate_distribution(r_ref, r_gen)
print("参考对半切：")
for k, v in result.items():
    print(f"  {k}: {v['value']}")

# 加偏移
r_gen_bias = r_ref + 0.5
result_bias = evaluate_distribution(r_ref, r_gen_bias)
print("\n加偏移 0.5：")
for k, v in result_bias.items():
    print(f"  {k}: {v['value']}")
```

---

## 5. 推荐实施顺序

1. 6.1 建目录 + `__init__.py`
2. 6.2~6.5 逐个实现四个指标函数（每个写完简单测一下）
3. 6.6 写汇总函数
4. 6.7 写自测脚本，跑通双向自测
5. 全部通过后写 Day 6 日志
