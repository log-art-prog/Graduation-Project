# Day 5 任务表 — 阶段一全量验收与文档收尾

> 父计划：[PHASE1_TASKS.md](PHASE1_TASKS.md) 阶段一 Day 5 节
> 前置条件：Day 4 G4 已通过（npz 40.1 MB、同种子可复现、meta 齐全）
> 目标：**一键验收 G1~G4、验证"仰角→信道强度"物理单调性、补齐阶段文档**，给阶段二（CGAN）一个干净的地基。
> 本日无新模型代码，全部是验证、统计、画图、写文档。

---

## 0. 本日核心概念（动手前先看懂）

### 0.1 为什么要"一键验收"

Day 1~4 的验收分散在 4 个脚本里，每次改动后逐个跑容易漏。写一个 `validate_phase1.py` 把 G1~G4 全部检查串起来，PASS/FAIL 一屏看完——以后阶段二改了底层代码，跑它一遍就知道有没有把上层搞坏（回归测试的雏形）。

### 0.2 为什么要验 Spearman 单调性

数据集是仰角条件生成的。物理上仰角越高信道越好（Corazza 模型的核心假设），如果数据集里"平均包络 vs 仰角"不是单调上升的，说明生成链路有 bug 或标准化出错。Spearman 相关衡量的是**单调性**（不是线性），目标 > 0.95（学长论文实测 0.998）。

### 0.3 为什么要留"未参与训练的仰角"参考样本

训练集仰角是 [20,80] 连续均匀采样，几乎不可能正好包含 75.0°。单独生成一批 75° 的参考序列存盘，将来阶段四/五评估"连续仰角泛化"时，它是模型没见过的标准答案。今天只是生成存盘 + 写说明，不做评估。

---

## 1. 任务表

| ☐ | 编号 | 操作 | 文件/位置 |
|---|------|------|-----------|
| ☑ | 5.1 | 一键验收脚本：依次跑 G1（参数曲线合理性）→ G2（i.i.d. CDF KS）→ G3（ACF/LCR）→ G4（npz/meta/复现性），全部打印 PASS/FAIL，最后汇总退出码 | `generative/data/validate_phase1.py` |
| ☑ | 5.2 | 从固化数据集按仰角分箱（如每 10° 一箱），还原物理包络后统计每箱 mean/std，验证单调性，算 Spearman（目标 > 0.95）；画 mean±std vs 仰角图 | `scripts/day5_binned_stats.py` → `figs/p1_mean_std_vs_alpha.png` |
| ☑ | 5.3 | 用 `generate_corazza_sequence(75°)` 单独生成一批参考序列存 `datasets/alpha75_ref.npz`，写一段"泛化接口"说明 | 脚本 + 说明（并入 5.4 笔记） |
| ☑ | 5.4 | 写阶段笔记：公式→代码对应关系、单位与勘误结论、时序参数出处、三张复现图、自检数值汇总、已知局限 | `docs/notes/12_阶段1_数据层实现说明.md` |
| ☑ | 5.5 | 在 [IMPROVEMENT_PLAN.md](IMPROVEMENT_PLAN.md) 里程碑 M1 勾选完成、填写实际日期 2026-09-23 | 文档更新 |

> 编号说明：父计划 5.4 写的是 notes/10_阶段1...，但 10 已被"多普勒参数敏感性"占用、11 被自测题库占用，故本日笔记用 **12_** 前缀，笔记开头注明此偏差。

---

## 2. 验收关卡（阶段出口，全过才算阶段一完成）

| 检查项 | 标准 | 方法 |
|--------|------|------|
| 一键验收 | validate_phase1.py 所有检查 PASS | 脚本退出码 0 |
| Spearman | mean 包络 vs 仰角 Spearman > 0.95 | scipy.stats.spearmanr |
| 物理单调性 | 分箱 mean 随仰角上升；低仰角深衰落概率高于高仰角 | 分箱统计 + P(r < r₀) 对比 |
| 参考样本 | alpha75_ref.npz 可加载、含生成参数 | np.load 检查 |
| 文档 | 笔记 12 完成、M1 勾选 | 人工 |

---

## 3. 依赖的已有函数

| 函数 | 文件 | Day |
|------|------|-----|
| `corazza_params` | `corazza.py` | Day 1 |
| `sample_corazza_iid` / `corazza_cdf_theory` | `corazza.py` | Day 2 |
| `generate_corazza_sequence` / `colored_gaussian_*` | `corazza.py` | Day 3 |
| `fit_standardizer` / `standardize` / `inverse_standardize` / `normalize_alpha` | `transforms.py` | Day 2 |
| `build_dataset` / `load_dataset` / `save_dataset` | `dataset.py` | Day 4 |
| `make_rng` | `utils/seed.py` | Day 1 |
| `scipy.stats.spearmanr` | scipy | 外部 |

> G2/G3 的检查逻辑分散在 day2/day3 验证脚本里。5.1 可以把核心检查函数（不是整个脚本）抽成可 import 的形式调用，或在 validate 脚本里重新实现核心断言——**优先选择：把 day2/day3 脚本里的检查逻辑抽成函数放回对应模块或 validate 里复用，避免复制粘贴两份数字将来不同步。**

---

## 4. 手把手实现步骤（自己写，不抄答案）

### 4.1 任务 5.1：validate_phase1.py 骨架

```python
"""阶段一一键验收：G1~G4 全部检查。全 PASS 退出码 0，否则 1。"""
CHECKS = []

def check(name: str, ok: bool, detail: str = ""):
    CHECKS.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}")

def validate_g1():   # 参数曲线合理性
    # TODO: K(20) < K(80)（低仰角 K 约 1.7，高仰角约 11.9）
    # TODO: sigma(20) > sigma(80)（约 3.5 dB 降到 0.5 dB）
    ...

def validate_g2():   # i.i.d. CDF：KS < 0.01（复用 day2 逻辑，至少验 2 个仰角）
    ...

def validate_g3():   # ACF RMSE < 0.02、LCR 误差 < 15%（复用 day3 逻辑）
    ...

def validate_g4():   # npz/meta/shape/dtype/复现性（复用 day4 逻辑）
    ...

if __name__ == "__main__":
    validate_g1(); validate_g2(); validate_g3(); validate_g4()
    n_pass = sum(CHECKS)
    print(f"\n==== 阶段一验收：{n_pass}/{len(CHECKS)} PASS ====")
    sys.exit(0 if all(CHECKS) else 1)
```

**坑点**：
- G2/G3 有随机性，检查阈值要留余量（如 ACF 用 0.01 而不是贴着 0.02 报线），否则偶尔抖过线误报 FAIL。
- 每个 validate_gx 内部多打 detail 数值，FAIL 时能直接看出差多少。

### 4.2 任务 5.2：分箱统计

**步骤**：
1. `load_dataset` 读全量 npz，取 `alpha_deg`、`x_std`、`mu_log`、`sigma_log`。
2. 分箱：`bins = np.arange(20, 81, 10)`（20~80 每 10° 一箱），用 `np.digitize(alpha_deg, bins)` 给每条样本分箱号。
3. 对每箱：`inverse_standardize` 还原该箱所有样本（或先对 x_std 统计再还原 mean），算 mean/std。
4. 深衰落概率：全库取 r₀ = 5% 分位数，统计每箱 P(r < r₀)，确认**低仰角箱更大**。
5. `spearmanr(bin_centers, bin_means)`，打印相关系数。
6. 画图：x 轴箱中心仰角，y 轴 mean（errorbar 用 std），存 `figs/p1_mean_std_vs_alpha.png`（记得中文字体两行）。

**坑点**：
- 分箱用 **alpha_deg 原始角度**，不要用 alpha_norm（[-1,1] 归一化值，语义易混）。
- 还原必须用 npz 里存的 mu_log/sigma_log，**不要重新 fit**（否则违背全库统一原则）。
- 边缘箱（20~30、70~80）样本少是正常的，每箱仍有上千条，统计够稳。

### 4.3 任务 5.3：75° 参考样本

**步骤**：
1. `make_rng(固定种子如 2026)`，循环 `generate_corazza_sequence(75.0, ...)` 生成 50 条（默认 fs/fd/tau_shadow 与训练集一致）。
2. 用训练集的 mu_log/sigma_log 标准化后转 float32，和 alpha 数组一起 `np.savez` 到 `datasets/alpha75_ref.npz`。
3. 简单自检：shape (50, 1000)、加载往返一致。

**为什么用不同种子**：参考样本和训练集应相互独立，种子不同保证随机数流不重叠。

### 4.4 任务 5.4/5.5：笔记与里程碑

笔记 `12_阶段1_数据层实现说明.md` 建议目录：
1. 模型总览：r = R·S，两层时间尺度
2. 公式→代码对应表（式 2-23/2-26/Table I ↔ 函数名）
3. 单位与勘误结论（K 线性/μ Np/σ dB；K0=2.731）
4. 时序参数出处（fs/fd/tau_shadow 标注"复现设定"+ 文献）
5. 验收数值汇总表（G1~G4 全部数字）
6. 已知局限（循环卷积边界效应、均匀散射假设、npz 无 mmap 等）

5.5 打开 IMPROVEMENT_PLAN.md 找 M1，勾选 + 日期。

---

## 5. 推荐实施顺序

1. 5.1 validate_phase1（核心工作量在复用已有检查逻辑）
2. 5.2 分箱统计 + 图（Spearman 数字出来）
3. 5.3 75° 参考样本（小活）
4. 5.4 笔记（把 5.2/5.3 结果写进去）
5. 5.5 里程碑勾选
6. 全部完成后更新本表勾选、写 Day 5 日志——阶段一收官
