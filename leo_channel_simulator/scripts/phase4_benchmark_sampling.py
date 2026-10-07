"""阶段四补充实验：CGAN vs 扩散模型采样耗时基准（论文质量-成本 trade-off 表）。

协议：
- 吞吐：每仰角 n=500、batch_size=64，7 仰角（20..80），与验收生成协议一致
- 延迟：单条 n=1（batch=1），45°，重复 5 次取中位数
- 计时：torch.cuda.synchronize() + perf_counter，仅纯生成（不含指标计算与模型加载）
- 预热：正式计时前各跑一次小批量，摊销 CUDA kernel 编译与显存分配
- 模型：CGAN=checkpoints/cgan/cgan_final.pt（seed42 论文基线，EMA 权重）；
        扩散=checkpoints/ddpm/ddpm_final.pt（EMA 权重），DDIM 20/50/100 与 DDPM 1000

输出：results/phase4/sampling_benchmark.csv + 终端汇总表
运行：python scripts/phase4_benchmark_sampling.py   （预计 ~30 分钟，DDPM 占大头）
"""
from __future__ import annotations

import csv
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from generative.sample import load_generator
from generative.sample import generate_envelopes as gen_cgan
from generative.sample_ddpm import load_diffusion
from generative.sample_ddpm import generate_envelopes as gen_diff

ALPHAS = [20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]
N_THROUGHPUT = 500
BATCH = 64
N_LATENCY_REPS = 5
ALPHA_LAT = 45.0  # 单条延迟与仰角无关（计算量相同），固定 45°


def timed(fn, *args, **kwargs):
    """GPU 计时包装：同步→计时→同步，返回 (秒, 结果)。"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter() - t0, out


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows: list[dict] = []

    def record(model: str, alpha: float, n: int, batch: int, seconds: float) -> None:
        rows.append({"model": model, "alpha": alpha, "n": n,
                     "batch_size": batch,
                     "gen_time_s": round(seconds, 4),
                     "throughput_sps": round(n / seconds, 1)})
        print(f"[{model:<9}] α={alpha:5.1f}° n={n:4d} batch={batch:3d} "
              f"{seconds:9.3f}s  ({n / seconds:9.1f} 条/s)", flush=True)

    def latency(model: str, fn, *args, **kwargs) -> None:
        dts = []
        for _ in range(N_LATENCY_REPS):
            dt, _ = timed(fn, *args, **kwargs)
            dts.append(dt)
        med = statistics.median(dts)
        record(model, ALPHA_LAT, 1, 1, med)
        print(f"[{model:<9}] 单条延迟中位数 = {med * 1000:.2f} ms", flush=True)

    # ---------- CGAN ----------
    G, meta = load_generator(ROOT / "checkpoints" / "cgan" / "cgan_final.pt", device)
    gen_cgan(G, meta, 45.0, 16, batch_size=16, seed=0)  # 预热
    for a in ALPHAS:
        dt, _ = timed(gen_cgan, G, meta, a, N_THROUGHPUT, batch_size=BATCH, seed=0)
        record("CGAN", a, N_THROUGHPUT, BATCH, dt)
    latency("CGAN", gen_cgan, G, meta, ALPHA_LAT, 1, batch_size=1, seed=0)
    del G
    torch.cuda.empty_cache()

    # ---------- 扩散（一次加载，四套采样配置） ----------
    diff, dmeta = load_diffusion(ROOT / "checkpoints" / "ddpm" / "ddpm_final.pt",
                                 device)
    gen_diff(diff, dmeta, 45.0, 8, sampler="ddim", ddim_steps=20,
             batch_size=8, seed=0)  # 预热
    configs = [("DDIM-20", "ddim", 20), ("DDIM-50", "ddim", 50),
               ("DDIM-100", "ddim", 100), ("DDPM-1000", "ddpm", None)]
    for name, sampler, steps in configs:
        kw = {"sampler": sampler, "batch_size": BATCH, "seed": 0}
        if steps is not None:
            kw["ddim_steps"] = steps
        for a in ALPHAS:
            dt, _ = timed(gen_diff, diff, dmeta, a, N_THROUGHPUT, **kw)
            record(name, a, N_THROUGHPUT, BATCH, dt)
        kw1 = dict(kw, batch_size=1, seed=0)
        latency(name, gen_diff, diff, dmeta, ALPHA_LAT, 1, **kw1)

    # ---------- 存表 ----------
    out = ROOT / "results" / "phase4" / "sampling_benchmark.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ---------- 汇总 ----------
    print("\n===== 汇总：每仰角 500 条 × 7 仰角，batch=64 =====")
    print(f"{'模型':<10} {'总生成耗时(s)':>14} {'吞吐(条/s)':>12} {'单条延迟(ms)':>14}")
    for name, _, _ in [("CGAN", None, None)] + configs:
        th = [r for r in rows if r["model"] == name and r["n"] == N_THROUGHPUT]
        lat = next(r for r in rows if r["model"] == name and r["n"] == 1)
        total = sum(r["gen_time_s"] for r in th)
        print(f"{name:<10} {total:>14.2f} {N_THROUGHPUT * len(th) / total:>12.1f} "
              f"{lat['gen_time_s'] * 1000:>14.2f}")
    print(f"\n明细已存：{out}")


if __name__ == "__main__":
    main()
