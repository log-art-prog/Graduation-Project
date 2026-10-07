"""阶段三诊断（续）：中间 checkpoint 趋势 — 判断 20°/25° 劣化是否随训练改善。

运行：
  python scripts/phase3_diag_ckpt_trend.py                      # 默认 run（seed42）
  python scripts/phase3_diag_ckpt_trend.py --dir checkpoints/cgan_seed1234

判读：
  - KS 随步数持续下降 → 条件映射仍在精化，续训有望修复（支持欠训练假设）
  - KS 早已平台/反弹 → 续训无用，需换策略（如低仰角条件重加权）
30° 为对照组（验收已达标，应保持稳定）。
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from generative.metrics.distribution import evaluate_distribution
from generative.sample import load_generator, generate_envelopes

ALPHAS_DIAG = [20.0, 25.0, 30.0]  # 30° 为对照组
N = 500


def discover_checkpoints(ckpt_dir: Path):
    """按步数排序发现 cgan_step{N}.pt，并把 cgan_final.pt 排在末尾。"""
    stepped = []
    for p in ckpt_dir.glob("cgan_step*.pt"):
        m = re.search(r"cgan_step(\d+)\.pt$", p.name)
        if m:
            stepped.append((int(m.group(1)), f"step{m.group(1)}", p))
    stepped.sort(key=lambda t: t[0])
    items = [(name, path) for _, name, path in stepped]
    final = ckpt_dir / "cgan_final.pt"
    if final.exists():
        items.append(("final", final))
    return items


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dir", type=str,
                   default=str(ROOT / "checkpoints" / "cgan"))
    args = p.parse_args()
    ckpt_dir = Path(args.dir)
    checkpoints = discover_checkpoints(ckpt_dir)
    if not checkpoints:
        raise SystemExit(f"目录 {ckpt_dir} 下未发现 cgan_step*.pt")

    from scripts.phase3_diagnose_20deg import get_reference  # 复用参考缓存协议

    print(f"run: {ckpt_dir.name}")
    print(f"{'ckpt':>10s} | " + " | ".join(
        f"α={a:.0f}° KS / log均值偏差" for a in ALPHAS_DIAG))
    for name, path in checkpoints:
        G, meta = load_generator(path)
        cells = []
        for alpha in ALPHAS_DIAG:
            r_gen = generate_envelopes(G, meta, alpha, N, seed=1000 + int(alpha))
            r_ref = get_reference(alpha, N)
            ks = evaluate_distribution(r_ref, r_gen)["ks"]["value"]
            lm_gen = float(np.log(r_gen + 1e-6).mean())
            lm_ref = float(np.log(r_ref + 1e-6).mean())
            cells.append(f"{ks:.4f} / {lm_gen - lm_ref:+.3f}")
        print(f"{name:>10s} | " + " | ".join(cells))


if __name__ == "__main__":
    main()
