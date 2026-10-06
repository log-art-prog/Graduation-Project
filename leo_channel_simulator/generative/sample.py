"""阶段三：CGAN 采样 — 加载训练好的生成器（EMA 权重），按仰角生成包络序列。

用法：
    # 作为模块被评估脚本调用
    from generative.sample import load_generator, generate_envelopes
    G, meta = load_generator("checkpoints/cgan/cgan_final.pt")
    r = generate_envelopes(G, meta, alpha_deg=45.0, n=500)

    # 命令行快速查看
    python -m generative.sample --alpha 45 --n 8
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from generative.data.dataset import load_dataset
from generative.data.transforms import inverse_standardize, normalize_alpha
from generative.models.cgan import SequenceNoiseFilterGenerator, NOISE_DIM, SEQ_LEN


def load_generator(ckpt_path: str | Path,
                   device: torch.device | str | None = None):
    """加载 checkpoint，返回用 EMA 权重武装好的生成器与元信息 dict。"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    G = SequenceNoiseFilterGenerator(noise_dim=ckpt.get("noise_dim", NOISE_DIM),
                                     seq_len=ckpt.get("seq_len", SEQ_LEN))
    G.load_state_dict(ckpt["g_ema"])  # 采样一律用 EMA 权重（论文设定）
    G.to(device).eval()

    # 标准化常量从冻结数据集元信息读取（训练时固化的那一组）
    d = load_dataset(ROOT / "datasets" / "corazza_train.npz")
    meta = {
        "mu_log": float(d["mu_log"]),
        "sigma_log": float(d["sigma_log"]),
        "eps": float(d["eps"]),
        "device": device,
        "noise_dim": ckpt.get("noise_dim", NOISE_DIM),
        "seq_len": ckpt.get("seq_len", SEQ_LEN),
    }
    return G, meta


@torch.no_grad()
def generate_envelopes(G, meta: dict, alpha_deg: float, n: int,
                       batch_size: int = 256,
                       seed: int | None = None) -> np.ndarray:
    """生成指定仰角的 n 条物理包络序列（已逆标准化）。

    Args:
        G:         load_generator 返回的生成器
        meta:      load_generator 返回的元信息
        alpha_deg: 仰角（度），任意 [20,80] 内连续值（可未参与训练）
        n:         生成条数
        batch_size: 分批大小（防爆显存）
        seed:      torch 采样种子（None 则沿用全局状态）

    Returns:
        (n, seq_len) float64 物理包络
    """
    device = meta["device"]
    c_val = float(normalize_alpha(alpha_deg))
    outs = []
    if seed is not None:
        torch.manual_seed(seed)
    for start in range(0, n, batch_size):
        bs = min(batch_size, n - start)
        z = torch.randn(bs, meta["noise_dim"], meta["seq_len"], device=device)
        c = torch.full((bs,), c_val, device=device)
        x_std = G(z, c).squeeze(1).cpu().numpy().astype(np.float64)
        outs.append(x_std)
    x_std = np.concatenate(outs, axis=0)
    return inverse_standardize(x_std, meta["mu_log"], meta["sigma_log"],
                               meta["eps"])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str,
                   default=str(ROOT / "checkpoints" / "cgan" / "cgan_final.pt"))
    p.add_argument("--alpha", type=float, default=45.0)
    p.add_argument("--n", type=int, default=8)
    args = p.parse_args()

    G, meta = load_generator(args.ckpt)
    r = generate_envelopes(G, meta, args.alpha, args.n, seed=0)
    print(f"仰角 {args.alpha}° 生成 {args.n} 条："
          f"mean={r.mean():.4f}  std={r.std():.4f}  "
          f"min={r.min():.4f}  max={r.max():.4f}")


if __name__ == "__main__":
    main()
