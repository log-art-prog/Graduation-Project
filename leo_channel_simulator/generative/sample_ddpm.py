"""阶段四：扩散模型采样 — 加载 EMA 去噪网络，按仰角生成物理包络。

与 generative/sample.py（CGAN）接口对称，供阶段四/五评估脚本调用：
    from generative.sample_ddpm import load_diffusion, generate_envelopes
    diff, meta = load_diffusion("checkpoints/ddpm/ddpm_final.pt")
    r = generate_envelopes(diff, meta, 45.0, 500, sampler="ddim", ddim_steps=50)

命令行：
    python -m generative.sample_ddpm --alpha 45 --n 8 --sampler ddim --ddim-steps 50
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
from generative.models.ddim import ddim_sample
from generative.models.ddpm import GaussianDiffusion
from generative.models.schedule import DiffusionSchedule
from generative.models.unet1d import ConditionalUNet1D, SEQ_LEN


def load_diffusion(ckpt_path: str | Path,
                   device: torch.device | str | None = None,
                   use_ema: bool = True):
    """重建调度表+去噪网络（默认武装 EMA 权重），返回 (diffusion, meta)。"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    channels = tuple(ckpt.get("channels", (64, 128, 256)))
    model = ConditionalUNet1D(
        channels=channels,
        kernel=ckpt.get("kernel", 15),
        time_dim=ckpt.get("time_dim", 128),
        seq_len=ckpt.get("seq_len", SEQ_LEN),
    )
    model.load_state_dict(ckpt["model_ema"] if use_ema else ckpt["model_raw"])
    schedule = DiffusionSchedule(T=ckpt.get("T", 1000),
                                 kind=ckpt.get("schedule", "linear"))
    diff = GaussianDiffusion(model, schedule).to(device).eval()

    d = load_dataset(ROOT / "datasets" / "corazza_train.npz")
    meta = {
        "mu_log": float(d["mu_log"]),
        "sigma_log": float(d["sigma_log"]),
        "eps": float(d["eps"]),
        "device": device,
        "seq_len": ckpt.get("seq_len", SEQ_LEN),
        "T": ckpt.get("T", 1000),
        "schedule": ckpt.get("schedule", "linear"),
    }
    return diff, meta


@torch.no_grad()
def generate_envelopes(diff: GaussianDiffusion, meta: dict, alpha_deg: float,
                       n: int, sampler: str = "ddpm",
                       ddim_steps: int = 50, eta: float = 0.0,
                       clip_x0: float | None = 30.0,
                       batch_size: int = 64,
                       seed: int | None = None) -> np.ndarray:
    """生成指定仰角的 n 条物理包络（已逆标准化）。

    Args:
        sampler:    "ddpm"（T 步随机祖先采样）或 "ddim"（确定性跳步加速）
        ddim_steps: DDIM 跳步步数（仅 sampler="ddim" 时生效，规划取 20/50/100）
        clip_x0:    反向过程 x̂₀ 裁剪门限（标准化 log 域），默认 30.0。
                    真实数据范围 [-16.7, 3.5]，该门限保留全部真实尾部，
                    仅驯服数值爆炸，保证逆标准化 exp 不溢出；None 关闭。
    Returns:
        (n, seq_len) float64 物理包络
    """
    device = meta["device"]
    seq_len = meta["seq_len"]
    c_val = float(normalize_alpha(alpha_deg))
    if seed is not None:
        torch.manual_seed(seed)

    outs = []
    for start in range(0, n, batch_size):
        bs = min(batch_size, n - start)
        c = torch.full((bs,), c_val, device=device)
        if sampler == "ddpm":
            x_std = diff.sample(c, bs, seq_len=seq_len, clip_x0=clip_x0)
        elif sampler == "ddim":
            xT = torch.randn(bs, 1, seq_len, device=device)
            x_std = ddim_sample(diff.model, diff.schedule, c, xT,
                                num_steps=ddim_steps, eta=eta,
                                clip_x0=clip_x0)
        else:
            raise ValueError(f"未知采样器 {sampler!r}（支持 ddpm / ddim）")
        outs.append(x_std.squeeze(1).cpu().numpy().astype(np.float64))
    x_std = np.concatenate(outs, axis=0)
    r = inverse_standardize(x_std, meta["mu_log"], meta["sigma_log"],
                            meta["eps"])
    # 物理域契约：包络必须非负。exp(·)−eps 在极负尾部（x<-26.6，仅数值
    # 裁剪才会到达）产生 -eps 量级伪负值，投影回 0；真实范围内不受影响
    return np.maximum(r, 0.0)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str,
                   default=str(ROOT / "checkpoints" / "ddpm" / "ddpm_final.pt"))
    p.add_argument("--alpha", type=float, default=45.0)
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--sampler", choices=["ddpm", "ddim"], default="ddim")
    p.add_argument("--ddim-steps", type=int, default=50)
    args = p.parse_args()

    diff, meta = load_diffusion(args.ckpt)
    r = generate_envelopes(diff, meta, args.alpha, args.n,
                           sampler=args.sampler, ddim_steps=args.ddim_steps,
                           seed=0)
    print(f"仰角 {args.alpha}° {args.sampler}"
          f"{('/'+str(args.ddim_steps)+'步') if args.sampler=='ddim' else ''} "
          f"生成 {args.n} 条：mean={r.mean():.4f}  std={r.std():.4f}  "
          f"min={r.min():.4f}  max={r.max():.4f}")


if __name__ == "__main__":
    main()
