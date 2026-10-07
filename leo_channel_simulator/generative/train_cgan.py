"""阶段三：CGAN 训练脚本（WGAN-GP + TTUR + EMA [+ 条件单调一致性正则]）。

运行（leo_gen 环境）：
    python -m generative.train_cgan --steps 30000 --seed 42                     # 基线复现
    python -m generative.train_cgan --seed 42 --lambda-mono 5.0 --out-dir checkpoints/cgan_mono  # 方案 R

数据：datasets/corazza_train.npz（10000×1000，float32 全量驻留 GPU，仅 40MB）
基线超参：严格按学长论文表 3-2（见 generative/models/cgan.py 文档字符串）。

方案 R（--lambda-mono>0 启用）：双种子复现（seed42/seed1234）证实 WGAN-GP
存在随种子漂移的条件均值交错（KS≈0.22 的失败仰角分别为 20°/40°）。加入物理
先验正则 monotone_condition_penalty：同一 z 在有序仰角网格上的包络均值必须
随仰角非降，铰链惩罚逆序对。该目标函数改动在论文中如实报告，基线权重已冻结
于 checkpoints/cgan/，对照不受影响。

产物（--out-dir 下）：
  cgan_final.pt       最终权重（含 EMA 生成器，采样用它）
  cgan_step{N}.pt     周期存档
  train_log.csv       每 100 步的 d_loss / g_loss / w_dist / mono
  train_curves.png    训练曲线（对标论文图 3-5）
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from generative.data.dataset import load_dataset
from generative.models.cgan import (
    EMA, LargeReceptiveFieldDiscriminator, SequenceNoiseFilterGenerator,
    gradient_penalty, monotone_condition_penalty,
    NOISE_DIM, SEQ_LEN,
)
from generative.utils.seed import seed_everything

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=30000, help="生成器训练步数")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr-g", type=float, default=1e-4)
    p.add_argument("--lr-d", type=float, default=3e-4)
    p.add_argument("--n-critic", type=int, default=5)
    p.add_argument("--lambda-gp", type=float, default=10.0)
    p.add_argument("--lambda-mono", type=float, default=0.0,
                   help="条件单调一致性正则权重；0 关闭（基线），方案 R 取 5.0")
    p.add_argument("--mono-grid", type=int, default=4,
                   help="单调正则有序条件网格点数（4 → 6 个条件对）")
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--save-every", type=int, default=5000)
    p.add_argument("--out-dir", type=str, default=str(ROOT / "checkpoints" / "cgan"))
    return p.parse_args()


def save_ckpt(path: Path, G, D, ema: EMA, args, step: int) -> None:
    torch.save({
        "step": step,
        "g_ema": ema.state_dict(),          # 采样/评估用
        "g_raw": G.state_dict(),
        "d": D.state_dict(),
        "config": vars(args),
        "noise_dim": NOISE_DIM,
        "seq_len": SEQ_LEN,
    }, path)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    # ---- 数据全量驻留 GPU（10000×1000 float32 ≈ 40MB）----
    d = load_dataset(ROOT / "datasets" / "corazza_train.npz")
    x_all = torch.from_numpy(d["x_std"]).to(device)          # (N, L)
    c_all = torch.from_numpy(d["alpha_norm"]).to(device)     # (N,)
    n_total = x_all.shape[0]
    print(f"数据集: {n_total} 条 × {SEQ_LEN} 点（已驻留 {device}）")

    G = SequenceNoiseFilterGenerator().to(device)
    D = LargeReceptiveFieldDiscriminator().to(device)
    n_g = sum(p.numel() for p in G.parameters())
    n_d = sum(p.numel() for p in D.parameters())
    print(f"参数量: G={n_g/1e6:.2f}M  D={n_d/1e6:.2f}M")

    opt_g = torch.optim.Adam(G.parameters(), lr=args.lr_g, betas=(0.0, 0.9))
    opt_d = torch.optim.Adam(D.parameters(), lr=args.lr_d, betas=(0.0, 0.9))
    ema = EMA(G, decay=args.ema_decay)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.csv"

    perm = torch.randperm(n_total, device=device)  # 手动洗牌批采样
    cursor = 0
    history = {"step": [], "d_loss": [], "g_loss": [], "w_dist": [], "mono": []}
    t0 = time.time()

    def next_batch(bs: int):
        nonlocal perm, cursor
        if cursor + bs > n_total:
            perm = torch.randperm(n_total, device=device)
            cursor = 0
        idx = perm[cursor:cursor + bs]
        cursor += bs
        # x: (B, L) → (B, 1, L)
        return x_all[idx].unsqueeze(1), c_all[idx]

    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "d_loss", "g_loss", "w_dist", "mono", "elapsed_s"])

        for step in range(1, args.steps + 1):
            # ---------- D 训练 n_critic 次 ----------
            for _ in range(args.n_critic):
                x_real, c = next_batch(args.batch_size)
                z = torch.randn(args.batch_size, NOISE_DIM, SEQ_LEN, device=device)
                x_fake = G(z, c)
                d_real = D(x_real, c).mean()
                d_fake = D(x_fake, c).mean()
                gp = gradient_penalty(D, x_real, x_fake, c, args.lambda_gp)
                d_loss = d_fake - d_real + gp
                opt_d.zero_grad(set_to_none=True)
                d_loss.backward()
                opt_d.step()

            # ---------- G 训练 1 次 ----------
            _, c_g = next_batch(args.batch_size)
            z = torch.randn(args.batch_size, NOISE_DIM, SEQ_LEN, device=device)
            g_adv = -D(G(z, c_g), c_g).mean()
            if args.lambda_mono > 0.0:
                # 同一 z 跨条件复用，惩罚仰角-均值逆序对（方案 R）
                mono = monotone_condition_penalty(
                    G, z, n_grid=args.mono_grid, margin=0.0)
                g_loss = g_adv + args.lambda_mono * mono
            else:
                mono = torch.zeros((), device=device)
                g_loss = g_adv
            opt_g.zero_grad(set_to_none=True)
            g_loss.backward()
            opt_g.step()
            ema.update(G)

            w_dist = (d_real - d_fake).item()  # Wasserstein 距离估计

            if step % args.log_every == 0:
                el = time.time() - t0
                writer.writerow([step, f"{d_loss.item():.4f}",
                                 f"{g_loss.item():.4f}", f"{w_dist:.4f}",
                                 f"{mono.item():.6f}", f"{el:.1f}"])
                f.flush()
                history["step"].append(step)
                history["d_loss"].append(d_loss.item())
                history["g_loss"].append(g_loss.item())
                history["w_dist"].append(w_dist)
                history["mono"].append(mono.item())
                print(f"step {step:6d}/{args.steps}  d={d_loss.item():+.3f}  "
                      f"g={g_loss.item():+.3f}  W={w_dist:.4f}  "
                      f"mono={mono.item():.4f}  ({el:.0f}s)")

            if step % args.save_every == 0:
                save_ckpt(out_dir / f"cgan_step{step}.pt", G, D, ema, args, step)

    save_ckpt(out_dir / "cgan_final.pt", G, D, ema, args, args.steps)

    # ---- 训练曲线（对标论文图 3-5；启用正则时多画 mono 面板）----
    keys = ["d_loss", "g_loss", "w_dist"]
    titles = ["判别器损失", "生成器损失", "Wasserstein 距离估计"]
    if args.lambda_mono > 0.0:
        keys.append("mono")
        titles.append(f"单调惩罚（λ={args.lambda_mono:g}）")
    fig, axes = plt.subplots(1, len(keys), figsize=(5 * len(keys), 4))
    for ax, key, title in zip(axes, keys, titles):
        ax.plot(history["step"], history[key], lw=0.8)
        ax.set_xlabel("生成器步数")
        ax.set_title(title)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "train_curves.png", dpi=120)
    plt.close()

    print(f"完成：{out_dir / 'cgan_final.pt'}（耗时 {time.time()-t0:.0f}s）")


if __name__ == "__main__":
    main()
