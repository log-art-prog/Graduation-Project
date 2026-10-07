"""阶段四：条件扩散模型训练脚本（DDPM ε-prediction + FiLM + EMA）。

运行（leo_gen 环境，与 CGAN 公平对齐的默认超参）：
    python -m generative.train_ddpm --steps 30000 --schedule linear

数据：datasets/corazza_train.npz（与 CGAN 同一份冻结数据 + 同一 log 标准化）
超参：Adam lr=2e-4, β=(0.9,0.999)（标准 DDPM，与 CGAN 的 WGAN 设置不同，
      训练设定差异在论文中如实报告）；batch=64；生成器(去噪网络) EMA=0.999。

公平性口径一（训练同步数）：默认 30000 优化步。注意 CGAN 一步含 6 次
网络前向（n_critic+1），DDPM 一步仅 1 次前向；"同墙钟"口径二由阶段 5
记录实际耗时后补报。

产物（checkpoints/ddpm/ 下）：
    ddpm_final.pt      最终权重（含 EMA 去噪网络 + 全部重建参数）
    ddpm_step{N}.pt    周期存档（可画收敛曲线）
    train_log.csv      每 100 步的 ε-MSE（只有一条平滑曲线，无对抗博弈）
    train_curve.png    损失曲线
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
import torch

from generative.data.dataset import load_dataset
from generative.models.ddpm import GaussianDiffusion
from generative.models.schedule import DiffusionSchedule
from generative.models.unet1d import ConditionalUNet1D
from generative.models.cgan import EMA
from generative.utils.seed import seed_everything

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def parse_channels(text: str) -> tuple:
    return tuple(int(x.strip()) for x in text.split(",") if x.strip())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=30000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--T", type=int, default=1000, help="前向扩散总步数")
    p.add_argument("--schedule", choices=["linear", "cosine"], default="linear")
    p.add_argument("--channels", type=str, default="64,128,256",
                   help="U-Net 三级通道，逗号分隔")
    p.add_argument("--kernel", type=int, default=15)
    p.add_argument("--time-dim", type=int, default=128)
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--save-every", type=int, default=5000)
    p.add_argument("--out-dir", type=str,
                   default=str(ROOT / "checkpoints" / "ddpm"))
    return p.parse_args()


def save_ckpt(path: Path, diff: GaussianDiffusion, ema: EMA,
              args: argparse.Namespace, step: int) -> None:
    torch.save({
        "step": step,
        "model_ema": ema.state_dict(),
        "model_raw": diff.model.state_dict(),
        "config": vars(args),
        "T": args.T,
        "schedule": args.schedule,
        "channels": parse_channels(args.channels),
        "kernel": args.kernel,
        "time_dim": args.time_dim,
        "seq_len": 1000,
    }, path)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    d = load_dataset(ROOT / "datasets" / "corazza_train.npz")
    x_all = torch.from_numpy(d["x_std"]).to(device)
    c_all = torch.from_numpy(d["alpha_norm"]).to(device)
    n_total = x_all.shape[0]
    print(f"数据集: {n_total} 条 × 1000 点；调度: {args.schedule} T={args.T}")

    channels = parse_channels(args.channels)
    model = ConditionalUNet1D(channels=channels, kernel=args.kernel,
                              time_dim=args.time_dim)
    schedule = DiffusionSchedule(T=args.T, kind=args.schedule)
    diff = GaussianDiffusion(model, schedule).to(device)
    print(f"去噪网络参数量: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 betas=(0.9, 0.999))
    ema = EMA(model, decay=args.ema_decay)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.csv"

    perm = torch.randperm(n_total, device=device)
    cursor = 0
    history = {"step": [], "loss": []}
    t0 = time.time()

    def next_batch(bs: int):
        nonlocal perm, cursor
        if cursor + bs > n_total:
            perm = torch.randperm(n_total, device=device)
            cursor = 0
        idx = perm[cursor:cursor + bs]
        cursor += bs
        return x_all[idx].unsqueeze(1), c_all[idx]

    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "loss", "elapsed_s"])

        for step in range(1, args.steps + 1):
            x0, c = next_batch(args.batch_size)
            loss = diff.training_loss(x0, c)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            ema.update(model)

            if step % args.log_every == 0:
                el = time.time() - t0
                writer.writerow([step, f"{loss.item():.6f}", f"{el:.1f}"])
                f.flush()
                history["step"].append(step)
                history["loss"].append(loss.item())
                print(f"step {step:6d}/{args.steps}  ε-MSE={loss.item():.4f}  "
                      f"({el:.0f}s)")

            if step % args.save_every == 0:
                save_ckpt(out_dir / f"ddpm_step{step}.pt", diff, ema, args, step)

    save_ckpt(out_dir / "ddpm_final.pt", diff, ema, args, args.steps)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history["step"], history["loss"], lw=0.9)
    ax.set_xlabel("优化步数")
    ax.set_ylabel("ε-MSE 训练损失")
    ax.set_title(f"DDPM 训练曲线（{args.schedule}, T={args.T}, seed={args.seed}）")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "train_curve.png", dpi=120)
    plt.close()

    print(f"完成：{out_dir / 'ddpm_final.pt'}（耗时 {time.time()-t0:.0f}s）")


if __name__ == "__main__":
    main()
