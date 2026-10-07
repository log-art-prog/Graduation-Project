"""阶段四端到端冒烟：真实冻结数据上小模型快训 + DDPM/DDIM 采样链路。

运行（约 3 分钟，GPU；可与正式训练并行，显存占用很小）：
    python scripts/phase4_smoke.py

验证项：
  1. ε-MSE 从初始 ~1.0 显著下降（模型确实在学，非形状跑通而已）
  2. checkpoint 保存/加载（EMA 权重）闭环
  3. DDIM 20 步：形状/有限/物理非负/同种子可复现（链路机制；玩具模型
     欠训练，η=0 确定性链对大 t 误差敏感，分布保真度不在此判定，留给
     正式 30k 模型的 phase4 验收做 20/50/100 步 Pareto 对比）
  4. DDPM T 步：形状/有限/物理非负，并打印两采样器耗时对比

注：EMA decay 取 0.995 而非正式训练的 0.999——0.999^1500≈0.22，EMA 权重
仍被 zero-init 初值主导；0.995^1500≈6e-4，有效平均窗约 200 步，与冒烟
训练长度匹配。正式训练 30000 步仍用 0.999（有效窗 1000 步）。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from generative.data.dataset import load_dataset
from generative.models.cgan import EMA
from generative.models.ddpm import GaussianDiffusion
from generative.models.schedule import DiffusionSchedule
from generative.models.unet1d import ConditionalUNet1D
from generative.sample_ddpm import load_diffusion, generate_envelopes
from generative.utils.seed import seed_everything

SMOKE_DIR = ROOT / "results" / "phase4_smoke"
STEPS = 1500
T_SMOKE = 100
N_TRAIN = 2000
BATCH = 64
EMA_DECAY = 0.995


def main() -> None:
    seed_everything(2026)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SMOKE_DIR.mkdir(parents=True, exist_ok=True)

    # ---- 真实冻结数据抽子集 ----
    d = load_dataset(ROOT / "datasets" / "corazza_train.npz")
    x_all = torch.from_numpy(d["x_std"][:N_TRAIN]).to(device)
    c_all = torch.from_numpy(d["alpha_norm"][:N_TRAIN]).to(device)
    print(f"设备 {device}；冒烟数据 {N_TRAIN} 条 × {x_all.shape[1]} 点")

    # ---- 小模型：通道 16/32/64，T=100 ----
    model = ConditionalUNet1D(channels=(16, 32, 64), kernel=15, time_dim=64)
    schedule = DiffusionSchedule(T=T_SMOKE, kind="cosine")
    diff = GaussianDiffusion(model, schedule).to(device)
    n_param = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"冒烟模型参数量 {n_param:.2f}M")

    opt = torch.optim.Adam(model.parameters(), lr=2e-4, betas=(0.9, 0.999))
    ema = EMA(model, decay=EMA_DECAY)
    losses = []
    t0 = time.time()
    for step in range(1, STEPS + 1):
        idx = torch.randint(0, N_TRAIN, (BATCH,), device=device)
        loss = diff.training_loss(x_all[idx].unsqueeze(1), c_all[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        ema.update(model)
        losses.append(loss.item())
        if step % 250 == 0:
            print(f"  step {step:4d}  ε-MSE={loss.item():.4f}")
    print(f"训练 {STEPS} 步耗时 {time.time()-t0:.0f}s")

    early = float(np.mean(losses[:50]))
    late = float(np.mean(losses[-50:]))
    print(f"前50步均值 loss={early:.4f}；后50步均值 loss={late:.4f}")
    assert early < 1.15, f"初始损失异常（应≈1）：{early}"
    assert late < early - 0.15, "损失未显著下降：模型可能没学到东西"
    assert late < 0.75, f"收敛不足：{late:.4f}"
    print("✓ 损失显著下降")

    # ---- 存盘并走与正式评估完全相同的加载路径 ----
    ckpt = SMOKE_DIR / "smoke_ddpm.pt"
    torch.save({
        "step": STEPS, "model_ema": ema.state_dict(),
        "model_raw": model.state_dict(),
        "T": T_SMOKE, "schedule": "cosine",
        "channels": (16, 32, 64), "kernel": 15, "time_dim": 64,
        "seq_len": 1000,
    }, ckpt)
    diff2, meta = load_diffusion(ckpt)

    # ---- DDIM 20 步（玩具模型只验链路机制：形状/有限/非负/确定性）----
    t1 = time.time()
    r_ddim_a = generate_envelopes(diff2, meta, 45.0, 16, sampler="ddim",
                                  ddim_steps=20, seed=11)
    t_ddim = time.time() - t1
    assert r_ddim_a.shape == (16, 1000)
    assert np.isfinite(r_ddim_a).all()
    assert (r_ddim_a >= 0).all(), "物理包络必须处处非负"
    r_ddim_b = generate_envelopes(diff2, meta, 45.0, 16, sampler="ddim",
                                  ddim_steps=20, seed=11)
    assert np.allclose(r_ddim_a, r_ddim_b, atol=1e-5), "DDIM 同种子须可复现"
    print(f"✓ DDIM 20步：16条 {t_ddim:.1f}s，mean={r_ddim_a.mean():.3f}，"
          f"链路合法且确定性可复现")

    # ---- DDPM 100 步（随机）----
    t1 = time.time()
    r_ddpm = generate_envelopes(diff2, meta, 45.0, 8, sampler="ddpm", seed=22)
    t_ddpm = time.time() - t1
    assert r_ddpm.shape == (8, 1000)
    assert np.isfinite(r_ddpm).all()
    assert (r_ddpm >= 0).all(), "物理包络必须处处非负"
    print(f"✓ DDPM {T_SMOKE}步：8条 {t_ddpm:.1f}s，mean={r_ddpm.mean():.3f}")

    # ---- 仰角条件方向正确（粗验）：80° 均值应高于 20° ----
    r20 = generate_envelopes(diff2, meta, 20.0, 8, sampler="ddim",
                             ddim_steps=20, seed=1)
    r80 = generate_envelopes(diff2, meta, 80.0, 8, sampler="ddim",
                             ddim_steps=20, seed=2)
    print(f"  仰角方向：20° mean={r20.mean():.3f} vs 80° mean={r80.mean():.3f}")
    print(f"\n冒烟通过 ✅ 产物：{ckpt}")


if __name__ == "__main__":
    main()
