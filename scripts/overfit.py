"""Sanidade do M2: o modelo precisa decorar 8 imagens de treino (Dice > 0,9), sem augmentation."""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from seg2av.data import KvasirSeg, load_split, resolve_root
from seg2av.evaluate import evaluate
from seg2av.losses import DiceBCELoss
from seg2av.models import build_model
from seg2av.train import load_config, pick_device, seed_everything

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
parser.add_argument("--steps", type=int, default=150)
args = parser.parse_args()

cfg = load_config(args.config)
seed_everything(0)
device = pick_device()
d = cfg["data"]
names = load_split(Path(d["split_file"]), Path(d["split_sha256_file"]))["train"][:8]
loader = DataLoader(KvasirSeg(resolve_root(d["root"]), names, d["image_size"]), batch_size=8)
model = build_model(cfg["model"]).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"])
loss_fn = DiceBCELoss()
image, mask = (x.to(device) for x in next(iter(loader)))
for step in range(1, args.steps + 1):
    model.train()
    opt.zero_grad()
    loss = loss_fn(model(image), mask)
    loss.backward()
    opt.step()
    if step % 25 == 0:
        dice = evaluate(model, loader, device)["dice"].mean()
        print(f"{cfg['scenario']} passo {step}: loss {loss.item():.3f}, Dice {dice:.3f}")
print("OK" if dice > 0.9 else "FALHOU", f"Dice final {dice:.3f} (meta > 0,9)")
