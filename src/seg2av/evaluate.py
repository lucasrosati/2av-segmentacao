"""Métricas por imagem (Dice, IoU com limiar 0,5), latência de inferência e baseline zero."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from seg2av.data import KvasirSeg, load_split, resolve_root


def per_image_scores(pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """pred/target binários (N,1,H,W). Imagem sem pólipo e sem predição conta como acerto (1)."""
    pred, target = pred.flatten(1).float(), target.flatten(1).float()
    inter = (pred * target).sum(1)
    total = pred.sum(1) + target.sum(1)
    union = total - inter
    dice = torch.where(total > 0, 2 * inter / total.clamp(min=1), torch.ones_like(total))
    iou = torch.where(union > 0, inter / union.clamp(min=1), torch.ones_like(union))
    return dice, iou


@torch.no_grad()
def evaluate(model, loader, device, amp: bool = False) -> dict[str, np.ndarray]:
    model.eval()
    dices, ious = [], []
    for image, mask in loader:
        with torch.autocast(device.type, enabled=amp):
            logits = model(image.to(device))
        d, i = per_image_scores(logits > 0, mask.to(device) > 0.5)  # logit > 0 ⇔ sigmoide > 0,5
        dices.append(d.cpu())
        ious.append(i.cpu())
    return {"dice": torch.cat(dices).numpy(), "iou": torch.cat(ious).numpy()}


@torch.no_grad()
def latency_ms(model, device, image_size: int, amp: bool = False, warmup=10, iters=50) -> float:
    """Mediana do tempo de inferência de 1 imagem (batch 1), em ms, após aquecimento."""
    model.eval()
    x = torch.randn(1, 3, image_size, image_size, device=device)
    sync = torch.cuda.synchronize if device.type == "cuda" else lambda: None
    times = []
    for i in range(warmup + iters):
        sync()
        t0 = time.perf_counter()
        with torch.autocast(device.type, enabled=amp):
            model(x)
        sync()
        if i >= warmup:
            times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times))


def zero_mask_baseline(loader) -> dict[str, float]:
    """Piso de sanidade: prever 'sem pólipo' em toda imagem."""
    dices, ious = [], []
    for _, mask in loader:
        d, i = per_image_scores(torch.zeros_like(mask, dtype=torch.bool), mask > 0.5)
        dices.append(d)
        ious.append(i)
    return {"dice": float(torch.cat(dices).mean()), "iou": float(torch.cat(ious).mean())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cnn_resnet34.yaml")
    parser.add_argument("--out", type=Path, default=Path("results/baselines/zero_mask.json"))
    args = parser.parse_args()
    d = yaml.safe_load(Path(args.config).read_text())["data"]
    names = load_split(Path(d["split_file"]), Path(d["split_sha256_file"]))["test"]
    loader = DataLoader(KvasirSeg(resolve_root(d["root"]), names, d["image_size"]), batch_size=32)
    result = {
        "baseline": "zero_mask",
        "split": "test",
        "n": len(names),
        **zero_mask_baseline(loader),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(result)


if __name__ == "__main__":
    main()
