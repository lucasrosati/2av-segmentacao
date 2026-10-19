"""Métricas por imagem (Dice, IoU com limiar 0,5) e avaliação de um modelo num DataLoader."""

import numpy as np
import torch


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
