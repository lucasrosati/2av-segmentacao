"""Cenários: U-Net (A) + encoder pré-treinado no ImageNet (B) + ResNet-34 (C) ou MiT-B0 (C')."""

import segmentation_models_pytorch as smp
from torch import nn


def build_model(model_cfg: dict) -> nn.Module:
    assert model_cfg["arch"] == "Unet", "o desenho fixa A = U-Net nos dois cenários"
    return smp.Unet(
        encoder_name=model_cfg["encoder_name"],
        encoder_weights=model_cfg["encoder_weights"],
        classes=model_cfg["classes"],
    )


def n_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())
