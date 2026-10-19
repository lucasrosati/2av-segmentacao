"""Loss Dice + BCE (pesos iguais), sobre logits."""

import segmentation_models_pytorch as smp
from torch import nn


class DiceBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice = smp.losses.DiceLoss(mode="binary", from_logits=True)
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, target):
        logits = logits.float()  # fora do autocast: loss em fp32
        return self.dice(logits, target) + self.bce(logits, target)
