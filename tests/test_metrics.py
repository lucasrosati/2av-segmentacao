import pytest
import torch

from seg2av.evaluate import latency_ms, per_image_scores, zero_mask_baseline


def _mask(*pixels, size=4):
    m = torch.zeros(1, 1, size, size, dtype=torch.bool)
    for r, c in pixels:
        m[0, 0, r, c] = True
    return m


def test_perfeito_vazio_e_parcial():
    full = _mask((0, 0), (0, 1), (1, 0), (1, 1))
    half = _mask((0, 0), (0, 1))
    empty = _mask()
    pred = torch.cat([full, empty, half, empty, full])
    target = torch.cat([full, empty, full, full, empty])
    dice, iou = per_image_scores(pred, target)
    # perfeito, ambos vazios, metade, predição vazia, alarme falso
    assert torch.allclose(dice, torch.tensor([1, 1, 2 * 2 / 6, 0, 0]))
    assert torch.allclose(iou, torch.tensor([1, 1, 0.5, 0, 0]))


def test_baseline_mascara_zero_acerta_so_imagem_vazia():
    target = torch.cat([_mask((0, 0)), _mask(), _mask((1, 1), (2, 2))]).float()
    loader = [(None, target[:2]), (None, target[2:])]
    assert zero_mask_baseline(loader) == pytest.approx({"dice": 1 / 3, "iou": 1 / 3})


def test_latencia_positiva_em_cpu():
    model = torch.nn.Conv2d(3, 1, 3, padding=1)
    assert latency_ms(model, torch.device("cpu"), 32, warmup=1, iters=3) > 0
