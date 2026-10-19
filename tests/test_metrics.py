import torch

from seg2av.evaluate import per_image_scores


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
