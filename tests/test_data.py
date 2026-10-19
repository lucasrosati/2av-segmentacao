import json

import numpy as np
import pytest
import torch

from seg2av.data import (
    LOCAL_ROOT,
    KvasirSeg,
    build_transform,
    list_names,
    load_split,
    split_names,
    write_split,
)
from seg2av.env import PYPROJECT

SPLIT = PYPROJECT.parent / "splits/split_v1.json"
NAMES = [f"img{i:04d}" for i in range(1000)]


def test_split_e_deterministico_e_80_10_10():
    a, b = split_names(NAMES, seed=0), split_names(NAMES, seed=0)
    assert a == b
    assert [len(a[k]) for k in ("train", "val", "test")] == [800, 100, 100]
    assert split_names(NAMES, seed=1) != a


def test_split_versionado_sem_vazamento_e_hash_confere():
    split = load_split(SPLIT)
    sets = [set(split[k]) for k in ("train", "val", "test")]
    assert [len(s) for s in sets] == [800, 100, 100]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    assert len(set().union(*sets)) == 1000


def test_hash_divergente_aborta(tmp_path):
    path = tmp_path / "split.json"
    write_split(split_names(NAMES), path)
    path.write_text(json.dumps({"train": [], "val": [], "test": []}))
    with pytest.raises(ValueError, match="alterado"):
        load_split(path)


def test_augmentation_geometrica_aplica_igual_na_imagem_e_na_mascara():
    rng = np.random.default_rng(0)
    mask = (rng.random((64, 64)) > 0.5).astype(np.uint8)
    image = np.zeros((64, 64, 3), np.uint8)
    image[..., 0] = mask * 255  # canal R codifica a máscara
    tf = build_transform(64, ["hflip", "vflip", "rotate90"])
    for _ in range(20):
        out = tf(image=image, mask=mask)
        red_on = out["image"][0] > 0  # normalizado: 255 vira positivo, 0 vira negativo
        assert torch.equal(red_on, out["mask"].bool())


@pytest.mark.skipif(not LOCAL_ROOT.exists(), reason="dados locais ausentes (make data)")
def test_dataset_real_mascara_binaria_e_alinhada():
    assert len(list_names(LOCAL_ROOT)) == 1000
    names = load_split(SPLIT)["train"][:8]
    ds = KvasirSeg(LOCAL_ROOT, names, 256, ["hflip", "vflip", "rotate90", "color_jitter_light"])
    for image, mask in ds:
        assert image.shape == (3, 256, 256) and mask.shape == (1, 256, 256)
        assert set(mask.unique().tolist()) <= {0.0, 1.0}
        assert mask.sum() > 0  # todo exemplo do Kvasir-SEG tem pólipo
