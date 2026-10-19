import json

import pytest

from seg2av.data import LOCAL_ROOT
from seg2av.env import PYPROJECT
from seg2av.train import load_config, run


@pytest.mark.skipif(not LOCAL_ROOT.exists(), reason="dados locais ausentes (make data)")
@pytest.mark.parametrize("config", ["cnn_resnet34.yaml", "transformer_mitb0.yaml"])
def test_uma_epoca_em_16_imagens_grava_run_completa(config, tmp_path, monkeypatch):
    monkeypatch.chdir(PYPROJECT.parent)
    cfg = load_config(f"configs/{config}")
    cfg["train"].update(epochs=1, num_workers=0)
    assert run(cfg, seed=0, out_dir=tmp_path, limit=16) == "finished"
    run_dir = tmp_path / f"{cfg['scenario']}_seed0"
    files = {p.name for p in run_dir.iterdir()}
    assert {"config.json", "metrics.json", "history.csv", "test_per_image.csv", "DONE"} <= files
    assert "last.pt" not in files and "best.pt" in files
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["n_test"] == 16 and 0 <= metrics["test_dice"] <= 1
    assert run(cfg, seed=0, out_dir=tmp_path, limit=16) == "done"  # idempotente


@pytest.mark.skipif(not LOCAL_ROOT.exists(), reason="dados locais ausentes (make data)")
def test_retoma_run_interrompida(tmp_path, monkeypatch):
    monkeypatch.chdir(PYPROJECT.parent)
    cfg = load_config("configs/transformer_mitb0.yaml")
    cfg["train"].update(epochs=2, num_workers=0)
    assert run(cfg, 0, tmp_path, deadline=0, limit=16) == "interrupted"  # para antes da 1ª época
    assert run(cfg, 0, tmp_path, limit=16) == "finished"
    history = (tmp_path / "transformer_mitb0_seed0" / "history.csv").read_text().splitlines()
    assert len(history) == 3  # cabeçalho + 2 épocas
