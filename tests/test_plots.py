import json

import cv2
import numpy as np
import pandas as pd
import torch

from seg2av.models import build_model
from seg2av.plots import median_seed, paper_figures, pick_examples
from seg2av.stats import ALT, BASE, compare

SEEDS = range(5)
NAMES = [f"img{i}" for i in range(6)]
MODEL = {BASE: "resnet34", ALT: "mit_b0"}


def _fake_results(tmp_path, rng):
    """results/ sintético no layout D3: runs.csv, tables/stats.csv e uma pasta por run."""
    rows = []
    for s in (BASE, ALT):
        for seed in SEEDS:
            run_dir = tmp_path / "runs" / f"{s}_seed{seed}"
            run_dir.mkdir(parents=True)
            epochs = int(rng.integers(5, 12))  # early stopping: comprimentos diferentes
            pd.DataFrame(
                {
                    "epoch": range(1, epochs + 1),
                    "train_loss": np.linspace(1, 0.2, epochs),
                    "val_dice": np.linspace(0.7, 0.9, epochs) + rng.normal(0, 0.01, epochs),
                }
            ).to_csv(run_dir / "history.csv", index=False)
            pd.DataFrame({"name": NAMES, "dice": rng.random(len(NAMES))}).to_csv(
                run_dir / "test_per_image.csv", index=False
            )
            rows.append(
                {
                    "scenario": s,
                    "seed": seed,
                    "test_dice": 0.9 + (0.01 if s == ALT else 0) + rng.normal(0, 0.01),
                    "test_iou": 0.8 + rng.normal(0, 0.01),
                    "n_params": 24_000_000 if s == BASE else 5_500_000,
                    "latency_ms": 9.0 if s == BASE else 14.5,
                    "latency_device": "Tesla T4",
                }
            )
    runs = pd.DataFrame(rows)
    runs.to_csv(tmp_path / "runs.csv", index=False)
    (tmp_path / "tables").mkdir()
    compare(runs).to_csv(tmp_path / "tables" / "stats.csv", index=False)
    return runs


def _fake_dataset(root, rng):
    for sub in ("images", "masks"):
        (root / sub).mkdir(parents=True)
    for n in NAMES:
        cv2.imwrite(str(root / "images" / f"{n}.jpg"), rng.integers(0, 255, (80, 96, 3), np.uint8))
        mask = np.zeros((80, 96), np.uint8)
        mask[20:60, 30:70] = 255
        cv2.imwrite(str(root / "masks" / f"{n}.jpg"), mask)


def _fake_checkpoint(run_dir, encoder):
    model_cfg = {"arch": "Unet", "encoder_name": encoder, "encoder_weights": None, "classes": 1}
    cfg = {"model": model_cfg, "data": {"image_size": 64}}
    (run_dir / "config.json").write_text(json.dumps({"config": cfg}))
    torch.save(build_model(model_cfg).state_dict(), run_dir / "best.pt")


def test_paper_figures_from_synthetic_results(tmp_path):
    rng = np.random.default_rng(0)
    runs = _fake_results(tmp_path, rng)
    _fake_dataset(tmp_path / "data", rng)
    for s in (BASE, ALT):
        _fake_checkpoint(tmp_path / "runs" / f"{s}_seed{median_seed(runs, s)}", MODEL[s])
    paper_figures(tmp_path, root=tmp_path / "data")
    for name in ("dice_box", "paired_diff", "curves", "dice_latency", "qualitative"):
        assert (tmp_path / "figures" / f"{name}.png").stat().st_size > 10_000, name


def test_pick_examples_best_and_worst():
    per_image = {
        s: [pd.DataFrame({"name": ["a", "b", "c", "d", "e"], "dice": [0.9, 0.1, 0.5, 0.8, 0.2]})]
        for s in (BASE, ALT)
    }
    assert pick_examples(per_image) == ["a", "d", "e", "b"]


def test_median_seed():
    runs = pd.DataFrame({"scenario": [BASE] * 3, "seed": [0, 1, 2], "test_dice": [0.7, 0.9, 0.8]})
    assert median_seed(runs, BASE) == 2
