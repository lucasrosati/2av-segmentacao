import numpy as np
import pandas as pd
import pytest
from test_plots import MODEL, NAMES, _fake_checkpoint, _fake_dataset, _fake_results

from seg2av.env import PYPROJECT
from seg2av.plots import median_seed
from seg2av.slides import EXPECTED, build, crop_header
from seg2av.stats import ALT, BASE


@pytest.fixture
def package(tmp_path, monkeypatch):
    monkeypatch.chdir(PYPROJECT.parent)  # configs/ e splits/ reais
    rng = np.random.default_rng(0)
    runs = _fake_results(tmp_path, rng)
    _fake_dataset(tmp_path / "data", rng)
    for s in (BASE, ALT):
        _fake_checkpoint(tmp_path / "runs" / f"{s}_seed{median_seed(runs, s)}", MODEL[s])
    missing = build(
        tmp_path, root=tmp_path / "data", pdf=tmp_path / "sem.pdf", sample_names=NAMES[:4]
    )
    return tmp_path / "slides", missing


def test_expected_files_exist_and_pngs_are_1920x1080(package):
    out, missing = package
    assert missing == ["paper_header.png"]  # sem PDF do paper neste teste
    for f in EXPECTED:
        assert f in missing or (out / f).stat().st_size > 0, f
    import cv2

    for f in EXPECTED:
        if f.endswith(".png") and f not in missing:
            assert cv2.imread(str(out / f)).shape[:2] == (1080, 1920), f


def test_numbers_trace_to_runs_csv(package, tmp_path):
    out, _ = package
    runs = pd.read_csv(tmp_path / "runs.csv")
    key = (out / "key_numbers.md").read_text()
    for s in (BASE, ALT):
        r = runs[runs.scenario == s]
        assert f"{r.test_dice.mean():.4f}".replace(".", ",") in key
        assert f"{r.test_dice.std():.4f}".replace(".", ",") in key
    compare = pd.read_csv(out / "table_compare.csv")
    assert list(compare.columns) == ["Atributo", "ResNet-34", "MiT-B0"]
    assert "TODO: pendente" in (out / "HANDOFF.md").read_text()  # header e GIF da demo
    assert "não\n  estão no GitHub" in (out / "HANDOFF.md").read_text()
    assert "H0" in (out / "problem.md").read_text()


def test_crop_header_trims_margins():
    page = np.full((1000, 800, 3), 255, np.uint8)
    page[100:150, 200:600] = 0  # "título"
    page[900:950, 100:700] = 0  # texto fora do topo: descartado
    crop = crop_header(page, pad=10)
    assert crop.shape[:2] == (50 + 20, 400 + 20)
