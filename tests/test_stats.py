import numpy as np
import pandas as pd
import pytest

from seg2av.stats import ALT, BASE, compare, main, paired_stats

N = 30


def _pair(rng, effect, sd=0.01):
    a = rng.normal(0.9, 0.02, N)
    return a, a + effect + rng.normal(0, sd, N)


def test_h0_true_rarely_rejected():
    rng = np.random.default_rng(0)
    rejects = [paired_stats(*_pair(rng, 0.0), n_boot=200)["reject_h0"] for _ in range(300)]
    assert np.mean(rejects) < 0.10  # α nominal = 0,05


def test_large_effect_rejected_and_estimated():
    s = paired_stats(*_pair(np.random.default_rng(1), 0.02))
    assert s["reject_h0"] and s["t_p"] < 1e-6 and s["wilcoxon_p"] < 1e-6
    assert s["ci_low"] < 0.02 < s["ci_high"] and s["ci_low"] > 0
    assert s["d_z"] == pytest.approx(2.0, abs=0.7)  # d_z verdadeiro = 0,02 / 0,01


def test_known_values():
    a = np.zeros(N)
    d = np.linspace(-1, 3, N)  # média 1, dp amostral conhecido
    s = paired_stats(a, d, n_boot=500)
    assert s["mean_diff"] == pytest.approx(1.0)
    assert s["d_z"] == pytest.approx(1.0 / d.std(ddof=1))
    assert s["primary_test"] == "t"  # uniforme simétrica com n=30 passa no Shapiro


def test_compare_rejects_unpaired_seeds():
    runs = pd.DataFrame(
        {"scenario": [BASE, BASE, ALT], "seed": [0, 1, 0], "test_dice": [0.9, 0.8, 0.85]}
    )
    with pytest.raises(ValueError, match="sem par"):
        compare(runs, metrics=("test_dice",))


def test_main_writes_tables(tmp_path, monkeypatch):
    rng = np.random.default_rng(2)
    a, b = _pair(rng, 0.01)
    runs = pd.DataFrame(
        {
            "scenario": [BASE] * N + [ALT] * N,
            "seed": list(range(N)) * 2,
            "test_dice": np.r_[a, b],
            "test_iou": np.r_[a, b] - 0.05,
        }
    )
    runs.to_csv(tmp_path / "runs.csv", index=False)
    out = tmp_path / "tables"
    monkeypatch.setattr(
        "sys.argv", ["stats", "--runs", str(tmp_path / "runs.csv"), "--out-dir", str(out)]
    )
    main()
    assert len(pd.read_csv(out / "stats.csv")) == 2
    tex = (out / "stats.tex").read_text()
    assert r"\begin{tabular}" in tex and "Dice" in tex
    macros = (out / "numbers.tex").read_text()
    assert r"\newcommand{\DiceDecision}" in macros and r"\newcommand{\NPairs}{30}" in macros
    assert r"\newcommand{\DiceCIt}" in macros and r"\newcommand{\IoUDecision}" in macros
