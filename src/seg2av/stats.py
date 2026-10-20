"""M6: comparação pareada por seed entre os cenários (H0: mesma média de Dice no teste).

Diferença = transformer − cnn. Decisão pelo t pareado se Shapiro-Wilk não rejeita a normalidade
das diferenças, senão pelo Wilcoxon pareado; os dois são sempre reportados.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

BASE, ALT = "cnn_resnet34", "transformer_mitb0"
METRICS = ("test_dice", "test_iou")
ALPHA = 0.05  # bilateral, decidido no plano
LABELS = {"test_dice": "Dice", "test_iou": "IoU"}


def paired_stats(
    a, b, alpha: float = ALPHA, n_boot: int = 10_000, seed: int = 0
) -> dict[str, float | int | str | bool]:
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    n = len(d)
    sd = d.std(ddof=1)
    shapiro_p = stats.shapiro(d).pvalue
    t = stats.ttest_rel(b, a)
    w = stats.wilcoxon(b, a)
    boot = d[np.random.default_rng(seed).integers(0, n, (n_boot, n))].mean(axis=1)
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    primary = "t" if shapiro_p >= alpha else "wilcoxon"
    p = t.pvalue if primary == "t" else w.pvalue
    return {
        "n": n,
        "mean_base": a.mean(),
        "sd_base": a.std(ddof=1),
        "mean_alt": b.mean(),
        "sd_alt": b.std(ddof=1),
        "mean_diff": d.mean(),
        "sd_diff": sd,
        "ci_low": lo,
        "ci_high": hi,
        "d_z": d.mean() / sd,
        "shapiro_p": shapiro_p,
        "t": t.statistic,
        "t_p": t.pvalue,
        "wilcoxon_w": w.statistic,
        "wilcoxon_p": w.pvalue,
        "primary_test": primary,
        "reject_h0": bool(p < alpha),
    }


def compare(runs: pd.DataFrame, metrics=METRICS) -> pd.DataFrame:
    rows = []
    for m in metrics:
        wide = runs.pivot(index="seed", columns="scenario", values=m)
        unpaired = wide[wide[[BASE, ALT]].isna().any(axis=1)].index.tolist()
        if unpaired:
            raise ValueError(f"seeds sem par entre {BASE} e {ALT}: {unpaired}")
        rows.append({"metric": m, **paired_stats(wide[BASE], wide[ALT])})
    return pd.DataFrame(rows)


def _p(x: float) -> str:
    return "< 0,001" if x < 1e-3 else f"{x:.3f}".replace(".", ",")


def to_latex(df: pd.DataFrame) -> str:
    def f(x, k=4):
        return f"{x:.{k}f}".replace(".", ",")

    lines = [
        r"\begin{tabular}{lcccccc}",
        r"\hline",
        r"Métrica & ResNet-34 & MiT-B0 & $\bar{d}$ [IC 95\%] & $d_z$ & $p_t$ / $p_W$ & H0 \\",
        r"\hline",
    ]
    for r in df.itertuples():
        lines.append(
            f"{LABELS[r.metric]} "
            f"& {f(r.mean_base)} $\\pm$ {f(r.sd_base)} & {f(r.mean_alt)} $\\pm$ {f(r.sd_alt)} "
            f"& {f(r.mean_diff)} [{f(r.ci_low)}; {f(r.ci_high)}] & {f(r.d_z, 2)} "
            f"& {_p(r.t_p)} / {_p(r.wilcoxon_p)} "
            f"& {'rejeita' if r.reject_h0 else 'não rejeita'} ({r.primary_test}) \\\\"
        )
    lines += [r"\hline", r"\end{tabular}", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, default=Path("results/runs.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/tables"))
    args = parser.parse_args()
    df = compare(pd.read_csv(args.runs))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_dir / "stats.csv", index=False)
    (args.out_dir / "stats.tex").write_text(to_latex(df))
    print(df.T.to_string())


if __name__ == "__main__":
    main()
