"""M6: comparação pareada por seed entre os cenários (H0: mesma média de Dice no teste).

Diferença = transformer − cnn. Decisão pelo t pareado se Shapiro-Wilk não rejeita a normalidade
das diferenças, senão pelo Wilcoxon pareado; os dois são sempre reportados.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
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


def _f(x, k=4) -> str:
    return f"{x:.{k}f}".replace(".", ",").replace("-", "\\textminus{}")


def _metric_macros(r, prefix: str) -> dict[str, str]:
    test = "teste \\textit{t} pareado" if r.primary_test == "t" else "teste de Wilcoxon pareado"
    half = stats.t.ppf(1 - ALPHA / 2, r.n - 1) * r.sd_diff / np.sqrt(r.n)
    return {
        f"{prefix}Base": f"{_f(r.mean_base)} $\\pm$ {_f(r.sd_base)}",
        f"{prefix}Alt": f"{_f(r.mean_alt)} $\\pm$ {_f(r.sd_alt)}",
        f"{prefix}Diff": _f(r.mean_diff),
        f"{prefix}CI": f"[{_f(r.ci_low)}; {_f(r.ci_high)}]",
        # IC t da mesma diferença: só para discutir a divergência com o bootstrap, não decide nada
        f"{prefix}CIt": f"[{_f(r.mean_diff - half)}; {_f(r.mean_diff + half)}]",
        f"{prefix}Dz": _f(r.d_z, 2),
        f"{prefix}ShapiroP": _p(r.shapiro_p),
        f"{prefix}TP": _p(r.t_p),
        f"{prefix}WP": _p(r.wilcoxon_p),
        f"{prefix}Test": test,
        f"{prefix}Decision": "rejeita-se H0" if r.reject_h0 else "não se rejeita H0",
    }


def _scenario_macros(runs: pd.DataFrame) -> dict[str, str]:
    out = {}
    for s, suffix in ((BASE, "Base"), (ALT, "Alt")):
        r = runs[runs.scenario == s]
        out |= {
            f"Params{suffix}": _f(r.n_params.iloc[0] / 1e6, 1),
            f"ParamsEnc{suffix}": _f(r.n_params_encoder.iloc[0] / 1e6, 1),
            f"Lat{suffix}": _f(r.latency_ms.median(), 1),
            f"TrainMin{suffix}": _f(r.train_time_s.mean() / 60, 1),
            f"EpochTime{suffix}": _f(r.epoch_time_s.mean(), 1),
            f"Epochs{suffix}": _f(r.epochs_run.mean(), 1),
            f"BestEpoch{suffix}": _f(r.best_epoch.mean(), 1),
            f"EarlyStop{suffix}": str(int(r.early_stopped.sum())),
            f"BestVal{suffix}": _f(r.best_val_dice.mean()),
        }
    lat = runs.groupby("scenario").latency_ms.median()
    params = runs.groupby("scenario").n_params.first()
    d = runs.pivot(index="seed", columns="scenario", values="test_dice")
    out |= {
        "LatRatio": _f(lat[ALT] / lat[BASE], 1),
        "ParamsRatio": _f(params[BASE] / params[ALT], 1),
        "LatDevice": str(runs.latency_device.iloc[0]),
        "NFavorAlt": str(int((d[ALT] > d[BASE]).sum())),
        "NTest": str(int(runs.n_test.iloc[0])),
    }
    return out


def to_macros(df: pd.DataFrame, runs: pd.DataFrame | None = None, extra=None) -> str:
    """Macros para o texto do paper (\\input em main.tex), sem número digitado à mão."""
    m = df.set_index("metric")
    macros = _metric_macros(m.loc["test_dice"], "Dice") | {"NPairs": str(int(m.loc["test_dice"].n))}
    if "test_iou" in m.index:
        macros |= _metric_macros(m.loc["test_iou"], "IoU")
    if runs is not None and "n_params" in runs:  # runs.csv completo do collect
        macros |= _scenario_macros(runs)
    macros |= extra or {}
    lines = ["% gerado por make stats a partir de results/ e configs/; não editar"]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in macros.items()]
    return "\n".join(lines) + "\n"


def _protocol_macros(config: Path, split: Path, baseline: Path, budget: Path) -> dict[str, str]:
    """Protocolo e custo para o paper; cada arquivo ausente só deixa de gerar as suas macros."""
    out = {}
    if config.exists():
        c = yaml.safe_load(config.read_text())
        t = c["train"]
        mant, exp = f"{t['lr']:.0e}".split("e")
        out |= {
            "ImageSize": str(c["data"]["image_size"]),
            "MaxEpochs": str(t["epochs"]),
            "BatchSize": str(t["batch_size"]),
            "LR": f"${mant} \\times 10^{{{int(exp)}}}$",
            "WeightDecay": f"$10^{{{int(np.log10(t['weight_decay']))}}}$",
            "Patience": str(t["early_stopping"]["patience"]),
        }
    if split.exists():
        sp = json.loads(split.read_text())
        out |= {f"N{k.capitalize()}": str(len(v)) for k, v in sp.items()}
    if baseline.exists():
        b = json.loads(baseline.read_text())
        out |= {"ZeroDice": _f(b["dice"]), "ZeroIoU": _f(b["iou"])}
    if budget.exists():
        out["GPUHours"] = _f(pd.read_csv(budget).hours.sum(), 1)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, default=Path("results/runs.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/tables"))
    parser.add_argument("--config", type=Path, default=Path("configs/cnn_resnet34.yaml"))
    parser.add_argument("--baseline", type=Path, default=Path("results/baselines/zero_mask.json"))
    args = parser.parse_args()
    runs = pd.read_csv(args.runs)
    df = compare(runs)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_dir / "stats.csv", index=False)
    (args.out_dir / "stats.tex").write_text(to_latex(df))
    extra = _protocol_macros(
        args.config,
        Path("splits/split_v1.json"),
        args.baseline,
        args.runs.parent / "gpu_budget.csv",
    )
    (args.out_dir / "numbers.tex").write_text(to_macros(df, runs, extra))
    print(df.T.to_string())


if __name__ == "__main__":
    main()
