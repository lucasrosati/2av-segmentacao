"""Figuras do paper (M7) e dos slides (M7b), todas a partir de results/ (D3)."""

import argparse
import json
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import ScalarFormatter  # noqa: E402

from seg2av.data import KvasirSeg, load_split, read_pair, resolve_root  # noqa: E402
from seg2av.models import build_model  # noqa: E402
from seg2av.stats import ALT, BASE  # noqa: E402

LABEL = {BASE: "ResNet-34", ALT: "MiT-B0"}
COLOR = {BASE: "#1f77b4", ALT: "#d62728"}
SCENARIOS = (BASE, ALT)
# paper: coluna IEEE a 300 dpi; slides: 1920×1080 com fonte ≥ 24 pt ("small" = 0,83 × 30)
STYLE = {
    "paper": {"font.size": 9, "lines.markersize": 3.2, "savefig.dpi": 300},
    "slides": {
        "font.size": 30,
        "lines.markersize": 10,
        "lines.linewidth": 3,
        "savefig.dpi": 100,
        **{f"boxplot.{p}props.linewidth": 3 for p in ("box", "whisker", "cap", "median")},
    },
}
SLIDE = (19.2, 10.8)


def _br(x: float, fmt: str) -> str:
    return f"{x:{fmt}}".replace(".", ",").replace("-", "−")  # pt-BR, sinal de menos


class _BrFormatter(ScalarFormatter):
    def __call__(self, x, pos=None):
        return super().__call__(x, pos).replace(".", ",")


def _dot() -> float:
    return plt.rcParams["lines.markersize"] ** 2


def _save(fig, out: Path, title: str | None = None) -> None:
    if title:
        fig.suptitle(title, fontweight="bold")
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            if type(axis.get_major_formatter()) is ScalarFormatter:  # preserva rótulos fixos
                axis.set_major_formatter(_BrFormatter())
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def samples_figure(
    root: Path, names: list[str], out: Path, size: int = 256, figsize=None, title=None
) -> None:
    fig, axes = plt.subplots(2, len(names), figsize=figsize or (2.2 * len(names), 4.6))
    for col, name in enumerate(names):
        image, mask = read_pair(root, name)
        image = cv2.resize(image, (size, size))
        mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
        overlay = image.copy()
        overlay[mask == 1] = (0.5 * overlay[mask == 1] + [0, 0, 127]).astype("uint8")
        axes[0, col].imshow(image)
        axes[1, col].imshow(overlay)
        for ax in axes[:, col]:
            ax.axis("off")
    axes[0, 0].set_title("Imagem", loc="left")
    axes[1, 0].set_title("Máscara (azul)", loc="left")
    _save(fig, out, title)


def dice_box_figure(runs: pd.DataFrame, out: Path, figsize=(3.5, 3.2), title=None) -> None:
    """Boxplot do Dice de teste por cenário, com uma linha ligando cada par de seeds."""
    wide = runs.pivot(index="seed", columns="scenario", values="test_dice")[list(SCENARIOS)]
    fig, ax = plt.subplots(figsize=figsize)
    ax.boxplot([wide[s] for s in SCENARIOS], widths=0.5, showfliers=False)
    for _, row in wide.iterrows():
        ax.plot([1, 2], row.values, color="0.7", lw=0.4 * plt.rcParams["lines.linewidth"], zorder=1)
    for i, s in enumerate(SCENARIOS, start=1):
        ax.scatter(np.full(len(wide), i), wide[s], s=_dot(), color=COLOR[s], zorder=2)
    ax.set_xticks([1, 2], [LABEL[s] for s in SCENARIOS])
    ax.set_ylabel("Dice no teste (por run)")
    ax.set_title(f"{len(wide)} seeds pareados")
    _save(fig, out, title)


def paired_diff_figure(
    runs: pd.DataFrame, stats_row: pd.Series, out: Path, figsize=(3.5, 3.2), title=None
) -> None:
    """Diferença por seed (MiT-B0 − ResNet-34), com média e IC 95% bootstrap do stats.py."""
    wide = runs.pivot(index="seed", columns="scenario", values="test_dice")
    diff = (wide[ALT] - wide[BASE]).sort_values()
    fig, ax = plt.subplots(figsize=figsize)
    ax.axhspan(stats_row["ci_low"], stats_row["ci_high"], color="0.85", label="IC 95% da média")
    ax.axhline(stats_row["mean_diff"], color="k", lw=1, label="diferença média")
    ax.axhline(0, color="k", lw=0.6, ls="--")
    ax.scatter(range(len(diff)), diff, s=_dot(), color=np.where(diff > 0, COLOR[ALT], COLOR[BASE]))
    ax.set_xticks([])
    ax.set_xlabel("seeds (ordenados pela diferença)")
    ax.set_ylabel(f"Dice {LABEL[ALT]} − {LABEL[BASE]}")
    ax.legend(fontsize="small", loc="upper left")
    _save(fig, out, title)


def curves_figure(
    histories: dict[str, list[pd.DataFrame]], out: Path, figsize=(7.0, 2.8), title=None
) -> None:
    """Média ± dp por época entre as runs que chegaram àquela época (early stopping encurta)."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    for s in SCENARIOS:
        g = pd.concat(histories[s]).groupby("epoch")
        for ax, col in zip(axes, ("train_loss", "val_dice"), strict=True):
            mean, sd = g[col].mean(), g[col].std().fillna(0)
            ax.plot(mean.index, mean, color=COLOR[s], label=LABEL[s])
            ax.fill_between(mean.index, mean - sd, mean + sd, color=COLOR[s], alpha=0.2)
    axes[0].set_ylabel("loss de treino (Dice + BCE)")
    axes[1].set_ylabel("Dice de validação")
    for ax in axes:
        ax.set_xlabel("época")
    axes[1].legend(fontsize="small")
    _save(fig, out, title)


def dice_latency_figure(runs: pd.DataFrame, out: Path, figsize=(3.5, 3.0), title=None) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    for s in SCENARIOS:
        r = runs[runs.scenario == s]
        x, y = r.latency_ms.median(), r.test_dice.mean()
        ax.errorbar(x, y, yerr=r.test_dice.std(), fmt="o", color=COLOR[s], capsize=3)
        ax.annotate(
            f"{LABEL[s]}\n{_br(r.n_params.iloc[0] / 1e6, '.1f')} M parâm.",
            (x, y),
            textcoords="offset points",
            xytext=(8, -4),
            fontsize="small",
        )
    ax.set_xlabel(f"latência mediana, lote 1 ({runs.latency_device.iloc[0]}) [ms]")
    ax.set_ylabel("Dice no teste (média ± dp)")
    ax.margins(x=0.4, y=0.3)
    _save(fig, out, title)


def pick_examples(per_image: dict[str, list[pd.DataFrame]], k: int = 2) -> list[str]:
    """k imagens de maior e k de menor Dice médio entre todas as runs dos dois cenários."""
    mean = pd.concat([df for s in SCENARIOS for df in per_image[s]]).groupby("name").dice.mean()
    ranked = mean.sort_values(ascending=False).index.tolist()
    return ranked[:k] + ranked[-k:]


def median_seed(runs: pd.DataFrame, scenario: str) -> int:
    """Run representativa: a de Dice de teste mais próximo da mediana do cenário."""
    r = runs[runs.scenario == scenario]
    return int(r.loc[(r.test_dice - r.test_dice.median()).abs().idxmin(), "seed"])


@torch.no_grad()
def predict_masks(run_dir: Path, root: Path, names: list[str]) -> np.ndarray:
    cfg = json.loads((run_dir / "config.json").read_text())["config"]
    model = build_model({**cfg["model"], "encoder_weights": None})  # pesos vêm do best.pt
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location="cpu"))
    model.eval()
    ds = KvasirSeg(root, names, cfg["data"]["image_size"])
    images = torch.stack([ds[i][0] for i in range(len(ds))])
    return (model(images)[:, 0] > 0).numpy().astype(np.uint8)


def qualitative_figure(
    images, masks, preds: dict[str, np.ndarray], dices, out: Path, figsize=None, title=None
) -> None:
    """Linhas = imagens; colunas = imagem, máscara real e a predição de cada cenário."""
    cols = ["Imagem", "Máscara real", *(LABEL[s] for s in SCENARIOS)]
    fig, axes = plt.subplots(len(images), len(cols), figsize=figsize or (7.0, 1.8 * len(images)))
    for i, (image, mask) in enumerate(zip(images, masks, strict=True)):
        axes[i, 0].imshow(image)
        axes[i, 1].imshow(mask, cmap="gray")
        for j, s in enumerate(SCENARIOS, start=2):
            axes[i, j].imshow(preds[s][i], cmap="gray")
            axes[i, j].set_title(f"Dice {_br(dices[s][i], '.3f')}", fontsize="small")
        for ax in axes[i]:
            ax.set_xticks([])
            ax.set_yticks([])
    for j, c in enumerate(cols):
        axes[0, j].set_title(f"{c}\n{axes[0, j].get_title()}".strip())
    _save(fig, out, title)


TITLES = {
    "dice_box": "Dice no teste por cenário",
    "paired_diff": "Diferença pareada por seed",
    "curves": "Curvas de treino (média ± dp)",
    "dice_latency": "Desempenho × custo de inferência",
    "qualitative": "Exemplos qualitativos: melhores e piores imagens",
}


def result_figures(
    results: Path = Path("results"),
    out_dir: Path | None = None,
    root: Path | None = None,
    target: str = "paper",
) -> None:
    """Figuras de resultados; `slides` gera 1920×1080 com título, `paper` o tamanho de coluna."""
    out_dir = out_dir or results / "figures"
    slides = target == "slides"
    kw = {k: {"figsize": SLIDE, "title": t} if slides else {} for k, t in TITLES.items()}
    runs = pd.read_csv(results / "runs.csv")
    stats = pd.read_csv(results / "tables" / "stats.csv").set_index("metric")
    dirs = {
        s: [results / "runs" / f"{s}_seed{n}" for n in runs[runs.scenario == s].seed]
        for s in SCENARIOS
    }
    with plt.rc_context(STYLE[target]):
        dice_box_figure(runs, out_dir / "dice_box.png", **kw["dice_box"])
        paired_diff_figure(
            runs, stats.loc["test_dice"], out_dir / "paired_diff.png", **kw["paired_diff"]
        )
        curves_figure(
            {s: [pd.read_csv(d / "history.csv") for d in dirs[s]] for s in SCENARIOS},
            out_dir / "curves.png",
            **kw["curves"],
        )
        dice_latency_figure(runs, out_dir / "dice_latency.png", **kw["dice_latency"])

        per_image = {s: [pd.read_csv(d / "test_per_image.csv") for d in dirs[s]] for s in SCENARIOS}
        rep = {s: results / "runs" / f"{s}_seed{median_seed(runs, s)}" for s in SCENARIOS}
        root = root or resolve_root()
        if not root.exists() or not all((d / "best.pt").exists() for d in rep.values()):
            print(
                "qualitativos pulados: faltam os dados (make data) ou o best.pt das runs medianas"
            )
            return
        names = pick_examples(per_image)
        pairs = [read_pair(root, n) for n in names]
        cfg = json.loads((rep[BASE] / "config.json").read_text())["config"]
        size = int(cfg["data"]["image_size"])
        images = [cv2.resize(im, (size, size)) for im, _ in pairs]
        masks = [cv2.resize(m, (size, size), interpolation=cv2.INTER_NEAREST) for _, m in pairs]
        preds = {s: predict_masks(rep[s], root, names) for s in SCENARIOS}
        dices = {
            s: pd.read_csv(rep[s] / "test_per_image.csv").set_index("name").dice[names].tolist()
            for s in SCENARIOS
        }
        qualitative_figure(
            images, masks, preds, dices, out_dir / "qualitative.png", **kw["qualitative"]
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=["samples", "paper"], required=True)
    args = parser.parse_args()
    if args.target == "paper":
        result_figures()
        return
    train = load_split(Path("splits/split_v1.json"))["train"]
    samples_figure(resolve_root(), train[:4], Path("results/figures/samples.png"))


if __name__ == "__main__":
    main()
