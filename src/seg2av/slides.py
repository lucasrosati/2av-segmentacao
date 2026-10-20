"""M7b: pacote para a equipe de slides em results/slides/ (make slides-assets).

Todo número sai de results/ (runs.csv, tables/stats.csv), dos configs ou do split versionado;
nada é digitado à mão. Os CSV trazem exatamente o texto que aparece nos PNG e no key_numbers.md.
"""

import argparse
import json
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch
from scipy import stats as st

from seg2av.data import load_split, resolve_root
from seg2av.plots import (
    COLOR,
    LABEL,
    SCENARIOS,
    SLIDE,
    STYLE,
    _br,
    _save,
    result_figures,
    samples_figure,
)
from seg2av.stats import ALPHA, ALT, BASE
from seg2av.train import load_config

CONFIGS = {BASE: Path("configs/cnn_resnet34.yaml"), ALT: Path("configs/transformer_mitb0.yaml")}
WORDS = {
    "imagenet": "ImageNet",
    "adamw": "AdamW",
    "cosine": "cosseno",
    "dice_bce": "Dice + BCE",
    "hflip": "flip horizontal",
    "vflip": "flip vertical",
    "rotate90": "rotação de 90°",
    "color_jitter_light": "cor leve",
}
FIGURES = ["samples", "flowchart", "dice_box", "paired_diff", "curves", "qualitative"]
EXPECTED = [
    *(f"{f}.png" for f in [*FIGURES, "dice_latency", "paper_header"]),
    *(f"table_{t}.{e}" for t in ("main", "stats", "compare") for e in ("csv", "png")),
    "problem.md",
    "key_numbers.md",
    "method_summary.md",
    "HANDOFF.md",
]


def _pm(mean: float, sd: float, k: int = 4) -> str:
    return f"{_br(mean, f'.{k}f')} ± {_br(sd, f'.{k}f')}"


def _p(p: float) -> str:
    return "< 0,001" if p < 1e-3 else _br(p, ".3f")


def _git(*args: str) -> str:
    out = subprocess.run(["git", *args], capture_output=True, text=True)
    return out.stdout.strip() or "desconhecido"


def summary(runs: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por cenário com tudo que os slides citam."""
    rows = {}
    for s in SCENARIOS:
        r = runs[runs.scenario == s]
        n, mean, sd = len(r), r.test_dice.mean(), r.test_dice.std()
        lo, hi = st.t.interval(1 - ALPHA, n - 1, loc=mean, scale=sd / np.sqrt(n))
        rows[s] = {
            "n_runs": n,
            "dice_mean": mean,
            "dice_sd": sd,
            "dice_ci_low": lo,
            "dice_ci_high": hi,
            "iou_mean": r.test_iou.mean(),
            "iou_sd": r.test_iou.std(),
            "n_params": r.n_params.iloc[0],
            "n_params_encoder": r.n_params_encoder.iloc[0],
            "latency_ms": r.latency_ms.median(),
            "latency_device": r.latency_device.iloc[0],
            "train_min_mean": r.train_time_s.mean() / 60,
            "train_min_sd": r.train_time_s.std() / 60,
            "best_epoch_mean": r.best_epoch.mean(),
            "epochs_run_mean": r.epochs_run.mean(),
        }
    return pd.DataFrame(rows).T


def _by_scenario(rows: dict) -> pd.DataFrame:
    """Linhas = atributos, colunas = cenários lado a lado."""
    return pd.DataFrame(
        {"Atributo": list(rows), **{LABEL[s]: [f(s) for f in rows.values()] for s in SCENARIOS}}
    )


def table_main(sm: pd.DataFrame) -> pd.DataFrame:
    return _by_scenario(
        {
            "Runs válidas": lambda s: str(int(sm.n_runs[s])),
            "Dice (média ± dp)": lambda s: _pm(sm.dice_mean[s], sm.dice_sd[s]),
            "IoU (média ± dp)": lambda s: _pm(sm.iou_mean[s], sm.iou_sd[s]),
            "Melhor época (média)": lambda s: _br(sm.best_epoch_mean[s], ".1f"),
            "Épocas rodadas (média)": lambda s: _br(sm.epochs_run_mean[s], ".1f"),
            "Treino por run (min)": lambda s: _pm(sm.train_min_mean[s], sm.train_min_sd[s], 1),
        }
    )


def table_stats(stats: pd.DataFrame) -> pd.DataFrame:
    """Linhas = estatísticas, colunas = métricas (Dice é a primária)."""
    col = {"test_dice": "Dice (primária)", "test_iou": "IoU"}
    out = {
        "Estatística": [
            "Pares (seeds)",
            f"Diferença média ({LABEL[ALT]} − {LABEL[BASE]})",
            "IC 95% da diferença (bootstrap)",
            "Tamanho de efeito d_z",
            "Shapiro-Wilk nas diferenças (p)",
            "t pareado (p)",
            "Wilcoxon pareado (p)",
            "Teste que decide",
            f"Decisão (α = {_br(ALPHA, 'g')})",
        ]
    }
    for m, r in stats.iterrows():
        out[col[m]] = [
            str(int(r.n)),
            _br(r.mean_diff, ".4f"),
            f"[{_br(r.ci_low, '.4f')}; {_br(r.ci_high, '.4f')}]",
            _br(r.d_z, ".2f"),
            _p(r.shapiro_p),
            _p(r.t_p),
            _p(r.wilcoxon_p),
            "t pareado" if r.primary_test == "t" else "Wilcoxon",
            "rejeita H0" if r.reject_h0 else "não rejeita H0",
        ]
    return pd.DataFrame(out)


def table_compare(sm: pd.DataFrame, cfgs: dict) -> pd.DataFrame:
    """Lado a lado: linhas = atributos, colunas = cenários (pedido do grupo em 20/10)."""
    return _by_scenario(
        {
            "Arquitetura": lambda s: cfgs[s]["model"]["arch"].replace("Unet", "U-Net"),
            "Encoder": lambda s: LABEL[s],
            "Pré-treino": lambda s: WORDS[cfgs[s]["model"]["encoder_weights"]],
            "Parâmetros (total)": lambda s: f"{_br(sm.n_params[s] / 1e6, '.1f')} M",
            "Parâmetros (encoder)": lambda s: f"{_br(sm.n_params_encoder[s] / 1e6, '.1f')} M",
            "Dice no teste (média ± dp)": lambda s: _pm(sm.dice_mean[s], sm.dice_sd[s]),
            "IC 95% da média (t)": lambda s: (
                f"[{_br(sm.dice_ci_low[s], '.4f')}; {_br(sm.dice_ci_high[s], '.4f')}]"
            ),
            "Latência (mediana, lote 1)": lambda s: (
                f"{_br(sm.latency_ms[s], '.1f')} ms ({sm.latency_device[s]})"
            ),
        }
    )


def table_png(df: pd.DataFrame, out: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=SLIDE)
    ax.axis("off")
    tbl = ax.table(cellText=df.values, colLabels=df.columns, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(26)
    tbl.scale(1, 3.2)
    tbl.auto_set_column_width(range(len(df.columns)))
    for (row, _), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_text_props(fontweight="bold")
            cell.set_facecolor("#e8eef6")
    _save(fig, out, title)


def flowchart_figure(cfgs: dict, split: dict, sm: pd.DataFrame, seeds: list[int], out: Path):
    """Fluxograma vertical da metodologia, no estilo do exemplo mostrado em aula."""
    c, d, t = cfgs[BASE], cfgs[BASE]["data"], cfgs[BASE]["train"]
    n_tr, n_va, n_te = (len(split[k]) for k in ("train", "val", "test"))
    augs = ", ".join(WORDS[a] for a in d["augment"])
    size = d["image_size"]
    level = [
        f"Kvasir-SEG: {n_tr + n_va + n_te} imagens com máscara\n"
        f"divisão fixa: treino {n_tr} / validação {n_va} / teste {n_te}",
        f"Pré-processamento: {size}×{size} px, normalização ImageNet\naumento só no treino: {augs}",
        None,  # os dois cenários lado a lado
        f"Treino comum: {WORDS[t['optimizer']]} (lr {_br(t['lr'], 'g')}), {WORDS[t['scheduler']]}, "
        f"{WORDS[t['loss']]}, lote {t['batch_size']}, até {t['epochs']} épocas,\n"
        f"early stopping na Dice de validação (paciência {t['early_stopping']['patience']})",
        f"Avaliação no teste ({n_te} imagens): Dice e IoU por imagem,\n"
        "com os pesos da melhor época de validação",
        f"{len(seeds)} pares (seeds {min(seeds)}–{max(seeds)}) = {2 * len(seeds)} treinos",
        f"t pareado bilateral + Wilcoxon pareado · IC 95% (bootstrap) · d_z · "
        f"α = {_br(ALPHA, 'g')}",
    ]
    fig, ax = plt.subplots(figsize=SLIDE)
    ax.set(xlim=(0, 100), ylim=(0, 100))
    ax.axis("off")
    ys = np.linspace(92, 7, len(level))
    h = 9.0

    def box(x, y, w, text, color="#f4f6f9", edge="0.3"):
        ax.add_patch(
            FancyBboxPatch(
                (x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.4", fc=color, ec=edge, lw=2
            )
        )
        ax.text(x, y, text, ha="center", va="center", fontsize="small")

    def arrow(x0, y0, x1, y1):
        ax.annotate(
            "",
            (x1, y1 + h / 2 + 0.6),
            (x0, y0 - h / 2 - 0.6),
            arrowprops={"arrowstyle": "-|>", "lw": 2},
        )

    for i, (y, text) in enumerate(zip(ys, level, strict=True)):
        if text is None:
            for x, s in ((27, BASE), (73, ALT)):
                box(
                    x,
                    y,
                    42,
                    f"U-Net + {LABEL[s]} (pesos {WORDS[c['model']['encoder_weights']]})\n"
                    f"{_br(sm.n_params[s] / 1e6, '.1f')} M parâmetros",
                    color=COLOR[s] + "22",
                    edge=COLOR[s],
                )
                arrow(50, ys[i - 1], x, y)
                arrow(x, y, 50, ys[i + 1])
            continue
        box(50, y, 96, text)
        if i and level[i - 1] is not None:
            arrow(50, ys[i - 1], 50, y)
    _save(fig, out, "Metodologia")


def crop_header(page: np.ndarray, frac: float = 0.5, pad: int = 40) -> np.ndarray:
    """Recorta o topo da 1ª página (título, autores, resumo) e apara a margem branca."""
    top = page[: int(page.shape[0] * frac)]
    blank = np.flatnonzero(top.min(axis=(1, 2)) >= 200)
    top = top[: blank[-1]]  # termina num espaço entre linhas, sem cortar texto
    ink = np.argwhere(top.min(axis=2) < 200)
    (y0, x0), (y1, x1) = ink.min(axis=0), ink.max(axis=0)
    crop = top[y0 : y1 + 1, x0 : x1 + 1]
    return cv2.copyMakeBorder(crop, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(255, 255, 255))


def paper_header(pdf: Path, out: Path) -> bool:
    # shortcut: exige o Ghostscript (gs); trocar por pypdfium2 se rodar fora desta máquina
    if not pdf.exists() or not shutil.which("gs"):
        return False
    with tempfile.TemporaryDirectory() as tmp:
        png = Path(tmp) / "p1.png"
        subprocess.run(
            [
                "gs",
                "-q",
                "-dSAFER",
                "-dBATCH",
                "-dNOPAUSE",
                "-sDEVICE=png16m",
                "-r200",
                "-dFirstPage=1",
                "-dLastPage=1",
                f"-sOutputFile={png}",
                str(pdf),
            ],
            check=True,
        )
        cv2.imwrite(str(out), crop_header(cv2.imread(str(png))))
    return True


def problem_md(n_images: int, n_pairs: int) -> str:
    a, b = LABEL[BASE], LABEL[ALT]
    return f"""# Definição do problema

**Problema.** Segmentação binária de pólipos em imagens de colonoscopia: para cada pixel, decidir
se ele pertence a um pólipo. A tarefa apoia a triagem do câncer colorretal.

**Dados.** Kvasir-SEG: {n_images} imagens de colonoscopia com máscara do pólipo (Jha et al., 2020).

**Pergunta de pesquisa.** Com a U-Net e o transfer learning do ImageNet fixos, trocar o encoder
CNN ({a}) por um encoder Transformer hierárquico ({b}) muda o desempenho de segmentação?

**Hipóteses** (métrica: Dice médio no teste por run; {n_pairs} seeds pareados; bilateral,
α = {_br(ALPHA, "g")}):

- **H0:** μ_Dice({a}) = μ_Dice({b})
- **H1:** μ_Dice({a}) ≠ μ_Dice({b})
"""


def key_numbers_md(sm, stats, baseline, commits) -> str:
    dice, iou = stats.loc["test_dice"], stats.loc["test_iou"]
    decision = "rejeita H0" if dice.reject_h0 else "não rejeita H0"
    test = "t pareado" if dice.primary_test == "t" else "Wilcoxon pareado"
    lines = [
        "# Números-chave",
        "",
        "Gerado por `make slides-assets` a partir de `results/runs.csv` e "
        "`results/tables/stats.csv`. Não recalcule nem arredonde de outro jeito.",
        f"Código das runs: {', '.join(commits)}.",
        "",
        "## Por cenário (conjunto de teste)",
        "",
    ]
    for s in SCENARIOS:
        lines += [
            f"**{LABEL[s]}** ({int(sm.n_runs[s])} runs válidas)",
            f"- Dice: {_pm(sm.dice_mean[s], sm.dice_sd[s])} "
            f"(IC 95% da média: [{_br(sm.dice_ci_low[s], '.4f')}; "
            f"{_br(sm.dice_ci_high[s], '.4f')}])",
            f"- IoU: {_pm(sm.iou_mean[s], sm.iou_sd[s])}",
            f"- Parâmetros: {_br(sm.n_params[s] / 1e6, '.1f')} M no total, "
            f"{_br(sm.n_params_encoder[s] / 1e6, '.1f')} M no encoder",
            f"- Latência: {_br(sm.latency_ms[s], '.1f')} ms por imagem (mediana, lote 1, "
            f"{sm.latency_device[s]})",
            f"- Treino: {_pm(sm.train_min_mean[s], sm.train_min_sd[s], 1)} min por run; melhor "
            f"época média {_br(sm.best_epoch_mean[s], '.1f')} de "
            f"{_br(sm.epochs_run_mean[s], '.1f')} rodadas",
            "",
        ]
    lines += [
        f"## Comparação pareada ({LABEL[ALT]} − {LABEL[BASE]}, {int(dice.n)} pares)",
        "",
        f"- Diferença média do Dice: {_br(dice.mean_diff, '.4f')} "
        f"(IC 95% bootstrap: [{_br(dice.ci_low, '.4f')}; {_br(dice.ci_high, '.4f')}])",
        f"- Tamanho de efeito d_z: {_br(dice.d_z, '.2f')}",
        f"- Shapiro-Wilk nas diferenças: p = {_p(dice.shapiro_p)}",
        f"- t pareado: p = {_p(dice.t_p)}; Wilcoxon pareado: p = {_p(dice.wilcoxon_p)}",
        f"- **Decisão (α = {_br(ALPHA, 'g')}, decidida pelo {test}): {decision}**",
        f"- IoU (secundária): diferença {_br(iou.mean_diff, '.4f')}, "
        f"p (t) = {_p(iou.t_p)}, p (Wilcoxon) = {_p(iou.wilcoxon_p)}",
        "",
    ]
    if baseline:
        lines += [
            f"## Referência trivial: máscara toda zero no teste (n = {baseline['n']})",
            "",
            f"- Dice: {_br(baseline['dice'], '.4f')}; IoU: {_br(baseline['iou'], '.4f')}",
            "",
        ]
    return "\n".join(lines)


def method_summary_md(cfgs, split, seeds) -> str:
    d, t = cfgs[BASE]["data"], cfgs[BASE]["train"]
    n_tr, n_va, n_te = (len(split[k]) for k in ("train", "val", "test"))
    augs = ", ".join(WORDS[a] for a in d["augment"])
    a, b = LABEL[BASE], LABEL[ALT]
    return f"""# Resumo do método

**Problema.** Segmentação binária de pólipos em colonoscopia (Kvasir-SEG, {n_tr + n_va + n_te}
imagens com máscara).

**Técnicas.** (A) U-Net como arquitetura de segmentação; (B) transfer learning, com o encoder
pré-treinado no {WORDS[cfgs[BASE]["model"]["encoder_weights"]]}; (C) encoder CNN {a} contra (C')
encoder Transformer hierárquico {b}. Só o encoder muda entre os cenários.

**Dados.** Divisão fixa, igual em todas as execuções: treino {n_tr}, validação {n_va},
teste {n_te}. Imagens redimensionadas para {d["image_size"]}×{d["image_size"]}; aumento de dados
só no treino ({augs}).

**Treino (igual nos dois cenários).** {WORDS[t["optimizer"]]}, lr {_br(t["lr"], "g")}, agenda
{WORDS[t["scheduler"]]}, perda {WORDS[t["loss"]]}, lote {t["batch_size"]}, até {t["epochs"]}
épocas com early stopping na Dice de validação (paciência {t["early_stopping"]["patience"]}).
O teste é avaliado uma única vez, com os pesos da melhor época de validação.

**Protocolo.** {len(seeds)} simulações pareadas por seed ({min(seeds)} a {max(seeds)}): o mesmo
seed nos dois cenários, totalizando {2 * len(seeds)} treinos. Métrica primária: Dice médio no
teste por run.

**Hipóteses e testes.** H0: μ_Dice({a}) = μ_Dice({b}); H1: diferentes (bilateral,
α = {_br(ALPHA, "g")}). Shapiro-Wilk nas diferenças pareadas; t pareado se forem aproximadamente
normais, senão Wilcoxon pareado (os dois são reportados). IC 95% da diferença média por
bootstrap e tamanho de efeito d_z de Cohen.
"""


def handoff_md(missing: list[str], commits: list[str]) -> str:
    slides = [
        ("Título", "paper_header.png"),
        ("Definição do problema, pergunta e hipóteses", "problem.md"),
        ("Dataset", "samples.png"),
        ("Cenários comparados", "table_compare.png"),
        ("Metodologia", "flowchart.png, method_summary.md"),
        ("Resultados", "dice_box.png, table_main.png"),
        ("Estatística", "paired_diff.png, table_stats.png"),
        ("Treino e custo", "curves.png, dice_latency.png"),
        ("Exemplos qualitativos", "qualitative.png"),
        ("Demo, conclusão e limitações", "demo.gif (make demo-gif); key_numbers.md"),
    ]
    lines = [
        "# Handoff para a equipe de slides",
        "",
        f"- Gerado em {datetime.now():%d/%m/%Y %H:%M} pelo commit {_git('rev-parse', 'HEAD')}.",
        f"- Código que treinou as runs: {', '.join(commits)}.",
        "- Todos os números estão em `key_numbers.md`. Não recalcule nem arredonde diferente;",
        "  dúvidas vão para o Lucas.",
        "- Figuras: PNG 1920×1080, fundo claro, cores fixas (azul = ResNet-34, vermelho = MiT-B0).",
        "- `samples.png`, `qualitative.png` e `demo.gif` mostram fotos do Kvasir-SEG e não",
        "  estão no GitHub: vão pela pasta compartilhada. Uso só acadêmico, citando",
        "  Jha et al. (MMM 2020).",
        "",
        "## Estrutura sugerida (10 slides)",
        "",
        "| # | Slide | Arquivos |",
        "|---|---|---|",
        *(f"| {i} | {name} | {files} |" for i, (name, files) in enumerate(slides, start=1)),
        "",
    ]
    if missing:
        lines += ["## Ainda não gerados", "", *(f"- {m}: TODO: pendente" for m in missing), ""]
    return "\n".join(lines)


def build(
    results: Path = Path("results"),
    out: Path | None = None,
    root: Path | None = None,
    pdf: Path = Path("paper/main.pdf"),
    split_file: Path = Path("splits/split_v1.json"),
    sample_names: list[str] | None = None,
) -> list[str]:
    """Gera o pacote e devolve os arquivos esperados que não puderam ser gerados."""
    out = out or results / "slides"
    out.mkdir(parents=True, exist_ok=True)
    runs = pd.read_csv(results / "runs.csv")
    stats = pd.read_csv(results / "tables" / "stats.csv").set_index("metric")
    cfgs = {s: load_config(p) for s, p in CONFIGS.items()}
    split = load_split(split_file)
    seeds = sorted(runs.seed.unique().tolist())
    sm = summary(runs)
    commits = sorted(
        {
            json.loads(p.read_text())["env"]["git_commit"][:7]
            for p in (results / "runs").glob("*/config.json")
            if "env" in json.loads(p.read_text())
        }
    ) or ["desconhecido"]
    root = root or resolve_root()

    result_figures(results, out, root, target="slides")
    with plt.rc_context(STYLE["slides"]):
        flowchart_figure(cfgs, split, sm, seeds, out / "flowchart.png")
        if root.exists():
            names = sample_names or split["train"][:4]
            samples_figure(
                root, names, out / "samples.png", figsize=SLIDE, title="Kvasir-SEG: exemplos"
            )
        for name, df, title in (
            ("main", table_main(sm), "Resultados no conjunto de teste"),
            ("stats", table_stats(stats), "Teste de hipótese pareado"),
            ("compare", table_compare(sm, cfgs), "Cenários lado a lado"),
        ):
            df.to_csv(out / f"table_{name}.csv", index=False)
            table_png(df, out / f"table_{name}.png", title)
    paper_header(pdf, out / "paper_header.png")

    baseline_file = results / "baselines" / "zero_mask.json"
    baseline = json.loads(baseline_file.read_text()) if baseline_file.exists() else None
    n_images = sum(len(v) for v in split.values())
    (out / "problem.md").write_text(problem_md(n_images, len(seeds)))
    (out / "key_numbers.md").write_text(key_numbers_md(sm, stats, baseline, commits))
    (out / "method_summary.md").write_text(method_summary_md(cfgs, split, seeds))
    missing = [f for f in EXPECTED if f != "HANDOFF.md" and not (out / f).exists()]
    (out / "HANDOFF.md").write_text(handoff_md(missing, commits))
    return missing


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args()
    missing = build(args.results)
    print("pacote em", args.results / "slides", "| faltando:", missing or "nada")


if __name__ == "__main__":
    main()
