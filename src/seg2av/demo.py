"""M8: inferência da demo. Carrega o best.pt da run de Dice mediano de cada cenário e sobrepõe a
máscara prevista na imagem. A interface Gradio fica em demo/app.py; aqui não depende do Gradio."""

import argparse
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image

from seg2av.data import build_transform, load_split, read_pair, resolve_root
from seg2av.evaluate import per_image_scores
from seg2av.models import build_model
from seg2av.plots import LABEL, SCENARIOS, SLIDE, STYLE, _br, median_seed

OVERLAY = (255, 0, 0)  # vermelho, opaco a 45%


def load_models(results: Path = Path("results")) -> dict[str, dict]:
    runs = pd.read_csv(results / "runs.csv")
    models = {}
    for s in SCENARIOS:
        seed = median_seed(runs, s)
        run_dir = results / "runs" / f"{s}_seed{seed}"
        cfg = json.loads((run_dir / "config.json").read_text())["config"]
        model = build_model({**cfg["model"], "encoder_weights": None})  # pesos vêm do best.pt
        model.load_state_dict(torch.load(run_dir / "best.pt", map_location="cpu"))
        size = int(cfg["data"]["image_size"])
        models[s] = {"model": model.eval(), "transform": build_transform(size), "seed": seed}
    return models


@torch.no_grad()
def predict(entry: dict, image: np.ndarray) -> np.ndarray:
    """Máscara binária no tamanho original da imagem RGB uint8."""
    x = entry["transform"](image=image)["image"][None]
    mask = (entry["model"](x)[0, 0] > 0).numpy().astype(np.uint8)
    h, w = image.shape[:2]
    return cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)


def overlay(image: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    out = image.copy()
    m = mask.astype(bool)
    out[m] = ((1 - alpha) * image[m] + alpha * np.array(OVERLAY)).astype(np.uint8)
    return out


def segment(models: dict[str, dict], image: np.ndarray) -> dict[str, tuple[np.ndarray, float]]:
    """Por cenário: imagem com a máscara sobreposta e fração da imagem marcada como pólipo."""
    out = {}
    for s, entry in models.items():
        mask = predict(entry, image)
        out[s] = (overlay(image, mask), float(mask.mean()))
    return out


def _dice(pred: np.ndarray, mask: np.ndarray) -> float:
    """Dice na resolução original (o da avaliação oficial é em image_size; difere pouco)."""
    return per_image_scores(torch.from_numpy(pred)[None], torch.from_numpy(mask)[None])[0].item()


def demo_gif(models, root: Path, names: list[str], out: Path, ms_per_frame: int = 1800) -> None:
    """Uma imagem do teste por quadro: original | ResNet-34 | MiT-B0, com o Dice de cada um."""
    frames = []
    with plt.rc_context(STYLE["slides"]):
        for name in names:
            image, mask = read_pair(root, name)
            fig, axes = plt.subplots(1, 1 + len(models), figsize=SLIDE)
            axes[0].imshow(overlay(image, mask))
            axes[0].set_title("Máscara real")
            for ax, (s, entry) in zip(axes[1:], models.items(), strict=True):
                pred = predict(entry, image)
                ax.imshow(overlay(image, pred))
                ax.set_title(f"{LABEL[s]}\nDice {_br(_dice(pred, mask), '.3f')}")
            for ax in axes:
                ax.axis("off")
            fig.suptitle(f"Demo: segmentação de pólipo (teste, {name})")
            fig.tight_layout()
            fig.canvas.draw()
            frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3]))
            plt.close(fig)
    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=ms_per_frame, loop=0)


def test_names(n: int, split: Path = Path("splits/split_v1.json")) -> list[str]:
    """n imagens do teste igualmente espaçadas na ordem do split (sem escolher a dedo)."""
    test = load_split(split)["test"]
    return [test[i] for i in np.linspace(0, len(test) - 1, n).astype(int)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gif", type=Path, default=Path("results/slides/demo.gif"))
    parser.add_argument("--frames", type=int, default=5)
    args = parser.parse_args()
    demo_gif(load_models(), resolve_root(), test_names(args.frames), args.gif)
    print(f"GIF em {args.gif}")


if __name__ == "__main__":
    main()
