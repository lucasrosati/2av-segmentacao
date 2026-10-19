"""Figuras do paper e dos slides (M1: amostras do dataset; resto no M7)."""

import argparse
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from seg2av.data import load_split, read_pair, resolve_root  # noqa: E402


def samples_figure(root: Path, names: list[str], out: Path, size: int = 256) -> None:
    fig, axes = plt.subplots(2, len(names), figsize=(2.2 * len(names), 4.6))
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
    axes[0, 0].set_title("Imagem", loc="left", fontsize=9)
    axes[1, 0].set_title("Máscara (azul)", loc="left", fontsize=9)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=["samples", "paper", "slides"], required=True)
    args = parser.parse_args()
    if args.target != "samples":
        raise NotImplementedError("M7/M7b")
    train = load_split(Path("splits/split_v1.json"))["train"]
    samples_figure(resolve_root(), train[:4], Path("results/figures/samples.png"))


if __name__ == "__main__":
    main()
