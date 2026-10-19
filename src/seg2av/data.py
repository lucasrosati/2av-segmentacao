"""Kvasir-SEG: localização dos dados, split fixo com hash (D2) e Dataset com augmentations."""

import hashlib
import json
import os
import random
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

LOCAL_ROOT = Path("data/raw/kvasir-seg/Kvasir-SEG")

AUGMENTS = {
    "hflip": lambda: A.HorizontalFlip(p=0.5),
    "vflip": lambda: A.VerticalFlip(p=0.5),
    "rotate90": lambda: A.RandomRotate90(p=0.5),
    "color_jitter_light": lambda: A.ColorJitter(
        brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02, p=0.5
    ),
}


def resolve_root(root: str = "auto") -> Path:
    """`auto`: dataset privado montado no Kaggle, senão data/raw local."""
    if root != "auto":
        return Path(root)
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        # o Kaggle mudou o ponto de montagem (/kaggle/input/datasets/<user>/...), então procura
        for images in sorted(Path("/kaggle/input").glob("**/kvasir-seg/**/images")):
            if (images.parent / "masks").is_dir():
                return images.parent
    return LOCAL_ROOT


def list_names(root: Path) -> list[str]:
    return sorted(p.stem for p in (root / "images").glob("*.jpg"))


def split_names(names: list[str], seed: int = 0, fracs=(0.8, 0.1, 0.1)) -> dict[str, list[str]]:
    shuffled = sorted(names)
    random.Random(seed).shuffle(shuffled)
    n_train, n_val = round(fracs[0] * len(names)), round(fracs[1] * len(names))
    return {
        "train": sorted(shuffled[:n_train]),
        "val": sorted(shuffled[n_train : n_train + n_val]),
        "test": sorted(shuffled[n_train + n_val :]),
    }


def write_split(split: dict, out: Path) -> str:
    out.write_text(json.dumps(split, indent=1) + "\n")
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    out.with_suffix(".sha256").write_text(f"{digest}  {out.name}\n")
    return digest


def load_split(path: Path, sha_path: Path | None = None) -> dict[str, list[str]]:
    """Carrega o split e aborta se o SHA-256 não bater com o registrado."""
    sha_path = sha_path or path.with_suffix(".sha256")
    want = sha_path.read_text().split()[0]
    got = hashlib.sha256(path.read_bytes()).hexdigest()
    if got != want:
        raise ValueError(f"split {path} alterado: sha256 {got} != {want}")
    return json.loads(path.read_text())


def build_transform(image_size: int, augment=()) -> A.Compose:
    return A.Compose(
        [A.Resize(image_size, image_size), *(AUGMENTS[a]() for a in augment)]
        + [A.Normalize(), ToTensorV2()]  # média/dp do ImageNet, iguais nos dois encoders
    )


def read_pair(root: Path, name: str) -> tuple[np.ndarray, np.ndarray]:
    image = cv2.cvtColor(cv2.imread(str(root / "images" / f"{name}.jpg")), cv2.COLOR_BGR2RGB)
    # máscaras são JPEG: compressão deixa tons intermediários, binariza no meio da escala
    gray = cv2.imread(str(root / "masks" / f"{name}.jpg"), cv2.IMREAD_GRAYSCALE)
    return image, (gray > 127).astype(np.uint8)


class KvasirSeg(Dataset):
    def __init__(self, root: Path, names: list[str], image_size: int = 256, augment=()):
        self.root, self.names = Path(root), names
        self.transform = build_transform(image_size, augment)

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, i: int):
        image, mask = read_pair(self.root, self.names[i])
        out = self.transform(image=image, mask=mask)
        return out["image"], out["mask"][None].float()
