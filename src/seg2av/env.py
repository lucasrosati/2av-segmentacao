"""Confere libs instaladas x pins do pyproject (detecta mudança na imagem do Kaggle)."""

import importlib.metadata as md
import sys
import tomllib
from pathlib import Path

PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"


def pins(pyproject: Path = PYPROJECT) -> dict[str, str]:
    deps = tomllib.loads(pyproject.read_text())["project"]["dependencies"]
    return dict(d.split("==") for d in deps)


def installed_version(name: str) -> str | None:
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None


def mismatches(pinned: dict[str, str], get=installed_version) -> list[str]:
    # "+cu128" é sufixo local do build do Kaggle, não muda a versão
    return [
        f"{name}: pin {want}, instalado {have}"
        for name, want in pinned.items()
        if (have := get(name)) is None or have.split("+")[0] != want
    ]


def main() -> None:
    bad = mismatches(pins())
    print("\n".join(bad) or "ambiente bate com o pyproject")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
