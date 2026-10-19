"""Gera o split fixo 80/10/10 (D2) e o SHA-256 que todo run confere ao iniciar."""

import argparse
from pathlib import Path

from seg2av.data import list_names, resolve_root, split_names, write_split

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out", type=Path, default=Path("splits/split_v1.json"))
parser.add_argument("--root", default="auto")
args = parser.parse_args()

names = list_names(resolve_root(args.root))
assert len(names) == 1000, f"esperado 1000 imagens, achei {len(names)}"
split = split_names(names, seed=args.seed)
digest = write_split(split, args.out)
print({k: len(v) for k, v in split.items()}, args.out, digest)
