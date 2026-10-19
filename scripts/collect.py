"""Consolida <runs>/*/metrics.json de runs concluídas (DONE) num CSV (D3: fonte da verdade)."""

import argparse
import json
from pathlib import Path

import pandas as pd


def collect(runs_dir: Path) -> pd.DataFrame:
    rows = [
        json.loads((d / "metrics.json").read_text())
        for d in sorted(runs_dir.iterdir())
        if (d / "DONE").exists()
    ]
    return pd.DataFrame(rows).sort_values(["scenario", "seed"], ignore_index=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, default=Path("results/runs"))
    parser.add_argument("--out", type=Path, default=Path("results/runs.csv"))
    args = parser.parse_args()
    df = collect(args.runs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(df.groupby("scenario").size().to_string(), "->", args.out)
