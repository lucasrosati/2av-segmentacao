"""Roda seeds de um cenário em sequência. Pula runs com DONE e retoma parciais (--resume);
para com folga antes do limite da sessão do Kaggle (--time-budget-h)."""

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

from seg2av.train import load_config, pick_device, run


def parse_seeds(spec: str) -> list[int]:
    lo, _, hi = spec.partition("-")
    return list(range(int(lo), int(hi) + 1)) if hi else [int(s) for s in spec.split(",")]


parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
parser.add_argument("--seeds", default="0-29", help="intervalo 0-29 ou lista 0,3,7")
parser.add_argument("--resume", action="store_true", help="retoma runs parciais")
parser.add_argument("--time-budget-h", type=float, default=None)
parser.add_argument("--out-dir", default=None)
args = parser.parse_args()

cfg = load_config(args.config)
out_dir = Path(args.out_dir or cfg["tracking"]["out_dir"])
start = time.time()
deadline = start + args.time_budget_h * 3600 if args.time_budget_h else None
done = []
for seed in parse_seeds(args.seeds):
    run_dir = out_dir / f"{cfg['scenario']}_seed{seed}"
    if run_dir.exists() and not (run_dir / "DONE").exists() and not args.resume:
        raise SystemExit(f"{run_dir} existe incompleta: use --resume para retomar")
    status = run(cfg, seed, out_dir, deadline)
    print(f"{run_dir.name}: {status} ({(time.time() - start) / 60:.1f} min desde o início)")
    if status == "interrupted":
        break
    if status == "finished":
        done.append(seed)

budget = out_dir.parent / "gpu_budget.csv"
new = not budget.exists()
with budget.open("a", newline="") as f:
    w = csv.writer(f)
    if new:
        w.writerow(["started_at", "scenario", "hours", "runs_finished", "seeds", "device"])
    w.writerow(
        [
            datetime.fromtimestamp(start).isoformat(timespec="seconds"),
            cfg["scenario"],
            round((time.time() - start) / 3600, 3),
            len(done),
            " ".join(map(str, done)),
            pick_device().type,
        ]
    )
