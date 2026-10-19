"""Uma run (cenário × seed): treino retomável, early stopping na val Dice, teste uma vez no melhor
checkpoint. Saídas em <out_dir>/<cenário>_seed<n>/ (D3)."""

import argparse
import csv
import hashlib
import json
import platform
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, get_worker_info

from seg2av.data import KvasirSeg, load_split, resolve_root
from seg2av.env import installed_version, pins
from seg2av.evaluate import evaluate
from seg2av.losses import DiceBCELoss
from seg2av.models import build_model, n_params


def load_config(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text())


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # também semeia todas as GPUs
    # cudnn determinístico; use_deterministic_algorithms não dá: o backward do upsample da U-Net
    # na CUDA não tem versão determinística, então restam diferenças mínimas entre execuções
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _seed_worker(_: int) -> None:
    # albumentations não trata workers: sem isso todos repetiriam a mesma sequência de augmentation
    info = get_worker_info()
    info.dataset.transform.set_random_seed(info.seed % 2**32)


def make_loaders(cfg: dict, seed: int, limit: int | None = None):
    d, t = cfg["data"], cfg["train"]
    split = load_split(Path(d["split_file"]), Path(d["split_sha256_file"]))
    root = resolve_root(d["root"])
    names = {k: v[:limit] for k, v in split.items()} if limit else split
    gen = torch.Generator().manual_seed(seed)
    kw = dict(
        batch_size=t["batch_size"],
        num_workers=t["num_workers"],
        worker_init_fn=_seed_worker,
        pin_memory=torch.cuda.is_available(),
    )
    train_ds = KvasirSeg(root, names["train"], d["image_size"], d["augment"], seed=seed)
    loaders = [DataLoader(train_ds, shuffle=True, generator=gen, **kw)]
    loaders += [
        DataLoader(KvasirSeg(root, names[k], d["image_size"]), **kw) for k in ("val", "test")
    ]
    return (*loaders, gen)


def _rng_state(gen: torch.Generator) -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "loader": gen.get_state(),
    }


def _set_rng_state(state: dict, gen: torch.Generator) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    gen.set_state(state["loader"])


def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
        return out.stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def environment(device: torch.device) -> dict:
    return {
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "libs": {name: installed_version(name) for name in pins()},
        "device": torch.cuda.get_device_name() if device.type == "cuda" else device.type,
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def train_one_epoch(model, loader, loss_fn, opt, scaler, device, amp) -> float:
    model.train()
    total, n = 0.0, 0
    for image, mask in loader:
        image, mask = image.to(device, non_blocking=True), mask.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device.type, enabled=amp):
            loss = loss_fn(model(image), mask)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        total += loss.item() * len(image)
        n += len(image)
    return total / n


def run(
    cfg: dict, seed: int, out_dir: str | Path, deadline: float | None = None, limit=None
) -> str:
    """Devolve 'done' (já existia), 'finished' ou 'interrupted' (estourou o deadline)."""
    run_dir = Path(out_dir) / f"{cfg['scenario']}_seed{seed}"
    if (run_dir / "DONE").exists():
        return "done"
    run_dir.mkdir(parents=True, exist_ok=True)
    t = cfg["train"]

    seed_everything(seed)
    device = pick_device()
    amp = bool(t["amp"]) and device.type == "cuda"
    train_dl, val_dl, test_dl, gen = make_loaders(cfg, seed, limit)
    model = build_model(cfg["model"]).to(device)
    loss_fn = DiceBCELoss()
    opt = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=t["epochs"])
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    state = {"epoch": 0, "best_val_dice": -1.0, "best_epoch": 0, "bad_epochs": 0, "train_s": 0.0}
    history: list[dict] = []
    last = run_dir / "last.pt"
    if last.exists():  # retoma run parcial
        ck = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        state, history = ck["state"], ck["history"]
        _set_rng_state(ck["rng"], gen)
    else:
        env = environment(device)
        (run_dir / "config.json").write_text(
            json.dumps({"config": cfg, "seed": seed, "limit": limit, "env": env}, indent=2)
        )

    patience = t["early_stopping"]["patience"]
    while state["epoch"] < t["epochs"] and state["bad_epochs"] < patience:
        if deadline is not None and time.time() > deadline:
            return "interrupted"
        t0 = time.time()
        train_loss = train_one_epoch(model, train_dl, loss_fn, opt, scaler, device, amp)
        val = evaluate(model, val_dl, device, amp)
        lr = sched.get_last_lr()[0]
        sched.step()
        epoch_s = time.time() - t0
        state["epoch"] += 1
        state["train_s"] += epoch_s
        val_dice = float(val["dice"].mean())
        history.append(
            {
                "epoch": state["epoch"],
                "train_loss": train_loss,
                "val_dice": val_dice,
                "val_iou": float(val["iou"].mean()),
                "lr": lr,
                "epoch_s": epoch_s,
            }
        )
        if val_dice > state["best_val_dice"]:
            state.update(best_val_dice=val_dice, best_epoch=state["epoch"], bad_epochs=0)
            torch.save(model.state_dict(), run_dir / "best.pt")
        else:
            state["bad_epochs"] += 1
        ck = {
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "sched": sched.state_dict(),
            "scaler": scaler.state_dict(),
            "state": state,
            "history": history,
            "rng": _rng_state(gen),
        }
        torch.save(ck, last)

    # teste avaliado uma única vez, no checkpoint de melhor validação (regra 4)
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device))
    test = evaluate(model, test_dl, device, amp)
    names = test_dl.dataset.names
    _write_csv(run_dir / "history.csv", history)
    _write_csv(
        run_dir / "test_per_image.csv",
        [
            {"name": n, "dice": float(d), "iou": float(i)}
            for n, d, i in zip(names, test["dice"], test["iou"], strict=True)
        ],
    )
    split_file = Path(cfg["data"]["split_file"])
    metrics = {
        "scenario": cfg["scenario"],
        "seed": seed,
        "test_dice": float(test["dice"].mean()),
        "test_iou": float(test["iou"].mean()),
        "n_test": len(names),
        "best_val_dice": state["best_val_dice"],
        "best_epoch": state["best_epoch"],
        "epochs_run": state["epoch"],
        "early_stopped": state["bad_epochs"] >= patience,
        "train_time_s": state["train_s"],
        "epoch_time_s": state["train_s"] / state["epoch"],
        "n_params": n_params(model),
        "n_params_encoder": n_params(model.encoder),
        "split_sha256": hashlib.sha256(split_file.read_bytes()).hexdigest(),
        "limit": limit,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    last.unlink()  # guarda só o melhor checkpoint por run
    (run_dir / "DONE").touch()
    return "finished"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    print(run(cfg, args.seed, args.out_dir or cfg["tracking"]["out_dir"]))


if __name__ == "__main__":
    main()
