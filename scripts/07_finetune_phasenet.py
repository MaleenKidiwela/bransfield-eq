"""
Fine-tune SeisBench PhaseNet on the Bransfield mag07 dataset.

Loads `data/seisbench/bransfield/{metadata.csv,waveforms.hdf5}` (built by
`06_build_finetune_dataset.py`), trains with a standard SeisBench Generator
pipeline + RealNoise from `bransfield_noise.{csv,hdf5}`, fine-tunes PhaseNet
from the `instance` pretrained weights.

Pilot usage (small dataset, fast):
    python scripts/07_finetune_phasenet.py --epochs 5 --batch-size 32

Full run (after rebuilding dataset without --max-events):
    python scripts/07_finetune_phasenet.py --epochs 30 --batch-size 64

The fine-tuned weights save to `models/phasenet_bransfield_<timestamp>/`.
Use them at inference with `--weights local:<path>` (not yet wired into
`03_run_phasenet.py`; manual load via `torch.load`).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import seisbench.data as sbd
import seisbench.generate as sbg
import seisbench.models as sbm

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data" / "seisbench"


def build_generator(dataset, noise_dataset, augment: bool, window_len: int = 3001,
                    sigma: int = 20):
    """Compose the SeisBench Generator pipeline."""
    gen = sbg.GenericGenerator(dataset)

    # Window selection: cut around the P pick (or S pick if no P), with random jitter.
    # We allow the pick to land anywhere in the central 60% of the window so the
    # model sees picks at different positions.
    gen.add_augmentations([
        sbg.WindowAroundSample(
            metadata_keys=["trace_p_arrival_sample", "trace_s_arrival_sample"],
            samples_before=window_len,            # candidate range start
            windowlen=2 * window_len,             # 2x window so RandomWindow has slack
            selection="random" if augment else "first",
            strategy="variable",
        ),
        sbg.RandomWindow(
            windowlen=window_len,
            strategy="pad",
            low=0,
            high=window_len,                      # jitter pick position freely
        ),
    ])

    # Optional: mix real OBS noise at variable SNR (only on positives during train).
    if augment and noise_dataset is not None and len(noise_dataset) > 0:
        try:
            gen.add_augmentations([
                sbg.OneOf(
                    augmentations=[
                        sbg.RealNoise(noise_dataset=noise_dataset, scale=(0.0, 1.5)),
                        sbg.NullAugmentation(),
                    ],
                    probabilities=[0.5, 0.5],
                )
            ])
        except Exception as e:
            print(f"  warning: skipping RealNoise (dataset incompatible): {e}")

    gen.add_augmentations([
        sbg.ChangeDtype(np.float32),
        sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type="peak"),
        sbg.ProbabilisticLabeller(
            label_columns={"trace_p_arrival_sample": "P",
                           "trace_s_arrival_sample": "S"},
            sigma=sigma,
            dim=0,
        ),
    ])
    return gen


def loss_fn(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Cross-entropy at every sample. PhaseNet predicts (B,3,T): N/P/S logits.
    Target is (B,3,T) probabilities; we use KL/CE loss."""
    # Standard PhaseNet training: KL divergence between target and pred (softmax).
    log_pred = torch.log_softmax(pred, dim=1)
    return -(target * log_pred).sum(dim=1).mean()


def evaluate(model, loader, device) -> dict:
    """One-pass eval: loss + per-class AUC-style soft hit rate."""
    model.eval()
    total_loss = 0.0
    n = 0
    p_hits = s_hits = p_total = s_total = 0
    with torch.no_grad():
        for batch in loader:
            x = batch["X"].to(device)
            y = batch["y"].to(device)
            pred = model(x)
            total_loss += loss_fn(pred, y).item() * x.size(0)
            n += x.size(0)
            # Soft "did the model peak near the labeled pick" check
            prob = torch.softmax(pred, dim=1)
            for ch, hits, totals in [(1, "p_hits", "p_total"), (2, "s_hits", "s_total")]:
                tgt_max = y[:, ch].max(dim=1).values
                pred_max = prob[:, ch].max(dim=1).values
                mask = tgt_max > 0.5
                if ch == 1:
                    p_total += mask.sum().item()
                    p_hits += ((pred_max > 0.3) & mask).sum().item()
                else:
                    s_total += mask.sum().item()
                    s_hits += ((pred_max > 0.3) & mask).sum().item()
    return {
        "loss": total_loss / max(n, 1),
        "p_softhit": p_hits / max(p_total, 1),
        "s_softhit": s_hits / max(s_total, 1),
        "n": n,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-name", default="bransfield",
                    help="data dir under data/seisbench/<name>/")
    ap.add_argument("--noise-csv", default="bransfield_noise.csv")
    ap.add_argument("--noise-hdf5", default="bransfield_noise.hdf5")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=None,
                    help="output dir (default: models/phasenet_bransfield_<timestamp>)")
    args = ap.parse_args()

    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device
    print(f"device={device}  epochs={args.epochs}  batch_size={args.batch_size}  lr={args.lr}")

    # Load datasets
    base = DATA / args.data_name
    ds = sbd.WaveformDataset(path=base, name=args.data_name, sampling_rate=100,
                             component_order="ZNE")
    train_ds = ds.train()
    dev_ds = ds.dev()
    print(f"train={len(train_ds)}  dev={len(dev_ds)}  test={len(ds.test())}")

    # Noise dataset for RealNoise augmentation (optional)
    noise_dir = DATA / "bransfield_noise_local"
    noise_dir.mkdir(parents=True, exist_ok=True)
    (noise_dir / "metadata.csv").unlink(missing_ok=True)
    (noise_dir / "waveforms.hdf5").unlink(missing_ok=True)
    (noise_dir / "metadata.csv").symlink_to(DATA / args.noise_csv)
    (noise_dir / "waveforms.hdf5").symlink_to(DATA / args.noise_hdf5)
    try:
        noise_ds = sbd.WaveformDataset(path=noise_dir, name="noise", sampling_rate=100,
                                        component_order="ZNE")
        print(f"noise={len(noise_ds)} traces available")
    except Exception as e:
        print(f"  noise dataset unavailable: {e}")
        noise_ds = None

    # Generators
    train_gen = build_generator(train_ds, noise_ds, augment=True)
    dev_gen = build_generator(dev_ds, None, augment=False)

    train_loader = DataLoader(train_gen, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, drop_last=True, pin_memory=True)
    dev_loader = DataLoader(dev_gen, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    # Load pretrained PhaseNet (instance — best for Bransfield P)
    print("Loading PhaseNet 'instance' weights ...")
    model = sbm.PhaseNet.from_pretrained("instance")
    model.to(device)
    model.train()

    # Optimizer
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Output dir
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = Path(args.out) if args.out else REPO / "models" / f"phasenet_bransfield_{ts}"
    out.mkdir(parents=True, exist_ok=True)
    print(f"checkpoints -> {out}")

    # Initial eval (sanity check on pretrained)
    init_eval = evaluate(model, dev_loader, device)
    print(f"  init dev: loss={init_eval['loss']:.4f}  "
          f"P-softhit={init_eval['p_softhit']:.3f}  S-softhit={init_eval['s_softhit']:.3f}")

    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        n_batches = 0
        train_loss = 0.0
        for batch in train_loader:
            x = batch["X"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            pred = model(x)
            loss = loss_fn(pred, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss += loss.item()
            n_batches += 1
        train_loss /= max(n_batches, 1)
        ev = evaluate(model, dev_loader, device)
        dt = time.time() - t0
        print(f"epoch {epoch:3d}/{args.epochs}  "
              f"train_loss={train_loss:.4f}  dev_loss={ev['loss']:.4f}  "
              f"P-softhit={ev['p_softhit']:.3f}  S-softhit={ev['s_softhit']:.3f}  "
              f"({dt:.1f}s)")
        if ev["loss"] < best_loss:
            best_loss = ev["loss"]
            ckpt = out / "best.pt"
            torch.save({"model": model.state_dict(),
                        "config": vars(args), "epoch": epoch,
                        "dev_loss": ev["loss"]}, ckpt)
            print(f"  saved best: {ckpt}")

    final = out / "final.pt"
    torch.save({"model": model.state_dict(),
                "config": vars(args), "epoch": args.epochs}, final)
    print(f"\nDone. Final weights: {final}\nBest weights: {out/'best.pt'}")


if __name__ == "__main__":
    main()
