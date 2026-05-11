"""
Phase 1b: fine-tune SeisBench DeepDenoiser on Bransfield OBS data.

DeepDenoiser is a STFT U-Net that outputs a time-frequency mask separating
signal from noise. SeisBench's STFTDenoiserLabeller builds the supervisory mask
from (clean event waveform + random noise window) pairs.

Inputs (from Phase 1a):
  data/seisbench/bransfield_events/  (clean events, 3-component, 100 Hz)
  data/seisbench/bransfield_noise/   (quiet OBS noise windows)

Output:
  models/deepdenoiser_obs/best.pt    state-dict by min val_loss
  figures/dd_train_loss.png          train + val loss curve

Train/val split is BY STATION (not random per-window) so val measures
generalisation to held-out stations, not just held-out windows of trained
stations.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import seisbench.data as sbd
import seisbench.generate as sbg
import seisbench.models as sbm

REPO = Path(__file__).resolve().parent.parent


def split_by_station(ds: sbd.WaveformDataset, val_frac: float = 0.1, seed: int = 0):
    """Hold out ~val_frac of stations entirely from training."""
    stations = sorted(ds.metadata.station_code.unique())
    rng = np.random.default_rng(seed)
    rng.shuffle(stations)
    n_val = max(1, int(round(len(stations) * val_frac)))
    val_stations = set(stations[:n_val])
    train_stations = set(stations[n_val:])
    train_mask = ds.metadata.station_code.isin(train_stations).values
    val_mask = ds.metadata.station_code.isin(val_stations).values
    print(f"  train stations ({len(train_stations)}): {sorted(train_stations)}")
    print(f"  val stations   ({len(val_stations)}): {sorted(val_stations)}")
    return train_mask, val_mask, train_stations, val_stations


def make_generator(events_ds: sbd.WaveformDataset, noise_ds: sbd.WaveformDataset):
    """SeisBench generator: per sample mix event+noise via STFTDenoiserLabeller."""
    gen = sbg.GenericGenerator(events_ds)
    gen.add_augmentations([
        sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type="peak"),
        sbg.STFTDenoiserLabeller(
            noise_dataset=noise_ds,
            scale=(0.3, 2.0),       # event vs noise scale range; pairs roughly to SNR ∈ [0.5, 10]
            scaling_type="peak",
            component="ZNE",
        ),
    ])
    return gen


def epoch_pass(model, loader, optim, device, train: bool):
    model.train(train)
    total = 0.0
    n_batches = 0
    for batch in loader:
        x = batch["X"].to(device, non_blocking=True).float()
        y = batch["y"].to(device, non_blocking=True).float()
        if train:
            optim.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            pred = model(x)
            # MSE between predicted mask and ground-truth mask
            loss = F.mse_loss(pred, y)
            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
        total += loss.item()
        n_batches += 1
    return total / max(n_batches, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default=str(REPO / "data/seisbench/bransfield_events"))
    ap.add_argument("--noise", default=str(REPO / "data/seisbench/bransfield_noise"))
    ap.add_argument("--out", default=str(REPO / "models/deepdenoiser_obs"))
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = ("cuda" if (args.device == "auto" and torch.cuda.is_available())
              else args.device)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={device}  epochs={args.epochs}  batch={args.batch_size}  lr={args.lr}")

    # Load datasets
    print("Loading event dataset ...")
    events = sbd.WaveformDataset(path=args.events, sampling_rate=100,
                                  component_order="ZNE")
    print(f"  events: {len(events)} traces, {events.metadata.station_code.nunique()} stations")
    print("Loading noise dataset ...")
    noise = sbd.WaveformDataset(path=args.noise, sampling_rate=100,
                                 component_order="ZNE")
    print(f"  noise:  {len(noise)} traces, {noise.metadata.station_code.nunique()} stations")
    print()

    # Train / val split by station
    print("Splitting events by station ...")
    train_mask, val_mask, _, _ = split_by_station(events, args.val_frac, args.seed)
    train_ds = events.filter(train_mask, inplace=False)
    val_ds = events.filter(val_mask, inplace=False)
    print(f"  train traces: {len(train_ds)}  val traces: {len(val_ds)}")
    print()

    # Generators
    train_gen = make_generator(train_ds, noise)
    val_gen = make_generator(val_ds, noise)

    train_loader = DataLoader(train_gen, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_gen, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    # Model
    print("Loading DeepDenoiser('original') ...")
    model = sbm.DeepDenoiser.from_pretrained("original")
    model.to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    train_losses, val_losses = [], []
    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss = epoch_pass(model, train_loader, optim, device, train=True)
        vl_loss = epoch_pass(model, val_loader, optim, device, train=False)
        dt = time.time() - t0
        train_losses.append(tr_loss); val_losses.append(vl_loss)
        marker = ""
        if vl_loss < best_val:
            best_val = vl_loss
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "val_loss": vl_loss, "config": vars(args)},
                       out_dir / "best.pt")
            marker = "  *saved*"
        print(f"epoch {epoch:3d}/{args.epochs}  train={tr_loss:.5f}  "
              f"val={vl_loss:.5f}  ({dt:.1f}s){marker}", flush=True)

    torch.save({"model": model.state_dict(), "epoch": args.epochs,
                "val_loss": vl_loss, "config": vars(args)},
               out_dir / "final.pt")

    # Loss curves
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label="train")
    ax.plot(epochs, val_losses, label="val (held-out stations)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE on STFT mask")
    ax.set_title(f"OBS DeepDenoiser fine-tuning (best val={best_val:.5f})")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig_path = REPO / "figures" / "dd_train_loss.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=140)
    plt.close(fig)
    print(f"\nDone. best.pt → {out_dir / 'best.pt'}  loss curve → {fig_path}")


if __name__ == "__main__":
    main()
