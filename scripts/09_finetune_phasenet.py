"""
Phase 3: fine-tune PhaseNet on the augmented Bransfield dataset.

Loads data/seisbench/bransfield_aug (built by 08_build_augmented_dataset.py),
fine-tunes PhaseNet from `instance` weights, saves Lightning-style checkpoint
to models/phasenet_bransfield_v1/best.ckpt.

Uses a self-contained PyTorch training loop with SeisBench's GenericGenerator
+ ProbabilisticLabeller pipeline. The state dict in the checkpoint is wrapped
to look like a Lightning checkpoint (state_dict with 'model.' prefix) so
03_run_phasenet.py's --weights ckpt:<path> can load it cleanly.

Online augmentation per epoch (in addition to the pre-materialized variants):
- RandomWindow jitter
- ChannelDropout
- Optional extra noise mixing
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
WINDOW_SAMPLES = 3001  # PhaseNet expects 3001
TARGET_RATE = 100.0


def make_generator(ds, augment: bool, noise_dataset=None, sigma: int = 20):
    gen = sbg.GenericGenerator(ds)
    augs = [
        sbg.WindowAroundSample(
            metadata_keys=["trace_p_arrival_sample", "trace_s_arrival_sample"],
            samples_before=WINDOW_SAMPLES,
            windowlen=2 * WINDOW_SAMPLES,
            selection="random" if augment else "first",
            strategy="variable",
        ),
        sbg.RandomWindow(
            windowlen=WINDOW_SAMPLES,
            strategy="pad",
            low=0,
            high=WINDOW_SAMPLES,
        ),
    ]
    # NOTE: We deliberately DON'T add online RealNoise here — we already have
    # 30 pre-materialized noise variants per pick (Phase 2). RealNoise also
    # requires noise samples >= window length which the pre-materialized
    # noise pool doesn't satisfy (3000 vs 3001).
    augs.extend([
        sbg.ChangeDtype(np.float32),
        sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type="peak"),
        sbg.ProbabilisticLabeller(
            label_columns={"trace_p_arrival_sample": "P",
                            "trace_s_arrival_sample": "S"},
            sigma=sigma,
            dim=0,
        ),
    ])
    if augment:
        # Channel dropout — randomly zero some components
        augs.append(sbg.ChannelDropout(axis=-2))
    gen.add_augmentations(augs)
    return gen


def loss_fn(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """KL-divergence between target and softmax(pred). Standard for PhaseNet."""
    log_pred = torch.log_softmax(pred, dim=1)
    return -(target * log_pred).sum(dim=1).mean()


@torch.no_grad()
def evaluate(model, loader, device, tol_samples: int = 50):
    model.eval()
    total_loss, n = 0.0, 0
    p_tp = p_fn = s_tp = s_fn = 0
    for batch in loader:
        x = batch["X"].to(device).float()
        y = batch["y"].to(device).float()
        pred = model(x)
        total_loss += loss_fn(pred, y).item() * x.size(0); n += x.size(0)
        prob = torch.softmax(pred, dim=1)
        for ch, tp_var, fn_var in [(1, "p", "p"), (2, "s", "s")]:
            t_argmax = y[:, ch].argmax(dim=-1)
            t_max = y[:, ch].max(dim=-1).values
            p_argmax = prob[:, ch].argmax(dim=-1)
            mask = t_max > 0.5
            within = (t_argmax - p_argmax).abs() <= tol_samples
            tp = (mask & within).sum().item()
            fn = (mask & ~within).sum().item()
            if ch == 1:
                p_tp += tp; p_fn += fn
            else:
                s_tp += tp; s_fn += fn
    return {
        "loss": total_loss / max(n, 1),
        "p_recall": p_tp / max(p_tp + p_fn, 1),
        "s_recall": s_tp / max(s_tp + s_fn, 1),
        "n": n,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(REPO / "data/seisbench/bransfield_aug"))
    ap.add_argument("--noise", default=str(REPO / "data/seisbench/bransfield_noise"))
    ap.add_argument("--out", default=str(REPO / "models/phasenet_bransfield_v1"))
    ap.add_argument("--pretrained", default="instance")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--val-frac", type=float, default=0.15,
                    help="Fraction of source events held out for val (by event_id hash)")
    args = ap.parse_args()

    device = ("cuda" if (args.device == "auto" and torch.cuda.is_available())
              else args.device)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={device} epochs={args.epochs} batch={args.batch_size} lr={args.lr}")

    # Load dataset
    print("Loading augmented dataset ...")
    ds = sbd.WaveformDataset(path=args.data, sampling_rate=100,
                              component_order="ZNE")
    print(f"  total: {len(ds)} variants")

    # Train/dev split by source event_id (extract original event id from
    # source_id, which is "<eventid>__<variant>" for variants — double underscore).
    meta = ds.metadata
    if "source_id" in meta.columns:
        base_ids = meta.source_id.astype(str).str.split("__", n=1).str[0]
    else:
        base_ids = pd.Series(range(len(meta)))
    import hashlib
    def is_val(eid):
        return (int(hashlib.md5(eid.encode()).hexdigest(), 16) % 100) < int(args.val_frac * 100)
    val_mask = base_ids.apply(is_val).values
    train_mask = ~val_mask
    train_ds = ds.filter(train_mask, inplace=False)
    val_ds = ds.filter(val_mask, inplace=False)
    print(f"  train: {len(train_ds)}  val: {len(val_ds)}")

    # Optional online-noise dataset
    noise_ds = None
    if Path(args.noise).exists():
        try:
            noise_ds = sbd.WaveformDataset(path=args.noise, sampling_rate=100,
                                           component_order="ZNE")
            print(f"  online-noise pool: {len(noise_ds)} traces")
        except Exception as e:
            print(f"  warn: noise dataset unavailable: {e}")

    train_gen = make_generator(train_ds, augment=True, noise_dataset=noise_ds)
    val_gen = make_generator(val_ds, augment=False)
    train_loader = DataLoader(train_gen, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_gen, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    # Model
    print(f"Loading PhaseNet({args.pretrained!r}) ...")
    model = sbm.PhaseNet.from_pretrained(args.pretrained)
    model.to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Sanity baseline
    init = evaluate(model, val_loader, device)
    print(f"  init val: loss={init['loss']:.4f}  P-rec={init['p_recall']:.3f}  S-rec={init['s_recall']:.3f}")

    train_losses, val_losses, p_recs, s_recs = [], [], [init["p_recall"]], [init["s_recall"]]
    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0; nb = 0
        for batch in train_loader:
            x = batch["X"].to(device, non_blocking=True).float()
            y = batch["y"].to(device, non_blocking=True).float()
            pred = model(x)
            loss = loss_fn(pred, y)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            train_loss += loss.item(); nb += 1
        train_loss /= max(nb, 1)
        ev = evaluate(model, val_loader, device)
        dt = time.time() - t0
        train_losses.append(train_loss); val_losses.append(ev["loss"])
        p_recs.append(ev["p_recall"]); s_recs.append(ev["s_recall"])
        marker = ""
        if ev["loss"] < best_loss:
            best_loss = ev["loss"]
            ckpt = {
                "state_dict": {f"model.{k}": v for k, v in model.state_dict().items()},
                "model_class": "PhaseNet",
                "pretrained_base": args.pretrained,
                "epoch": epoch,
                "val_loss": ev["loss"],
                "val_p_recall": ev["p_recall"],
                "val_s_recall": ev["s_recall"],
                "config": vars(args),
            }
            torch.save(ckpt, out_dir / "best.ckpt")
            marker = "  *saved*"
        print(f"epoch {epoch:3d}/{args.epochs}  train={train_loss:.4f}  "
              f"val={ev['loss']:.4f}  P-rec={ev['p_recall']:.3f}  "
              f"S-rec={ev['s_recall']:.3f}  ({dt:.1f}s){marker}", flush=True)

    # Final
    torch.save({
        "state_dict": {f"model.{k}": v for k, v in model.state_dict().items()},
        "model_class": "PhaseNet",
        "pretrained_base": args.pretrained,
        "epoch": args.epochs,
        "config": vars(args),
    }, out_dir / "final.ckpt")

    # Loss + recall curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(train_losses, label="train"); axes[0].plot(val_losses, label="val")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("KL loss"); axes[0].legend(); axes[0].set_title("Loss")
    axes[1].plot(p_recs, "r-", label="P recall"); axes[1].plot(s_recs, "b-", label="S recall")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Recall (val, ±0.5s)"); axes[1].set_ylim(0, 1)
    axes[1].legend(); axes[1].set_title("Validation recall")
    fig.suptitle(f"PhaseNet fine-tune (best val loss={best_loss:.4f})")
    fig.tight_layout()
    fig_path = REPO / "figures" / "finetune_train.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=140)
    plt.close(fig)
    print(f"\nDone. best.ckpt → {out_dir/'best.ckpt'}  loss curve → {fig_path}")


if __name__ == "__main__":
    import pandas as pd  # for the split-by-event hashing line at module bottom
    main()
