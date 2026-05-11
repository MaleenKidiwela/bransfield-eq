"""
Visual sanity check of the trained OBS-DeepDenoiser (Phase 1b output).

Same windows as scripts/test_deepdenoiser.py, but uses our fine-tuned weights
loaded from models/deepdenoiser_obs/best.pt. Compares side-by-side to the
off-the-shelf 'original' denoiser to show whether retraining changed anything.

Outputs: figures/dd_obs_sanity/event_*.png, noise_*.png — each shows
3 columns: original | off-the-shelf denoised | OBS-fine-tuned denoised
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from obspy import UTCDateTime, read

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
from bransfield_eq.config import mseed_path  # noqa: E402

import seisbench.models as sbm  # noqa: E402

OUT_DIR = REPO / "figures" / "dd_obs_sanity"
WINDOW_SEC = 30.0
TARGET_RATE = 100.0
PRE_SEC = 5.0


def load_3c(net, sta, day, t_center,
            picking_glob="EH?,HH?,BH?,SH?,EL?,HL?,BL?,SL?"):
    import fnmatch
    mp = mseed_path(net, sta, day)
    if not mp.exists() or mp.stat().st_size == 0:
        return None
    t0, t1 = t_center - PRE_SEC, t_center - PRE_SEC + WINDOW_SEC
    try:
        st = read(str(mp), starttime=t0, endtime=t1).copy()
    except Exception:
        return None
    st.traces = [tr for tr in st
                 if any(fnmatch.fnmatchcase(tr.stats.channel, g.strip())
                        for g in picking_glob.split(","))]
    if not st:
        return None
    st.merge(method=1, fill_value=0)
    for tr in st:
        if abs(tr.stats.sampling_rate - TARGET_RATE) > 1e-6:
            tr.resample(TARGET_RATE)
    return st


def plot_compare(stream_in, stream_off, stream_obs, title, out_path):
    n_chan = min(3, len(stream_in))
    fig, axes = plt.subplots(n_chan, 3, figsize=(13, 2.0 * n_chan + 0.6),
                              sharex=True, squeeze=False)
    for i in range(n_chan):
        tr_in = stream_in[i]
        chan = tr_in.stats.channel

        def _match(stream):
            m = [t for t in stream if t.stats.channel == chan]
            if not m:
                m = [t for t in stream if t.stats.channel.endswith(chan[-1])]
            return m[0] if m else None

        tr_off = _match(stream_off)
        tr_obs = _match(stream_obs)

        n = min(tr_in.stats.npts,
                 tr_off.stats.npts if tr_off else 1,
                 tr_obs.stats.npts if tr_obs else 1)
        t = np.arange(n) / TARGET_RATE
        amp_in = tr_in.data[:n].astype(np.float32)
        peak = max(abs(amp_in).max(), 1e-12)

        axes[i, 0].plot(t, amp_in / peak, "k", lw=0.6)
        axes[i, 0].set_ylabel(chan, rotation=0, ha="right", va="center", fontsize=9)
        axes[i, 0].set_ylim(-1.2, 1.2); axes[i, 0].set_yticks([])
        if i == 0: axes[i, 0].set_title("Original")

        if tr_off:
            axes[i, 1].plot(t, tr_off.data[:n] / peak, "C0", lw=0.6)
        axes[i, 1].set_ylim(-1.2, 1.2); axes[i, 1].set_yticks([])
        if i == 0: axes[i, 1].set_title("Off-the-shelf 'original' (land STEAD)")

        if tr_obs:
            axes[i, 2].plot(t, tr_obs.data[:n] / peak, "C2", lw=0.6)
        axes[i, 2].set_ylim(-1.2, 1.2); axes[i, 2].set_yticks([])
        if i == 0: axes[i, 2].set_title("OBS-fine-tuned (Phase 1b)")

    axes[-1, 1].set_xlabel("Seconds in window")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    warnings.filterwarnings("ignore", category=FutureWarning)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading off-the-shelf DeepDenoiser('original') ...")
    model_off = sbm.DeepDenoiser.from_pretrained("original")

    print("Loading OBS-fine-tuned DeepDenoiser ...")
    model_obs = sbm.DeepDenoiser.from_pretrained("original")
    ck = torch.load(REPO / "models/deepdenoiser_obs/best.pt", weights_only=False)
    model_obs.load_state_dict(ck["model"])

    # On CPU to avoid GPU contention with concurrent fine-tune
    print("(running on CPU to avoid GPU contention with active fine-tune)")
    print()

    m = pd.read_csv(REPO / "catalogs" / "manual_picks.csv",
                    parse_dates=["origin_time", "pick_time"])
    m07 = m[(m.source_file == "nllmaleen_mag07_202210.out") &
            (m.uncertainty_s <= 0.1)].copy()

    # 3 events (use HELD-OUT VAL stations BRA13 + BRA22 to test generalization)
    print("=== EVENT windows (val stations BRA13 + BRA22) ===")
    val_stations = ["BRA13", "BRA22"]
    n = 0
    for _, r in m07[m07.station.isin(val_stations)].iterrows():
        if n >= 3:
            break
        net, sta = r.network, r.station
        t_center = UTCDateTime(r.pick_time.to_pydatetime())
        day = UTCDateTime(t_center.date)
        st = load_3c(net, sta, day, t_center)
        if st is None or len(st) < 3:
            continue
        try:
            den_off = model_off.annotate(st.copy())
            den_obs = model_obs.annotate(st.copy())
        except Exception as e:
            print(f"  [skip] {e}")
            continue
        if not den_off or not den_obs:
            continue
        title = f"Event {r.event_id}  |  {net}.{sta}  |  {r.phase}-pick @ {t_center.strftime('%Y-%m-%dT%H:%M:%S')}"
        out = OUT_DIR / f"event_{n+1:02d}_{net}_{sta}.png"
        plot_compare(st, den_off, den_obs, title, out)
        n += 1

    print()
    print("=== NOISE windows (val station BRA13 on a quiet day) ===")
    rng = np.random.default_rng(0)
    nn = 0
    for attempt in range(15):
        if nn >= 3:
            break
        day = UTCDateTime("2019-08-15")
        t_center = day + rng.uniform(0, 86400 - WINDOW_SEC)
        st = load_3c("ZX", "BRA13", day, t_center)
        if st is None or len(st) < 3:
            continue
        try:
            den_off = model_off.annotate(st.copy())
            den_obs = model_obs.annotate(st.copy())
        except Exception as e:
            print(f"  [skip] {e}")
            continue
        if not den_off or not den_obs:
            continue
        out = OUT_DIR / f"noise_{nn+1:02d}_BRA13_{t_center.strftime('%H%M%S')}.png"
        title = f"NOISE  |  ZX.BRA13  |  {t_center.strftime('%Y-%m-%dT%H:%M:%S')}"
        plot_compare(st, den_off, den_obs, title, out)
        nn += 1

    print(f"\nDone.  {OUT_DIR}")


if __name__ == "__main__":
    main()
