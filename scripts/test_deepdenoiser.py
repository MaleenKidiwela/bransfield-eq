"""
Phase 0: sanity-check off-the-shelf DeepDenoiser ('original' weights, land-trained)
on Bransfield OBS waveforms.

Pulls 3 high-confidence mag07 event windows + 3 noise windows from disk, applies
DeepDenoiser, plots original / denoised / residual side by side.

Outputs: figures/dd_sanity/event_*.png, figures/dd_sanity/noise_*.png

Decision rule: if denoised events look mangled (impulsive arrivals lost), Phase 1
retraining is mandatory. If reasonable, retraining is still planned but we have a
cheap fallback option.
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
from obspy import UTCDateTime, read

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
from bransfield_eq.config import mseed_path  # noqa: E402

import seisbench.models as sbm  # noqa: E402

OUT_DIR = REPO / "figures" / "dd_sanity"
WINDOW_SEC = 30.0
TARGET_RATE = 100.0
PRE_SEC = 5.0  # seconds before pick


def load_3c(net: str, sta: str, day: UTCDateTime, t_center: UTCDateTime,
            picking_glob: str = "EH?,HH?,BH?,SH?,EL?,HL?,BL?,SL?"):
    """Read mseed for the day, trim around t_center, resample, return 3-component obspy stream."""
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
    # Need at least one Z and two horizontals; let DeepDenoiser handle component_order
    return st


def plot_triplet(stream_in, stream_out, title: str, out_path: Path):
    """3-row plot (Z, H1, H2): original, denoised, residual."""
    n_chan = min(3, len(stream_in))
    fig, axes = plt.subplots(n_chan, 3, figsize=(13, 2.0 * n_chan + 0.6),
                              sharex=True, squeeze=False)
    for i in range(n_chan):
        tr_in = stream_in[i]
        chan = tr_in.stats.channel
        # Find matching channel in output by suffix letter (Z, N, E, 1, 2)
        matching = [t for t in stream_out if t.stats.channel == chan]
        if not matching:
            # fallback: same component letter
            matching = [t for t in stream_out
                        if t.stats.channel.endswith(chan[-1])]
        if not matching:
            continue
        tr_out = matching[0]
        n = min(tr_in.stats.npts, tr_out.stats.npts)
        t = np.arange(n) / TARGET_RATE
        amp_in = tr_in.data[:n].astype(np.float32)
        amp_out = tr_out.data[:n].astype(np.float32)
        amp_res = amp_in - amp_out
        peak = max(abs(amp_in).max(), 1e-12)

        axes[i, 0].plot(t, amp_in / peak, "k", lw=0.6)
        axes[i, 0].set_ylabel(f"{chan}", rotation=0, ha="right", va="center", fontsize=9)
        axes[i, 0].set_ylim(-1.2, 1.2); axes[i, 0].set_yticks([])
        if i == 0: axes[i, 0].set_title("Original")

        axes[i, 1].plot(t, amp_out / peak, "C0", lw=0.6)
        axes[i, 1].set_ylim(-1.2, 1.2); axes[i, 1].set_yticks([])
        if i == 0: axes[i, 1].set_title("Denoised (DeepDenoiser 'original')")

        axes[i, 2].plot(t, amp_res / peak, "C3", lw=0.6)
        axes[i, 2].set_ylim(-1.2, 1.2); axes[i, 2].set_yticks([])
        if i == 0: axes[i, 2].set_title("Residual = original − denoised")

    axes[-1, 1].set_xlabel("Seconds in window")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading DeepDenoiser('original') ...")
    model = sbm.DeepDenoiser.from_pretrained("original")
    print(f"  in_samples={model.in_samples}, sampling_rate={model.sampling_rate}")
    print()

    # Pick 3 mag07 events on stations whose mseed is available, and 3 noise windows.
    m = pd.read_csv(REPO / "catalogs" / "manual_picks.csv",
                    parse_dates=["origin_time", "pick_time"])
    m07 = m[(m.source_file == "nllmaleen_mag07_202210.out") &
            (m.uncertainty_s <= 0.1)].copy()
    # Prefer day-26 + a couple of high-density days we know are downloaded
    candidates = m07[m07.pick_time.dt.date.isin([
        pd.Timestamp("2019-12-26").date(),
        pd.Timestamp("2019-02-13").date(),
        pd.Timestamp("2019-08-20").date(),
    ])].copy()

    print("=== EVENT windows ===")
    n_events_done = 0
    for _, r in candidates.iterrows():
        if n_events_done >= 3:
            break
        net, sta = r.network, r.station
        t_center = UTCDateTime(r.pick_time.to_pydatetime())
        day = UTCDateTime(t_center.date)
        st = load_3c(net, sta, day, t_center)
        if st is None or len(st) < 3:
            continue
        try:
            denoised = model.annotate(st)
        except Exception as e:
            print(f"  [skip] {net}.{sta} {t_center}: {e}")
            continue
        if denoised is None or len(denoised) == 0:
            continue
        title = (f"Event {r.event_id}  |  {net}.{sta}  |  {r.phase}-pick @ "
                 f"{t_center.strftime('%Y-%m-%dT%H:%M:%S')} UTC")
        out = OUT_DIR / f"event_{n_events_done+1:02d}_{net}_{sta}_{t_center.strftime('%Y%m%dT%H%M%S')}.png"
        plot_triplet(st, denoised, title, out)
        n_events_done += 1

    print()
    print("=== NOISE windows (away from any pick by ≥60 s) ===")
    # Pick 3 random noise windows from BRA13/16/19 (all complete) on a non-pick time.
    rng = np.random.default_rng(0)
    pick_times_by_sta = (m07.assign(t=m07.pick_time.values.astype("datetime64[s]").astype(np.int64))
                              .groupby(["network", "station"])["t"].apply(np.array)
                              .to_dict())
    n_noise_done = 0
    candidate_stations = [("ZX", "BRA13"), ("ZX", "BRA16"), ("ZX", "BRA19"),
                          ("ZX", "BRA22"), ("ZX", "BRA25")]
    for (net, sta) in candidate_stations:
        if n_noise_done >= 3:
            break
        # Pick a random hour-of-day on 2019-08-15 (a quiet day)
        for attempt in range(20):
            day = UTCDateTime("2019-08-15")
            t_center = day + rng.uniform(0, 86400 - WINDOW_SEC)
            sta_picks = pick_times_by_sta.get((net, sta), np.array([]))
            if len(sta_picks):
                ts = int(t_center.timestamp)
                idx = np.searchsorted(sta_picks, ts)
                near = min(abs(sta_picks[max(idx - 1, 0)] - ts),
                           abs(sta_picks[min(idx, len(sta_picks) - 1)] - ts))
                if near < 60:
                    continue
            st = load_3c(net, sta, day, t_center)
            if st is None or len(st) < 3:
                continue
            try:
                denoised = model.annotate(st)
            except Exception as e:
                print(f"  [skip] {net}.{sta} noise: {e}")
                break
            if denoised is None or len(denoised) == 0:
                continue
            title = (f"NOISE  |  {net}.{sta}  |  "
                     f"{t_center.strftime('%Y-%m-%dT%H:%M:%S')} UTC (no manual pick within 60s)")
            out = OUT_DIR / f"noise_{n_noise_done+1:02d}_{net}_{sta}_{t_center.strftime('%Y%m%dT%H%M%S')}.png"
            plot_triplet(st, denoised, title, out)
            n_noise_done += 1
            break

    print()
    print(f"Done.  wrote {n_events_done} event plots + {n_noise_done} noise plots to {OUT_DIR}")


if __name__ == "__main__":
    main()
