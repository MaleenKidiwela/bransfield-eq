"""Pull a representative tomo-shot pyocto event and a representative
earthquake, read the full station-day mseed for each, and plot side-by-side
spectrograms ±15 s around the picked arrival.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from obspy import read
from scipy.signal import spectrogram

REPO = Path(__file__).resolve().parent.parent
WAVE_DIR = REPO / "data" / "waveforms"
OUT = REPO / "notes" / "figures" / "comparison" / "spectrogram_shot_vs_eq.png"


def waveform_path(network, station, year, doy):
    return WAVE_DIR / network / station / f"{network}.{station}.{year}.{doy:03d}.mseed"


def get_station_day(net, sta, year, doy, target_t, half_sec=20.0, fs=100.0):
    """Read mseed, resample, slice ±half_sec around target_t. Returns
    (data, t0_unix, fs)."""
    p = waveform_path(net, sta, year, doy)
    st = read(str(p))
    z = [tr for tr in st if tr.stats.channel.endswith("Z")]
    tr = z[0] if z else st.merge(fill_value=0)[0]
    if tr.stats.sampling_rate != fs:
        tr.resample(fs, no_filter=True)
    t0 = tr.stats.starttime.timestamp
    c = int(round((target_t.timestamp() - t0) * fs))
    n_each = int(round(half_sec * fs))
    s0, s1 = max(0, c - n_each), min(len(tr.data), c + n_each)
    return np.asarray(tr.data[s0:s1], dtype=np.float32), s0 / fs + t0, fs


def pick_one_event(ev_df, pk_df, station_pref=None, n=1):
    """Return list of (event_idx, station, network, pick_time, phase) tuples
    for the first n events in ev_df (already filtered by caller)."""
    candidates = ev_df
    if len(candidates) == 0:
        return []
    out = []
    for _, row in candidates.iterrows():
        eid = row["event_idx"]
        picks = pk_df[(pk_df.event_idx == eid) & (pk_df.phase == "P")]
        if station_pref is not None:
            picks_pref = picks[picks.station.str.endswith("." + station_pref)]
            if len(picks_pref) > 0:
                picks = picks_pref
        if len(picks) == 0:
            continue
        # Pick the highest-prob station
        p = picks.sort_values("prob", ascending=False).iloc[0]
        net, sta = p["station"].split(".")
        out.append({
            "event_idx": eid, "network": net, "station": sta,
            "pick_time": pd.to_datetime(p["time"], unit="s", utc=True),
            "phase": p["phase"], "prob": p["prob"],
        })
        if len(out) >= n:
            break
    return out


def main():
    print("Loading events + picks ...")
    ev = pd.read_csv(REPO / "catalogs" / "pyocto_events_picker_only_with_shot_flag_v2.csv",
                     low_memory=False)
    ev["origin_time"] = pd.to_datetime(ev.origin_time, utc=True, format="mixed", errors="coerce")
    pk = pd.read_csv(REPO / "catalogs" / "pyocto_picks_picker_only.csv")

    # Tomo shot: temporally flagged, survey=='orca_tomo', high prob_shot
    shot_mask = (ev["shot_survey"] == "orca_tomo") & ev["flag_shot"].astype(bool)
    # Sort by classifier confidence so we pick a very-confident shot
    shot_pool = ev[shot_mask].sort_values("prob_shot", ascending=False)
    shot_picks = pick_one_event(shot_pool, pk, station_pref="BYE", n=3)
    if not shot_picks:
        shot_picks = pick_one_event(shot_pool, pk, n=3)

    # Earthquake: outside shot window, low prob_shot
    eq_mask = (
        (ev["origin_time"] < pd.Timestamp("2019-01-21", tz="UTC")) &
        (ev["prob_shot"].fillna(0) < 0.1) &
        ev["origin_time"].notna()
    )
    eq_pool = ev[eq_mask].sort_values("prob_shot")
    eq_picks = pick_one_event(eq_pool, pk, station_pref="BYE", n=3)
    if not eq_picks:
        eq_picks = pick_one_event(eq_pool, pk, n=3)

    print(f"shot candidates: {len(shot_picks)},  eq candidates: {len(eq_picks)}")
    print(f"shot picked: {shot_picks[0] if shot_picks else None}")
    print(f"eq picked:   {eq_picks[0] if eq_picks else None}")

    # Read mseed for both
    shot = shot_picks[0]
    eq   = eq_picks[0]

    shot_wf, shot_t0, fs = get_station_day(
        shot["network"], shot["station"],
        shot["pick_time"].year, shot["pick_time"].dayofyear,
        shot["pick_time"], half_sec=20.0)
    eq_wf, eq_t0, _ = get_station_day(
        eq["network"], eq["station"],
        eq["pick_time"].year, eq["pick_time"].dayofyear,
        eq["pick_time"], half_sec=20.0)

    # Spectrograms
    nperseg = 128   # 1.28 s windows at 100 Hz
    noverlap = 96

    fig, axes = plt.subplots(2, 2, figsize=(15, 9),
                              gridspec_kw={"height_ratios": [1, 2]})
    for col, (wf, t0, info, title) in enumerate([
        (shot_wf, shot_t0, shot, "TOMO SHOT"),
        (eq_wf, eq_t0, eq, "EARTHQUAKE"),
    ]):
        # Time axis
        t_vec = np.arange(len(wf)) / fs - (info["pick_time"].timestamp() - t0)
        # Top: time series
        ax = axes[0, col]
        ax.plot(t_vec, wf, color="black", linewidth=0.7)
        ax.axvline(0, color="red", linestyle="--", linewidth=1.0,
                   label=f"pick (prob={info['prob']:.2f})")
        ax.set_xlim(t_vec.min(), t_vec.max())
        ax.set_xlabel("time relative to P pick (s)")
        ax.set_ylabel("amplitude (counts)")
        ax.set_title(f"{title}  —  {info['network']}.{info['station']}  "
                     f"@ {info['pick_time']}")
        ax.legend(loc="upper right")
        ax.grid(alpha=0.3)

        # Bottom: spectrogram
        ax = axes[1, col]
        f, t_spec, Sxx = spectrogram(wf, fs=fs, nperseg=nperseg, noverlap=noverlap)
        t_rel = t_spec - (info["pick_time"].timestamp() - t0)
        Sxx_db = 10 * np.log10(Sxx + 1e-10)
        vmin, vmax = np.percentile(Sxx_db, [5, 99])
        im = ax.pcolormesh(t_rel, f, Sxx_db, shading="auto",
                            cmap="viridis", vmin=vmin, vmax=vmax)
        ax.axvline(0, color="red", linestyle="--", linewidth=1.0)
        ax.set_xlabel("time relative to P pick (s)")
        ax.set_ylabel("frequency (Hz)")
        ax.set_ylim(0, 50)
        plt.colorbar(im, ax=ax, label="power (dB)")

    fig.suptitle("Spectrogram comparison: airgun shot vs natural earthquake\n"
                 "Shots: narrow-band, harmonic spacing from airgun bubble pulses. "
                 "EQs: broadband, decaying after onset.",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(OUT, dpi=140, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
