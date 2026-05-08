"""
Stage 1.a — Bransfield EQ data inventory.

Reads `configs/targets.yaml` and queries each FDSN data center for station
metadata only (no waveforms). Writes per-network StationXML, a flat
`station_inventory.csv`, and a disk-space estimate based on actual
operational overlap of each channel-epoch with the target window.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from obspy import UTCDateTime

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from bransfield_eq.config import (
    Config, XML_DIR, get_client,
)

CATALOG_DIR = REPO / "catalogs"
XML_DIR.mkdir(parents=True, exist_ok=True)
CATALOG_DIR.mkdir(parents=True, exist_ok=True)

# MSEED Steim2 typical compression for seismic int32 data.
COMPRESSION_RATIO = 3.5


def query_target(client, net: str, sta: str, channels: str,
                 starttime, endtime, bbox: dict | None) -> "obspy.Inventory":
    kwargs = dict(network=net, station=sta, level="channel",
                  channel=channels, starttime=starttime, endtime=endtime)
    if bbox:
        kwargs.update(bbox)
    return client.get_stations(**kwargs)


def inventory_to_df(inv) -> pd.DataFrame:
    rows = []
    for net in inv:
        for sta in net:
            for ch in sta:
                rows.append({
                    "network": net.code,
                    "station": sta.code,
                    "location": ch.location_code,
                    "channel": ch.code,
                    "latitude": ch.latitude,
                    "longitude": ch.longitude,
                    "elevation_m": ch.elevation,
                    "depth_m": ch.depth,
                    "sample_rate_hz": ch.sample_rate,
                    "start": str(ch.start_date) if ch.start_date else None,
                    "end": str(ch.end_date) if ch.end_date else None,
                })
    return pd.DataFrame(rows)


def channel_bytes(sample_rate_hz: float, seconds: float) -> float:
    return (sample_rate_hz * 4 * seconds) / COMPRESSION_RATIO


def overlap_seconds(row, win_start: UTCDateTime, win_end: UTCDateTime) -> float:
    s = pd.to_datetime(row["start"]) if row["start"] else None
    e = pd.to_datetime(row["end"]) if row["end"] else None
    if s is None:
        return 0.0
    s = UTCDateTime(s.to_pydatetime())
    e = UTCDateTime(e.to_pydatetime()) if e is not None else win_end
    a = max(s, win_start)
    b = min(e, win_end)
    return max(0.0, b - a)


def main() -> None:
    cfg = Config.load()
    clients: dict = {}
    frames = []

    for tgt in cfg.targets:
        print(f"[{tgt.data_center}] {tgt.network}.{tgt.station} — {tgt.label} — querying ...")
        if tgt.data_center not in clients:
            clients[tgt.data_center] = get_client(tgt.data_center)
        try:
            inv = query_target(
                clients[tgt.data_center],
                tgt.network, tgt.station, cfg.channels,
                cfg.start, cfg.end,
                cfg.bbox if tgt.bbox_filter else None,
            )
        except Exception as e:
            print(f"  FAILED: {e}")
            continue

        sta_safe = tgt.station.replace("*", "ALL")
        xml_path = XML_DIR / f"{tgt.network}_{sta_safe}_{tgt.data_center}.xml"
        inv.write(str(xml_path), format="STATIONXML")
        df = inventory_to_df(inv)
        df["target_label"] = tgt.label
        df["data_center"] = tgt.data_center
        frames.append(df)
        print(f"  stations={df['station'].nunique()}  "
              f"channel-epochs={len(df)}  "
              f"saved={xml_path.relative_to(REPO)}")

    if not frames:
        print("No metadata returned.")
        return

    full = pd.concat(frames, ignore_index=True)
    full["overlap_s"] = full.apply(
        lambda r: overlap_seconds(r, cfg.start, cfg.end), axis=1)
    full["bytes_in_window"] = full.apply(
        lambda r: channel_bytes(r["sample_rate_hz"], r["overlap_s"]), axis=1)

    out_csv = CATALOG_DIR / "station_inventory.csv"
    full.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv.relative_to(REPO)}  ({len(full)} rows)")

    by_net = full.groupby("network")["bytes_in_window"].sum() / 1e9
    print(f"\nEstimated MSEED footprint over {cfg.start.date} → {cfg.end.date} "
          f"(compressed, GB):")
    for net, gb in by_net.items():
        n_active = full[(full.network == net) & (full.overlap_s > 0)]["station"].nunique()
        print(f"  {net:>4} : {gb:>8.1f} GB   ({n_active} stations active in window)")
    total_gb = by_net.sum()
    print(f"  ---- : {total_gb:>8.1f} GB total")
    print(f"\nRecommend free disk: ~{int(total_gb * 2.5):d} GB "
          "(waveforms + working copies + picks)")


if __name__ == "__main__":
    main()
