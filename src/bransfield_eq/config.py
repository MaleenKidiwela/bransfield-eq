"""Shared config + path helpers."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import yaml
from obspy import UTCDateTime
from obspy.clients.fdsn import Client

REPO = Path(__file__).resolve().parents[2]
DATA_DIR = REPO / "data"
WAVE_DIR = DATA_DIR / "waveforms"
XML_DIR = DATA_DIR / "stationxml"
PICK_DIR = REPO / "catalogs" / "picks"

RASPISHAKE_URL = "https://data.raspberryshake.org"


@dataclass
class Target:
    data_center: str
    network: str
    station: str
    label: str
    bbox_filter: bool


@dataclass
class Config:
    start: UTCDateTime
    end: UTCDateTime
    channels: str          # download channels (seismic + hydrophone)
    picking_channels: str  # PhaseNet input channels (seismic only)
    bbox: dict
    targets: list[Target]

    @classmethod
    def load(cls, path: Path | str | None = None) -> "Config":
        path = Path(path) if path else REPO / "configs" / "targets.yaml"
        raw = yaml.safe_load(path.read_text())
        return cls(
            start=UTCDateTime(raw["window"]["start"]),
            end=UTCDateTime(raw["window"]["end"]),
            channels=raw["channels"],
            picking_channels=raw.get("picking_channels", raw["channels"]),
            bbox=raw["bbox"],
            targets=[Target(**t) for t in raw["targets"]],
        )


def get_client(data_center: str) -> Client:
    if data_center == "RASPISHAKE":
        return Client(RASPISHAKE_URL)
    return Client(data_center)


def daterange(start: UTCDateTime, end: UTCDateTime) -> Iterator[UTCDateTime]:
    """Yield UTC midnights from start (inclusive) to end (exclusive)."""
    t = UTCDateTime(start.date)
    while t < end:
        yield t
        t += 86400


def mseed_path(network: str, station: str, day: UTCDateTime) -> Path:
    """Per-station-day MSEED location: data/waveforms/NET/STA/NET.STA.YYYY.JJJ.mseed"""
    return (WAVE_DIR / network / station /
            f"{network}.{station}.{day.year}.{day.julday:03d}.mseed")


def pick_csv_path(network: str, station: str, day: UTCDateTime) -> Path:
    return (PICK_DIR / f"{network}.{station}" /
            f"{day.year}-{day.julday:03d}.csv")
