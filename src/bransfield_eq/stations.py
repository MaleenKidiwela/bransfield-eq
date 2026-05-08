"""Resolve YAML targets into a concrete (data_center, net, sta) station list."""
from __future__ import annotations

from dataclasses import dataclass

from .config import Config, get_client


@dataclass
class StationRef:
    data_center: str
    network: str
    station: str
    label: str


def resolve_stations(cfg: Config) -> list[StationRef]:
    """Expand wildcard targets into individual stations using FDSN metadata."""
    out: list[StationRef] = []
    seen: set[tuple[str, str, str]] = set()

    for tgt in cfg.targets:
        client = get_client(tgt.data_center)
        kwargs = dict(
            network=tgt.network, station=tgt.station, level="station",
            channel=cfg.channels, starttime=cfg.start, endtime=cfg.end,
        )
        if tgt.bbox_filter:
            kwargs.update(cfg.bbox)
        try:
            inv = client.get_stations(**kwargs)
        except Exception as e:
            print(f"  [{tgt.data_center}] {tgt.network}.{tgt.station}: {e}")
            continue
        for net in inv:
            for sta in net:
                key = (tgt.data_center, net.code, sta.code)
                if key in seen:
                    continue
                seen.add(key)
                out.append(StationRef(tgt.data_center, net.code, sta.code, tgt.label))
    return out
