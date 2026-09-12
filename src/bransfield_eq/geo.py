"""Fixed map projection for the Bransfield catalogue.

Why this exists
---------------
`17_pyocto_associate.py` used to build its transverse-Mercator origin from
``stations.latitude.mean()`` over *only the stations that had picks in that
run*. Two consequences, both silent:

1. **The origin moved between runs.** Different pick sets select different
   station subsets, which shifts the mean. Measured 2026-09-12: the two
   validation months differed by 2 stations (35 vs 37) and their bounding
   boxes were offset ~5 km; two different pick pools on the same day differed
   by **19.9 km**, so a naive x/y comparison matched *zero* of 117 events.
2. **The origin was never written down.** The event catalogue schema was
   ``idx,time,x,y,z,picks`` -- projected kilometres with no record of what
   they were measured from, and no lat/lon. The coordinates could not be
   interpreted, or corrected, without re-deriving which stations happened to
   have picks in that exact run.

The danger case is a chunked full-year run: OBS instruments come and go over a
14-month deployment, so chunks would land on different origins and the year
catalogue would be stitched from mismatched frames -- no error, events simply
a few km out of register with each other, corrupting every downstream stage
that reasons about inter-event distance (XC pair selection, ph2dt neighbour
search, hypoDD clustering).

The fix: one frozen origin, defined here, imported everywhere. Never derived
from data.

Freezing the origin
-------------------
ORIGIN_LAT/ORIGIN_LON are **constants and must never be recomputed**, not even
from the full station list -- ``catalogs/station_geometry.csv`` can change, and
a "recompute from all stations" rule would silently reintroduce the same bug.
They are deliberately round numbers near the centre of the 38-station network
(full-set mean is -62.5274, -58.8027) so it is obvious they are chosen, not
derived. Changing them invalidates every x/y ever written; write lat/lon and
re-project instead.
"""
from __future__ import annotations

import numpy as np

# Frozen 2026-09-12. Do not recompute. See module docstring.
ORIGIN_LAT = -62.5
ORIGIN_LON = -58.8
PROJ4 = f"+proj=tmerc +lat_0={ORIGIN_LAT} +lon_0={ORIGIN_LON} +ellps=WGS84"

_FWD = None   # lat/lon -> x/y
_INV = None   # x/y -> lat/lon


def _transformers():
    global _FWD, _INV
    if _FWD is None:
        from pyproj import CRS, Transformer
        crs = CRS.from_proj4(PROJ4)
        _FWD = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        _INV = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    return _FWD, _INV


def make_crs():
    """The projected CRS itself, for callers that need to hand it to a library."""
    from pyproj import CRS
    return CRS.from_proj4(PROJ4)


def to_xy(lat, lon):
    """(lat, lon) degrees -> (x, y) kilometres from the frozen origin."""
    fwd, _ = _transformers()
    x, y = fwd.transform(np.asarray(lon, dtype=float), np.asarray(lat, dtype=float))
    return np.asarray(x) / 1e3, np.asarray(y) / 1e3


def to_latlon(x_km, y_km):
    """(x, y) kilometres from the frozen origin -> (lat, lon) degrees."""
    _, inv = _transformers()
    lon, lat = inv.transform(np.asarray(x_km, dtype=float) * 1e3,
                             np.asarray(y_km, dtype=float) * 1e3)
    return np.asarray(lat), np.asarray(lon)


def origin_metadata() -> dict:
    """Stamp for writing next to any file that contains projected coordinates."""
    return {"origin_lat": ORIGIN_LAT, "origin_lon": ORIGIN_LON,
            "proj4": PROJ4, "units": "km",
            "note": "frozen origin; x/y are km from it. See bransfield_eq.geo"}


def assert_roundtrip(tol_m: float = 1.0) -> None:
    """Fail loudly if projection round-trip drifts. Cheap; call at startup."""
    lat = np.array([-63.5, -62.5, -62.0]); lon = np.array([-61.0, -58.8, -57.4])
    x, y = to_xy(lat, lon)
    lat2, lon2 = to_latlon(x, y)
    dlat = np.abs(lat2 - lat).max() * 111_000
    dlon = np.abs(lon2 - lon).max() * 111_000 * np.cos(np.radians(-62.5))
    if max(dlat, dlon) > tol_m:
        raise AssertionError(f"projection round-trip off by {max(dlat, dlon):.3f} m")
