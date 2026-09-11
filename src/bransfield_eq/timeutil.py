"""Unit-safe epoch conversion.

Why this module exists: `pd.Series.astype("int64")` on a datetime column returns
the integer in the column's *stored* resolution, not nanoseconds. pandas 2
defaulted `to_datetime` to `datetime64[ns]`; pandas 3 (in this image since
2026-06-29) defaults to `datetime64[us]`. Every `.astype("int64") / 1e9` in the
pipeline therefore became 1000x too small — silently, with no error — which
collapsed a year of pick times into ~9 hours of apparent time.

Use these helpers instead of `.astype("int64")` anywhere a datetime becomes a
number. They pin the unit explicitly and behave identically on both versions.
"""
from __future__ import annotations

import pandas as pd


def epoch_ns(s) -> pd.Series:
    """Datetime-like Series (or anything to_datetime accepts) -> int64 ns since epoch."""
    t = pd.to_datetime(s, utc=True, format="ISO8601"
                       ) if _is_stringy(s) else pd.to_datetime(s, utc=True)
    return t.dt.as_unit("ns").astype("int64")


def epoch_seconds(s) -> pd.Series:
    """Datetime-like Series -> float64 seconds since epoch."""
    return epoch_ns(s) / 1e9


def epoch_ms(s) -> pd.Series:
    """Datetime-like Series -> int64 milliseconds since epoch (for dedup keys)."""
    return epoch_ns(s) // 10**6


def _is_stringy(s) -> bool:
    try:
        return pd.api.types.is_string_dtype(s) or pd.api.types.is_object_dtype(s)
    except Exception:
        return False


def assert_nanosecond_sanity() -> None:
    """Preflight guard: fail loudly if epoch conversion is off by a unit factor.

    Cheap enough to call at the top of any stage that converts pick times.
    """
    t = pd.Series(pd.to_datetime(["2019-07-11T00:00:00Z", "2019-07-11T00:00:01Z"], utc=True))
    dt = epoch_seconds(t).diff().iloc[1]
    if abs(dt - 1.0) > 1e-6:
        raise RuntimeError(
            f"epoch conversion is broken: 1 s apart measured as {dt} s. "
            f"pandas={pd.__version__}"
        )
