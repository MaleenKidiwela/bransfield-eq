"""Zero-phase bandpass applied before waveform cross-correlation.

Why this exists
---------------
Raw Bransfield OBS records are swell-dominated. Measured over 19,912 pick
windows from ``growclust/picker_only/pick_windows.npy`` (pre-pick half vs
post-pick half, so the ratio is a true SNR):

    below 5 Hz:  27.5% of signal power but 64.4% of the NOISE power
    SNR peaks near 6 Hz at +10 dB; SNR > 3 dB only above ~3.3 Hz

Correlating unfiltered data therefore correlates mostly swell, which makes the
CC peak nearly flat.  ``xcorr_with_lag`` normalises each window to unit L2 norm
*once* over the full window and then calls ``correlate(..., mode="full")``, so
the effective normalisation at lag L is short by a factor ~(n-|L|)/n -- a
triangular taper.  Against a flat peak that taper wins and drags the apparent
maximum toward lag 0, biasing differential times toward zero.

Measured leakage (same data displaced by a known lag, so all error is leakage;
2,989 windows x 4 offsets of +-10..20 ms):

    band          RMS err    %err>5ms   mean cc
    unfiltered     4.58 ms     10.2%     0.993
    2-15 Hz        0.53 ms      0.2%     0.992
    3-20 Hz        0.34 ms      0.1%     0.993
    4-25 Hz        0.25 ms      0.0%     0.993

The bad measurements come back with cc ~ 0.99, so neither ``CC_THRESH`` nor
GrowClust's ``rmin`` screens them out -- they are silent.

DEFAULT_BAND is 3-20 Hz: within 0.1 ms of the best band tested, while keeping a
low corner that still passes the longer-period energy of the larger events.
Filtering a whole station-day at once (rather than per window) costs one filter
pass per day and leaves no transients inside any window.
"""
from __future__ import annotations

import numpy as np

DEFAULT_BAND = (3.0, 20.0)
DEFAULT_ORDER = 4


def make_sos(lo: float, hi: float, fs: float, order: int = DEFAULT_ORDER):
    """Butterworth bandpass as second-order sections. Corners are clamped into
    (0, nyquist) so a caller cannot silently build an invalid filter."""
    from scipy.signal import butter

    nyq = fs / 2.0
    lo_n = max(lo / nyq, 1e-6)
    hi_n = min(hi / nyq, 0.999999)
    if not (0.0 < lo_n < hi_n < 1.0):
        raise ValueError(f"invalid band {lo}-{hi} Hz at fs={fs} Hz")
    return butter(order, [lo_n, hi_n], btype="band", output="sos")


def bandpass(data: np.ndarray, lo: float, hi: float, fs: float,
             order: int = DEFAULT_ORDER, sos=None) -> np.ndarray:
    """Zero-phase bandpass a 1-D array, returned as float32.

    Zero-phase (``sosfiltfilt``) so no timing shift is introduced -- critical
    here, since the whole point is differential-time accuracy.

    Returns the input unchanged if it is too short for the filter's padding or
    if it is not finite; callers treat waveform loading as best-effort and a
    short/degenerate day should not raise.
    """
    if data is None or data.size == 0:
        return data
    from scipy.signal import sosfiltfilt

    if sos is None:
        sos = make_sos(lo, hi, fs, order)
    # sosfiltfilt needs more samples than its default padlen (3 * n_sections * 2).
    if data.size <= 6 * sos.shape[0] + 1:
        return np.asarray(data, dtype=np.float32)
    if not np.isfinite(data).all():
        return np.asarray(data, dtype=np.float32)
    out = sosfiltfilt(sos, np.asarray(data, dtype=np.float64))
    return np.ascontiguousarray(out, dtype=np.float32)


def add_cli(ap):
    """Attach --bandpass / --no-bandpass to an ArgumentParser."""
    lo, hi = DEFAULT_BAND
    ap.add_argument("--bandpass", nargs=2, type=float, metavar=("LO", "HI"),
                    default=None,
                    help=f"XC bandpass corners in Hz (default {lo}-{hi})")
    ap.add_argument("--no-bandpass", action="store_true",
                    help="Correlate unfiltered data. Reproduces pre-2026-09-12 "
                         "runs; biases differential times toward zero.")
    return ap


def band_from_args(args):
    """Resolve (lo, hi) or None from parsed args."""
    if getattr(args, "no_bandpass", False):
        return None
    band = getattr(args, "bandpass", None)
    return tuple(band) if band else DEFAULT_BAND
