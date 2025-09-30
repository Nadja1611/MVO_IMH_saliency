from __future__ import annotations

import numpy as np
import scipy.interpolate

def extract_window(x: np.ndarray, idx: int, nbefore: int, nafter: int) -> np.ndarray:
    """Extract window from signal around a single peak location.

    The signal is zero-padded if needed. To extract waveforms for multiple peaks at
    once, use the `extract_waveforms` function.

    Args:
        x: n-dimensional input. First axis must be time.
        idx: index around which to extract the window
        nbefore: number of samples before the peak
        nafter: number of samples after the peak

    Returns:
        extracted window
    """
    lpad = max(0, nbefore - idx)
    rpad = max(0, idx + nafter - len(x))
    snip = x[idx - nbefore + lpad : idx + nafter - rpad]
    return np.r_[np.zeros_like(x[:lpad]), snip, np.zeros_like(x[:rpad])]


def extract_waveform(
    x: np.ndarray,
    idx: int,
    fs: float,
    before: float,
    after: float | None = None,
) -> np.ndarray:
    """Extract waveform from signal around peak locations.

    Convenience function working with sample rate and time units instead of samples.
    For bulk extraction, use the `extract_waveforms` function.
    """
    if after is None:
        after = before
    nbefore = int(before * fs)
    nafter = int(after * fs)
    return extract_window(x, idx, nbefore, nafter)


def extract_waveforms(
    x: np.ndarray,
    indices: Sequence | np.ndarray,
    fs: float,
    before: float,
    after: float | None = None,
) -> np.ndarray:
    """Extract waveforms from signal around peak locations.

    Args:
        x: n-dimensional input. First axis must be time.
        indices: array of peak indices
        fs: sample rate of the input signal
        before: included time before the peak (in seconds)
        after: included time after the peak (in seconds)

    Returns:
        array of extracted waveforms NxTxC (N: number of peaks, T: time, C: channels)
    """
    return np.array([extract_waveform(x, idx, fs, before, after) for idx in indices])


def extract_normalized_waveforms(
    x, rpeaks, newsize: int = 400, tb=0.4, ta=0.60
) -> tuple[np.ndarray, np.ndarray]:
    """Extract time-normalized (based on IBI) waveforms.

    Args:
        x: n-dimensional input signal. First axis must be time.
        rpeaks: array of peak indices
        newsize (optional): resample waves to common size. Defaults to 400.
        tb (optional): proportion of IBI before each peak. Defaults to 0.4.
        ta (float, optional): proportion of IBI after each peak. Defaults to 0.60.

    Returns:
        Normalized beats (beats x time x channels) and time vector
    """
    ibi = np.diff(rpeaks).mean()
    before, after = ibi * tb, ibi * ta
    waves = extract_waveforms(x, rpeaks, 1, before, after)

    # resample to uniform length, to avoid jagged arrays later on.
    f = scipy.interpolate.interp1d(np.linspace(0, 1, waves.shape[1]), waves, axis=1)
    waves = f(np.linspace(0, 1, newsize))

    return waves, np.linspace(-before, after, newsize)