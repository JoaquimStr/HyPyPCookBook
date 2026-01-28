"""
Amplitude envelope extraction and analysis functions.

This module provides utilities for extracting, smoothing, and analyzing
amplitude envelopes from neural signals, commonly used in hyperscanning
and connectivity studies.
"""

from typing import Any, Dict, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, filtfilt, hilbert, welch


def extract_envelope(
    signal: NDArray[np.floating[Any]],
    fs: float,
    band: Tuple[float, float],
    filter_order: int = 4
) -> NDArray[np.floating[Any]]:
    """
    Extract amplitude envelope from a signal in a specific frequency band.

    Applies bandpass filtering, Hilbert transform, and takes the magnitude
    to obtain the instantaneous amplitude envelope.

    Parameters
    ----------
    signal : NDArray[np.floating[Any]]
        Input signal (1D array).
    fs : float
        Sampling frequency in Hz.
    band : Tuple[float, float]
        Frequency band as (low_freq, high_freq) in Hz.
    filter_order : int, optional
        Order of the Butterworth bandpass filter. Default is 4.

    Returns
    -------
    NDArray[np.floating[Any]]
        Amplitude envelope of the filtered signal.

    Examples
    --------
    >>> import numpy as np
    >>> fs = 250
    >>> t = np.arange(0, 2, 1/fs)
    >>> signal = np.sin(2*np.pi*10*t)  # 10 Hz signal
    >>> envelope = extract_envelope(signal, fs, (8, 13))  # Alpha band
    >>> envelope.shape == signal.shape
    True
    """
    # Bandpass filter
    nyq = fs / 2
    low = band[0] / nyq
    high = band[1] / nyq
    b, a = butter(filter_order, [low, high], btype='band')
    filtered = filtfilt(b, a, signal)

    # Hilbert transform and envelope
    analytic = hilbert(filtered)
    envelope = np.abs(analytic)

    return envelope


def compute_envelope_psd(
    envelope: NDArray[np.floating[Any]],
    fs: float,
    nperseg: int | None = None
) -> Tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
    """
    Compute power spectral density of an amplitude envelope.

    Uses Welch's method to estimate the PSD, revealing the temporal
    dynamics of envelope fluctuations.

    Parameters
    ----------
    envelope : NDArray[np.floating[Any]]
        Amplitude envelope signal.
    fs : float
        Sampling frequency in Hz.
    nperseg : int | None, optional
        Length of each segment for Welch's method. Default is fs*2.

    Returns
    -------
    freqs : NDArray[np.floating[Any]]
        Frequency values in Hz.
    psd : NDArray[np.floating[Any]]
        Power spectral density values.

    Examples
    --------
    >>> import numpy as np
    >>> fs = 250
    >>> t = np.arange(0, 10, 1/fs)
    >>> envelope = 1 + 0.5*np.sin(2*np.pi*0.5*t)  # 0.5 Hz modulation
    >>> freqs, psd = compute_envelope_psd(envelope, fs)
    >>> freqs[np.argmax(psd)]  # Should be near 0.5 Hz
    """
    if nperseg is None:
        nperseg = int(fs * 2)

    freqs, psd = welch(envelope, fs=fs, nperseg=nperseg)

    return freqs, psd


def smooth_envelope_moving_average(
    envelope: NDArray[np.floating[Any]],
    window_samples: int
) -> NDArray[np.floating[Any]]:
    """
    Smooth envelope using a simple moving average filter.

    Parameters
    ----------
    envelope : NDArray[np.floating[Any]]
        Input amplitude envelope.
    window_samples : int
        Size of the moving average window in samples.

    Returns
    -------
    NDArray[np.floating[Any]]
        Smoothed envelope (same length as input).

    Notes
    -----
    Uses 'same' mode convolution, which may introduce edge effects.
    Consider using Gaussian smoothing for smoother results.

    Examples
    --------
    >>> import numpy as np
    >>> envelope = np.array([1, 2, 3, 4, 5, 4, 3, 2, 1])
    >>> smoothed = smooth_envelope_moving_average(envelope, 3)
    >>> len(smoothed) == len(envelope)
    True
    """
    kernel = np.ones(window_samples) / window_samples
    smoothed = np.convolve(envelope, kernel, mode='same')
    return smoothed


def smooth_envelope_gaussian(
    envelope: NDArray[np.floating[Any]],
    sigma_samples: int
) -> NDArray[np.floating[Any]]:
    """
    Smooth envelope using a Gaussian filter.

    Parameters
    ----------
    envelope : NDArray[np.floating[Any]]
        Input amplitude envelope.
    sigma_samples : int
        Standard deviation of Gaussian kernel in samples.
        To convert from milliseconds: sigma_samples = sigma_ms * fs / 1000

    Returns
    -------
    NDArray[np.floating[Any]]
        Gaussian-smoothed envelope.

    Notes
    -----
    Gaussian smoothing provides better frequency characteristics than
    moving average and produces smoother output without sharp transitions.

    Examples
    --------
    >>> import numpy as np
    >>> envelope = np.random.randn(1000) + 5  # Noisy envelope
    >>> smoothed = smooth_envelope_gaussian(envelope, sigma_samples=25)
    >>> np.std(smoothed) < np.std(envelope)  # Should be smoother
    True
    """
    smoothed = gaussian_filter1d(envelope, sigma=sigma_samples)
    return smoothed


def smooth_envelope_lowpass(
    envelope: NDArray[np.floating[Any]],
    fs: float,
    cutoff: float = 2.0,
    order: int = 4
) -> NDArray[np.floating[Any]]:
    """
    Smooth envelope using a low-pass Butterworth filter.

    Parameters
    ----------
    envelope : NDArray[np.floating[Any]]
        Input amplitude envelope.
    fs : float
        Sampling frequency in Hz.
    cutoff : float, optional
        Cutoff frequency in Hz. Default is 2.0 Hz.
    order : int, optional
        Filter order. Default is 4.

    Returns
    -------
    NDArray[np.floating[Any]]
        Low-pass filtered envelope.

    Notes
    -----
    Low-pass filtering provides a principled way to remove high-frequency
    noise from envelopes while preserving physiologically relevant dynamics.
    Typical envelope dynamics are below 2-3 Hz for most cognitive processes.

    Examples
    --------
    >>> import numpy as np
    >>> fs = 250
    >>> envelope = np.random.randn(fs*10) + 5  # 10 seconds
    >>> smoothed = smooth_envelope_lowpass(envelope, fs, cutoff=1.0)
    """
    nyq = fs / 2
    normalized_cutoff = cutoff / nyq
    b, a = butter(order, normalized_cutoff, btype='low')
    smoothed = filtfilt(b, a, envelope)
    return smoothed


def compute_envelope_correlation(
    signal1: NDArray[np.floating[Any]],
    signal2: NDArray[np.floating[Any]],
    fs: float,
    band: Tuple[float, float],
    filter_order: int = 4
) -> float:
    """
    Compute Pearson correlation between amplitude envelopes of two signals.

    Extracts envelopes from both signals in the specified frequency band,
    then computes their correlation coefficient.

    Parameters
    ----------
    signal1 : NDArray[np.floating[Any]]
        First input signal.
    signal2 : NDArray[np.floating[Any]]
        Second input signal (same length as signal1).
    fs : float
        Sampling frequency in Hz.
    band : Tuple[float, float]
        Frequency band as (low_freq, high_freq) in Hz.
    filter_order : int, optional
        Order of the Butterworth bandpass filter. Default is 4.

    Returns
    -------
    float
        Pearson correlation coefficient between envelopes (-1 to +1).

    Notes
    -----
    Envelope correlation is a measure of amplitude coupling between signals.
    High values indicate that the signals increase and decrease in power
    together, suggesting coordinated neural activity.

    Caution: Volume conduction can cause spurious envelope correlations
    between nearby electrodes.

    Examples
    --------
    >>> import numpy as np
    >>> fs = 250
    >>> t = np.arange(0, 5, 1/fs)
    >>> mod = 0.5 + 0.5*np.sin(2*np.pi*0.3*t)
    >>> s1 = mod * np.sin(2*np.pi*10*t)
    >>> s2 = mod * np.sin(2*np.pi*10*t + 0.5)  # Same modulation
    >>> corr = compute_envelope_correlation(s1, s2, fs, (8, 13))
    >>> corr > 0.9  # Should be highly correlated
    True
    """
    env1 = extract_envelope(signal1, fs, band, filter_order)
    env2 = extract_envelope(signal2, fs, band, filter_order)

    correlation = float(np.corrcoef(env1, env2)[0, 1])

    return correlation


def compute_envelope_statistics(
    envelope: NDArray[np.floating[Any]]
) -> Dict[str, float]:
    """
    Compute descriptive statistics of an amplitude envelope.

    Parameters
    ----------
    envelope : NDArray[np.floating[Any]]
        Input amplitude envelope.

    Returns
    -------
    Dict[str, float]
        Dictionary containing:
        - 'mean': Mean amplitude
        - 'std': Standard deviation
        - 'cv': Coefficient of variation (std/mean)
        - 'median': Median amplitude
        - 'p25': 25th percentile
        - 'p75': 75th percentile
        - 'iqr': Interquartile range (p75 - p25)

    Notes
    -----
    The coefficient of variation (CV) is useful for comparing envelope
    variability across conditions or participants, as it normalizes by
    the mean amplitude.

    Examples
    --------
    >>> import numpy as np
    >>> envelope = np.abs(np.random.randn(1000)) + 1
    >>> stats = compute_envelope_statistics(envelope)
    >>> 'mean' in stats and 'cv' in stats
    True
    """
    mean_val = float(np.mean(envelope))
    std_val = float(np.std(envelope))
    cv = std_val / mean_val if mean_val > 0 else 0.0
    median_val = float(np.median(envelope))
    p25 = float(np.percentile(envelope, 25))
    p75 = float(np.percentile(envelope, 75))

    return {
        'mean': mean_val,
        'std': std_val,
        'cv': cv,
        'median': median_val,
        'p25': p25,
        'p75': p75,
        'iqr': p75 - p25
    }
