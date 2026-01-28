"""Spectral analysis functions for frequency domain operations.

This module provides functions for computing FFT, amplitude spectrum,
phase spectrum, power spectral density, and band power analysis.
"""

import numpy as np
from numpy.typing import NDArray
from scipy.fft import fft, fftfreq
from scipy.signal import welch


def compute_fft(
    signal: NDArray[np.floating],
    fs: float,
) -> tuple[NDArray[np.floating], NDArray[np.complexfloating]]:
    """Compute the Fast Fourier Transform of a signal.

    Parameters
    ----------
    signal : NDArray[np.floating]
        Input signal in the time domain.
    fs : float
        Sampling frequency in Hz.

    Returns
    -------
    frequencies : NDArray[np.floating]
        Array of frequency values in Hz.
    fft_values : NDArray[np.complexfloating]
        Complex FFT coefficients.

    Examples
    --------
    >>> t = np.linspace(0, 1, 1000)
    >>> signal = np.sin(2 * np.pi * 10 * t)
    >>> frequencies, fft_values = compute_fft(signal, fs=1000)
    """
    n_samples = len(signal)
    fft_values = fft(signal)
    frequencies = fftfreq(n_samples, 1 / fs)
    return frequencies, fft_values


def compute_amplitude_spectrum(
    signal: NDArray[np.floating],
    fs: float,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Compute the amplitude spectrum (positive frequencies only).

    The amplitude is properly scaled to recover the original signal amplitudes.

    Parameters
    ----------
    signal : NDArray[np.floating]
        Input signal in the time domain.
    fs : float
        Sampling frequency in Hz.

    Returns
    -------
    frequencies : NDArray[np.floating]
        Array of positive frequency values in Hz.
    amplitude : NDArray[np.floating]
        Amplitude at each frequency (properly scaled).

    Examples
    --------
    >>> t = np.linspace(0, 1, 1000)
    >>> signal = 2.0 * np.sin(2 * np.pi * 10 * t)  # Amplitude = 2
    >>> freqs, amps = compute_amplitude_spectrum(signal, fs=1000)
    >>> # Peak at 10 Hz should be approximately 2.0
    """
    n_samples = len(signal)
    frequencies, fft_values = compute_fft(signal, fs)

    # Keep only positive frequencies
    positive_mask = frequencies >= 0
    frequencies_pos = frequencies[positive_mask]
    fft_pos = fft_values[positive_mask]

    # Scale: divide by N and multiply by 2 (except DC)
    amplitude = np.abs(fft_pos) * 2 / n_samples
    amplitude[0] /= 2  # DC component shouldn't be doubled

    return frequencies_pos, amplitude


def compute_phase_spectrum(
    signal: NDArray[np.floating],
    fs: float,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Compute the phase spectrum (positive frequencies only).

    Parameters
    ----------
    signal : NDArray[np.floating]
        Input signal in the time domain.
    fs : float
        Sampling frequency in Hz.

    Returns
    -------
    frequencies : NDArray[np.floating]
        Array of positive frequency values in Hz.
    phase : NDArray[np.floating]
        Phase angle at each frequency in radians (-π to π).

    Examples
    --------
    >>> t = np.linspace(0, 1, 1000)
    >>> signal = np.sin(2 * np.pi * 10 * t + np.pi/4)  # Phase = π/4
    >>> freqs, phases = compute_phase_spectrum(signal, fs=1000)
    """
    frequencies, fft_values = compute_fft(signal, fs)

    # Keep only positive frequencies
    positive_mask = frequencies >= 0
    frequencies_pos = frequencies[positive_mask]
    fft_pos = fft_values[positive_mask]

    # Extract phase
    phase = np.angle(fft_pos)

    return frequencies_pos, phase


def compute_frequency_resolution(fs: float, n_samples: int) -> float:
    """Compute the frequency resolution of an FFT.

    The frequency resolution (Δf) determines the spacing between
    frequency bins and the ability to distinguish nearby frequencies.

    Parameters
    ----------
    fs : float
        Sampling frequency in Hz.
    n_samples : int
        Number of samples in the signal.

    Returns
    -------
    float
        Frequency resolution in Hz (Δf = fs / N).

    Notes
    -----
    To resolve two frequencies that are Δf apart, you need at least
    1/Δf seconds of data. For example, to resolve 1 Hz difference,
    you need at least 1 second of data.

    Examples
    --------
    >>> compute_frequency_resolution(fs=1000, n_samples=1000)
    1.0
    >>> compute_frequency_resolution(fs=1000, n_samples=2000)
    0.5
    """
    return fs / n_samples


# =============================================================================
# Power Spectral Density Functions (A03)
# =============================================================================


def compute_psd_fft(
    signal: NDArray[np.floating],
    fs: float,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Compute Power Spectral Density using the periodogram method (direct FFT).

    This is a simple but high-variance estimator. For more robust estimation,
    use compute_psd_welch() instead.

    Parameters
    ----------
    signal : NDArray[np.floating]
        Input signal in the time domain.
    fs : float
        Sampling frequency in Hz.

    Returns
    -------
    frequencies : NDArray[np.floating]
        Array of positive frequency values in Hz.
    psd : NDArray[np.floating]
        Power spectral density values in V²/Hz.

    Examples
    --------
    >>> t = np.linspace(0, 1, 1000, endpoint=False)
    >>> signal = np.sin(2 * np.pi * 10 * t)
    >>> freqs, psd = compute_psd_fft(signal, fs=1000)
    """
    n_samples = len(signal)
    frequencies, fft_values = compute_fft(signal, fs)

    # Keep only positive frequencies
    positive_mask = frequencies >= 0
    frequencies_pos = frequencies[positive_mask]
    fft_pos = fft_values[positive_mask]

    # Compute PSD: |X(f)|² / (fs * N), multiply by 2 for one-sided
    psd = (np.abs(fft_pos) ** 2) / (fs * n_samples)
    psd[1:] *= 2  # Double for one-sided (except DC)

    return frequencies_pos, psd


def compute_psd_welch(
    signal: NDArray[np.floating],
    fs: float,
    nperseg: int | None = None,
    noverlap: int | None = None,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Compute Power Spectral Density using Welch's method.

    Welch's method reduces variance by averaging periodograms of overlapping
    segments, at the cost of some frequency resolution.

    Parameters
    ----------
    signal : NDArray[np.floating]
        Input signal in the time domain.
    fs : float
        Sampling frequency in Hz.
    nperseg : int | None, optional
        Length of each segment. If None, defaults to 256.
    noverlap : int | None, optional
        Number of points to overlap between segments.
        If None, defaults to nperseg // 2 (50% overlap).

    Returns
    -------
    frequencies : NDArray[np.floating]
        Array of frequency values in Hz.
    psd : NDArray[np.floating]
        Power spectral density values in V²/Hz.

    Examples
    --------
    >>> signal = np.random.randn(10000)
    >>> freqs, psd = compute_psd_welch(signal, fs=1000, nperseg=256)
    """
    if nperseg is None:
        nperseg = min(256, len(signal))
    if noverlap is None:
        noverlap = nperseg // 2

    frequencies, psd = welch(signal, fs=fs, nperseg=nperseg, noverlap=noverlap)

    return frequencies, psd


# =============================================================================
# Band Power Functions (A03)
# =============================================================================


def compute_band_power(
    psd: NDArray[np.floating],
    freqs: NDArray[np.floating],
    freq_range: tuple[float, float],
) -> float:
    """Compute the power in a specific frequency band using trapezoidal integration.

    Parameters
    ----------
    psd : NDArray[np.floating]
        Power spectral density values.
    freqs : NDArray[np.floating]
        Frequency values corresponding to PSD.
    freq_range : tuple[float, float]
        Tuple of (low_freq, high_freq) defining the band.

    Returns
    -------
    float
        Total power in the specified frequency band.

    Examples
    --------
    >>> freqs = np.array([0, 1, 2, 3, 4, 5])
    >>> psd = np.array([1, 1, 1, 1, 1, 1])
    >>> compute_band_power(psd, freqs, (1, 4))
    3.0
    """
    f_low, f_high = freq_range

    # Find indices within the frequency range
    band_mask = (freqs >= f_low) & (freqs <= f_high)
    freqs_band = freqs[band_mask]
    psd_band = psd[band_mask]

    if len(freqs_band) < 2:
        return 0.0

    # Trapezoidal integration
    band_power = np.trapz(psd_band, freqs_band)

    return float(band_power)


def compute_all_band_powers(
    psd: NDArray[np.floating],
    freqs: NDArray[np.floating],
    bands: dict[str, tuple[float, float]] | None = None,
) -> dict[str, float]:
    """Compute absolute power for all frequency bands.

    Parameters
    ----------
    psd : NDArray[np.floating]
        Power spectral density values.
    freqs : NDArray[np.floating]
        Frequency values corresponding to PSD.
    bands : dict[str, tuple[float, float]] | None, optional
        Dictionary mapping band names to (low_freq, high_freq) tuples.
        If None, uses standard EEG bands.

    Returns
    -------
    dict[str, float]
        Dictionary mapping band names to their absolute power values.

    Examples
    --------
    >>> freqs, psd = compute_psd_welch(signal, fs=256)
    >>> powers = compute_all_band_powers(psd, freqs)
    >>> print(powers["alpha"])
    """
    if bands is None:
        bands = {
            "delta": (1.0, 4.0),
            "theta": (4.0, 8.0),
            "alpha": (8.0, 13.0),
            "beta": (13.0, 30.0),
            "gamma": (30.0, 100.0),
        }

    band_powers = {}
    for band_name, freq_range in bands.items():
        band_powers[band_name] = compute_band_power(psd, freqs, freq_range)

    return band_powers


def compute_relative_band_power(
    psd: NDArray[np.floating],
    freqs: NDArray[np.floating],
    freq_range: tuple[float, float],
    total_range: tuple[float, float] = (1.0, 100.0),
) -> float:
    """Compute the relative power of a frequency band as a percentage of total power.

    Parameters
    ----------
    psd : NDArray[np.floating]
        Power spectral density values.
    freqs : NDArray[np.floating]
        Frequency values corresponding to PSD.
    freq_range : tuple[float, float]
        Tuple of (low_freq, high_freq) defining the band of interest.
    total_range : tuple[float, float], optional
        Frequency range for computing total power. Default is (1, 100) Hz.

    Returns
    -------
    float
        Relative power as a percentage (0-100).

    Examples
    --------
    >>> freqs, psd = compute_psd_welch(signal, fs=256)
    >>> alpha_relative = compute_relative_band_power(psd, freqs, (8, 13))
    >>> print(f"Alpha: {alpha_relative:.1f}%")
    """
    band_power = compute_band_power(psd, freqs, freq_range)
    total_power = compute_band_power(psd, freqs, total_range)

    if total_power == 0:
        return 0.0

    return 100.0 * band_power / total_power


def power_to_db(
    power: NDArray[np.floating],
    ref: float | None = None,
    min_db: float = -100.0,
) -> NDArray[np.floating]:
    """Convert power values to decibels (dB).

    Parameters
    ----------
    power : NDArray[np.floating]
        Power values to convert.
    ref : float | None, optional
        Reference power value. If None, uses the maximum power value.
        Common choices: 1.0 (absolute), max(power) (relative to peak).
    min_db : float, optional
        Minimum dB value to return (clips very small values). Default is -100.

    Returns
    -------
    NDArray[np.floating]
        Power values in decibels.

    Examples
    --------
    >>> power = np.array([1, 10, 100, 1000])
    >>> power_to_db(power, ref=1.0)
    array([ 0., 10., 20., 30.])

    Notes
    -----
    - Uses 10*log10 (power ratio), not 20*log10 (amplitude ratio)
    - Zero or negative power values are clipped to min_db
    """
    power = np.asarray(power, dtype=np.float64)

    if ref is None:
        ref = np.max(power)

    # Avoid log of zero or negative values
    power_safe = np.maximum(power, np.finfo(float).tiny)

    db_values = 10.0 * np.log10(power_safe / ref)

    # Clip to minimum dB
    db_values = np.maximum(db_values, min_db)

    return db_values
