"""
Wavelet analysis functions for time-frequency decomposition.

This module provides functions for wavelet-based time-frequency analysis,
particularly using complex Morlet wavelets which are standard in EEG research.

Functions
---------
create_morlet_wavelet
    Create a complex Morlet wavelet at a given frequency.
wavelet_convolution
    Convolve a signal with a complex wavelet using FFT.
compute_wavelet_transform
    Compute the full wavelet transform (time-frequency representation).
compute_wavelet_power
    Compute time-frequency power from wavelet transform.
compute_wavelet_phase
    Extract instantaneous phase at multiple frequencies.
compute_adaptive_cycles
    Compute frequency-adaptive number of cycles.
compute_edge_samples
    Calculate samples affected by edge effects.
"""

from typing import Tuple, Union, Optional

import numpy as np
from numpy.typing import NDArray


def create_morlet_wavelet(
    frequency: float,
    fs: float,
    n_cycles: float = 5.0,
    return_time: bool = False
) -> Union[NDArray[np.complex128], Tuple[NDArray[np.complex128], NDArray[np.float64]]]:
    """
    Create a complex Morlet wavelet at a specified frequency.

    The Morlet wavelet is a complex sinusoid tapered by a Gaussian envelope.
    It is widely used in EEG analysis for time-frequency decomposition.

    Parameters
    ----------
    frequency : float
        Center frequency of the wavelet in Hz.
    fs : float
        Sampling frequency in Hz.
    n_cycles : float, optional
        Number of cycles in the Gaussian envelope. Controls the trade-off
        between time and frequency resolution. Default is 5.0.
    return_time : bool, optional
        If True, also return the time vector. Default is False.

    Returns
    -------
    wavelet : ndarray of complex128
        Complex Morlet wavelet.
    time : ndarray of float64, optional
        Time vector in seconds (only if return_time=True).

    Notes
    -----
    The wavelet is constructed as:

        w(t) = exp(2πift) · exp(-t²/(2σ²))

    where σ = n_cycles / (2πf) controls the Gaussian width.

    Examples
    --------
    >>> wavelet = create_morlet_wavelet(10.0, 256, n_cycles=5)
    >>> len(wavelet)
    128
    >>> wavelet, time = create_morlet_wavelet(10.0, 256, return_time=True)
    """
    # Gaussian width in seconds
    sigma_t = n_cycles / (2 * np.pi * frequency)

    # Time vector (±3 sigma covers >99% of Gaussian)
    duration = 6 * sigma_t
    n_samples = int(np.ceil(duration * fs))
    # Ensure odd number for symmetry
    if n_samples % 2 == 0:
        n_samples += 1

    time = np.arange(n_samples) / fs - duration / 2

    # Create complex Morlet wavelet
    # Gaussian envelope
    gaussian = np.exp(-time**2 / (2 * sigma_t**2))
    # Complex sinusoid
    sinusoid = np.exp(2j * np.pi * frequency * time)
    # Combine
    wavelet = gaussian * sinusoid

    # Normalize to unit energy
    wavelet = wavelet / np.sqrt(np.sum(np.abs(wavelet)**2))

    if return_time:
        return wavelet, time
    return wavelet


def wavelet_convolution(
    signal: NDArray[np.float64],
    wavelet: NDArray[np.complex128],
    mode: str = 'same'
) -> NDArray[np.complex128]:
    """
    Convolve a signal with a complex wavelet using FFT for efficiency.

    Parameters
    ----------
    signal : ndarray of float64
        Input signal (1D array).
    wavelet : ndarray of complex128
        Complex wavelet (e.g., Morlet wavelet).
    mode : str, optional
        Convolution mode. 'same' returns output with same length as signal.
        Default is 'same'.

    Returns
    -------
    result : ndarray of complex128
        Complex-valued convolution result. The magnitude gives power,
        and the angle gives instantaneous phase.

    Notes
    -----
    Uses FFT-based convolution (convolution theorem) for efficiency:

        conv(s, w) = ifft(fft(s) * fft(w))

    Examples
    --------
    >>> signal = np.sin(2 * np.pi * 10 * np.linspace(0, 1, 256))
    >>> wavelet = create_morlet_wavelet(10, 256, n_cycles=5)
    >>> result = wavelet_convolution(signal, wavelet)
    >>> power = np.abs(result) ** 2
    """
    # Determine FFT size (next power of 2 for efficiency)
    n_signal = len(signal)
    n_wavelet = len(wavelet)
    n_conv = n_signal + n_wavelet - 1
    n_fft = int(2 ** np.ceil(np.log2(n_conv)))

    # FFT of signal and wavelet
    signal_fft = np.fft.fft(signal, n=n_fft)
    wavelet_fft = np.fft.fft(wavelet, n=n_fft)

    # Multiply in frequency domain (convolution theorem)
    result_fft = signal_fft * wavelet_fft

    # Inverse FFT
    result = np.fft.ifft(result_fft)

    # Trim to match 'same' mode
    if mode == 'same':
        # Remove half the wavelet length from each end
        start = (n_wavelet - 1) // 2
        result = result[start:start + n_signal]

    return result


def compute_wavelet_transform(
    signal: NDArray[np.float64],
    frequencies: NDArray[np.float64],
    fs: float,
    n_cycles: Union[float, NDArray[np.float64]] = 5.0
) -> NDArray[np.complex128]:
    """
    Compute the wavelet transform of a signal at multiple frequencies.

    Parameters
    ----------
    signal : ndarray of float64
        Input signal (1D array).
    frequencies : ndarray of float64
        Array of frequencies to analyze (in Hz).
    fs : float
        Sampling frequency in Hz.
    n_cycles : float or ndarray, optional
        Number of cycles for the wavelets. Can be a single value
        or an array matching the length of frequencies. Default is 5.0.

    Returns
    -------
    tfr : ndarray of complex128
        Time-frequency representation, shape (n_frequencies, n_times).

    Notes
    -----
    For each frequency, creates a Morlet wavelet and convolves it with
    the signal. The result is complex-valued: magnitude gives power,
    angle gives phase.

    Examples
    --------
    >>> signal = np.random.randn(1000)
    >>> freqs = np.arange(5, 40, 1)
    >>> tfr = compute_wavelet_transform(signal, freqs, fs=256)
    >>> power = np.abs(tfr) ** 2
    """
    n_freqs = len(frequencies)
    n_times = len(signal)

    # Handle n_cycles as array
    if np.isscalar(n_cycles):
        n_cycles_arr = np.full(n_freqs, n_cycles)
    else:
        n_cycles_arr = np.asarray(n_cycles)

    # Initialize output
    tfr = np.zeros((n_freqs, n_times), dtype=np.complex128)

    # Compute wavelet transform for each frequency
    for i, (freq, nc) in enumerate(zip(frequencies, n_cycles_arr)):
        wavelet = create_morlet_wavelet(freq, fs, n_cycles=nc)
        tfr[i, :] = wavelet_convolution(signal, wavelet)

    return tfr


def compute_wavelet_power(
    signal: NDArray[np.float64],
    frequencies: NDArray[np.float64],
    fs: float,
    n_cycles: Union[float, NDArray[np.float64]] = 5.0,
    baseline: Optional[Tuple[float, float]] = None,
    baseline_mode: str = 'percent'
) -> NDArray[np.float64]:
    """
    Compute time-frequency power from wavelet transform.

    Parameters
    ----------
    signal : ndarray of float64
        Input signal (1D array).
    frequencies : ndarray of float64
        Array of frequencies to analyze (in Hz).
    fs : float
        Sampling frequency in Hz.
    n_cycles : float or ndarray, optional
        Number of cycles for the wavelets. Default is 5.0.
    baseline : tuple of (start, end), optional
        Baseline period in seconds for normalization.
    baseline_mode : str, optional
        Baseline normalization mode: 'percent', 'zscore', or 'ratio'.
        Default is 'percent'.

    Returns
    -------
    power : ndarray of float64
        Time-frequency power, shape (n_frequencies, n_times).

    Notes
    -----
    Power is computed as the squared magnitude of the wavelet transform:

        power = |W(t, f)|²

    Baseline normalization is applied if a baseline period is specified.

    Examples
    --------
    >>> signal = np.random.randn(1000)
    >>> freqs = np.arange(5, 40, 1)
    >>> power = compute_wavelet_power(signal, freqs, fs=256)
    """
    tfr = compute_wavelet_transform(signal, frequencies, fs, n_cycles)
    power = np.abs(tfr) ** 2

    # Apply baseline normalization
    if baseline is not None:
        times = np.arange(len(signal)) / fs
        baseline_mask = (times >= baseline[0]) & (times <= baseline[1])
        baseline_power = power[:, baseline_mask].mean(axis=1, keepdims=True)

        if baseline_mode == 'ratio':
            power = power / baseline_power
        elif baseline_mode == 'zscore':
            baseline_std = power[:, baseline_mask].std(axis=1, keepdims=True)
            power = (power - baseline_power) / baseline_std
        elif baseline_mode == 'percent':
            power = (power - baseline_power) / baseline_power * 100

    return power


def compute_wavelet_phase(
    signal: NDArray[np.float64],
    frequencies: NDArray[np.float64],
    fs: float,
    n_cycles: Union[float, NDArray[np.float64]] = 5.0
) -> NDArray[np.float64]:
    """
    Compute instantaneous phase at multiple frequencies using wavelets.

    Parameters
    ----------
    signal : ndarray of float64
        Input signal (1D array).
    frequencies : ndarray of float64
        Array of frequencies to analyze (in Hz).
    fs : float
        Sampling frequency in Hz.
    n_cycles : float or ndarray, optional
        Number of cycles for the wavelets. Default is 5.0.

    Returns
    -------
    phase : ndarray of float64
        Phase values in radians, shape (n_frequencies, n_times).
        Values are in [-π, π].

    Notes
    -----
    Phase is extracted from the complex wavelet transform:

        phase = arctan(Im(W) / Re(W)) = angle(W)

    This is useful for phase-based connectivity metrics like PLV.

    Examples
    --------
    >>> signal = np.sin(2 * np.pi * 10 * np.linspace(0, 1, 256))
    >>> freqs = np.array([10.0])
    >>> phase = compute_wavelet_phase(signal, freqs, fs=256)
    """
    tfr = compute_wavelet_transform(signal, frequencies, fs, n_cycles)
    return np.angle(tfr)


def compute_adaptive_cycles(
    frequencies: NDArray[np.float64],
    min_cycles: float = 3.0,
    max_cycles: float = 10.0,
    scaling: str = 'linear'
) -> NDArray[np.float64]:
    """
    Compute frequency-adaptive number of cycles for wavelet analysis.

    Parameters
    ----------
    frequencies : ndarray of float64
        Array of frequencies in Hz.
    min_cycles : float, optional
        Minimum number of cycles. Default is 3.0.
    max_cycles : float, optional
        Maximum number of cycles. Default is 10.0.
    scaling : str, optional
        Scaling method: 'linear' or 'log'. Default is 'linear'.

    Returns
    -------
    n_cycles : ndarray of float64
        Array of n_cycles values, one per frequency.

    Notes
    -----
    - Linear scaling: n_cycles = freq / 2, bounded by min/max.
    - Log scaling: n_cycles scales with log2(freq).

    Using adaptive n_cycles helps maintain consistent time-frequency
    resolution across the frequency spectrum.

    Examples
    --------
    >>> freqs = np.arange(5, 50, 5)
    >>> n_cycles = compute_adaptive_cycles(freqs)
    """
    frequencies = np.asarray(frequencies)

    if scaling == 'linear':
        n_cycles = frequencies / 2.0
    elif scaling == 'log':
        n_cycles = np.log2(frequencies) * 2
    else:
        raise ValueError(f"Unknown scaling: {scaling}")

    # Apply bounds
    n_cycles = np.clip(n_cycles, min_cycles, max_cycles)

    return n_cycles


def compute_edge_samples(
    frequency: float,
    fs: float,
    n_cycles: float = 5.0,
    n_sigma: float = 3.0
) -> int:
    """
    Compute the number of samples affected by edge effects.

    Parameters
    ----------
    frequency : float
        Wavelet center frequency in Hz.
    fs : float
        Sampling frequency in Hz.
    n_cycles : float, optional
        Number of cycles in the wavelet. Default is 5.0.
    n_sigma : float, optional
        Number of standard deviations to consider. Default is 3.0
        (covers >99% of Gaussian).

    Returns
    -------
    n_edge : int
        Number of samples affected by edge effects at each end.

    Notes
    -----
    The edge effect extends approximately:

        N_edge = n_sigma * n_cycles * fs / (2π * f)

    Lower frequencies have longer wavelets and thus more edge effects.

    Examples
    --------
    >>> n_edge = compute_edge_samples(10.0, 256, n_cycles=5)
    >>> print(f"Exclude first and last {n_edge} samples")
    """
    sigma_t = n_cycles / (2 * np.pi * frequency)
    n_edge = int(np.ceil(n_sigma * sigma_t * fs))
    return n_edge
