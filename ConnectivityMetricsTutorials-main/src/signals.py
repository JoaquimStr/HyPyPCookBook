"""Signal generation utilities for the Hyperscanning Workshop."""

from typing import Optional

import numpy as np
from numpy.typing import NDArray


def generate_time_vector(
    duration: float,
    fs: float,
) -> NDArray[np.float64]:
    """
    Generate a time vector for signal creation.

    Parameters
    ----------
    duration : float
        Duration of the time vector in seconds.
    fs : float
        Sampling frequency in Hz.

    Returns
    -------
    NDArray[np.float64]
        Time vector from 0 to duration (exclusive) with spacing 1/fs.
    """
    return np.arange(0, duration, 1 / fs)


def generate_sine_wave(
    t: NDArray[np.float64],
    frequency: float,
    amplitude: float = 1.0,
    phase: float = 0.0,
) -> NDArray[np.float64]:
    """
    Generate a sine wave signal.

    Parameters
    ----------
    t : NDArray[np.float64]
        Time vector in seconds.
    frequency : float
        Frequency of the sine wave in Hz.
    amplitude : float, optional
        Peak amplitude of the sine wave. Default is 1.0.
    phase : float, optional
        Phase offset in radians. Default is 0.0.

    Returns
    -------
    NDArray[np.float64]
        Sine wave signal values.
    """
    return amplitude * np.sin(2 * np.pi * frequency * t + phase)


def generate_white_noise(
    n_samples: int,
    amplitude: float = 1.0,
    seed: Optional[int] = None,
) -> NDArray[np.float64]:
    """
    Generate white noise signal.

    White noise has equal power at all frequencies, resulting in
    a flat power spectrum.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate.
    amplitude : float, optional
        Standard deviation of the noise. Default is 1.0.
    seed : int, optional
        Random seed for reproducibility. Default is None.

    Returns
    -------
    NDArray[np.float64]
        White noise signal.
    """
    rng = np.random.default_rng(seed)
    return amplitude * rng.standard_normal(n_samples)


def generate_pink_noise(
    n_samples: int,
    amplitude: float = 1.0,
    seed: Optional[int] = None,
) -> NDArray[np.float64]:
    """
    Generate pink (1/f) noise signal.

    Pink noise has a power spectrum that decreases with frequency,
    with power proportional to 1/f. This is more representative of
    real EEG signals than white noise.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate.
    amplitude : float, optional
        Scaling factor for the noise amplitude. Default is 1.0.
    seed : int, optional
        Random seed for reproducibility. Default is None.

    Returns
    -------
    NDArray[np.float64]
        Pink noise signal.
    """
    rng = np.random.default_rng(seed)

    # Generate white noise in frequency domain
    white = rng.standard_normal(n_samples)

    # Compute FFT
    fft_white = np.fft.rfft(white)

    # Create 1/f filter (avoiding division by zero)
    frequencies = np.fft.rfftfreq(n_samples)
    frequencies[0] = 1  # Avoid division by zero
    pink_filter = 1 / np.sqrt(frequencies)
    pink_filter[0] = 0  # Remove DC component

    # Apply filter and inverse FFT
    fft_pink = fft_white * pink_filter
    pink = np.fft.irfft(fft_pink, n=n_samples)

    # Normalize and scale
    pink = amplitude * pink / np.std(pink)

    return pink


def generate_composite_signal(
    t: NDArray[np.float64],
    frequencies: list[float],
    amplitudes: list[float],
    phases: Optional[list[float]] = None,
) -> NDArray[np.float64]:
    """
    Generate a composite signal as a sum of sine waves.

    Parameters
    ----------
    t : NDArray[np.float64]
        Time vector in seconds.
    frequencies : list[float]
        List of frequencies in Hz for each component.
    amplitudes : list[float]
        List of amplitudes for each component.
    phases : list[float], optional
        List of phase offsets in radians. Default is all zeros.

    Returns
    -------
    NDArray[np.float64]
        Composite signal.
    """
    if phases is None:
        phases = [0.0] * len(frequencies)

    if len(frequencies) != len(amplitudes) or len(frequencies) != len(phases):
        raise ValueError(
            "frequencies, amplitudes, and phases must have the same length"
        )

    signal = np.zeros_like(t)
    for freq, amp, phase in zip(frequencies, amplitudes, phases):
        signal += generate_sine_wave(t, freq, amp, phase)

    return signal


def compute_aliased_frequency(
    true_freq: float,
    fs: float,
) -> float:
    """
    Compute the aliased frequency when sampling violates Nyquist.

    Parameters
    ----------
    true_freq : float
        The true frequency of the signal in Hz.
    fs : float
        The sampling frequency in Hz.

    Returns
    -------
    float
        The frequency that will appear in the sampled signal.
    """
    nyquist = fs / 2
    # Fold the frequency back into the Nyquist range
    aliased = true_freq % fs
    if aliased > nyquist:
        aliased = fs - aliased
    return aliased
