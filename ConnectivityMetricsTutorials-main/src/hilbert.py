"""
Hilbert transform utilities for extracting amplitude and phase.

This module provides functions for computing the analytic signal,
extracting instantaneous amplitude (envelope) and phase from signals.
"""

from typing import Tuple, Optional
import numpy as np
from numpy.typing import NDArray
from scipy.signal import hilbert


def compute_analytic_signal(
    signal_data: NDArray[np.floating],
    axis: int = -1
) -> NDArray[np.complexfloating]:
    """
    Compute the analytic signal using the Hilbert transform.
    
    The analytic signal is z(t) = x(t) + i*H{x(t)}, where H{} is
    the Hilbert transform.
    
    Parameters
    ----------
    signal_data : NDArray[np.floating]
        Input real-valued signal.
    axis : int, optional
        Axis along which to compute the analytic signal. Default is -1.
    
    Returns
    -------
    analytic : NDArray[np.complexfloating]
        Complex-valued analytic signal.
    
    Examples
    --------
    >>> import numpy as np
    >>> t = np.linspace(0, 1, 250)
    >>> signal = np.sin(2 * np.pi * 10 * t)
    >>> analytic = compute_analytic_signal(signal)
    >>> envelope = np.abs(analytic)
    """
    return hilbert(signal_data, axis=axis)


def compute_hilbert_transform(
    signal_data: NDArray[np.floating],
    axis: int = -1
) -> NDArray[np.floating]:
    """
    Compute the Hilbert transform of a signal.
    
    The Hilbert transform shifts each frequency component by 90 degrees.
    For a sine wave, this produces a negative cosine.
    
    Parameters
    ----------
    signal_data : NDArray[np.floating]
        Input real-valued signal.
    axis : int, optional
        Axis along which to compute. Default is -1.
    
    Returns
    -------
    hilbert_transformed : NDArray[np.floating]
        Hilbert transform of the input signal.
    
    Examples
    --------
    >>> t = np.linspace(0, 1, 250)
    >>> signal = np.sin(2 * np.pi * 10 * t)
    >>> h_signal = compute_hilbert_transform(signal)
    >>> # h_signal ≈ -cos(2 * np.pi * 10 * t)
    """
    analytic = hilbert(signal_data, axis=axis)
    return np.imag(analytic)


def compute_envelope(
    signal_data: NDArray[np.floating],
    axis: int = -1
) -> NDArray[np.floating]:
    """
    Compute the instantaneous amplitude (envelope) of a signal.
    
    The envelope is the magnitude of the analytic signal:
    A(t) = |z(t)| = sqrt(x(t)^2 + H{x(t)}^2)
    
    Parameters
    ----------
    signal_data : NDArray[np.floating]
        Input real-valued signal. Should be narrowband filtered.
    axis : int, optional
        Axis along which to compute. Default is -1.
    
    Returns
    -------
    envelope : NDArray[np.floating]
        Instantaneous amplitude envelope.
    
    Notes
    -----
    For meaningful results, the input signal should be band-pass
    filtered to a narrow frequency range (bandwidth/center < 0.5).
    
    Examples
    --------
    >>> from filtering import bandpass_filter
    >>> filtered = bandpass_filter(raw_signal, 8, 12, fs=250)
    >>> envelope = compute_envelope(filtered)
    """
    analytic = hilbert(signal_data, axis=axis)
    return np.abs(analytic)


def compute_instantaneous_phase(
    signal_data: NDArray[np.floating],
    axis: int = -1,
    unwrap: bool = False
) -> NDArray[np.floating]:
    """
    Compute the instantaneous phase of a signal.
    
    The instantaneous phase is the angle of the analytic signal:
    φ(t) = arg(z(t)) = atan2(H{x(t)}, x(t))
    
    Parameters
    ----------
    signal_data : NDArray[np.floating]
        Input real-valued signal. Should be narrowband filtered.
    axis : int, optional
        Axis along which to compute. Default is -1.
    unwrap : bool, optional
        If True, unwrap phase to remove discontinuities. Default is False.
    
    Returns
    -------
    phase : NDArray[np.floating]
        Instantaneous phase in radians. Range is [-π, π] if unwrap=False,
        or continuous if unwrap=True.
    
    Notes
    -----
    Phase is only meaningful when the signal has sufficient amplitude.
    For meaningful results, the input should be band-pass filtered.
    
    Examples
    --------
    >>> from filtering import bandpass_filter
    >>> filtered = bandpass_filter(raw_signal, 8, 12, fs=250)
    >>> phase = compute_instantaneous_phase(filtered)
    """
    analytic = hilbert(signal_data, axis=axis)
    phase = np.angle(analytic)
    
    if unwrap:
        phase = np.unwrap(phase, axis=axis)
    
    return phase


def compute_instantaneous_frequency(
    signal_data: NDArray[np.floating],
    fs: float,
    axis: int = -1
) -> NDArray[np.floating]:
    """
    Compute the instantaneous frequency from a signal.
    
    Instantaneous frequency is the derivative of unwrapped phase:
    f(t) = (1/2π) * dφ/dt
    
    Parameters
    ----------
    signal_data : NDArray[np.floating]
        Input real-valued signal. Should be narrowband filtered.
    fs : float
        Sampling frequency in Hz.
    axis : int, optional
        Axis along which to compute. Default is -1.
    
    Returns
    -------
    inst_freq : NDArray[np.floating]
        Instantaneous frequency in Hz.
    
    Notes
    -----
    Instantaneous frequency is very sensitive to noise and only
    meaningful for narrowband signals with sufficient amplitude.
    
    Examples
    --------
    >>> from filtering import bandpass_filter
    >>> filtered = bandpass_filter(raw_signal, 8, 12, fs=250)
    >>> inst_freq = compute_instantaneous_frequency(filtered, fs=250)
    """
    # Get unwrapped phase
    phase = compute_instantaneous_phase(signal_data, axis=axis, unwrap=True)
    
    # Compute derivative (phase difference)
    phase_diff = np.diff(phase, axis=axis)
    
    # Convert to frequency
    inst_freq = fs * phase_diff / (2 * np.pi)
    
    # Pad to match original length (repeat first value)
    pad_shape = list(inst_freq.shape)
    pad_shape[axis] = 1
    first_slice = [slice(None)] * inst_freq.ndim
    first_slice[axis] = slice(0, 1)
    padding = inst_freq[tuple(first_slice)]
    inst_freq = np.concatenate([padding, inst_freq], axis=axis)
    
    return inst_freq


def extract_band_amplitude(
    signal_data: NDArray[np.floating],
    low_freq: float,
    high_freq: float,
    fs: float,
    filter_func: Optional[callable] = None
) -> NDArray[np.floating]:
    """
    Extract amplitude envelope for a specific frequency band.
    
    This is the standard workflow: filter → Hilbert → amplitude.
    
    Parameters
    ----------
    signal_data : NDArray[np.floating]
        Input raw signal.
    low_freq : float
        Lower cutoff frequency in Hz.
    high_freq : float
        Upper cutoff frequency in Hz.
    fs : float
        Sampling frequency in Hz.
    filter_func : callable, optional
        Band-pass filter function. If None, uses default from filtering module.
    
    Returns
    -------
    amplitude : NDArray[np.floating]
        Amplitude envelope of the filtered signal.
    
    Examples
    --------
    >>> alpha_amplitude = extract_band_amplitude(raw_eeg, 8, 13, fs=250)
    """
    # Import here to avoid circular imports
    if filter_func is None:
        from filtering import bandpass_filter
        filter_func = bandpass_filter
    
    # Filter to band
    filtered = filter_func(signal_data, low_freq, high_freq, fs)
    
    # Extract amplitude
    return compute_envelope(filtered)


def extract_band_phase(
    signal_data: NDArray[np.floating],
    low_freq: float,
    high_freq: float,
    fs: float,
    unwrap: bool = False,
    filter_func: Optional[callable] = None
) -> NDArray[np.floating]:
    """
    Extract instantaneous phase for a specific frequency band.
    
    This is the standard workflow: filter → Hilbert → phase.
    
    Parameters
    ----------
    signal_data : NDArray[np.floating]
        Input raw signal.
    low_freq : float
        Lower cutoff frequency in Hz.
    high_freq : float
        Upper cutoff frequency in Hz.
    fs : float
        Sampling frequency in Hz.
    unwrap : bool, optional
        If True, unwrap phase. Default is False.
    filter_func : callable, optional
        Band-pass filter function. If None, uses default from filtering module.
    
    Returns
    -------
    phase : NDArray[np.floating]
        Instantaneous phase of the filtered signal.
    
    Examples
    --------
    >>> alpha_phase = extract_band_phase(raw_eeg, 8, 13, fs=250)
    """
    # Import here to avoid circular imports
    if filter_func is None:
        from filtering import bandpass_filter
        filter_func = bandpass_filter
    
    # Filter to band
    filtered = filter_func(signal_data, low_freq, high_freq, fs)
    
    # Extract phase
    return compute_instantaneous_phase(filtered, unwrap=unwrap)


def extract_band_amplitude_phase(
    signal_data: NDArray[np.floating],
    low_freq: float,
    high_freq: float,
    fs: float,
    filter_func: Optional[callable] = None
) -> Tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """
    Extract both amplitude and phase for a specific frequency band.
    
    This is the complete workflow: filter → Hilbert → amplitude + phase.
    
    Parameters
    ----------
    signal_data : NDArray[np.floating]
        Input raw signal.
    low_freq : float
        Lower cutoff frequency in Hz.
    high_freq : float
        Upper cutoff frequency in Hz.
    fs : float
        Sampling frequency in Hz.
    filter_func : callable, optional
        Band-pass filter function. If None, uses default from filtering module.
    
    Returns
    -------
    filtered : NDArray[np.floating]
        Band-pass filtered signal.
    amplitude : NDArray[np.floating]
        Instantaneous amplitude envelope.
    phase : NDArray[np.floating]
        Instantaneous phase in radians [-π, π].
    
    Examples
    --------
    >>> filtered, amplitude, phase = extract_band_amplitude_phase(
    ...     raw_eeg, 8, 13, fs=250
    ... )
    """
    # Import here to avoid circular imports
    if filter_func is None:
        from filtering import bandpass_filter
        filter_func = bandpass_filter
    
    # Filter to band
    filtered = filter_func(signal_data, low_freq, high_freq, fs)
    
    # Compute analytic signal
    analytic = hilbert(filtered)
    
    # Extract amplitude and phase
    amplitude = np.abs(analytic)
    phase = np.angle(analytic)
    
    return filtered, amplitude, phase
