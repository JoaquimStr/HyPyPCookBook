"""
Filtering functions for EEG signal processing.

This module provides functions for designing and applying digital filters
to neural signals, with a focus on EEG preprocessing for connectivity analysis.

Functions
---------
design_fir_filter : Design a FIR filter using the window method
design_iir_filter : Design an IIR filter (Butterworth, Chebyshev, etc.)
apply_filter : Apply a filter to a signal with zero-phase option
lowpass_filter : Apply a lowpass filter
highpass_filter : Apply a highpass filter
bandpass_filter : Apply a bandpass filter
notch_filter : Remove a specific frequency
notch_filter_harmonics : Remove a frequency and its harmonics
mne_filter_data : Wrapper for MNE filtering
"""

from typing import Literal, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.signal import (
    butter,
    cheby1,
    cheby2,
    ellip,
    filtfilt,
    firwin,
    iirnotch,
    lfilter,
)


def design_iir_filter(
    cutoff: float | Tuple[float, float],
    fs: float,
    order: int = 4,
    btype: Literal["low", "high", "band", "bandstop"] = "low",
    ftype: Literal["butter", "cheby1", "cheby2", "ellip"] = "butter",
    rp: float = 1.0,
    rs: float = 40.0,
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Design an IIR filter.

    Parameters
    ----------
    cutoff : float or tuple of float
        Cutoff frequency in Hz. For bandpass/bandstop, provide (low, high).
    fs : float
        Sampling frequency in Hz.
    order : int, default=4
        Filter order.
    btype : {'low', 'high', 'band', 'bandstop'}, default='low'
        Filter type.
    ftype : {'butter', 'cheby1', 'cheby2', 'ellip'}, default='butter'
        IIR filter family.
    rp : float, default=1.0
        Maximum ripple in passband (dB). Used for cheby1 and ellip.
    rs : float, default=40.0
        Minimum attenuation in stopband (dB). Used for cheby2 and ellip.

    Returns
    -------
    b : ndarray
        Numerator coefficients.
    a : ndarray
        Denominator coefficients.

    Examples
    --------
    >>> b, a = design_iir_filter(30, fs=250, order=4, btype='low')
    >>> b, a = design_iir_filter((8, 13), fs=250, order=4, btype='band')
    """
    nyquist = fs / 2

    # Normalize cutoff frequency
    if isinstance(cutoff, (list, tuple)):
        normalized_cutoff: float | Tuple[float, float] = (
            cutoff[0] / nyquist,
            cutoff[1] / nyquist,
        )
    else:
        normalized_cutoff = cutoff / nyquist

    # Select filter design function
    if ftype == "butter":
        b, a = butter(order, normalized_cutoff, btype=btype)
    elif ftype == "cheby1":
        b, a = cheby1(order, rp, normalized_cutoff, btype=btype)
    elif ftype == "cheby2":
        b, a = cheby2(order, rs, normalized_cutoff, btype=btype)
    elif ftype == "ellip":
        b, a = ellip(order, rp, rs, normalized_cutoff, btype=btype)
    else:
        raise ValueError(f"Unknown filter type: {ftype}")

    return b, a


def design_fir_filter(
    cutoff: float | Tuple[float, float],
    fs: float,
    numtaps: int = 101,
    btype: Literal["low", "high", "band", "bandstop"] = "low",
    window: str = "hamming",
) -> NDArray[np.floating]:
    """
    Design a FIR filter using the window method.

    Parameters
    ----------
    cutoff : float or tuple of float
        Cutoff frequency in Hz. For bandpass/bandstop, provide (low, high).
    fs : float
        Sampling frequency in Hz.
    numtaps : int, default=101
        Number of filter coefficients (filter length).
        Should be odd for type I linear phase FIR.
    btype : {'low', 'high', 'band', 'bandstop'}, default='low'
        Filter type.
    window : str, default='hamming'
        Window function to use ('hamming', 'hann', 'blackman', 'kaiser', etc.).

    Returns
    -------
    h : ndarray
        FIR filter coefficients.

    Examples
    --------
    >>> h = design_fir_filter(30, fs=250, numtaps=101, btype='low')
    >>> h = design_fir_filter((8, 13), fs=250, numtaps=101, btype='band')
    """
    nyquist = fs / 2

    # Normalize cutoff frequency
    if isinstance(cutoff, (list, tuple)):
        normalized_cutoff: float | list[float] = [c / nyquist for c in cutoff]
    else:
        normalized_cutoff = cutoff / nyquist

    # Design filter
    if btype == "low":
        h = firwin(numtaps, normalized_cutoff, window=window)
    elif btype == "high":
        h = firwin(numtaps, normalized_cutoff, pass_zero=False, window=window)
    elif btype == "band":
        h = firwin(numtaps, normalized_cutoff, pass_zero=False, window=window)
    elif btype == "bandstop":
        h = firwin(numtaps, normalized_cutoff, pass_zero=True, window=window)
    else:
        raise ValueError(f"Unknown filter type: {btype}")

    return h


def apply_filter(
    signal: NDArray[np.floating],
    b: NDArray[np.floating],
    a: NDArray[np.floating] | None = None,
    zero_phase: bool = True,
) -> NDArray[np.floating]:
    """
    Apply a filter to a signal.

    Parameters
    ----------
    signal : ndarray
        Input signal to filter.
    b : ndarray
        Numerator coefficients of the filter.
    a : ndarray or None, default=None
        Denominator coefficients. If None, assumes FIR filter (a=[1]).
    zero_phase : bool, default=True
        If True, use zero-phase filtering (filtfilt).
        If False, use causal filtering (lfilter).

    Returns
    -------
    filtered : ndarray
        Filtered signal.

    Notes
    -----
    Zero-phase filtering applies the filter twice (forward and backward),
    which doubles the filter order but eliminates phase distortion.
    """
    if a is None:
        a = np.array([1.0])

    if zero_phase:
        # Zero-phase filtering (no phase distortion)
        return filtfilt(b, a, signal)
    else:
        # Causal filtering (introduces phase delay)
        return lfilter(b, a, signal)


def lowpass_filter(
    signal: NDArray[np.floating],
    cutoff: float,
    fs: float,
    order: int = 4,
    zero_phase: bool = True,
    fir: bool = False,
    numtaps: int = 101,
) -> NDArray[np.floating]:
    """
    Apply a lowpass filter to a signal.

    Parameters
    ----------
    signal : ndarray
        Input signal to filter.
    cutoff : float
        Cutoff frequency in Hz.
    fs : float
        Sampling frequency in Hz.
    order : int, default=4
        Filter order (for IIR) or ignored (for FIR).
    zero_phase : bool, default=True
        If True, use zero-phase filtering (filtfilt).
    fir : bool, default=False
        If True, use FIR filter. If False, use IIR (Butterworth).
    numtaps : int, default=101
        Number of FIR filter taps (only used if fir=True).

    Returns
    -------
    filtered : ndarray
        Lowpass filtered signal.

    Examples
    --------
    >>> filtered = lowpass_filter(signal, cutoff=30, fs=250)
    >>> filtered = lowpass_filter(signal, cutoff=30, fs=250, fir=True)
    """
    if fir:
        h = design_fir_filter(cutoff, fs, numtaps=numtaps, btype="low")
        return apply_filter(signal, h, zero_phase=zero_phase)
    else:
        b, a = design_iir_filter(cutoff, fs, order=order, btype="low")
        return apply_filter(signal, b, a, zero_phase=zero_phase)


def highpass_filter(
    signal: NDArray[np.floating],
    cutoff: float,
    fs: float,
    order: int = 4,
    zero_phase: bool = True,
    fir: bool = False,
    numtaps: int = 101,
) -> NDArray[np.floating]:
    """
    Apply a highpass filter to a signal.

    Parameters
    ----------
    signal : ndarray
        Input signal to filter.
    cutoff : float
        Cutoff frequency in Hz.
    fs : float
        Sampling frequency in Hz.
    order : int, default=4
        Filter order (for IIR) or ignored (for FIR).
    zero_phase : bool, default=True
        If True, use zero-phase filtering (filtfilt).
    fir : bool, default=False
        If True, use FIR filter. If False, use IIR (Butterworth).
    numtaps : int, default=101
        Number of FIR filter taps (only used if fir=True).

    Returns
    -------
    filtered : ndarray
        Highpass filtered signal.

    Examples
    --------
    >>> filtered = highpass_filter(signal, cutoff=1.0, fs=250)
    >>> filtered = highpass_filter(signal, cutoff=0.5, fs=250, fir=True)
    """
    if fir:
        h = design_fir_filter(cutoff, fs, numtaps=numtaps, btype="high")
        return apply_filter(signal, h, zero_phase=zero_phase)
    else:
        b, a = design_iir_filter(cutoff, fs, order=order, btype="high")
        return apply_filter(signal, b, a, zero_phase=zero_phase)


def bandpass_filter(
    signal: NDArray[np.floating],
    low_freq: float,
    high_freq: float,
    fs: float,
    order: int = 4,
    zero_phase: bool = True,
    fir: bool = False,
    numtaps: int = 101,
) -> NDArray[np.floating]:
    """
    Apply a bandpass filter to a signal.

    Parameters
    ----------
    signal : ndarray
        Input signal to filter.
    low_freq : float
        Lower cutoff frequency in Hz.
    high_freq : float
        Upper cutoff frequency in Hz.
    fs : float
        Sampling frequency in Hz.
    order : int, default=4
        Filter order (for IIR) or ignored (for FIR).
    zero_phase : bool, default=True
        If True, use zero-phase filtering (filtfilt).
    fir : bool, default=False
        If True, use FIR filter. If False, use IIR (Butterworth).
    numtaps : int, default=101
        Number of FIR filter taps (only used if fir=True).

    Returns
    -------
    filtered : ndarray
        Bandpass filtered signal.

    Examples
    --------
    >>> alpha = bandpass_filter(signal, low_freq=8, high_freq=13, fs=250)
    >>> standard_eeg = bandpass_filter(signal, low_freq=0.5, high_freq=40, fs=250)
    """
    if fir:
        h = design_fir_filter((low_freq, high_freq), fs, numtaps=numtaps, btype="band")
        return apply_filter(signal, h, zero_phase=zero_phase)
    else:
        b, a = design_iir_filter((low_freq, high_freq), fs, order=order, btype="band")
        return apply_filter(signal, b, a, zero_phase=zero_phase)


def notch_filter(
    signal: NDArray[np.floating],
    freq: float,
    fs: float,
    quality: float = 30.0,
    zero_phase: bool = True,
) -> NDArray[np.floating]:
    """
    Apply a notch filter to remove a specific frequency.

    Parameters
    ----------
    signal : ndarray
        Input signal to filter.
    freq : float
        Frequency to remove in Hz.
    fs : float
        Sampling frequency in Hz.
    quality : float, default=30.0
        Quality factor. Higher values create narrower notches.
        Bandwidth = freq / quality.
    zero_phase : bool, default=True
        If True, use zero-phase filtering (filtfilt).

    Returns
    -------
    filtered : ndarray
        Notch filtered signal.

    Examples
    --------
    >>> clean = notch_filter(signal, freq=50, fs=250)  # Remove 50 Hz
    >>> clean = notch_filter(signal, freq=60, fs=250, quality=50)  # Narrow notch
    """
    # Design notch filter
    b, a = iirnotch(freq, quality, fs)

    return apply_filter(signal, b, a, zero_phase=zero_phase)


def notch_filter_harmonics(
    signal: NDArray[np.floating],
    base_freq: float,
    fs: float,
    n_harmonics: int = 3,
    quality: float = 30.0,
    zero_phase: bool = True,
) -> NDArray[np.floating]:
    """
    Apply notch filters at a frequency and its harmonics.

    Parameters
    ----------
    signal : ndarray
        Input signal to filter.
    base_freq : float
        Base frequency to remove in Hz.
    fs : float
        Sampling frequency in Hz.
    n_harmonics : int, default=3
        Number of harmonics to remove (including base frequency).
    quality : float, default=30.0
        Quality factor for all notches.
    zero_phase : bool, default=True
        If True, use zero-phase filtering (filtfilt).

    Returns
    -------
    filtered : ndarray
        Signal with base frequency and harmonics removed.

    Examples
    --------
    >>> clean = notch_filter_harmonics(signal, 50, fs=500, n_harmonics=3)
    # Removes 50 Hz, 100 Hz, and 150 Hz
    """
    nyquist = fs / 2
    filtered = signal.copy()

    for i in range(1, n_harmonics + 1):
        harmonic_freq = base_freq * i
        if harmonic_freq < nyquist:
            filtered = notch_filter(filtered, harmonic_freq, fs, quality, zero_phase)

    return filtered


def mne_filter_data(
    data: NDArray[np.floating],
    fs: float,
    l_freq: float | None = None,
    h_freq: float | None = None,
    method: str = "fir",
    verbose: bool = False,
) -> NDArray[np.floating]:
    """
    Filter data using MNE's filter_data function.

    This is a wrapper around mne.filter.filter_data that provides
    MNE's robust filtering with simple numpy arrays.

    Parameters
    ----------
    data : ndarray
        Input data to filter. Can be 1D or 2D (channels x samples).
    fs : float
        Sampling frequency in Hz.
    l_freq : float or None
        Low cutoff frequency in Hz. If None, no highpass.
    h_freq : float or None
        High cutoff frequency in Hz. If None, no lowpass.
    method : str, default='fir'
        Filter method: 'fir' or 'iir'.
    verbose : bool, default=False
        If True, print MNE filter information.

    Returns
    -------
    filtered : ndarray
        Filtered data with same shape as input.

    Examples
    --------
    >>> # Bandpass filter 1-40 Hz
    >>> filtered = mne_filter_data(data, fs=250, l_freq=1, h_freq=40)

    >>> # Highpass only
    >>> filtered = mne_filter_data(data, fs=250, l_freq=0.5, h_freq=None)
    """
    import mne

    # Ensure 2D for MNE (channels x samples)
    was_1d = data.ndim == 1
    if was_1d:
        data = data.reshape(1, -1)

    # Use MNE's filter_data
    filtered = mne.filter.filter_data(
        data, sfreq=fs, l_freq=l_freq, h_freq=h_freq, method=method, verbose=verbose
    )

    # Return to original shape
    if was_1d:
        filtered = filtered.flatten()

    return filtered
