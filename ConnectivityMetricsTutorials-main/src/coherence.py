"""Spectral coherence analysis functions.

This module provides functions for computing coherence (correlation in the
frequency domain) between signals, including cross-spectral density, band
coherence, coherence matrices, and hyperscanning-specific analyses.
"""

import numpy as np
from numpy.typing import NDArray
from scipy.signal import csd, welch, coherence
from typing import Any


# =============================================================================
# Core Coherence Functions
# =============================================================================


def compute_cross_spectrum(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    fs: float,
    nperseg: int = 256,
    noverlap: int | None = None,
    window: str = "hann",
) -> tuple[NDArray[np.floating], NDArray[np.complexfloating]]:
    """Compute cross-spectral density using Welch's method.

    The cross-spectral density (CSD) measures the co-variation of two signals
    in the frequency domain. It is complex-valued, containing both magnitude
    (strength of relationship) and phase (timing relationship) information.

    Parameters
    ----------
    x : NDArray[np.floating]
        First input signal.
    y : NDArray[np.floating]
        Second input signal (same length as x).
    fs : float
        Sampling frequency in Hz.
    nperseg : int, optional
        Length of each segment for Welch's method. Default is 256.
    noverlap : int | None, optional
        Number of points to overlap between segments.
        If None, defaults to nperseg // 2.
    window : str, optional
        Window function to apply. Default is "hann".

    Returns
    -------
    frequencies : NDArray[np.floating]
        Array of frequency values in Hz.
    csd_values : NDArray[np.complexfloating]
        Complex cross-spectral density values.

    Examples
    --------
    >>> t = np.linspace(0, 2, 2000, endpoint=False)
    >>> x = np.sin(2 * np.pi * 10 * t)
    >>> y = np.sin(2 * np.pi * 10 * t + np.pi / 4)
    >>> freqs, csd_vals = compute_cross_spectrum(x, y, fs=1000)
    """
    if noverlap is None:
        noverlap = nperseg // 2

    frequencies, csd_values = csd(
        x, y,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        window=window,
    )

    return frequencies, csd_values


def compute_coherence(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    fs: float,
    nperseg: int = 256,
    noverlap: int | None = None,
    window: str = "hann",
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Compute magnitude-squared coherence using Welch's method.

    Coherence measures the linear correlation between two signals at each
    frequency. It is defined as:

        C_xy(f) = |S_xy(f)|² / (S_xx(f) × S_yy(f))

    where S_xy is the cross-spectral density and S_xx, S_yy are the power
    spectral densities.

    Parameters
    ----------
    x : NDArray[np.floating]
        First input signal.
    y : NDArray[np.floating]
        Second input signal (same length as x).
    fs : float
        Sampling frequency in Hz.
    nperseg : int, optional
        Length of each segment for Welch's method. Default is 256.
    noverlap : int | None, optional
        Number of points to overlap between segments.
        If None, defaults to nperseg // 2.
    window : str, optional
        Window function to apply. Default is "hann".

    Returns
    -------
    frequencies : NDArray[np.floating]
        Array of frequency values in Hz.
    coherence_values : NDArray[np.floating]
        Magnitude-squared coherence values (0 to 1).

    Notes
    -----
    - Coherence = 1 indicates perfect linear relationship at that frequency.
    - Coherence = 0 indicates no linear relationship.
    - Coherence is independent of phase (phase shift doesn't affect it).
    - Volume conduction can create spuriously high coherence at zero lag.

    Examples
    --------
    >>> t = np.linspace(0, 2, 2000, endpoint=False)
    >>> x = np.sin(2 * np.pi * 10 * t) + 0.5 * np.random.randn(2000)
    >>> y = np.sin(2 * np.pi * 10 * t) + 0.5 * np.random.randn(2000)
    >>> freqs, coh = compute_coherence(x, y, fs=1000)
    """
    if noverlap is None:
        noverlap = nperseg // 2

    frequencies, coherence_values = coherence(
        x, y,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        window=window,
    )

    return frequencies, coherence_values


def generate_coherent_signals(
    n_samples: int,
    fs: float,
    frequency: float,
    coherence_level: float,
    snr_db: float = 20.0,
    seed: int | None = None,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Generate two signals with specified coherence at a given frequency.

    Creates a pair of signals where the coherence at the target frequency
    can be controlled. The signals share a common sinusoidal component,
    and independent noise is added to each to achieve the desired coherence.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate.
    fs : float
        Sampling frequency in Hz.
    frequency : float
        Target frequency in Hz for the coherent component.
    coherence_level : float
        Desired coherence level (0 to 1). Higher values mean more shared
        signal relative to independent noise.
    snr_db : float, optional
        Signal-to-noise ratio in dB for the baseline noise. Default is 20.0.
    seed : int | None, optional
        Random seed for reproducibility.

    Returns
    -------
    x : NDArray[np.floating]
        First signal.
    y : NDArray[np.floating]
        Second signal with specified coherence to x at target frequency.

    Examples
    --------
    >>> x, y = generate_coherent_signals(
    ...     n_samples=2000, fs=500, frequency=10, coherence_level=0.8
    ... )
    >>> freqs, coh = compute_coherence(x, y, fs=500)
    >>> # Coherence at 10 Hz should be approximately 0.8
    """
    if seed is not None:
        np.random.seed(seed)

    t = np.arange(n_samples) / fs

    # Shared sinusoidal component
    shared_signal = np.sin(2 * np.pi * frequency * t)

    # Convert SNR to linear scale
    snr_linear = 10 ** (snr_db / 20)

    # Scale factors to achieve desired coherence
    # Coherence ≈ shared_power / (shared_power + noise_power)
    # With equal noise added to both signals
    shared_power = coherence_level
    noise_power = 1 - coherence_level

    # Scale shared component
    shared_scale = np.sqrt(shared_power)
    noise_scale = np.sqrt(noise_power) / snr_linear

    # Generate independent noise for each signal
    noise_x = np.random.randn(n_samples) * noise_scale
    noise_y = np.random.randn(n_samples) * noise_scale

    # Combine shared and independent components
    x = shared_scale * shared_signal + noise_x
    y = shared_scale * shared_signal + noise_y

    return x, y


# =============================================================================
# Band Coherence Functions
# =============================================================================


def compute_band_coherence(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    fs: float,
    band: tuple[float, float],
    nperseg: int = 256,
    method: str = "mean",
) -> float:
    """Compute average coherence in a specified frequency band.

    Parameters
    ----------
    x : NDArray[np.floating]
        First input signal.
    y : NDArray[np.floating]
        Second input signal.
    fs : float
        Sampling frequency in Hz.
    band : tuple[float, float]
        Frequency band as (low_freq, high_freq) in Hz.
    nperseg : int, optional
        Length of each segment for Welch's method. Default is 256.
    method : str, optional
        Averaging method: "mean" for simple average, "weighted" for
        power-weighted average. Default is "mean".

    Returns
    -------
    float
        Average coherence in the specified band.

    Examples
    --------
    >>> x, y = generate_coherent_signals(2000, 500, 10, 0.8)
    >>> alpha_coh = compute_band_coherence(x, y, fs=500, band=(8, 13))
    """
    frequencies, coh = compute_coherence(x, y, fs, nperseg=nperseg)

    # Select frequencies in band
    band_mask = (frequencies >= band[0]) & (frequencies <= band[1])
    coh_band = coh[band_mask]

    if len(coh_band) == 0:
        return 0.0

    if method == "mean":
        return float(np.mean(coh_band))
    elif method == "weighted":
        # Weight by combined power
        freqs_x, psd_x = welch(x, fs=fs, nperseg=nperseg)
        freqs_y, psd_y = welch(y, fs=fs, nperseg=nperseg)
        psd_combined = (psd_x[band_mask] + psd_y[band_mask]) / 2
        if np.sum(psd_combined) == 0:
            return float(np.mean(coh_band))
        return float(np.average(coh_band, weights=psd_combined))
    else:
        raise ValueError(f"Unknown method: {method}. Use 'mean' or 'weighted'.")


def compute_all_band_coherence(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    fs: float,
    bands: dict[str, tuple[float, float]] | None = None,
    nperseg: int = 256,
) -> dict[str, float]:
    """Compute coherence for all standard EEG frequency bands.

    Parameters
    ----------
    x : NDArray[np.floating]
        First input signal.
    y : NDArray[np.floating]
        Second input signal.
    fs : float
        Sampling frequency in Hz.
    bands : dict[str, tuple[float, float]] | None, optional
        Dictionary mapping band names to (low, high) frequency tuples.
        If None, uses standard EEG bands.
    nperseg : int, optional
        Length of each segment for Welch's method. Default is 256.

    Returns
    -------
    dict[str, float]
        Dictionary mapping band names to coherence values.

    Examples
    --------
    >>> x, y = generate_coherent_signals(2000, 500, 10, 0.8)
    >>> band_coh = compute_all_band_coherence(x, y, fs=500)
    >>> print(f"Alpha coherence: {band_coh['alpha']:.2f}")
    """
    if bands is None:
        bands = {
            "delta": (1.0, 4.0),
            "theta": (4.0, 8.0),
            "alpha": (8.0, 13.0),
            "beta": (13.0, 30.0),
            "gamma": (30.0, 100.0),
        }

    result = {}
    for band_name, band_range in bands.items():
        result[band_name] = compute_band_coherence(x, y, fs, band_range, nperseg)

    return result


# =============================================================================
# Coherence Matrix Functions
# =============================================================================


def compute_coherence_matrix(
    data: NDArray[np.floating],
    fs: float,
    band: tuple[float, float] | None = None,
    nperseg: int = 256,
) -> NDArray[np.floating]:
    """Compute coherence matrix for all channel pairs.

    Parameters
    ----------
    data : NDArray[np.floating]
        Multi-channel data array of shape (n_channels, n_samples).
    fs : float
        Sampling frequency in Hz.
    band : tuple[float, float] | None, optional
        If provided, returns band-averaged coherence matrix.
        If None, returns coherence at all frequencies (3D array).
    nperseg : int, optional
        Length of each segment for Welch's method. Default is 256.

    Returns
    -------
    NDArray[np.floating]
        If band is provided: (n_channels, n_channels) coherence matrix.
        If band is None: (n_channels, n_channels, n_freqs) coherence array.

    Notes
    -----
    - The matrix is symmetric: C_xy = C_yx.
    - Diagonal elements are 1 (self-coherence).

    Examples
    --------
    >>> data = np.random.randn(8, 2000)
    >>> coh_matrix = compute_coherence_matrix(data, fs=500, band=(8, 13))
    >>> print(coh_matrix.shape)  # (8, 8)
    """
    n_channels = data.shape[0]

    # First, compute one coherence to get frequency axis
    freqs, _ = compute_coherence(data[0], data[1], fs, nperseg=nperseg)
    n_freqs = len(freqs)

    if band is not None:
        # Return 2D matrix of band-averaged coherence
        coh_matrix = np.ones((n_channels, n_channels))

        for i in range(n_channels):
            for j in range(i + 1, n_channels):
                coh_value = compute_band_coherence(
                    data[i], data[j], fs, band, nperseg
                )
                coh_matrix[i, j] = coh_value
                coh_matrix[j, i] = coh_value

        return coh_matrix
    else:
        # Return 3D array with full spectrum
        coh_matrix = np.ones((n_channels, n_channels, n_freqs))

        for i in range(n_channels):
            for j in range(i + 1, n_channels):
                _, coh = compute_coherence(data[i], data[j], fs, nperseg=nperseg)
                coh_matrix[i, j, :] = coh
                coh_matrix[j, i, :] = coh

        return coh_matrix


def compute_coherence_matrix_bands(
    data: NDArray[np.floating],
    fs: float,
    bands: dict[str, tuple[float, float]] | None = None,
    nperseg: int = 256,
) -> dict[str, NDArray[np.floating]]:
    """Compute coherence matrices for all frequency bands.

    Parameters
    ----------
    data : NDArray[np.floating]
        Multi-channel data array of shape (n_channels, n_samples).
    fs : float
        Sampling frequency in Hz.
    bands : dict[str, tuple[float, float]] | None, optional
        Dictionary mapping band names to (low, high) frequency tuples.
        If None, uses standard EEG bands.
    nperseg : int, optional
        Length of each segment for Welch's method. Default is 256.

    Returns
    -------
    dict[str, NDArray[np.floating]]
        Dictionary mapping band names to coherence matrices.

    Examples
    --------
    >>> data = np.random.randn(8, 2000)
    >>> band_matrices = compute_coherence_matrix_bands(data, fs=500)
    >>> alpha_matrix = band_matrices["alpha"]
    """
    if bands is None:
        bands = {
            "delta": (1.0, 4.0),
            "theta": (4.0, 8.0),
            "alpha": (8.0, 13.0),
            "beta": (13.0, 30.0),
            "gamma": (30.0, 100.0),
        }

    result = {}
    for band_name, band_range in bands.items():
        result[band_name] = compute_coherence_matrix(data, fs, band_range, nperseg)

    return result


# =============================================================================
# Hyperscanning Coherence Functions
# =============================================================================


def compute_coherence_hyperscanning(
    data_p1: NDArray[np.floating],
    data_p2: NDArray[np.floating],
    fs: float,
    band: tuple[float, float],
    nperseg: int = 256,
) -> dict[str, NDArray[np.floating]]:
    """Compute coherence matrices for hyperscanning analysis.

    Computes within-participant and between-participant coherence matrices
    for dual-brain (hyperscanning) data.

    Parameters
    ----------
    data_p1 : NDArray[np.floating]
        Participant 1 data of shape (n_channels_p1, n_samples).
    data_p2 : NDArray[np.floating]
        Participant 2 data of shape (n_channels_p2, n_samples).
    fs : float
        Sampling frequency in Hz.
    band : tuple[float, float]
        Frequency band as (low_freq, high_freq) in Hz.
    nperseg : int, optional
        Length of each segment for Welch's method. Default is 256.

    Returns
    -------
    dict[str, NDArray[np.floating]]
        Dictionary containing:
        - "within_p1": Coherence matrix within participant 1
        - "within_p2": Coherence matrix within participant 2
        - "between": Between-participants coherence matrix (n_ch_p1 × n_ch_p2)
        - "full": Full combined matrix (n_total × n_total)

    Examples
    --------
    >>> data_p1 = np.random.randn(6, 2000)
    >>> data_p2 = np.random.randn(6, 2000)
    >>> coh = compute_coherence_hyperscanning(data_p1, data_p2, fs=500, band=(8, 13))
    >>> print(coh["between"].shape)  # (6, 6)
    """
    n_ch_p1 = data_p1.shape[0]
    n_ch_p2 = data_p2.shape[0]
    n_total = n_ch_p1 + n_ch_p2

    # Within-participant coherence
    within_p1 = compute_coherence_matrix(data_p1, fs, band, nperseg)
    within_p2 = compute_coherence_matrix(data_p2, fs, band, nperseg)

    # Between-participants coherence
    between = np.zeros((n_ch_p1, n_ch_p2))
    for i in range(n_ch_p1):
        for j in range(n_ch_p2):
            between[i, j] = compute_band_coherence(
                data_p1[i], data_p2[j], fs, band, nperseg
            )

    # Construct full matrix
    full = np.ones((n_total, n_total))
    full[:n_ch_p1, :n_ch_p1] = within_p1
    full[n_ch_p1:, n_ch_p1:] = within_p2
    full[:n_ch_p1, n_ch_p1:] = between
    full[n_ch_p1:, :n_ch_p1] = between.T

    return {
        "within_p1": within_p1,
        "within_p2": within_p2,
        "between": between,
        "full": full,
    }


def compute_global_coherence_hyperscanning(
    coherence_dict: dict[str, NDArray[np.floating]],
) -> dict[str, float]:
    """Compute summary statistics for hyperscanning coherence.

    Parameters
    ----------
    coherence_dict : dict[str, NDArray[np.floating]]
        Dictionary from compute_coherence_hyperscanning containing
        "within_p1", "within_p2", and "between" matrices.

    Returns
    -------
    dict[str, float]
        Dictionary containing:
        - "mean_within_p1": Mean within-P1 coherence (excluding diagonal)
        - "mean_within_p2": Mean within-P2 coherence (excluding diagonal)
        - "mean_between": Mean between-participants coherence
        - "ratio_between_within": Ratio of between to average within coherence

    Examples
    --------
    >>> coh = compute_coherence_hyperscanning(data_p1, data_p2, fs=500, band=(8, 13))
    >>> stats = compute_global_coherence_hyperscanning(coh)
    >>> print(f"Between/within ratio: {stats['ratio_between_within']:.2f}")
    """
    within_p1 = coherence_dict["within_p1"]
    within_p2 = coherence_dict["within_p2"]
    between = coherence_dict["between"]

    # Mean within-P1 (excluding diagonal)
    n_p1 = within_p1.shape[0]
    mask_p1 = ~np.eye(n_p1, dtype=bool)
    mean_within_p1 = float(np.mean(within_p1[mask_p1]))

    # Mean within-P2 (excluding diagonal)
    n_p2 = within_p2.shape[0]
    mask_p2 = ~np.eye(n_p2, dtype=bool)
    mean_within_p2 = float(np.mean(within_p2[mask_p2]))

    # Mean between
    mean_between = float(np.mean(between))

    # Ratio
    avg_within = (mean_within_p1 + mean_within_p2) / 2
    if avg_within > 0:
        ratio = mean_between / avg_within
    else:
        ratio = 0.0

    return {
        "mean_within_p1": mean_within_p1,
        "mean_within_p2": mean_within_p2,
        "mean_between": mean_between,
        "ratio_between_within": ratio,
    }


# =============================================================================
# Statistical Testing
# =============================================================================


def coherence_significance_threshold(
    n_segments: int,
    alpha: float = 0.05,
) -> float:
    """Compute theoretical significance threshold for coherence.

    For independent signals, coherence has a known distribution that depends
    on the number of segments used in Welch's method. This function returns
    the critical value above which coherence is statistically significant.

    Parameters
    ----------
    n_segments : int
        Number of segments used in Welch's method.
    alpha : float, optional
        Significance level. Default is 0.05.

    Returns
    -------
    float
        Coherence threshold for significance.

    Notes
    -----
    Based on the formula: threshold = 1 - alpha^(1/(n_segments-1))

    Examples
    --------
    >>> threshold = coherence_significance_threshold(n_segments=10, alpha=0.05)
    >>> print(f"Coherence above {threshold:.3f} is significant")
    """
    if n_segments <= 1:
        return 1.0

    return 1 - alpha ** (1 / (n_segments - 1))


def coherence_surrogate_test(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    fs: float,
    band: tuple[float, float],
    n_surrogates: int = 500,
    nperseg: int = 256,
    seed: int | None = None,
) -> dict[str, Any]:
    """Test coherence significance using surrogate data.

    Generates a null distribution by shuffling one signal to destroy
    the temporal relationship while preserving spectral properties.

    Parameters
    ----------
    x : NDArray[np.floating]
        First input signal.
    y : NDArray[np.floating]
        Second input signal.
    fs : float
        Sampling frequency in Hz.
    band : tuple[float, float]
        Frequency band as (low_freq, high_freq) in Hz.
    n_surrogates : int, optional
        Number of surrogate permutations. Default is 500.
    nperseg : int, optional
        Length of each segment for Welch's method. Default is 256.
    seed : int | None, optional
        Random seed for reproducibility.

    Returns
    -------
    dict[str, Any]
        Dictionary containing:
        - "observed": Observed band coherence
        - "null_mean": Mean of null distribution
        - "null_std": Standard deviation of null distribution
        - "pvalue": P-value (proportion of surrogates >= observed)
        - "threshold_95": 95th percentile of null distribution

    Examples
    --------
    >>> x, y = generate_coherent_signals(2000, 500, 10, 0.8, seed=42)
    >>> result = coherence_surrogate_test(x, y, fs=500, band=(8, 13), seed=42)
    >>> print(f"p = {result['pvalue']:.4f}")
    """
    if seed is not None:
        np.random.seed(seed)

    # Observed coherence
    observed = compute_band_coherence(x, y, fs, band, nperseg)

    # Generate null distribution
    null_distribution = np.zeros(n_surrogates)
    y_copy = y.copy()

    for i in range(n_surrogates):
        # Shuffle y to destroy relationship
        np.random.shuffle(y_copy)
        null_distribution[i] = compute_band_coherence(x, y_copy, fs, band, nperseg)

    # Statistics
    null_mean = float(np.mean(null_distribution))
    null_std = float(np.std(null_distribution))
    pvalue = float(np.mean(null_distribution >= observed))
    threshold_95 = float(np.percentile(null_distribution, 95))

    return {
        "observed": observed,
        "null_mean": null_mean,
        "null_std": null_std,
        "pvalue": pvalue,
        "threshold_95": threshold_95,
    }


# =============================================================================
# Validation Functions
# =============================================================================


def compare_with_scipy_coherence(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    fs: float,
    nperseg: int = 256,
) -> dict[str, Any]:
    """Compare our coherence implementation with scipy.signal.coherence.

    Validates that our implementation produces results consistent with
    the standard scipy implementation.

    Parameters
    ----------
    x : NDArray[np.floating]
        First input signal.
    y : NDArray[np.floating]
        Second input signal.
    fs : float
        Sampling frequency in Hz.
    nperseg : int, optional
        Length of each segment for Welch's method. Default is 256.

    Returns
    -------
    dict[str, Any]
        Dictionary containing:
        - "our_coherence": Our coherence values
        - "scipy_coherence": Scipy coherence values
        - "max_difference": Maximum absolute difference
        - "correlation": Correlation between the two

    Examples
    --------
    >>> x = np.random.randn(2000)
    >>> y = np.random.randn(2000)
    >>> result = compare_with_scipy_coherence(x, y, fs=500)
    >>> print(f"Max difference: {result['max_difference']:.2e}")
    """
    # Our implementation (which wraps scipy, so should match exactly)
    freqs_ours, coh_ours = compute_coherence(x, y, fs, nperseg=nperseg)

    # Direct scipy call
    freqs_scipy, coh_scipy = coherence(x, y, fs=fs, nperseg=nperseg)

    # Compare
    max_diff = float(np.max(np.abs(coh_ours - coh_scipy)))
    correlation = float(np.corrcoef(coh_ours, coh_scipy)[0, 1])

    return {
        "our_coherence": coh_ours,
        "scipy_coherence": coh_scipy,
        "frequencies": freqs_ours,
        "max_difference": max_diff,
        "correlation": correlation,
    }


# =============================================================================
# Imaginary Coherence Functions
# =============================================================================


def compute_imaginary_coherence(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    fs: float,
    nperseg: int = 256,
    noverlap: int | None = None,
    window: str = "hann",
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Compute imaginary coherence between two signals.

    Imaginary coherence uses only the imaginary part of the cross-spectrum,
    making it robust to volume conduction artifacts (zero-lag connections).

    Based on Nolte et al. (2004): "Identifying true brain interaction from EEG
    data using the imaginary part of coherency."

    Parameters
    ----------
    x : NDArray[np.floating]
        First input signal.
    y : NDArray[np.floating]
        Second input signal (same length as x).
    fs : float
        Sampling frequency in Hz.
    nperseg : int, optional
        Length of each segment for Welch's method. Default is 256.
    noverlap : int | None, optional
        Number of points to overlap between segments.
        If None, defaults to nperseg // 2.
    window : str, optional
        Window function to apply. Default is "hann".

    Returns
    -------
    frequencies : NDArray[np.floating]
        Array of frequency values in Hz.
    imcoh : NDArray[np.floating]
        Imaginary coherence values (-1 to +1).
        Positive: Y leads X. Negative: X leads Y.

    Examples
    --------
    >>> t = np.linspace(0, 2, 2000, endpoint=False)
    >>> x = np.sin(2 * np.pi * 10 * t)
    >>> y = np.sin(2 * np.pi * 10 * t + np.pi / 4)  # Phase lag
    >>> freqs, imcoh = compute_imaginary_coherence(x, y, fs=1000)
    >>> idx_10hz = np.argmin(np.abs(freqs - 10))
    >>> print(f"ImCoh at 10 Hz: {imcoh[idx_10hz]:.3f}")
    """
    if noverlap is None:
        noverlap = nperseg // 2

    # Compute cross-spectrum (complex)
    freqs, sxy = csd(x, y, fs=fs, nperseg=nperseg, noverlap=noverlap, window=window)

    # Compute power spectra
    _, sxx = welch(x, fs=fs, nperseg=nperseg, noverlap=noverlap, window=window)
    _, syy = welch(y, fs=fs, nperseg=nperseg, noverlap=noverlap, window=window)

    # Imaginary coherence = Im(Sxy) / sqrt(Sxx * Syy)
    imcoh = np.imag(sxy) / np.sqrt(sxx * syy)

    return freqs, imcoh


def compute_abs_imaginary_coherence(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    fs: float,
    nperseg: int = 256,
    noverlap: int | None = None,
    window: str = "hann",
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Compute absolute imaginary coherence (magnitude only).

    Same as compute_imaginary_coherence() but returns absolute values,
    discarding information about which signal leads.

    Parameters
    ----------
    x : NDArray[np.floating]
        First input signal.
    y : NDArray[np.floating]
        Second input signal.
    fs : float
        Sampling frequency in Hz.
    nperseg : int, optional
        Length of each segment. Default is 256.
    noverlap : int | None, optional
        Overlap between segments. Default is nperseg // 2.
    window : str, optional
        Window function. Default is "hann".

    Returns
    -------
    frequencies : NDArray[np.floating]
        Frequency values in Hz.
    abs_imcoh : NDArray[np.floating]
        Absolute imaginary coherence (0 to 1).

    Examples
    --------
    >>> t = np.linspace(0, 2, 2000, endpoint=False)
    >>> x = np.sin(2 * np.pi * 10 * t)
    >>> y = np.sin(2 * np.pi * 10 * t + np.pi / 4)
    >>> freqs, abs_imcoh = compute_abs_imaginary_coherence(x, y, fs=1000)
    """
    freqs, imcoh = compute_imaginary_coherence(x, y, fs, nperseg, noverlap, window)
    return freqs, np.abs(imcoh)


def compute_band_imaginary_coherence(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    fs: float,
    band: tuple[float, float] = (8.0, 13.0),
    nperseg: int = 256,
    noverlap: int | None = None,
    absolute: bool = False,
) -> float:
    """Compute imaginary coherence in a specific frequency band.

    Parameters
    ----------
    x : NDArray[np.floating]
        First input signal.
    y : NDArray[np.floating]
        Second input signal.
    fs : float
        Sampling frequency in Hz.
    band : tuple[float, float], optional
        Frequency band (low, high) in Hz. Default is (8, 13) for alpha band.
    nperseg : int, optional
        Length of each segment. Default is 256.
    noverlap : int | None, optional
        Overlap between segments. Default is nperseg // 2.
    absolute : bool, optional
        If True, return absolute value. Default is False.

    Returns
    -------
    imcoh_band : float
        Mean imaginary coherence in the specified band.

    Examples
    --------
    >>> # Alpha band imaginary coherence
    >>> imcoh_alpha = compute_band_imaginary_coherence(x, y, fs=500, band=(8, 13))
    """
    freqs, imcoh = compute_imaginary_coherence(x, y, fs, nperseg, noverlap)

    # Select frequency band
    band_mask = (freqs >= band[0]) & (freqs <= band[1])
    imcoh_band = np.mean(imcoh[band_mask])

    if absolute:
        return float(np.abs(imcoh_band))
    return float(imcoh_band)


def compute_all_band_imaginary_coherence(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    fs: float,
    bands: dict[str, tuple[float, float]] | None = None,
    nperseg: int = 256,
    noverlap: int | None = None,
    absolute: bool = False,
) -> dict[str, float]:
    """Compute imaginary coherence for multiple frequency bands.

    Parameters
    ----------
    x : NDArray[np.floating]
        First input signal.
    y : NDArray[np.floating]
        Second input signal.
    fs : float
        Sampling frequency in Hz.
    bands : dict[str, tuple[float, float]] | None, optional
        Dictionary mapping band names to (low, high) frequency tuples.
        If None, uses standard EEG bands.
    nperseg : int, optional
        Length of each segment. Default is 256.
    noverlap : int | None, optional
        Overlap between segments. Default is nperseg // 2.
    absolute : bool, optional
        If True, return absolute values. Default is False.

    Returns
    -------
    band_imcoh : dict[str, float]
        Imaginary coherence for each band.

    Examples
    --------
    >>> bands = {'alpha': (8, 13), 'beta': (13, 30)}
    >>> imcoh_bands = compute_all_band_imaginary_coherence(x, y, fs=500, bands=bands)
    >>> print(f"Alpha ImCoh: {imcoh_bands['alpha']:.3f}")
    """
    if bands is None:
        # Default EEG bands
        bands = {
            "delta": (1.0, 4.0),
            "theta": (4.0, 8.0),
            "alpha": (8.0, 13.0),
            "beta": (13.0, 30.0),
            "gamma": (30.0, 100.0),
        }

    band_imcoh = {}
    for band_name, band_range in bands.items():
        band_imcoh[band_name] = compute_band_imaginary_coherence(
            x, y, fs, band=band_range, nperseg=nperseg, noverlap=noverlap, absolute=absolute
        )

    return band_imcoh


def compute_imaginary_coherence_matrix(
    data: NDArray[np.floating],
    fs: float,
    band: tuple[float, float] | None = None,
    nperseg: int = 256,
    noverlap: int | None = None,
    absolute: bool = True,
) -> NDArray[np.floating]:
    """Compute imaginary coherence connectivity matrix.

    Parameters
    ----------
    data : NDArray[np.floating]
        Multi-channel data (n_channels, n_samples).
    fs : float
        Sampling frequency in Hz.
    band : tuple[float, float] | None, optional
        If provided, compute band-averaged imaginary coherence.
        If None, return full spectrum.
    nperseg : int, optional
        Length of each segment. Default is 256.
    noverlap : int | None, optional
        Overlap between segments. Default is nperseg // 2.
    absolute : bool, optional
        If True, return absolute values. Default is True.

    Returns
    -------
    matrix : NDArray[np.floating]
        Imaginary coherence matrix (n_channels, n_channels).
        Diagonal is zero.

    Examples
    --------
    >>> # 8-channel data
    >>> data = np.random.randn(8, 5000)
    >>> imcoh_matrix = compute_imaginary_coherence_matrix(data, fs=500, band=(8, 13))
    >>> print(f"Matrix shape: {imcoh_matrix.shape}")
    """
    n_channels = data.shape[0]
    matrix = np.zeros((n_channels, n_channels))

    for i in range(n_channels):
        for j in range(i + 1, n_channels):
            if band is not None:
                imcoh_val = compute_band_imaginary_coherence(
                    data[i], data[j], fs, band=band, nperseg=nperseg,
                    noverlap=noverlap, absolute=absolute
                )
            else:
                _, imcoh = compute_imaginary_coherence(
                    data[i], data[j], fs, nperseg=nperseg, noverlap=noverlap
                )
                imcoh_val = np.mean(np.abs(imcoh)) if absolute else np.mean(imcoh)

            matrix[i, j] = imcoh_val
            matrix[j, i] = imcoh_val

    return matrix


def compute_imaginary_coherence_hyperscanning(
    data_p1: NDArray[np.floating],
    data_p2: NDArray[np.floating],
    fs: float,
    band: tuple[float, float] | None = None,
    nperseg: int = 256,
    noverlap: int | None = None,
    absolute: bool = True,
) -> dict[str, NDArray[np.floating]]:
    """Compute imaginary coherence for hyperscanning (two-person) data.

    Returns within-person and between-person connectivity matrices.

    Parameters
    ----------
    data_p1 : NDArray[np.floating]
        Person 1 data (n_channels_p1, n_samples).
    data_p2 : NDArray[np.floating]
        Person 2 data (n_channels_p2, n_samples).
    fs : float
        Sampling frequency in Hz.
    band : tuple[float, float] | None, optional
        Frequency band for averaging. If None, uses full spectrum.
    nperseg : int, optional
        Length of each segment. Default is 256.
    noverlap : int | None, optional
        Overlap between segments. Default is nperseg // 2.
    absolute : bool, optional
        If True, return absolute values. Default is True.

    Returns
    -------
    result : dict[str, NDArray[np.floating]]
        Dictionary with keys:
        - 'within_p1': Within-person connectivity for P1
        - 'within_p2': Within-person connectivity for P2
        - 'between': Between-person connectivity (n_ch_p1, n_ch_p2)

    Examples
    --------
    >>> # 8 channels per person
    >>> data_p1 = np.random.randn(8, 10000)
    >>> data_p2 = np.random.randn(8, 10000)
    >>> result = compute_imaginary_coherence_hyperscanning(
    ...     data_p1, data_p2, fs=500, band=(8, 13)
    ... )
    >>> print(f"Between-person connectivity shape: {result['between'].shape}")
    """
    n_ch_p1 = data_p1.shape[0]
    n_ch_p2 = data_p2.shape[0]

    # Within-person connectivity
    within_p1 = compute_imaginary_coherence_matrix(
        data_p1, fs, band=band, nperseg=nperseg, noverlap=noverlap, absolute=absolute
    )
    within_p2 = compute_imaginary_coherence_matrix(
        data_p2, fs, band=band, nperseg=nperseg, noverlap=noverlap, absolute=absolute
    )

    # Between-person connectivity
    between = np.zeros((n_ch_p1, n_ch_p2))
    for i in range(n_ch_p1):
        for j in range(n_ch_p2):
            if band is not None:
                imcoh_val = compute_band_imaginary_coherence(
                    data_p1[i], data_p2[j], fs, band=band, nperseg=nperseg,
                    noverlap=noverlap, absolute=absolute
                )
            else:
                _, imcoh = compute_imaginary_coherence(
                    data_p1[i], data_p2[j], fs, nperseg=nperseg, noverlap=noverlap
                )
                imcoh_val = np.mean(np.abs(imcoh)) if absolute else np.mean(imcoh)

            between[i, j] = imcoh_val

    return {
        "within_p1": within_p1,
        "within_p2": within_p2,
        "between": between,
    }
