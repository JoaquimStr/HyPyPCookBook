# ============================================================================
# Volume Conduction Simulation and Robust Connectivity Metrics
# ============================================================================
"""
Functions for simulating volume conduction effects and computing
connectivity metrics that are robust to volume conduction artifacts.

Volume conduction is the instantaneous spread of electrical activity through
conductive brain tissue, causing spurious correlations between EEG electrodes.
"""

from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray
import scipy.signal


def simulate_volume_conduction(
    source: NDArray[np.floating],
    weights: NDArray[np.floating],
    noise_level: float = 0.1,
    seed: Optional[int] = None
) -> NDArray[np.floating]:
    """
    Simulate volume conduction from a single source to multiple electrodes.
    
    Each electrode receives a weighted copy of the source signal plus
    independent noise. This models the instantaneous spread of electrical
    activity through conductive tissue.
    
    Parameters
    ----------
    source : NDArray[np.floating]
        Source signal, shape (n_samples,).
    weights : NDArray[np.floating]
        Weight for each electrode, shape (n_electrodes,).
        Higher weight = electrode closer to source.
    noise_level : float, optional
        Standard deviation of added Gaussian noise. Default is 0.1.
    seed : Optional[int], optional
        Random seed for reproducibility. Default is None.
        
    Returns
    -------
    NDArray[np.floating]
        Electrode signals, shape (n_electrodes, n_samples).
        
    Examples
    --------
    >>> source = np.sin(2 * np.pi * 10 * np.arange(0, 1, 1/256))
    >>> weights = np.array([1.0, 0.7, 0.3])
    >>> electrodes = simulate_volume_conduction(source, weights, noise_level=0.1)
    >>> electrodes.shape
    (3, 256)
    """
    if seed is not None:
        np.random.seed(seed)
    
    n_electrodes = len(weights)
    n_samples = len(source)
    
    # Each electrode receives weighted source + independent noise
    electrodes = np.zeros((n_electrodes, n_samples))
    for i, w in enumerate(weights):
        electrodes[i] = w * source + noise_level * np.random.randn(n_samples)
    
    return electrodes


def create_mixing_matrix(
    source_positions: NDArray[np.floating],
    electrode_positions: NDArray[np.floating],
    falloff: float = 2.0
) -> NDArray[np.floating]:
    """
    Create a mixing matrix based on distance between sources and electrodes.
    
    The mixing weight decreases with distance according to an inverse
    power law, modeling the spatial spread of electrical fields.
    
    Parameters
    ----------
    source_positions : NDArray[np.floating]
        Source positions, shape (n_sources, 2) for 2D or (n_sources, 3) for 3D.
    electrode_positions : NDArray[np.floating]
        Electrode positions, shape (n_electrodes, 2) or (n_electrodes, 3).
    falloff : float, optional
        Power law exponent for distance falloff. Default is 2.0.
        Higher values = faster falloff with distance.
        
    Returns
    -------
    NDArray[np.floating]
        Mixing matrix, shape (n_electrodes, n_sources).
        Entry [i, j] is the weight from source j to electrode i.
        
    Examples
    --------
    >>> sources = np.array([[0.0, 0.5], [0.5, 0.5]])
    >>> electrodes = np.array([[0.0, 0.8], [0.3, 0.8], [0.6, 0.8]])
    >>> mixing = create_mixing_matrix(sources, electrodes)
    >>> mixing.shape
    (3, 2)
    """
    n_electrodes = len(electrode_positions)
    n_sources = len(source_positions)
    
    mixing = np.zeros((n_electrodes, n_sources))
    
    for i, elec_pos in enumerate(electrode_positions):
        for j, src_pos in enumerate(source_positions):
            distance = np.linalg.norm(elec_pos - src_pos)
            # Inverse power law with small offset to avoid division by zero
            mixing[i, j] = 1.0 / (distance + 0.1) ** falloff
    
    # Normalize each electrode's weights to sum to 1
    mixing = mixing / mixing.sum(axis=1, keepdims=True)
    
    return mixing


def apply_mixing(
    sources: NDArray[np.floating],
    mixing_matrix: NDArray[np.floating],
    noise_level: float = 0.1,
    seed: Optional[int] = None
) -> NDArray[np.floating]:
    """
    Apply a mixing matrix to source signals to simulate electrode recordings.
    
    This models volume conduction where each electrode receives a weighted
    sum of all source signals.
    
    Parameters
    ----------
    sources : NDArray[np.floating]
        Source signals, shape (n_sources, n_samples).
    mixing_matrix : NDArray[np.floating]
        Mixing matrix, shape (n_electrodes, n_sources).
    noise_level : float, optional
        Standard deviation of added noise. Default is 0.1.
    seed : Optional[int], optional
        Random seed for reproducibility. Default is None.
        
    Returns
    -------
    NDArray[np.floating]
        Electrode signals, shape (n_electrodes, n_samples).
        
    Examples
    --------
    >>> sources = np.random.randn(3, 256)  # 3 sources, 256 samples
    >>> mixing = np.array([[0.8, 0.15, 0.05], [0.1, 0.8, 0.1]])  # 2 electrodes
    >>> electrodes = apply_mixing(sources, mixing)
    >>> electrodes.shape
    (2, 256)
    """
    if seed is not None:
        np.random.seed(seed)
    
    n_electrodes = mixing_matrix.shape[0]
    n_samples = sources.shape[1]
    
    # Apply mixing: electrodes = mixing @ sources
    electrodes = mixing_matrix @ sources
    
    # Add independent noise to each electrode
    electrodes += noise_level * np.random.randn(n_electrodes, n_samples)
    
    return electrodes


def compute_cross_correlation(
    signal_1: NDArray[np.floating],
    signal_2: NDArray[np.floating],
    max_lag: Optional[int] = None
) -> Tuple[NDArray[np.floating], NDArray[np.integer]]:
    """
    Compute normalized cross-correlation between two signals.
    
    Cross-correlation measures similarity as a function of time lag.
    Volume conduction produces maximum correlation at zero lag.
    
    Parameters
    ----------
    signal_1 : NDArray[np.floating]
        First input signal.
    signal_2 : NDArray[np.floating]
        Second input signal (same length as signal_1).
    max_lag : Optional[int], optional
        Maximum lag to compute (in samples). Default is len(signal_1) // 2.
        
    Returns
    -------
    Tuple[NDArray[np.floating], NDArray[np.integer]]
        - Cross-correlation values, normalized to [-1, 1].
        - Corresponding lag values in samples.
        
    Examples
    --------
    >>> t = np.arange(0, 1, 1/256)
    >>> s1 = np.sin(2 * np.pi * 10 * t)
    >>> s2 = np.sin(2 * np.pi * 10 * t + 0.1)  # Slightly delayed
    >>> xcorr, lags = compute_cross_correlation(s1, s2)
    >>> peak_lag = lags[np.argmax(xcorr)]
    """
    if max_lag is None:
        max_lag = len(signal_1) // 2
    
    # Normalize signals
    s1_norm = (signal_1 - np.mean(signal_1)) / (np.std(signal_1) + 1e-10)
    s2_norm = (signal_2 - np.mean(signal_2)) / (np.std(signal_2) + 1e-10)
    
    # Full cross-correlation
    xcorr_full = scipy.signal.correlate(s1_norm, s2_norm, mode='full')
    xcorr_full /= len(signal_1)  # Normalize
    
    # Extract centered portion
    center = len(xcorr_full) // 2
    xcorr = xcorr_full[center - max_lag:center + max_lag + 1]
    lags = np.arange(-max_lag, max_lag + 1)
    
    return xcorr, lags


def compute_pli(
    signal_1: NDArray[np.floating],
    signal_2: NDArray[np.floating]
) -> np.floating:
    """
    Compute Phase Lag Index between two signals.
    
    PLI measures the asymmetry of the phase difference distribution.
    It is robust to volume conduction because zero-lag mixing produces
    symmetric phase differences (around 0), leading to PLI = 0.
    
    Parameters
    ----------
    signal_1 : NDArray[np.floating]
        First input signal.
    signal_2 : NDArray[np.floating]
        Second input signal.
        
    Returns
    -------
    np.floating
        Phase Lag Index value between 0 and 1.
        0 = no consistent lead/lag (or volume conduction)
        1 = perfect consistent lead/lag relationship
        
    Notes
    -----
    PLI is defined as:
        PLI = |mean(sign(Δφ))|
    
    where Δφ is the instantaneous phase difference.
    
    References
    ----------
    Stam, C. J., Nolte, G., & Daffertshofer, A. (2007). Phase lag index:
    assessment of functional connectivity from multi channel EEG and MEG
    with diminished bias from common sources. Human brain mapping, 28(11),
    1178-1193.
    
    Examples
    --------
    >>> t = np.arange(0, 2, 1/256)
    >>> s1 = np.sin(2 * np.pi * 10 * t)
    >>> s2 = np.sin(2 * np.pi * 10 * t + np.pi/4)  # 45° phase lag
    >>> pli = compute_pli(s1, s2)
    >>> pli > 0.8  # Should be high for consistent lag
    True
    """
    # Get instantaneous phases via Hilbert transform
    analytic_1 = scipy.signal.hilbert(signal_1)
    analytic_2 = scipy.signal.hilbert(signal_2)
    
    phase_1 = np.angle(analytic_1)
    phase_2 = np.angle(analytic_2)
    
    # Compute phase difference
    phase_diff = phase_2 - phase_1
    
    # Wrap to [-pi, pi]
    phase_diff = np.arctan2(np.sin(phase_diff), np.cos(phase_diff))
    
    # PLI = |mean(sign(phase_diff))|
    pli = np.abs(np.mean(np.sign(phase_diff)))
    
    return pli


def simulate_volume_conduction_scenario(
    fs: int = 256,
    duration: float = 5.0,
    freq: float = 10.0,
    mixing_strength: float = 0.5,
    noise_level: float = 0.1,
    seed: Optional[int] = None
) -> Tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """
    Simulate a volume conduction scenario with one source and two electrodes.
    
    Both electrodes receive the same source signal with different weights,
    creating spurious connectivity that is not due to neural communication.
    
    Parameters
    ----------
    fs : int, optional
        Sampling frequency in Hz. Default is 256.
    duration : float, optional
        Duration in seconds. Default is 5.0.
    freq : float, optional
        Frequency of the source signal in Hz. Default is 10.0.
    mixing_strength : float, optional
        How much the source contributes to electrode 2 (0 to 1). Default is 0.5.
    noise_level : float, optional
        Standard deviation of added noise. Default is 0.1.
    seed : Optional[int], optional
        Random seed for reproducibility. Default is None.
        
    Returns
    -------
    Tuple[NDArray, NDArray, NDArray]
        - Time vector.
        - Electrode 1 signal.
        - Electrode 2 signal.
        
    Examples
    --------
    >>> t, e1, e2 = simulate_volume_conduction_scenario(mixing_strength=0.8)
    >>> # e1 and e2 will show high PLV but low PLI
    """
    if seed is not None:
        np.random.seed(seed)
    
    t = np.arange(0, duration, 1/fs)
    
    # Source signal
    source = np.sin(2 * np.pi * freq * t)
    
    # Electrode signals (both receive the same source, different weights)
    electrode_1 = source + noise_level * np.random.randn(len(t))
    electrode_2 = mixing_strength * source + noise_level * np.random.randn(len(t))
    
    return t, electrode_1, electrode_2


def simulate_true_connectivity_scenario(
    fs: int = 256,
    duration: float = 5.0,
    freq: float = 10.0,
    phase_lag: float = np.pi/4,
    noise_level: float = 0.1,
    seed: Optional[int] = None
) -> Tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """
    Simulate a true connectivity scenario with consistent phase lag.
    
    The two signals have a consistent phase relationship that represents
    genuine neural communication, not volume conduction artifact.
    
    Parameters
    ----------
    fs : int, optional
        Sampling frequency in Hz. Default is 256.
    duration : float, optional
        Duration in seconds. Default is 5.0.
    freq : float, optional
        Frequency of the oscillation in Hz. Default is 10.0.
    phase_lag : float, optional
        Phase lag between signals in radians. Default is π/4 (45°).
    noise_level : float, optional
        Standard deviation of added noise. Default is 0.1.
    seed : Optional[int], optional
        Random seed for reproducibility. Default is None.
        
    Returns
    -------
    Tuple[NDArray, NDArray, NDArray]
        - Time vector.
        - Signal 1.
        - Signal 2 (phase-lagged relative to signal 1).
        
    Examples
    --------
    >>> t, s1, s2 = simulate_true_connectivity_scenario(phase_lag=np.pi/4)
    >>> # Both PLV and PLI should be high
    """
    if seed is not None:
        np.random.seed(seed)
    
    t = np.arange(0, duration, 1/fs)
    
    # Two signals with consistent phase lag
    signal_1 = np.sin(2 * np.pi * freq * t) + noise_level * np.random.randn(len(t))
    signal_2 = np.sin(2 * np.pi * freq * t + phase_lag) + noise_level * np.random.randn(len(t))
    
    return t, signal_1, signal_2
