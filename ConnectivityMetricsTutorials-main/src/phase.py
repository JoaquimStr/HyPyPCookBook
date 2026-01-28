"""
Phase Analysis Functions for Circular Statistics and Synchronization.

This module provides functions for working with phase as a circular variable,
including wrapping, unwrapping, circular statistics, and phase synchronization
metrics commonly used in hyperscanning and connectivity analysis.

Functions
---------
wrap_phase : Wrap phase to [-π, π]
unwrap_phase : Remove artificial 2π discontinuities
compute_phase_difference : Compute wrapped phase difference
circular_mean : Compute circular mean using vector averaging
resultant_vector_length : Compute R, the concentration measure
circular_variance : Compute circular variance (1 - R)
circular_std : Compute circular standard deviation
plot_phase_polar_histogram : Create polar histogram (rose plot)
plot_phase_on_circle : Scatter plot of phases on unit circle
mask_low_amplitude_phase : Mask unreliable phase estimates
compute_plv_simple : Compute Phase Locking Value
"""

from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from numpy.typing import NDArray


def wrap_phase(phase: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Wrap phase values to the interval [-π, π].

    Parameters
    ----------
    phase : NDArray[np.floating]
        Phase values in radians (can be any range).

    Returns
    -------
    NDArray[np.floating]
        Wrapped phase values in [-π, π].

    Examples
    --------
    >>> import numpy as np
    >>> phases = np.array([0, np.pi, 2*np.pi, 3*np.pi])
    >>> wrap_phase(phases)
    array([ 0.        ,  3.14159265,  0.        , -3.14159265])
    """
    return np.angle(np.exp(1j * phase))


def unwrap_phase(phase: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Unwrap phase to remove artificial 2π discontinuities.

    This function assumes the underlying phase is continuous and
    removes jumps greater than π by adding appropriate multiples of 2π.

    Parameters
    ----------
    phase : NDArray[np.floating]
        Wrapped phase values in radians.

    Returns
    -------
    NDArray[np.floating]
        Unwrapped (continuous) phase values.

    Notes
    -----
    Unwrapping can fail with noisy signals where the true phase
    change between samples exceeds π. Always validate results visually.

    Examples
    --------
    >>> import numpy as np
    >>> # Phase that wraps from π to -π
    >>> wrapped = np.array([2.9, 3.1, -3.1, -2.9])
    >>> unwrap_phase(wrapped)
    array([2.9, 3.1, 3.18..., 3.38...])
    """
    return np.unwrap(phase)


def compute_phase_difference(
    phase1: NDArray[np.floating],
    phase2: NDArray[np.floating],
) -> NDArray[np.floating]:
    """
    Compute the wrapped phase difference between two phase time series.

    Parameters
    ----------
    phase1 : NDArray[np.floating]
        First phase time series in radians.
    phase2 : NDArray[np.floating]
        Second phase time series in radians.

    Returns
    -------
    NDArray[np.floating]
        Wrapped phase difference (phase1 - phase2) in [-π, π].

    Examples
    --------
    >>> import numpy as np
    >>> p1 = np.array([0, np.pi/2, np.pi])
    >>> p2 = np.array([np.pi/4, np.pi/4, np.pi/4])
    >>> compute_phase_difference(p1, p2)
    array([-0.78539816,  0.78539816,  2.35619449])
    """
    return wrap_phase(phase1 - phase2)


def circular_mean(phases: NDArray[np.floating]) -> float:
    """
    Compute the circular mean of phase values using vector averaging.

    The circular mean is computed by converting phases to unit vectors,
    averaging the vectors, and finding the angle of the resultant.

    Parameters
    ----------
    phases : NDArray[np.floating]
        Array of phase values in radians.

    Returns
    -------
    float
        Circular mean in radians, in the range [-π, π].

    Notes
    -----
    If phases are uniformly distributed (R ≈ 0), the circular mean
    is poorly defined. Check resultant_vector_length before interpreting.

    Examples
    --------
    >>> import numpy as np
    >>> # Phases clustered around 0
    >>> phases = np.array([-0.1, 0, 0.1, 0.05, -0.05])
    >>> circular_mean(phases)
    0.0
    """
    mean_x = np.mean(np.cos(phases))
    mean_y = np.mean(np.sin(phases))
    return float(np.arctan2(mean_y, mean_x))


def resultant_vector_length(phases: NDArray[np.floating]) -> float:
    """
    Compute the resultant vector length R (phase concentration).

    R measures how concentrated the phases are around their circular mean.
    R = 1 means all phases are identical; R = 0 means uniform distribution.

    Parameters
    ----------
    phases : NDArray[np.floating]
        Array of phase values in radians.

    Returns
    -------
    float
        Resultant vector length in [0, 1].

    Notes
    -----
    R is equivalent to the Phase Locking Value (PLV) when computed
    on phase differences between two signals.

    Examples
    --------
    >>> import numpy as np
    >>> # Perfectly aligned phases
    >>> phases = np.array([0, 0, 0, 0])
    >>> resultant_vector_length(phases)
    1.0

    >>> # Uniformly distributed phases
    >>> phases = np.linspace(-np.pi, np.pi, 100)
    >>> r = resultant_vector_length(phases)
    >>> r < 0.1
    True
    """
    mean_x = np.mean(np.cos(phases))
    mean_y = np.mean(np.sin(phases))
    return float(np.sqrt(mean_x**2 + mean_y**2))


def circular_variance(phases: NDArray[np.floating]) -> float:
    """
    Compute the circular variance of phase values.

    Circular variance is defined as V = 1 - R, where R is the
    resultant vector length. V = 0 means no variance (all phases
    identical); V = 1 means maximum variance (uniform distribution).

    Parameters
    ----------
    phases : NDArray[np.floating]
        Array of phase values in radians.

    Returns
    -------
    float
        Circular variance in [0, 1].

    Examples
    --------
    >>> import numpy as np
    >>> # Identical phases: no variance
    >>> phases = np.array([0, 0, 0, 0])
    >>> circular_variance(phases)
    0.0
    """
    return 1.0 - resultant_vector_length(phases)


def circular_std(phases: NDArray[np.floating]) -> float:
    """
    Compute the circular standard deviation.

    Defined as sqrt(-2 * ln(R)), where R is the resultant vector length.
    This approximates the angular dispersion in radians.

    Parameters
    ----------
    phases : NDArray[np.floating]
        Array of phase values in radians.

    Returns
    -------
    float
        Circular standard deviation in radians.

    Notes
    -----
    When R → 0 (uniform distribution), circular_std → ∞.
    A small epsilon is added to R to avoid numerical issues.

    Examples
    --------
    >>> import numpy as np
    >>> # Highly concentrated phases
    >>> phases = np.array([0, 0.01, -0.01, 0.02])
    >>> std = circular_std(phases)
    >>> std < 0.1
    True
    """
    r = resultant_vector_length(phases)
    # Add small epsilon to avoid log(0)
    r = max(r, 1e-10)
    return float(np.sqrt(-2 * np.log(r)))


def plot_phase_polar_histogram(
    phases: NDArray[np.floating],
    n_bins: int = 24,
    ax: Optional[Axes] = None,
    color: str = "#3498DB",
    alpha: float = 0.7,
) -> Axes:
    """
    Create a polar histogram (rose plot) of phase values.

    Parameters
    ----------
    phases : NDArray[np.floating]
        Array of phase values in radians.
    n_bins : int, optional
        Number of angular bins (default: 24).
    ax : Axes, optional
        Matplotlib polar axes to plot on. If None, creates new figure.
    color : str, optional
        Bar color (default: "#3498DB").
    alpha : float, optional
        Bar transparency (default: 0.7).

    Returns
    -------
    Axes
        The matplotlib axes with the polar histogram.

    Examples
    --------
    >>> import numpy as np
    >>> phases = np.random.vonmises(0, 2, 100)
    >>> ax = plot_phase_polar_histogram(phases)
    """
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(6, 6), subplot_kw={"projection": "polar"})

    # Create histogram
    bin_edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    counts, _ = np.histogram(phases, bins=bin_edges)

    # Normalize to probability
    counts = counts / counts.sum()

    # Width of each bar
    width = 2 * np.pi / n_bins

    # Center of each bin
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Plot bars
    ax.bar(
        bin_centers,
        counts,
        width=width,
        color=color,
        alpha=alpha,
        edgecolor="black",
        linewidth=0.5,
    )

    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)

    return ax


def plot_phase_on_circle(
    phases: NDArray[np.floating],
    ax: Optional[Axes] = None,
    show_mean: bool = True,
    color: str = "#3498DB",
    mean_color: str = "#E74C3C",
    alpha: float = 0.6,
    marker_size: int = 50,
) -> Axes:
    """
    Plot phases as points on the unit circle.

    Parameters
    ----------
    phases : NDArray[np.floating]
        Array of phase values in radians.
    ax : Axes, optional
        Matplotlib polar axes to plot on. If None, creates new figure.
    show_mean : bool, optional
        Whether to show the circular mean as an arrow (default: True).
    color : str, optional
        Point color (default: "#3498DB").
    mean_color : str, optional
        Mean vector color (default: "#E74C3C").
    alpha : float, optional
        Point transparency (default: 0.6).
    marker_size : int, optional
        Size of scatter points (default: 50).

    Returns
    -------
    Axes
        The matplotlib axes with the scatter plot.

    Examples
    --------
    >>> import numpy as np
    >>> phases = np.random.vonmises(np.pi/4, 3, 50)
    >>> ax = plot_phase_on_circle(phases, show_mean=True)
    """
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(6, 6), subplot_kw={"projection": "polar"})

    # Plot phases on unit circle
    ax.scatter(
        phases,
        np.ones(len(phases)),
        color=color,
        alpha=alpha,
        s=marker_size,
        edgecolor="white",
        linewidth=0.5,
    )

    if show_mean:
        mean_angle = circular_mean(phases)
        r = resultant_vector_length(phases)

        # Draw mean vector
        ax.annotate(
            "",
            xy=(mean_angle, r),
            xytext=(0, 0),
            arrowprops={"arrowstyle": "->", "color": mean_color, "lw": 2},
        )

    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_ylim(0, 1.2)

    return ax


def mask_low_amplitude_phase(
    phase: NDArray[np.floating],
    amplitude: NDArray[np.floating],
    threshold_percentile: float = 20.0,
) -> NDArray[np.floating]:
    """
    Mask phase values where amplitude is below a threshold.

    Phase estimates are unreliable when the signal amplitude is low
    (dominated by noise). This function sets phase values to NaN
    where the amplitude is below a percentile threshold.

    Parameters
    ----------
    phase : NDArray[np.floating]
        Phase time series in radians.
    amplitude : NDArray[np.floating]
        Amplitude envelope (same length as phase).
    threshold_percentile : float, optional
        Percentile of amplitude below which to mask (default: 20).

    Returns
    -------
    NDArray[np.floating]
        Phase array with NaN values where amplitude was low.

    Examples
    --------
    >>> import numpy as np
    >>> phase = np.array([0, 1, 2, 3, 4])
    >>> amplitude = np.array([1.0, 0.1, 0.5, 0.05, 0.8])
    >>> masked = mask_low_amplitude_phase(phase, amplitude, threshold_percentile=25)
    >>> np.isnan(masked[1]) and np.isnan(masked[3])
    True
    """
    threshold = np.percentile(amplitude, threshold_percentile)
    masked_phase = phase.copy().astype(float)
    masked_phase[amplitude < threshold] = np.nan
    return masked_phase


def compute_plv_simple(
    phase1: NDArray[np.floating],
    phase2: NDArray[np.floating],
) -> float:
    """
    Compute the Phase Locking Value (PLV) between two phase time series.

    PLV is the resultant vector length of the phase differences,
    measuring how consistent the phase relationship is over time.
    PLV = 1 means perfect phase locking; PLV = 0 means no locking.

    Parameters
    ----------
    phase1 : NDArray[np.floating]
        First phase time series in radians.
    phase2 : NDArray[np.floating]
        Second phase time series in radians.

    Returns
    -------
    float
        Phase Locking Value in [0, 1].

    Notes
    -----
    This is a simplified PLV implementation. For robust connectivity
    analysis, see the dedicated PLV notebook (G01) which covers
    statistical significance, trial-based computation, and confounds.

    Examples
    --------
    >>> import numpy as np
    >>> # Perfectly locked (constant phase difference)
    >>> t = np.linspace(0, 1, 1000)
    >>> p1 = 2 * np.pi * 10 * t
    >>> p2 = 2 * np.pi * 10 * t + np.pi/4
    >>> plv = compute_plv_simple(p1, p2)
    >>> plv > 0.99
    True
    """
    phase_diff = phase1 - phase2
    return float(np.abs(np.mean(np.exp(1j * phase_diff))))
