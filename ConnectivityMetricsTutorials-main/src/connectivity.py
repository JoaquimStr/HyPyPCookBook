"""
Connectivity matrix computation and visualization utilities.

This module provides functions for computing, validating, and visualizing
connectivity matrices for multi-channel EEG and hyperscanning data.

Functions
---------
Matrix Operations:
    get_n_pairs : Compute number of unique channel pairs
    get_pair_indices : Get list of unique pair indices
    compute_connectivity_matrix : Compute full connectivity matrix
    get_upper_triangle_values : Extract upper triangle values
    upper_triangle_to_matrix : Reconstruct matrix from upper triangle

Region Analysis:
    define_channel_groups : Map channels to region groups
    compute_region_connectivity : Average connectivity by region

Hyperscanning:
    compute_hyperscanning_connectivity : Compute full hyperscanning analysis
    extract_between_participant_matrix : Extract inter-brain block

Visualization:
    plot_connectivity_matrix : Heatmap visualization
    plot_circular_connectivity : Network diagram
    plot_hyperscanning_matrix : Annotated hyperscanning heatmap
    plot_hyperscanning_circular : Two-brain circular plot

Global Metrics:
    compute_global_connectivity : Mean connectivity
    compute_connection_density : Proportion above threshold
    compute_hyperscanning_ratio : Between/within ratio

Validation:
    validate_connectivity_matrix : Check matrix properties
    get_matrix_statistics : Summary statistics
"""

from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.signal import butter, filtfilt, hilbert

from src.colors import (
    PRIMARY_BLUE,
    PRIMARY_GREEN,
    PRIMARY_RED,
    SECONDARY_PURPLE,
    SUBJECT_1,
    SUBJECT_2,
)


# =============================================================================
# Helper Functions
# =============================================================================


def _bandpass_filter(
    data: NDArray[np.floating],
    lowcut: float,
    highcut: float,
    fs: float,
    order: int = 4
) -> NDArray[np.floating]:
    """
    Apply bandpass filter to data.
    
    Parameters
    ----------
    data : NDArray[np.floating]
        Input signal.
    lowcut : float
        Low cutoff frequency in Hz.
    highcut : float
        High cutoff frequency in Hz.
    fs : float
        Sampling frequency in Hz.
    order : int, optional
        Filter order. Default is 4.
        
    Returns
    -------
    NDArray[np.floating]
        Filtered signal.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)


def _compute_plv_pair(
    signal_1: NDArray[np.floating],
    signal_2: NDArray[np.floating]
) -> float:
    """
    Compute Phase Locking Value between two signals.
    
    Parameters
    ----------
    signal_1 : NDArray[np.floating]
        First signal (should be bandpass filtered).
    signal_2 : NDArray[np.floating]
        Second signal (should be bandpass filtered).
        
    Returns
    -------
    float
        PLV value between 0 and 1.
    """
    # Extract phases using Hilbert transform
    phase_1 = np.angle(hilbert(signal_1))
    phase_2 = np.angle(hilbert(signal_2))
    
    # Compute phase difference
    phase_diff = phase_1 - phase_2
    
    # PLV = magnitude of mean phase difference vector
    plv = np.abs(np.mean(np.exp(1j * phase_diff)))
    
    return float(plv)


# =============================================================================
# Matrix Operations
# =============================================================================


def get_n_pairs(n_channels: int) -> int:
    """
    Compute the number of unique channel pairs.
    
    Parameters
    ----------
    n_channels : int
        Number of channels.
        
    Returns
    -------
    int
        Number of unique pairs: n(n-1)/2
        
    Examples
    --------
    >>> get_n_pairs(6)
    15
    >>> get_n_pairs(64)
    2016
    """
    return n_channels * (n_channels - 1) // 2


def get_pair_indices(n_channels: int) -> List[Tuple[int, int]]:
    """
    Get list of all unique channel pair indices.
    
    Parameters
    ----------
    n_channels : int
        Number of channels.
        
    Returns
    -------
    List[Tuple[int, int]]
        List of (i, j) tuples where i < j.
        
    Examples
    --------
    >>> get_pair_indices(3)
    [(0, 1), (0, 2), (1, 2)]
    """
    pairs = []
    for i in range(n_channels):
        for j in range(i + 1, n_channels):
            pairs.append((i, j))
    return pairs


def compute_connectivity_matrix(
    data: NDArray[np.floating],
    fs: float,
    band: Tuple[float, float],
    metric: str = "plv"
) -> NDArray[np.floating]:
    """
    Compute connectivity matrix for multi-channel data.
    
    Parameters
    ----------
    data : NDArray[np.floating]
        Multi-channel data, shape (n_channels, n_samples).
    fs : float
        Sampling frequency in Hz.
    band : Tuple[float, float]
        Frequency band (low, high) in Hz.
    metric : str, optional
        Connectivity metric. Currently only "plv" supported. Default is "plv".
        
    Returns
    -------
    NDArray[np.floating]
        Connectivity matrix, shape (n_channels, n_channels).
        Diagonal is NaN. Matrix is symmetric.
        
    Raises
    ------
    ValueError
        If unsupported metric is specified.
        
    Examples
    --------
    >>> data = np.random.randn(6, 1000)
    >>> matrix = compute_connectivity_matrix(data, fs=256, band=(8, 13))
    >>> matrix.shape
    (6, 6)
    """
    if metric != "plv":
        raise ValueError(f"Unsupported metric: {metric}. Currently only 'plv' supported.")
    
    n_channels = data.shape[0]
    matrix = np.zeros((n_channels, n_channels))
    
    # Bandpass filter all channels
    data_filtered = np.array([
        _bandpass_filter(ch, band[0], band[1], fs) for ch in data
    ])
    
    # Compute PLV for all pairs
    for i in range(n_channels):
        for j in range(i + 1, n_channels):
            plv = _compute_plv_pair(data_filtered[i], data_filtered[j])
            matrix[i, j] = plv
            matrix[j, i] = plv  # Symmetric
    
    # Diagonal = NaN
    np.fill_diagonal(matrix, np.nan)
    
    return matrix


def get_upper_triangle_values(
    matrix: NDArray[np.floating],
    k: int = 1
) -> NDArray[np.floating]:
    """
    Extract upper triangle values from a matrix.
    
    Parameters
    ----------
    matrix : NDArray[np.floating]
        Square matrix, shape (n, n).
    k : int, optional
        Diagonal offset. k=1 excludes the main diagonal (default).
        k=0 includes the diagonal.
        
    Returns
    -------
    NDArray[np.floating]
        1D array of upper triangle values.
        
    Notes
    -----
    For a symmetric matrix, this extracts all unique values.
    Number of values = n(n-1)/2 when k=1.
    
    Examples
    --------
    >>> matrix = np.array([[1, 2, 3], [2, 4, 5], [3, 5, 6]])
    >>> get_upper_triangle_values(matrix)
    array([2, 3, 5])
    """
    n = matrix.shape[0]
    indices = np.triu_indices(n, k=k)
    return matrix[indices]


def upper_triangle_to_matrix(
    values: NDArray[np.floating],
    n_channels: int,
    fill_diagonal: float = np.nan
) -> NDArray[np.floating]:
    """
    Reconstruct symmetric matrix from upper triangle values.
    
    Parameters
    ----------
    values : NDArray[np.floating]
        1D array of upper triangle values.
    n_channels : int
        Number of channels (matrix will be n_channels × n_channels).
    fill_diagonal : float, optional
        Value to fill diagonal. Default is NaN.
        
    Returns
    -------
    NDArray[np.floating]
        Symmetric matrix, shape (n_channels, n_channels).
        
    Raises
    ------
    ValueError
        If number of values doesn't match expected n(n-1)/2.
        
    Examples
    --------
    >>> values = np.array([0.5, 0.3, 0.8])
    >>> matrix = upper_triangle_to_matrix(values, 3)
    >>> matrix.shape
    (3, 3)
    """
    expected_n_values = n_channels * (n_channels - 1) // 2
    if len(values) != expected_n_values:
        raise ValueError(
            f"Expected {expected_n_values} values for {n_channels} channels, "
            f"got {len(values)}"
        )
    
    # Create empty matrix
    matrix = np.zeros((n_channels, n_channels))
    
    # Fill upper triangle
    indices = np.triu_indices(n_channels, k=1)
    matrix[indices] = values
    
    # Make symmetric
    matrix = matrix + matrix.T
    
    # Fill diagonal
    np.fill_diagonal(matrix, fill_diagonal)
    
    return matrix


# =============================================================================
# Region Analysis
# =============================================================================


def define_channel_groups(
    channel_names: List[str],
    group_definitions: Dict[str, List[str]]
) -> Dict[str, List[int]]:
    """
    Map channel group names to their indices.
    
    Parameters
    ----------
    channel_names : List[str]
        List of all channel names.
    group_definitions : Dict[str, List[str]]
        Mapping of group names to channel names.
        e.g., {"frontal": ["F3", "Fz", "F4"], "parietal": ["P3", "Pz", "P4"]}
        
    Returns
    -------
    Dict[str, List[int]]
        Mapping of group names to channel indices.
        
    Raises
    ------
    ValueError
        If a channel name in group_definitions is not found.
        
    Examples
    --------
    >>> channel_names = ['F3', 'F4', 'P3', 'P4']
    >>> groups = {"frontal": ["F3", "F4"], "parietal": ["P3", "P4"]}
    >>> define_channel_groups(channel_names, groups)
    {'frontal': [0, 1], 'parietal': [2, 3]}
    """
    result = {}
    for group_name, channels in group_definitions.items():
        indices = []
        for ch in channels:
            if ch not in channel_names:
                raise ValueError(f"Channel '{ch}' not found in channel_names")
            indices.append(channel_names.index(ch))
        result[group_name] = indices
    return result


def compute_region_connectivity(
    matrix: NDArray[np.floating],
    channel_groups: Dict[str, List[int]]
) -> Tuple[NDArray[np.floating], List[str]]:
    """
    Compute average connectivity between brain regions.
    
    Parameters
    ----------
    matrix : NDArray[np.floating]
        Full connectivity matrix, shape (n_channels, n_channels).
    channel_groups : Dict[str, List[int]]
        Mapping of group names to channel indices.
        
    Returns
    -------
    Tuple[NDArray[np.floating], List[str]]
        - Region connectivity matrix, shape (n_regions, n_regions)
        - List of region names
        
    Notes
    -----
    - Diagonal = mean connectivity WITHIN a region
    - Off-diagonal = mean connectivity BETWEEN regions
    
    Examples
    --------
    >>> matrix = np.random.rand(6, 6)
    >>> groups = {"A": [0, 1], "B": [2, 3], "C": [4, 5]}
    >>> region_matrix, names = compute_region_connectivity(matrix, groups)
    >>> region_matrix.shape
    (3, 3)
    """
    region_names = list(channel_groups.keys())
    n_regions = len(region_names)
    
    region_matrix = np.zeros((n_regions, n_regions))
    
    for i, region_i in enumerate(region_names):
        for j, region_j in enumerate(region_names):
            indices_i = channel_groups[region_i]
            indices_j = channel_groups[region_j]
            
            # Get all pairwise values between these regions
            values = []
            for idx_i in indices_i:
                for idx_j in indices_j:
                    if i == j and idx_i == idx_j:
                        # Skip self-connections within same region
                        continue
                    val = matrix[idx_i, idx_j]
                    if not np.isnan(val):
                        values.append(val)
            
            if values:
                region_matrix[i, j] = np.mean(values)
            else:
                region_matrix[i, j] = np.nan
    
    return region_matrix, region_names


# =============================================================================
# Hyperscanning
# =============================================================================


def compute_hyperscanning_connectivity(
    data_p1: NDArray[np.floating],
    data_p2: NDArray[np.floating],
    fs: float,
    band: Tuple[float, float],
    metric: str = "plv"
) -> Dict[str, NDArray[np.floating]]:
    """
    Compute connectivity matrices for hyperscanning data.
    
    Parameters
    ----------
    data_p1 : NDArray[np.floating]
        Participant 1 data, shape (n_channels, n_samples).
    data_p2 : NDArray[np.floating]
        Participant 2 data, shape (n_channels, n_samples).
    fs : float
        Sampling frequency in Hz.
    band : Tuple[float, float]
        Frequency band (low, high) in Hz.
    metric : str, optional
        Connectivity metric. Default is "plv".
        
    Returns
    -------
    Dict[str, NDArray[np.floating]]
        Dictionary with keys:
        - "within_p1": (n_ch, n_ch) connectivity within P1
        - "within_p2": (n_ch, n_ch) connectivity within P2
        - "between": (n_ch, n_ch) connectivity P1→P2
        - "full": (2*n_ch, 2*n_ch) complete hyperscanning matrix
        
    Raises
    ------
    ValueError
        If participants have different numbers of channels.
        
    Examples
    --------
    >>> data_p1 = np.random.randn(4, 1000)
    >>> data_p2 = np.random.randn(4, 1000)
    >>> results = compute_hyperscanning_connectivity(data_p1, data_p2, 256, (8, 13))
    >>> results['full'].shape
    (8, 8)
    """
    n_ch_p1 = data_p1.shape[0]
    n_ch_p2 = data_p2.shape[0]
    
    if n_ch_p1 != n_ch_p2:
        raise ValueError(
            f"Both participants must have same number of channels. "
            f"Got {n_ch_p1} and {n_ch_p2}."
        )
    
    n_ch = n_ch_p1
    
    # Compute within-participant connectivity
    within_p1 = compute_connectivity_matrix(data_p1, fs, band, metric)
    within_p2 = compute_connectivity_matrix(data_p2, fs, band, metric)
    
    # Compute between-participant connectivity
    # Filter all data first
    data_p1_filt = np.array([
        _bandpass_filter(ch, band[0], band[1], fs) for ch in data_p1
    ])
    data_p2_filt = np.array([
        _bandpass_filter(ch, band[0], band[1], fs) for ch in data_p2
    ])
    
    between = np.zeros((n_ch, n_ch))
    for i in range(n_ch):
        for j in range(n_ch):
            between[i, j] = _compute_plv_pair(data_p1_filt[i], data_p2_filt[j])
    
    # Build full matrix
    n_total = 2 * n_ch
    full = np.zeros((n_total, n_total))
    
    # Fill quadrants
    full[:n_ch, :n_ch] = within_p1              # Top-left
    full[n_ch:, n_ch:] = within_p2              # Bottom-right
    full[:n_ch, n_ch:] = between                # Top-right
    full[n_ch:, :n_ch] = between.T              # Bottom-left
    
    return {
        "within_p1": within_p1,
        "within_p2": within_p2,
        "between": between,
        "full": full
    }


def extract_between_participant_matrix(
    full_matrix: NDArray[np.floating],
    n_channels_per_participant: int
) -> NDArray[np.floating]:
    """
    Extract the between-participant block from a full hyperscanning matrix.
    
    Parameters
    ----------
    full_matrix : NDArray[np.floating]
        Full hyperscanning matrix, shape (2n, 2n).
    n_channels_per_participant : int
        Number of channels per participant.
        
    Returns
    -------
    NDArray[np.floating]
        Between-participant matrix, shape (n, n).
        Rows = P1 channels, Columns = P2 channels.
        
    Examples
    --------
    >>> full = np.random.rand(8, 8)
    >>> between = extract_between_participant_matrix(full, 4)
    >>> between.shape
    (4, 4)
    """
    n = n_channels_per_participant
    return full_matrix[:n, n:].copy()


# =============================================================================
# Visualization
# =============================================================================


def plot_connectivity_matrix(
    matrix: NDArray[np.floating],
    channel_names: Optional[List[str]] = None,
    ax: Optional[plt.Axes] = None,
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    mask_diagonal: bool = True,
    title: Optional[str] = None,
    show_values: bool = False
) -> plt.Axes:
    """
    Plot connectivity matrix as heatmap.
    
    Parameters
    ----------
    matrix : NDArray[np.floating]
        Connectivity matrix, shape (n, n).
    channel_names : Optional[List[str]], optional
        Channel labels. Default is None (uses indices).
    ax : Optional[plt.Axes], optional
        Matplotlib axes. If None, creates new figure.
    cmap : str, optional
        Colormap. Default is "viridis".
    vmin : Optional[float], optional
        Minimum value for colormap. Default is None (auto).
    vmax : Optional[float], optional
        Maximum value for colormap. Default is None (auto).
    mask_diagonal : bool, optional
        Whether to mask diagonal values. Default is True.
    title : Optional[str], optional
        Plot title. Default is None.
    show_values : bool, optional
        Whether to show values in cells. Default is False.
        
    Returns
    -------
    plt.Axes
        The matplotlib axes with the plot.
    """
    n = matrix.shape[0]
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 7))
    
    if channel_names is None:
        channel_names = [str(i) for i in range(n)]
    
    # Create masked array for diagonal if needed
    if mask_diagonal:
        plot_matrix = np.ma.masked_where(np.eye(n, dtype=bool), matrix)
    else:
        plot_matrix = matrix
    
    im = ax.imshow(plot_matrix, cmap=cmap, vmin=vmin, vmax=vmax)
    
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(channel_names)
    ax.set_yticklabels(channel_names)
    
    plt.colorbar(im, ax=ax, shrink=0.8)
    
    if show_values:
        for i in range(n):
            for j in range(n):
                if not (mask_diagonal and i == j):
                    val = matrix[i, j]
                    if not np.isnan(val):
                        color = 'white' if val > 0.5 else 'black'
                        ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                               fontsize=8, color=color)
    
    if title:
        ax.set_title(title, fontsize=13, fontweight='bold')
    
    return ax


def plot_circular_connectivity(
    matrix: NDArray[np.floating],
    channel_names: List[str],
    threshold: Optional[float] = None,
    ax: Optional[plt.Axes] = None,
    linewidth_scale: float = 3.0,
    node_colors: Optional[List[str]] = None,
    title: Optional[str] = None
) -> plt.Axes:
    """
    Plot connectivity as a circular graph.
    
    Parameters
    ----------
    matrix : NDArray[np.floating]
        Connectivity matrix, shape (n_channels, n_channels).
    channel_names : List[str]
        Channel labels.
    threshold : Optional[float], optional
        Only highlight connections above this value. Default is None.
    ax : Optional[plt.Axes], optional
        Matplotlib polar axes. If None, creates new figure.
    linewidth_scale : float, optional
        Scale factor for line width. Default is 3.0.
    node_colors : Optional[List[str]], optional
        Colors for each node. Default is None (uses primary blue).
    title : Optional[str], optional
        Plot title. Default is None.
        
    Returns
    -------
    plt.Axes
        The matplotlib axes with the plot.
    """
    n_channels = len(channel_names)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': 'polar'})
    
    if node_colors is None:
        node_colors = [PRIMARY_BLUE] * n_channels
    
    # Calculate node positions (evenly spaced around circle)
    angles = np.linspace(0, 2 * np.pi, n_channels, endpoint=False)
    
    # Plot nodes
    for i, (angle, name, color) in enumerate(zip(angles, channel_names, node_colors)):
        ax.scatter(angle, 1, s=300, c=color, zorder=5, edgecolors='white', linewidths=2)
        ax.text(angle, 1.15, name, ha='center', va='center', fontsize=11, fontweight='bold')
    
    # Plot connections (two passes: weak in grey, strong in color)
    for i in range(n_channels):
        for j in range(i + 1, n_channels):
            value = matrix[i, j]
            if np.isnan(value):
                continue
            
            # Draw arc between nodes
            angle_i, angle_j = angles[i], angles[j]
            
            # Create arc using bezier-like curve
            n_points = 50
            t_vals = np.linspace(0, 1, n_points)
            
            # Control point at center (r=0)
            r_vals = 1 - 0.5 * np.sin(np.pi * t_vals)  # Curve inward
            angle_vals = angle_i + t_vals * (angle_j - angle_i)
            
            # Adjust for shortest path
            if abs(angle_j - angle_i) > np.pi:
                if angle_j > angle_i:
                    angle_vals = angle_i + t_vals * (angle_j - 2*np.pi - angle_i)
                else:
                    angle_vals = angle_i + t_vals * (angle_j + 2*np.pi - angle_i)
            
            # Determine if connection is strong (above threshold)
            is_strong = threshold is None or value >= threshold
            
            if is_strong:
                # Strong connections: colored with variable width/alpha
                lw = value * linewidth_scale
                alpha = 0.3 + 0.7 * value
                color = SECONDARY_PURPLE
                zorder = 2
            else:
                # Weak connections: light grey, thin, subtle
                lw = 0.8
                alpha = 0.3
                color = '#CCCCCC'
                zorder = 1
            
            ax.plot(angle_vals, r_vals, color=color, 
                   linewidth=lw, alpha=alpha, zorder=zorder)
    
    # Clean up polar plot
    ax.set_ylim(0, 1.3)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.spines['polar'].set_visible(False)
    
    if title:
        ax.set_title(title, fontsize=13, fontweight='bold', pad=20)
    
    return ax


def plot_hyperscanning_matrix(
    full_matrix: NDArray[np.floating],
    channel_names_p1: List[str],
    channel_names_p2: List[str],
    ax: Optional[plt.Axes] = None,
    highlight_between: bool = True,
    cmap: str = 'viridis',
    title: Optional[str] = None
) -> plt.Axes:
    """
    Plot full hyperscanning matrix with quadrant annotations.
    
    Parameters
    ----------
    full_matrix : NDArray[np.floating]
        Full hyperscanning matrix, shape (2n, 2n).
    channel_names_p1 : List[str]
        Channel names for Participant 1.
    channel_names_p2 : List[str]
        Channel names for Participant 2.
    ax : Optional[plt.Axes], optional
        Matplotlib axes. If None, creates new figure.
    highlight_between : bool, optional
        Whether to highlight the between-participant block. Default True.
    cmap : str, optional
        Colormap. Default is 'viridis'.
    title : Optional[str], optional
        Plot title.
        
    Returns
    -------
    plt.Axes
        The matplotlib axes with the plot.
    """
    n_ch = len(channel_names_p1)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    
    im = ax.imshow(full_matrix, cmap=cmap, vmin=0, vmax=1)
    
    # Add dividing lines
    ax.axhline(n_ch - 0.5, color='white', linewidth=2)
    ax.axvline(n_ch - 0.5, color='white', linewidth=2)
    
    # Highlight between block
    if highlight_between:
        rect = plt.Rectangle(
            (n_ch - 0.5, -0.5), n_ch, n_ch,
            fill=False, edgecolor=PRIMARY_GREEN, linewidth=3, linestyle='--'
        )
        ax.add_patch(rect)
    
    # Labels
    all_labels = ([f'P1-{ch}' for ch in channel_names_p1] + 
                  [f'P2-{ch}' for ch in channel_names_p2])
    ax.set_xticks(range(2 * n_ch))
    ax.set_yticks(range(2 * n_ch))
    ax.set_xticklabels(all_labels, rotation=45, ha='right')
    ax.set_yticklabels(all_labels)
    
    # Color labels by participant
    for i, label in enumerate(ax.get_xticklabels()):
        label.set_color(SUBJECT_1 if i < n_ch else SUBJECT_2)
    for i, label in enumerate(ax.get_yticklabels()):
        label.set_color(SUBJECT_1 if i < n_ch else SUBJECT_2)
    
    plt.colorbar(im, ax=ax, shrink=0.8, label='PLV')
    
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    
    return ax


def plot_hyperscanning_circular(
    between_matrix: NDArray[np.floating],
    channel_names_p1: List[str],
    channel_names_p2: List[str],
    threshold: Optional[float] = None,
    ax: Optional[plt.Axes] = None,
    linewidth_scale: float = 3.0,
    title: Optional[str] = None
) -> plt.Axes:
    """
    Circular plot for hyperscanning with P1 on left, P2 on right.
    
    Parameters
    ----------
    between_matrix : NDArray[np.floating]
        Between-participant matrix, shape (n, n).
        Rows = P1 channels, Columns = P2 channels.
    channel_names_p1 : List[str]
        Channel names for Participant 1.
    channel_names_p2 : List[str]
        Channel names for Participant 2.
    threshold : Optional[float], optional
        Only highlight connections above this value. Default None.
    ax : Optional[plt.Axes], optional
        Polar axes. If None, creates new figure.
    linewidth_scale : float, optional
        Scale factor for line width. Default 3.0.
    title : Optional[str], optional
        Plot title.
        
    Returns
    -------
    plt.Axes
        The matplotlib polar axes with the plot.
    """
    n_ch = len(channel_names_p1)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 10), subplot_kw={'projection': 'polar'})
    
    # Position P1 on left side, P2 on right side
    angles_p1 = np.linspace(np.pi * 0.7, np.pi * 1.3, n_ch)
    angles_p2 = np.linspace(-np.pi * 0.3, np.pi * 0.3, n_ch)
    
    # Plot P1 nodes (left side)
    for i, (angle, name) in enumerate(zip(angles_p1, channel_names_p1)):
        ax.scatter(angle, 1, s=400, c=SUBJECT_1, zorder=5, 
                  edgecolors='white', linewidths=2)
        ax.text(angle, 1.2, f'P1-{name}', ha='center', va='center', 
                fontsize=10, fontweight='bold', color=SUBJECT_1)
    
    # Plot P2 nodes (right side)
    for i, (angle, name) in enumerate(zip(angles_p2, channel_names_p2)):
        ax.scatter(angle, 1, s=400, c=SUBJECT_2, zorder=5, 
                  edgecolors='white', linewidths=2)
        ax.text(angle, 1.2, f'P2-{name}', ha='center', va='center', 
                fontsize=10, fontweight='bold', color=SUBJECT_2)
    
    # Plot connections between P1 and P2
    for i in range(n_ch):
        for j in range(n_ch):
            value = between_matrix[i, j]
            if np.isnan(value):
                continue
            
            angle_i = angles_p1[i]
            angle_j = angles_p2[j]
            
            # Create arc
            n_points = 50
            t_vals = np.linspace(0, 1, n_points)
            r_vals = 1 - 0.4 * np.sin(np.pi * t_vals)
            angle_vals = angle_i + t_vals * (angle_j - angle_i)
            
            # Determine if strong connection
            is_strong = threshold is None or value >= threshold
            
            if is_strong:
                lw = value * linewidth_scale
                alpha = 0.4 + 0.6 * value
                color = SECONDARY_PURPLE
                zorder = 2
            else:
                lw = 0.8
                alpha = 0.2
                color = '#CCCCCC'
                zorder = 1
            
            ax.plot(angle_vals, r_vals, color=color, 
                   linewidth=lw, alpha=alpha, zorder=zorder)
    
    # Clean up
    ax.set_ylim(0, 1.4)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.spines['polar'].set_visible(False)
    
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    return ax


# =============================================================================
# Global Metrics
# =============================================================================


def compute_global_connectivity(
    matrix: NDArray[np.floating],
    exclude_diagonal: bool = True
) -> float:
    """
    Compute mean connectivity (global connectivity).
    
    Parameters
    ----------
    matrix : NDArray[np.floating]
        Connectivity matrix.
    exclude_diagonal : bool, optional
        Whether to exclude diagonal values. Default True.
        
    Returns
    -------
    float
        Mean connectivity value.
        
    Examples
    --------
    >>> matrix = np.array([[np.nan, 0.5, 0.3], [0.5, np.nan, 0.7], [0.3, 0.7, np.nan]])
    >>> compute_global_connectivity(matrix)
    0.5
    """
    if exclude_diagonal:
        # Get upper triangle values (excludes diagonal)
        values = get_upper_triangle_values(matrix, k=1)
    else:
        values = matrix.flatten()
    
    # Remove NaN values
    values = values[~np.isnan(values)]
    return float(np.mean(values))


def compute_connection_density(
    matrix: NDArray[np.floating],
    threshold: float,
    exclude_diagonal: bool = True
) -> float:
    """
    Compute proportion of connections exceeding threshold.
    
    Parameters
    ----------
    matrix : NDArray[np.floating]
        Connectivity matrix.
    threshold : float
        Connectivity threshold.
    exclude_diagonal : bool, optional
        Whether to exclude diagonal. Default True.
        
    Returns
    -------
    float
        Proportion of connections above threshold (0 to 1).
        
    Examples
    --------
    >>> matrix = np.array([[np.nan, 0.8, 0.3], [0.8, np.nan, 0.6], [0.3, 0.6, np.nan]])
    >>> compute_connection_density(matrix, 0.5)
    0.6666666666666666
    """
    if exclude_diagonal:
        values = get_upper_triangle_values(matrix, k=1)
    else:
        values = matrix.flatten()
    
    values = values[~np.isnan(values)]
    return float(np.mean(values > threshold))


def compute_hyperscanning_ratio(
    within_mean: float,
    between_mean: float
) -> float:
    """
    Compute ratio of between to within connectivity.
    
    Parameters
    ----------
    within_mean : float
        Mean within-participant connectivity.
    between_mean : float
        Mean between-participant connectivity.
        
    Returns
    -------
    float
        Ratio (between / within). 
        > 1 indicates stronger inter-brain than intra-brain connectivity.
        
    Examples
    --------
    >>> compute_hyperscanning_ratio(0.4, 0.6)
    1.5
    """
    if within_mean == 0:
        return np.inf if between_mean > 0 else 0.0
    return between_mean / within_mean


# =============================================================================
# Validation
# =============================================================================


def validate_connectivity_matrix(
    matrix: NDArray[np.floating],
    metric: str = "plv",
    tolerance: float = 1e-10
) -> Dict[str, Any]:
    """
    Validate connectivity matrix properties.
    
    Parameters
    ----------
    matrix : NDArray[np.floating]
        Connectivity matrix to validate.
    metric : str, optional
        Expected metric type. Default is "plv".
    tolerance : float, optional
        Tolerance for symmetry check. Default is 1e-10.
        
    Returns
    -------
    Dict[str, Any]
        Validation results with keys:
        - "is_square": bool
        - "is_symmetric": bool
        - "in_range": bool
        - "diagonal_is_nan": bool
        - "has_invalid_values": bool
        - "issues": str (description of any issues found)
        
    Examples
    --------
    >>> matrix = np.array([[np.nan, 0.5], [0.5, np.nan]])
    >>> result = validate_connectivity_matrix(matrix)
    >>> result['is_symmetric']
    True
    """
    issues = []
    
    # Check square
    is_square = matrix.shape[0] == matrix.shape[1]
    if not is_square:
        issues.append(f"Matrix is not square: {matrix.shape}")
    
    # Check symmetry (ignoring diagonal)
    if is_square:
        # Create a copy without diagonal for comparison
        m1 = matrix.copy()
        m2 = matrix.T.copy()
        np.fill_diagonal(m1, 0)
        np.fill_diagonal(m2, 0)
        # Handle NaN in comparison
        mask = ~(np.isnan(m1) | np.isnan(m2))
        is_symmetric = np.allclose(m1[mask], m2[mask], atol=tolerance)
        if not is_symmetric:
            issues.append("Matrix is not symmetric")
    else:
        is_symmetric = False
    
    # Check value range based on metric
    off_diag = matrix[~np.eye(matrix.shape[0], dtype=bool)]
    off_diag_valid = off_diag[~np.isnan(off_diag)]
    
    if metric == "plv":
        in_range = np.all((off_diag_valid >= 0) & (off_diag_valid <= 1))
        if not in_range:
            issues.append(f"PLV values out of [0, 1] range")
    elif metric == "correlation":
        in_range = np.all((off_diag_valid >= -1) & (off_diag_valid <= 1))
        if not in_range:
            issues.append(f"Correlation values out of [-1, 1] range")
    else:
        in_range = True  # Can't validate unknown metrics
    
    # Check diagonal
    diagonal = np.diag(matrix)
    diagonal_is_nan = np.all(np.isnan(diagonal))
    if not diagonal_is_nan:
        issues.append("Diagonal contains non-NaN values")
    
    # Check for invalid values (Inf)
    has_invalid_values = np.any(np.isinf(matrix))
    if has_invalid_values:
        issues.append("Matrix contains Inf values")
    
    return {
        "is_square": is_square,
        "is_symmetric": is_symmetric,
        "in_range": in_range,
        "diagonal_is_nan": diagonal_is_nan,
        "has_invalid_values": has_invalid_values,
        "issues": "; ".join(issues) if issues else ""
    }


def get_matrix_statistics(
    matrix: NDArray[np.floating],
    exclude_diagonal: bool = True
) -> Dict[str, float]:
    """
    Compute summary statistics of connectivity matrix.
    
    Parameters
    ----------
    matrix : NDArray[np.floating]
        Connectivity matrix.
    exclude_diagonal : bool, optional
        Whether to exclude diagonal values. Default True.
        
    Returns
    -------
    Dict[str, float]
        Statistics including mean, std, min, max, median, n_values.
        
    Examples
    --------
    >>> matrix = np.array([[np.nan, 0.5, 0.3], [0.5, np.nan, 0.7], [0.3, 0.7, np.nan]])
    >>> stats = get_matrix_statistics(matrix)
    >>> stats['mean']
    0.5
    """
    if exclude_diagonal:
        values = get_upper_triangle_values(matrix, k=1)
    else:
        values = matrix.flatten()
    
    # Remove NaN values
    values = values[~np.isnan(values)]
    
    if len(values) == 0:
        return {
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
            "median": np.nan,
            "n_values": 0
        }
    
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "median": float(np.median(values)),
        "n_values": len(values)
    }
