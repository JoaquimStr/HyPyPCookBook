"""Hyperscanning utilities for dual-brain connectivity analysis.

This module provides functions for organizing and analyzing hyperscanning data,
including data structure creation, connectivity block extraction, and
pseudo-pair analysis for statistical validation.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


def create_hyperscanning_data_structure(
    data_p1: NDArray[np.float64],
    data_p2: NDArray[np.float64],
    channel_names_p1: List[str],
    channel_names_p2: List[str],
) -> Dict[str, Any]:
    """Create a unified data structure for hyperscanning analysis.

    Combines data from two participants into a single structure with
    prefixed channel names (P1_, P2_) for clear identification.

    Parameters
    ----------
    data_p1 : NDArray[np.float64]
        EEG data from participant 1, shape (n_channels_p1, n_samples).
    data_p2 : NDArray[np.float64]
        EEG data from participant 2, shape (n_channels_p2, n_samples).
    channel_names_p1 : List[str]
        Channel names for participant 1.
    channel_names_p2 : List[str]
        Channel names for participant 2.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - "data_combined": Combined data array (n_total_channels, n_samples)
        - "channel_names": List of prefixed channel names
        - "n_channels_p1": Number of channels for P1
        - "n_channels_p2": Number of channels for P2
        - "participant_labels": Array of participant IDs (1 or 2)

    Raises
    ------
    ValueError
        If data dimensions don't match channel names or sample counts differ.

    Examples
    --------
    >>> data_p1 = np.random.randn(4, 1000)
    >>> data_p2 = np.random.randn(4, 1000)
    >>> channels = ["Fz", "Cz", "Pz", "Oz"]
    >>> result = create_hyperscanning_data_structure(
    ...     data_p1, data_p2, channels, channels
    ... )
    >>> result["data_combined"].shape
    (8, 1000)
    """
    # Validate inputs
    if data_p1.shape[0] != len(channel_names_p1):
        raise ValueError(
            f"P1 data has {data_p1.shape[0]} channels but "
            f"{len(channel_names_p1)} channel names provided"
        )
    if data_p2.shape[0] != len(channel_names_p2):
        raise ValueError(
            f"P2 data has {data_p2.shape[0]} channels but "
            f"{len(channel_names_p2)} channel names provided"
        )
    if data_p1.shape[1] != data_p2.shape[1]:
        raise ValueError(
            f"Sample counts differ: P1 has {data_p1.shape[1]}, "
            f"P2 has {data_p2.shape[1]}"
        )

    n_channels_p1 = data_p1.shape[0]
    n_channels_p2 = data_p2.shape[0]

    # Combine data
    data_combined = np.vstack([data_p1, data_p2])

    # Create prefixed channel names
    channel_names = [f"P1_{ch}" for ch in channel_names_p1] + [
        f"P2_{ch}" for ch in channel_names_p2
    ]

    # Create participant labels
    participant_labels = np.array(
        [1] * n_channels_p1 + [2] * n_channels_p2, dtype=np.int8
    )

    return {
        "data_combined": data_combined,
        "channel_names": channel_names,
        "n_channels_p1": n_channels_p1,
        "n_channels_p2": n_channels_p2,
        "participant_labels": participant_labels,
    }


def extract_connectivity_blocks(
    full_matrix: NDArray[np.float64],
    n_channels_p1: int,
) -> Dict[str, NDArray[np.float64]]:
    """Extract connectivity blocks from a combined hyperscanning matrix.

    Separates a full (n_p1 + n_p2) x (n_p1 + n_p2) connectivity matrix
    into within-participant and between-participant blocks.

    Parameters
    ----------
    full_matrix : NDArray[np.float64]
        Full connectivity matrix, shape (n_total, n_total).
    n_channels_p1 : int
        Number of channels for participant 1.

    Returns
    -------
    Dict[str, NDArray[np.float64]]
        Dictionary containing:
        - "within_p1": Connectivity within P1, shape (n_p1, n_p1)
        - "within_p2": Connectivity within P2, shape (n_p2, n_p2)
        - "between": Connectivity between P1 and P2, shape (n_p1, n_p2)

    Examples
    --------
    >>> matrix = np.random.rand(8, 8)
    >>> blocks = extract_connectivity_blocks(matrix, n_channels_p1=4)
    >>> blocks["within_p1"].shape
    (4, 4)
    >>> blocks["between"].shape
    (4, 4)
    """
    n_total = full_matrix.shape[0]
    n_channels_p2 = n_total - n_channels_p1

    within_p1 = full_matrix[:n_channels_p1, :n_channels_p1]
    within_p2 = full_matrix[n_channels_p1:, n_channels_p1:]
    between = full_matrix[:n_channels_p1, n_channels_p1:]

    return {
        "within_p1": within_p1,
        "within_p2": within_p2,
        "between": between,
    }


def combine_connectivity_blocks(
    within_p1: NDArray[np.float64],
    within_p2: NDArray[np.float64],
    between: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Combine connectivity blocks into a full hyperscanning matrix.

    Reconstructs a full (n_p1 + n_p2) x (n_p1 + n_p2) matrix from
    the within-participant and between-participant blocks.

    Parameters
    ----------
    within_p1 : NDArray[np.float64]
        Connectivity within P1, shape (n_p1, n_p1).
    within_p2 : NDArray[np.float64]
        Connectivity within P2, shape (n_p2, n_p2).
    between : NDArray[np.float64]
        Connectivity between P1 and P2, shape (n_p1, n_p2).

    Returns
    -------
    NDArray[np.float64]
        Full connectivity matrix, shape (n_p1 + n_p2, n_p1 + n_p2).

    Examples
    --------
    >>> within_p1 = np.eye(4)
    >>> within_p2 = np.eye(4)
    >>> between = np.ones((4, 4)) * 0.5
    >>> full = combine_connectivity_blocks(within_p1, within_p2, between)
    >>> full.shape
    (8, 8)
    """
    n_p1 = within_p1.shape[0]
    n_p2 = within_p2.shape[0]
    n_total = n_p1 + n_p2

    full_matrix = np.zeros((n_total, n_total), dtype=np.float64)

    # Fill blocks
    full_matrix[:n_p1, :n_p1] = within_p1
    full_matrix[n_p1:, n_p1:] = within_p2
    full_matrix[:n_p1, n_p1:] = between
    full_matrix[n_p1:, :n_p1] = between.T  # Symmetric assumption

    return full_matrix


def create_pseudo_pairs(
    participant_ids: List[str],
    real_pairs: List[Tuple[str, str]],
    n_pseudo: Optional[int] = None,
    seed: Optional[int] = None,
) -> List[Tuple[str, str]]:
    """Generate pseudo-pairs from participants who never interacted.

    Creates pairs of participants from different real pairs for use
    as a null distribution in statistical testing.

    Parameters
    ----------
    participant_ids : List[str]
        List of all participant IDs.
    real_pairs : List[Tuple[str, str]]
        List of tuples representing actual interaction pairs.
    n_pseudo : Optional[int]
        Number of pseudo-pairs to generate. If None, generates all possible.
    seed : Optional[int]
        Random seed for reproducibility.

    Returns
    -------
    List[Tuple[str, str]]
        List of pseudo-pair tuples.

    Examples
    --------
    >>> ids = ["A1", "A2", "B1", "B2", "C1", "C2"]
    >>> real = [("A1", "A2"), ("B1", "B2"), ("C1", "C2")]
    >>> pseudo = create_pseudo_pairs(ids, real, n_pseudo=5, seed=42)
    >>> len(pseudo)
    5
    """
    if seed is not None:
        np.random.seed(seed)

    # Create set of real pairs (both orderings)
    real_set = set()
    for p1, p2 in real_pairs:
        real_set.add((p1, p2))
        real_set.add((p2, p1))

    # Generate all possible pseudo-pairs
    pseudo_pairs = []
    for i, p1 in enumerate(participant_ids):
        for p2 in participant_ids[i + 1 :]:
            if (p1, p2) not in real_set:
                pseudo_pairs.append((p1, p2))

    # Subsample if requested
    if n_pseudo is not None and n_pseudo < len(pseudo_pairs):
        indices = np.random.choice(len(pseudo_pairs), n_pseudo, replace=False)
        pseudo_pairs = [pseudo_pairs[i] for i in indices]

    return pseudo_pairs


def compute_pseudo_pair_null(
    connectivity_values: Dict[Tuple[str, str], float],
    real_pairs: List[Tuple[str, str]],
) -> Dict[str, Any]:
    """Build null distribution from pseudo-pair connectivity values.

    Parameters
    ----------
    connectivity_values : Dict[Tuple[str, str], float]
        Dictionary mapping pair IDs to connectivity values.
    real_pairs : List[Tuple[str, str]]
        List of real interaction pairs.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - "null_distribution": Array of pseudo-pair values
        - "null_mean": Mean of null distribution
        - "null_std": Standard deviation of null distribution
        - "real_values": Array of real pair values
        - "real_mean": Mean of real pairs

    Examples
    --------
    >>> values = {("A", "B"): 0.8, ("C", "D"): 0.7, ("A", "C"): 0.3}
    >>> real = [("A", "B"), ("C", "D")]
    >>> result = compute_pseudo_pair_null(values, real)
    >>> result["null_mean"]
    0.3
    """
    real_set = set(real_pairs) | {(p2, p1) for p1, p2 in real_pairs}

    null_values = []
    real_values = []

    for pair, value in connectivity_values.items():
        if pair in real_set:
            real_values.append(value)
        else:
            null_values.append(value)

    null_array = np.array(null_values)
    real_array = np.array(real_values)

    return {
        "null_distribution": null_array,
        "null_mean": float(np.mean(null_array)) if len(null_array) > 0 else np.nan,
        "null_std": float(np.std(null_array)) if len(null_array) > 0 else np.nan,
        "real_values": real_array,
        "real_mean": float(np.mean(real_array)) if len(real_array) > 0 else np.nan,
    }


def test_against_pseudo_pairs(
    real_value: float,
    null_distribution: NDArray[np.float64],
) -> Dict[str, float]:
    """Test a real pair value against the pseudo-pair null distribution.

    Computes p-value, z-score, and percentile for a real pair's
    connectivity value relative to the null distribution.

    Parameters
    ----------
    real_value : float
        Connectivity value from a real pair.
    null_distribution : NDArray[np.float64]
        Array of connectivity values from pseudo-pairs.

    Returns
    -------
    Dict[str, float]
        Dictionary containing:
        - "pvalue": One-tailed p-value (proportion of null >= real)
        - "zscore": Z-score relative to null distribution
        - "percentile": Percentile of real value in null distribution

    Examples
    --------
    >>> null = np.random.normal(0.3, 0.1, 100)
    >>> result = test_against_pseudo_pairs(0.6, null)
    >>> result["pvalue"] < 0.05
    True
    """
    null_mean = np.mean(null_distribution)
    null_std = np.std(null_distribution)

    # Z-score
    zscore = (real_value - null_mean) / null_std if null_std > 0 else 0.0

    # One-tailed p-value (testing if real > null)
    pvalue = np.mean(null_distribution >= real_value)

    # Percentile
    percentile = np.mean(null_distribution <= real_value) * 100

    return {
        "pvalue": float(pvalue),
        "zscore": float(zscore),
        "percentile": float(percentile),
    }


def synchronize_recordings(
    data_p1: NDArray[np.float64],
    data_p2: NDArray[np.float64],
    timestamps_p1: NDArray[np.float64],
    timestamps_p2: NDArray[np.float64],
    target_fs: float,
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Synchronize two recordings to a common time base.

    Interpolates both recordings to a shared timeline with uniform
    sampling rate. Essential for hyperscanning when recordings
    have different start times or sampling rates.

    Parameters
    ----------
    data_p1 : NDArray[np.float64]
        EEG data from participant 1, shape (n_channels, n_samples).
    data_p2 : NDArray[np.float64]
        EEG data from participant 2, shape (n_channels, n_samples).
    timestamps_p1 : NDArray[np.float64]
        Timestamps for P1 samples (in seconds).
    timestamps_p2 : NDArray[np.float64]
        Timestamps for P2 samples (in seconds).
    target_fs : float
        Target sampling frequency (Hz).

    Returns
    -------
    Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]
        Tuple of (synchronized_p1, synchronized_p2, common_timestamps).

    Examples
    --------
    >>> data1 = np.random.randn(4, 1000)
    >>> data2 = np.random.randn(4, 1000)
    >>> t1 = np.linspace(0, 10, 1000)
    >>> t2 = np.linspace(0.1, 10.1, 1000)  # Offset by 0.1s
    >>> sync1, sync2, t_common = synchronize_recordings(
    ...     data1, data2, t1, t2, target_fs=100
    ... )
    >>> sync1.shape == sync2.shape
    True
    """
    from scipy.interpolate import interp1d

    # Find common time range
    t_start = max(timestamps_p1[0], timestamps_p2[0])
    t_end = min(timestamps_p1[-1], timestamps_p2[-1])

    # Create common time base
    n_samples = int((t_end - t_start) * target_fs)
    common_timestamps = np.linspace(t_start, t_end, n_samples)

    # Interpolate P1
    n_channels_p1 = data_p1.shape[0]
    sync_p1 = np.zeros((n_channels_p1, n_samples), dtype=np.float64)
    for ch in range(n_channels_p1):
        interp_func = interp1d(
            timestamps_p1, data_p1[ch, :], kind="linear", fill_value="extrapolate"
        )
        sync_p1[ch, :] = interp_func(common_timestamps)

    # Interpolate P2
    n_channels_p2 = data_p2.shape[0]
    sync_p2 = np.zeros((n_channels_p2, n_samples), dtype=np.float64)
    for ch in range(n_channels_p2):
        interp_func = interp1d(
            timestamps_p2, data_p2[ch, :], kind="linear", fill_value="extrapolate"
        )
        sync_p2[ch, :] = interp_func(common_timestamps)

    return sync_p1, sync_p2, common_timestamps


def check_synchronization(
    trigger_p1: NDArray[np.float64],
    trigger_p2: NDArray[np.float64],
    tolerance_samples: int = 5,
) -> Dict[str, Any]:
    """Verify that two recordings are properly synchronized.

    Compares trigger channels from both participants to check
    temporal alignment. Triggers should occur at the same time
    in both recordings.

    Parameters
    ----------
    trigger_p1 : NDArray[np.float64]
        Trigger channel from participant 1.
    trigger_p2 : NDArray[np.float64]
        Trigger channel from participant 2.
    tolerance_samples : int, optional
        Maximum allowed offset in samples, by default 5.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - "is_synchronized": Whether recordings are synchronized
        - "max_offset": Maximum offset found (samples)
        - "mean_offset": Mean offset (samples)
        - "n_triggers_matched": Number of matched triggers

    Examples
    --------
    >>> trigger1 = np.zeros(1000)
    >>> trigger1[[100, 300, 500]] = 1
    >>> trigger2 = np.zeros(1000)
    >>> trigger2[[101, 301, 501]] = 1  # 1 sample offset
    >>> result = check_synchronization(trigger1, trigger2)
    >>> result["is_synchronized"]
    True
    """
    # Find trigger onsets
    threshold = 0.5 * max(np.max(trigger_p1), np.max(trigger_p2))

    onsets_p1 = np.where(np.diff(trigger_p1 > threshold) > 0)[0]
    onsets_p2 = np.where(np.diff(trigger_p2 > threshold) > 0)[0]

    if len(onsets_p1) == 0 or len(onsets_p2) == 0:
        return {
            "is_synchronized": False,
            "max_offset": np.nan,
            "mean_offset": np.nan,
            "n_triggers_matched": 0,
            "message": "No triggers found in one or both recordings",
        }

    # Match triggers and compute offsets
    offsets = []
    n_matched = 0

    for onset_p1 in onsets_p1:
        # Find closest trigger in P2
        distances = np.abs(onsets_p2 - onset_p1)
        closest_idx = np.argmin(distances)
        offset = onsets_p2[closest_idx] - onset_p1

        if np.abs(offset) <= tolerance_samples * 2:  # Allow some slack for matching
            offsets.append(offset)
            n_matched += 1

    if len(offsets) == 0:
        return {
            "is_synchronized": False,
            "max_offset": np.nan,
            "mean_offset": np.nan,
            "n_triggers_matched": 0,
            "message": "No matching triggers found within tolerance",
        }

    offsets = np.array(offsets)
    max_offset = int(np.max(np.abs(offsets)))
    mean_offset = float(np.mean(offsets))

    is_synchronized = max_offset <= tolerance_samples

    return {
        "is_synchronized": is_synchronized,
        "max_offset": max_offset,
        "mean_offset": mean_offset,
        "n_triggers_matched": n_matched,
    }


def reject_epochs_both_participants(
    epochs_p1: NDArray[np.float64],
    epochs_p2: NDArray[np.float64],
    threshold_uv: float = 100.0,
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
    """Reject epochs where either participant has artifacts.

    For hyperscanning analysis, epochs must be clean for BOTH
    participants. This function identifies and removes epochs
    where either participant exceeds the amplitude threshold.

    Parameters
    ----------
    epochs_p1 : NDArray[np.float64]
        Epoched data for P1, shape (n_epochs, n_channels, n_samples).
    epochs_p2 : NDArray[np.float64]
        Epoched data for P2, shape (n_epochs, n_channels, n_samples).
    threshold_uv : float, optional
        Amplitude threshold in microvolts, by default 100.0.

    Returns
    -------
    Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]
        Tuple of (clean_epochs_p1, clean_epochs_p2, keep_mask).

    Examples
    --------
    >>> epochs1 = np.random.randn(100, 4, 256) * 20  # Normal amplitude
    >>> epochs2 = np.random.randn(100, 4, 256) * 20
    >>> epochs1[5, 0, :] = 200  # Artifact in epoch 5
    >>> clean1, clean2, mask = reject_epochs_both_participants(epochs1, epochs2)
    >>> mask[5]
    False
    """
    n_epochs = epochs_p1.shape[0]

    # Find max amplitude per epoch for each participant
    max_amp_p1 = np.max(np.abs(epochs_p1), axis=(1, 2))
    max_amp_p2 = np.max(np.abs(epochs_p2), axis=(1, 2))

    # Keep epoch only if BOTH participants are below threshold
    keep_p1 = max_amp_p1 < threshold_uv
    keep_p2 = max_amp_p2 < threshold_uv
    keep_mask = keep_p1 & keep_p2

    # Apply mask
    clean_epochs_p1 = epochs_p1[keep_mask]
    clean_epochs_p2 = epochs_p2[keep_mask]

    n_rejected = n_epochs - np.sum(keep_mask)
    if n_rejected > 0:
        print(
            f"Rejected {n_rejected}/{n_epochs} epochs "
            f"({100 * n_rejected / n_epochs:.1f}%)"
        )

    return clean_epochs_p1, clean_epochs_p2, keep_mask


def hyperscanning_analysis_pipeline(
    data_p1: NDArray[np.float64],
    data_p2: NDArray[np.float64],
    fs: float,
    channel_names: List[str],
    freq_band: Tuple[float, float],
    n_surrogates: int = 100,
) -> Dict[str, Any]:
    """Run a complete hyperscanning connectivity analysis pipeline.

    Performs filtering, connectivity computation, and statistical
    testing in one function. This is a demonstration/convenience
    function - for production, run steps separately with more control.

    Parameters
    ----------
    data_p1 : NDArray[np.float64]
        EEG data from participant 1, shape (n_channels, n_samples).
    data_p2 : NDArray[np.float64]
        EEG data from participant 2, shape (n_channels, n_samples).
    fs : float
        Sampling frequency (Hz).
    channel_names : List[str]
        Channel names (same for both participants).
    freq_band : Tuple[float, float]
        Frequency band of interest (low, high) in Hz.
    n_surrogates : int, optional
        Number of surrogates for statistical testing, by default 100.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - "connectivity_within_p1": Within-P1 connectivity matrix
        - "connectivity_within_p2": Within-P2 connectivity matrix
        - "connectivity_between": Between-brain connectivity matrix
        - "connectivity_full": Full combined matrix
        - "surrogate_mean": Mean of surrogate distribution
        - "surrogate_std": Std of surrogate distribution
        - "pvalues": P-values for between-brain connections
        - "significant_mask": Boolean mask of significant connections
        - "summary": Summary statistics dictionary

    Notes
    -----
    This function uses simple correlation as the connectivity metric.
    For other metrics (PLV, coherence), use the specific functions
    from the connectivity module.

    Examples
    --------
    >>> data1 = np.random.randn(4, 10000)
    >>> data2 = np.random.randn(4, 10000)
    >>> channels = ["Fz", "Cz", "Pz", "Oz"]
    >>> results = hyperscanning_analysis_pipeline(
    ...     data1, data2, fs=256, channel_names=channels,
    ...     freq_band=(8, 13), n_surrogates=50
    ... )
    >>> results["connectivity_between"].shape
    (4, 4)
    """
    from scipy.signal import butter, filtfilt
    from scipy.stats import pearsonr

    n_channels = data_p1.shape[0]

    # Step 1: Bandpass filter
    low, high = freq_band
    nyq = fs / 2
    b, a = butter(4, [low / nyq, high / nyq], btype="band")

    filtered_p1 = filtfilt(b, a, data_p1, axis=1)
    filtered_p2 = filtfilt(b, a, data_p2, axis=1)

    # Step 2: Compute connectivity (using correlation)
    def compute_correlation_matrix(
        data1: NDArray[np.float64], data2: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        n1, n2 = data1.shape[0], data2.shape[0]
        corr_mat = np.zeros((n1, n2))
        for i in range(n1):
            for j in range(n2):
                corr_mat[i, j], _ = pearsonr(data1[i], data2[j])
        return np.abs(corr_mat)  # Use absolute correlation

    within_p1 = compute_correlation_matrix(filtered_p1, filtered_p1)
    within_p2 = compute_correlation_matrix(filtered_p2, filtered_p2)
    between = compute_correlation_matrix(filtered_p1, filtered_p2)

    # Combine into full matrix
    full_matrix = combine_connectivity_blocks(within_p1, within_p2, between)

    # Step 3: Surrogate testing (phase shuffling)
    surrogate_between = np.zeros((n_surrogates, n_channels, n_channels))

    for s in range(n_surrogates):
        # Shuffle phases of P2
        shuffled_p2 = np.zeros_like(filtered_p2)
        for ch in range(n_channels):
            fft_result = np.fft.fft(filtered_p2[ch])
            phases = np.angle(fft_result)
            np.random.shuffle(phases)
            shuffled_fft = np.abs(fft_result) * np.exp(1j * phases)
            shuffled_p2[ch] = np.real(np.fft.ifft(shuffled_fft))

        surrogate_between[s] = compute_correlation_matrix(filtered_p1, shuffled_p2)

    # Compute p-values
    surrogate_mean = np.mean(surrogate_between, axis=0)
    surrogate_std = np.std(surrogate_between, axis=0)

    pvalues = np.zeros((n_channels, n_channels))
    for i in range(n_channels):
        for j in range(n_channels):
            pvalues[i, j] = np.mean(surrogate_between[:, i, j] >= between[i, j])

    # Significant connections (p < 0.05)
    significant_mask = pvalues < 0.05

    # Summary statistics
    summary = {
        "mean_within_p1": float(np.mean(within_p1[np.triu_indices(n_channels, k=1)])),
        "mean_within_p2": float(np.mean(within_p2[np.triu_indices(n_channels, k=1)])),
        "mean_between": float(np.mean(between)),
        "n_significant": int(np.sum(significant_mask)),
        "percent_significant": float(100 * np.sum(significant_mask) / between.size),
    }

    return {
        "connectivity_within_p1": within_p1,
        "connectivity_within_p2": within_p2,
        "connectivity_between": between,
        "connectivity_full": full_matrix,
        "surrogate_mean": surrogate_mean,
        "surrogate_std": surrogate_std,
        "pvalues": pvalues,
        "significant_mask": significant_mask,
        "summary": summary,
    }
