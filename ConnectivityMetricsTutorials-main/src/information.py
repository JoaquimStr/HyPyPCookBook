"""
Information theory functions for entropy, mutual information, and transfer entropy.

This module provides functions for computing various entropy, mutual information,
and transfer entropy measures used in signal analysis and connectivity metrics.

Functions
---------
Entropy Functions:
    compute_entropy_discrete
        Compute Shannon entropy of a discrete probability distribution.
    compute_entropy_from_counts
        Estimate entropy from observed counts.
    compute_max_entropy
        Compute maximum possible entropy for n states.
    compute_normalized_entropy
        Compute entropy normalized by maximum (range 0-1).
    binary_entropy
        Compute binary entropy function H(p).
    optimal_n_bins
        Compute optimal number of bins for entropy estimation.
    compute_entropy_continuous
        Estimate entropy of a continuous signal via binning.
    compute_entropy_miller_madow
        Compute entropy with Miller-Madow bias correction.
    compute_spectral_entropy
        Compute spectral entropy of a signal.
    compute_sample_entropy
        Compute sample entropy of a time series.

Mutual Information Functions:
    compute_joint_histogram
        Compute 2D histogram for joint probability estimation.
    compute_joint_entropy
        Compute joint entropy H(X, Y) of two signals.
    compute_conditional_entropy
        Compute conditional entropy H(Y|X).
    compute_mutual_information
        Compute mutual information I(X; Y) between two signals.
    compute_normalized_mi
        Compute normalized mutual information (range 0-1).
    mi_significance_test
        Test MI significance using surrogate data.
    compute_mi_sliding_window
        Compute MI in sliding windows for time-varying analysis.
    compute_time_lagged_mi
        Compute MI at different time lags for directionality.
    compute_mi_matrix
        Compute MI connectivity matrix for multiple signals.

Transfer Entropy Functions:
    create_embedding_vectors
        Create embedded state vectors for TE computation.
    compute_transfer_entropy
        Compute directed transfer entropy from X to Y.
    compute_net_transfer_entropy
        Compute TE in both directions and net flow.
    compute_te_surrogate
        Compute TE with shuffled source (null hypothesis).
    te_significance_test
        Test TE significance using surrogate distribution.
    compute_te_matrix
        Compute directed TE matrix for multi-channel data.
    compute_net_te_matrix
        Compute net TE matrix (antisymmetric).
    compute_te_hyperscanning
        Compute inter-brain TE for hyperscanning analysis.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Optional, Union
from scipy.signal import welch


def compute_entropy_discrete(
    probabilities: NDArray[np.float64],
    base: float = 2.0
) -> float:
    """
    Compute Shannon entropy of a discrete probability distribution.
    
    Parameters
    ----------
    probabilities : NDArray[np.float64]
        Probability distribution (must sum to 1).
    base : float, optional
        Logarithm base. Use 2 for bits, np.e for nats. Default is 2.
    
    Returns
    -------
    float
        Shannon entropy of the distribution.
    
    Examples
    --------
    >>> compute_entropy_discrete(np.array([0.5, 0.5]))
    1.0  # Fair coin = 1 bit
    >>> compute_entropy_discrete(np.array([0.25, 0.25, 0.25, 0.25]))
    2.0  # Uniform over 4 states = 2 bits
    """
    # Ensure probabilities sum to 1 (with tolerance)
    assert np.abs(np.sum(probabilities) - 1.0) < 1e-9, "Probabilities must sum to 1"
    
    # Filter out zeros to avoid log(0)
    p = probabilities[probabilities > 0]
    
    # Compute entropy
    if base == np.e:
        entropy = -np.sum(p * np.log(p))
    else:
        entropy = -np.sum(p * np.log(p) / np.log(base))
    
    return float(entropy)


def compute_entropy_from_counts(
    counts: NDArray[np.int64],
    base: float = 2.0
) -> float:
    """
    Estimate entropy from observed counts.
    
    Parameters
    ----------
    counts : NDArray[np.int64]
        Count of observations for each outcome.
    base : float, optional
        Logarithm base. Default is 2 (bits).
    
    Returns
    -------
    float
        Estimated Shannon entropy.
    
    Examples
    --------
    >>> compute_entropy_from_counts(np.array([50, 50]))
    1.0  # Equal counts = 1 bit
    """
    # Convert counts to probabilities
    total = np.sum(counts)
    if total == 0:
        return 0.0
    
    probabilities = counts / total
    return compute_entropy_discrete(probabilities, base)


def compute_max_entropy(n_states: int, base: float = 2.0) -> float:
    """
    Compute maximum possible entropy for n states.
    
    The maximum entropy is achieved when all states are equally likely
    (uniform distribution).
    
    Parameters
    ----------
    n_states : int
        Number of possible states/outcomes.
    base : float, optional
        Logarithm base. Default is 2 (bits).
    
    Returns
    -------
    float
        Maximum entropy = log(n_states).
    
    Examples
    --------
    >>> compute_max_entropy(2)
    1.0  # log2(2) = 1 bit
    >>> compute_max_entropy(8)
    3.0  # log2(8) = 3 bits
    """
    if n_states <= 0:
        raise ValueError("n_states must be positive")
    
    if base == np.e:
        return float(np.log(n_states))
    else:
        return float(np.log(n_states) / np.log(base))


def compute_normalized_entropy(
    probabilities: NDArray[np.float64],
    base: float = 2.0
) -> float:
    """
    Compute entropy normalized by maximum (range 0-1).
    
    Parameters
    ----------
    probabilities : NDArray[np.float64]
        Probability distribution.
    base : float, optional
        Logarithm base. Default is 2.
    
    Returns
    -------
    float
        Normalized entropy in range [0, 1].
        0 = deterministic, 1 = maximum uncertainty.
    
    Examples
    --------
    >>> compute_normalized_entropy(np.array([0.5, 0.5]))
    1.0  # Maximum entropy for 2 states
    >>> compute_normalized_entropy(np.array([1.0, 0.0]))
    0.0  # Deterministic
    """
    n_states = len(probabilities)
    if n_states <= 1:
        return 0.0
    
    H = compute_entropy_discrete(probabilities, base)
    H_max = compute_max_entropy(n_states, base)
    
    return H / H_max if H_max > 0 else 0.0


def binary_entropy(p: Union[float, NDArray[np.float64]]) -> Union[float, NDArray[np.float64]]:
    """
    Compute binary entropy function H(p) in bits.
    
    The binary entropy function gives the entropy of a Bernoulli
    random variable with probability p.
    
    Parameters
    ----------
    p : float or NDArray
        Probability value(s) in range [0, 1].
    
    Returns
    -------
    float or NDArray
        Binary entropy H(p) = -p*log2(p) - (1-p)*log2(1-p).
    
    Examples
    --------
    >>> binary_entropy(0.5)
    1.0  # Maximum at p=0.5
    >>> binary_entropy(0.0)
    0.0  # Deterministic
    """
    p = np.asarray(p)
    
    # Handle edge cases
    result = np.zeros_like(p, dtype=float)
    
    # Only compute for valid probabilities (not 0 or 1)
    valid = (p > 0) & (p < 1)
    p_valid = p[valid]
    result[valid] = -p_valid * np.log2(p_valid) - (1 - p_valid) * np.log2(1 - p_valid)
    
    # Return scalar if input was scalar
    if result.ndim == 0:
        return float(result)
    return result


def optimal_n_bins(n_samples: int, method: str = "sturges") -> int:
    """
    Compute optimal number of bins for entropy estimation.
    
    Parameters
    ----------
    n_samples : int
        Number of data samples.
    method : str, optional
        Method for determining bins:
        - "sturges": 1 + log2(n) (default, good for normal distributions)
        - "sqrt": sqrt(n) (simple rule)
        - "rice": 2 * n^(1/3) (better for larger datasets)
    
    Returns
    -------
    int
        Recommended number of bins.
    
    Examples
    --------
    >>> optimal_n_bins(1000, "sturges")
    11
    >>> optimal_n_bins(1000, "sqrt")
    32
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    
    if method == "sturges":
        return max(1, int(np.ceil(1 + np.log2(n_samples))))
    elif method == "sqrt":
        return max(1, int(np.ceil(np.sqrt(n_samples))))
    elif method == "rice":
        return max(1, int(np.ceil(2 * n_samples ** (1/3))))
    else:
        raise ValueError(f"Unknown method: {method}. Use 'sturges', 'sqrt', or 'rice'.")


def compute_entropy_continuous(
    signal: NDArray[np.float64],
    n_bins: Union[int, str] = "auto",
    method: str = "uniform"
) -> Tuple[float, int]:
    """
    Estimate entropy of a continuous signal via binning.
    
    Parameters
    ----------
    signal : NDArray[np.float64]
        Continuous signal to analyze.
    n_bins : int or str, optional
        Number of bins, or "auto"/"sturges"/"sqrt" for automatic.
        Default is "auto" (uses Sturges' rule).
    method : str, optional
        Binning method:
        - "uniform": Equal width bins (default)
        - "equiprobable": Equal count bins
    
    Returns
    -------
    Tuple[float, int]
        (entropy in bits, actual number of bins used)
    
    Examples
    --------
    >>> signal = np.random.randn(1000)
    >>> H, n_bins = compute_entropy_continuous(signal)
    """
    n_samples = len(signal)
    
    # Determine number of bins
    if isinstance(n_bins, str):
        if n_bins == "auto" or n_bins == "sturges":
            actual_bins = optimal_n_bins(n_samples, "sturges")
        elif n_bins == "sqrt":
            actual_bins = optimal_n_bins(n_samples, "sqrt")
        else:
            actual_bins = optimal_n_bins(n_samples, "sturges")
    else:
        actual_bins = int(n_bins)
    
    # Compute histogram
    if method == "uniform":
        counts, _ = np.histogram(signal, bins=actual_bins)
    elif method == "equiprobable":
        # Create bins with equal number of samples
        percentiles = np.linspace(0, 100, actual_bins + 1)
        bin_edges = np.percentile(signal, percentiles)
        counts, _ = np.histogram(signal, bins=bin_edges)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Compute entropy from counts
    entropy = compute_entropy_from_counts(counts)
    
    return entropy, actual_bins


def compute_entropy_miller_madow(
    signal: NDArray[np.float64],
    n_bins: int,
    base: float = 2.0
) -> float:
    """
    Compute entropy with Miller-Madow bias correction.
    
    The Miller-Madow correction adds (m-1)/(2n) to the raw entropy
    estimate, where m is the number of non-empty bins and n is the
    sample size.
    
    Parameters
    ----------
    signal : NDArray[np.float64]
        Continuous signal to analyze.
    n_bins : int
        Number of bins for discretization.
    base : float, optional
        Logarithm base. Default is 2 (bits).
    
    Returns
    -------
    float
        Bias-corrected entropy estimate.
    
    Examples
    --------
    >>> signal = np.random.uniform(0, 1, 100)
    >>> H_corrected = compute_entropy_miller_madow(signal, n_bins=20)
    """
    n_samples = len(signal)
    
    # Compute histogram
    counts, _ = np.histogram(signal, bins=n_bins)
    
    # Number of non-empty bins
    m = np.sum(counts > 0)
    
    # Raw entropy
    H_raw = compute_entropy_from_counts(counts, base)
    
    # Miller-Madow correction
    if base == np.e:
        correction = (m - 1) / (2 * n_samples)
    else:
        correction = (m - 1) / (2 * n_samples * np.log(base))
    
    return H_raw + correction


def compute_spectral_entropy(
    signal: NDArray[np.float64],
    fs: float,
    nperseg: int = 256,
    freq_range: Optional[Tuple[float, float]] = None,
    normalize: bool = True
) -> Tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    """
    Compute spectral entropy of a signal.
    
    Spectral entropy measures how spread the power is across frequencies.
    High spectral entropy indicates broadband signal (power spread),
    low spectral entropy indicates narrowband signal (power concentrated).
    
    Parameters
    ----------
    signal : NDArray[np.float64]
        Time series signal.
    fs : float
        Sampling frequency in Hz.
    nperseg : int, optional
        Length of each segment for Welch's method. Default is 256.
    freq_range : Tuple[float, float], optional
        Frequency range (fmin, fmax) to consider. Default is None (all).
    normalize : bool, optional
        Whether to normalize by maximum entropy. Default is True.
    
    Returns
    -------
    Tuple[float, NDArray, NDArray]
        (spectral_entropy, frequencies, psd)
    
    Examples
    --------
    >>> t = np.arange(0, 10, 1/256)
    >>> signal = np.sin(2 * np.pi * 10 * t)  # Pure sine
    >>> H_spec, freqs, psd = compute_spectral_entropy(signal, fs=256)
    >>> # H_spec will be low (narrowband)
    """
    # Compute PSD using Welch's method
    freqs, psd = welch(signal, fs=fs, nperseg=min(nperseg, len(signal)))
    
    # Apply frequency range if specified
    if freq_range is not None:
        mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
        freqs = freqs[mask]
        psd = psd[mask]
    
    # Normalize PSD to make it a probability distribution
    psd_norm = psd / np.sum(psd)
    
    # Remove zeros
    psd_valid = psd_norm[psd_norm > 0]
    
    # Compute entropy
    H_spectral = -np.sum(psd_valid * np.log2(psd_valid))
    
    # Normalize if requested
    if normalize:
        H_max = np.log2(len(psd_valid))
        if H_max > 0:
            H_spectral = H_spectral / H_max
    
    return float(H_spectral), freqs, psd


def compute_sample_entropy(
    signal: NDArray[np.float64],
    m: int = 2,
    r: Optional[float] = None
) -> float:
    """
    Compute sample entropy of a time series.
    
    Sample entropy measures the complexity/irregularity of a signal
    by comparing patterns at different time points. It does not
    require binning.
    
    Parameters
    ----------
    signal : NDArray[np.float64]
        Time series signal.
    m : int, optional
        Embedding dimension (pattern length). Default is 2.
    r : float, optional
        Tolerance for pattern matching. Default is 0.2 * std(signal).
    
    Returns
    -------
    float
        Sample entropy value.
        Higher values indicate more complexity/irregularity.
        Lower values indicate more self-similarity/regularity.
    
    Notes
    -----
    Computational complexity is O(n²), so use shorter signals
    (e.g., 500-2000 samples) for reasonable computation time.
    
    Examples
    --------
    >>> regular_signal = np.sin(np.linspace(0, 10*np.pi, 500))
    >>> random_signal = np.random.randn(500)
    >>> se_regular = compute_sample_entropy(regular_signal)
    >>> se_random = compute_sample_entropy(random_signal)
    >>> # se_random > se_regular
    """
    N = len(signal)
    
    if r is None:
        r = 0.2 * np.std(signal)
    
    def count_matches(template_length: int) -> int:
        """Count pairs of matching templates."""
        templates = np.array([
            signal[i:i + template_length] 
            for i in range(N - template_length)
        ])
        
        count = 0
        n_templates = len(templates)
        
        for i in range(n_templates):
            for j in range(i + 1, n_templates):
                # Chebyshev distance (max absolute difference)
                if np.max(np.abs(templates[i] - templates[j])) <= r:
                    count += 1
        
        return count
    
    # Count matches for m and m+1
    A = count_matches(m + 1)  # matches for length m+1
    B = count_matches(m)      # matches for length m
    
    # Sample entropy
    if A == 0 or B == 0:
        return np.inf  # No matches found
    
    return -np.log(A / B)


# =============================================================================
# MUTUAL INFORMATION FUNCTIONS
# =============================================================================


def compute_joint_histogram(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    n_bins: int = 20
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Compute 2D histogram for joint probability estimation.
    
    Parameters
    ----------
    x : NDArray[np.float64]
        First signal.
    y : NDArray[np.float64]
        Second signal.
    n_bins : int, optional
        Number of bins for each dimension. Default is 20.
    
    Returns
    -------
    hist_2d : NDArray[np.float64]
        2D histogram counts.
    x_edges : NDArray[np.float64]
        Bin edges for x.
    y_edges : NDArray[np.float64]
        Bin edges for y.
    
    Examples
    --------
    >>> x = np.random.randn(1000)
    >>> y = 0.5 * x + np.random.randn(1000)
    >>> hist, x_edges, y_edges = compute_joint_histogram(x, y)
    """
    hist_2d, x_edges, y_edges = np.histogram2d(x, y, bins=n_bins)
    return hist_2d, x_edges, y_edges


def compute_joint_entropy(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    n_bins: int = 20
) -> float:
    """
    Compute joint entropy H(X, Y) of two continuous signals.
    
    Joint entropy measures the total uncertainty of both variables
    together.
    
    Parameters
    ----------
    x : NDArray[np.float64]
        First signal.
    y : NDArray[np.float64]
        Second signal.
    n_bins : int, optional
        Number of bins for discretization. Default is 20.
    
    Returns
    -------
    float
        Joint entropy H(X, Y) in bits.
    
    Examples
    --------
    >>> x = np.random.randn(1000)
    >>> y = x.copy()  # Perfect correlation
    >>> H_xy = compute_joint_entropy(x, y)
    >>> # H_xy ≈ H(X) since Y = X
    """
    hist_2d, _, _ = compute_joint_histogram(x, y, n_bins)
    
    # Convert to probabilities
    p_xy = hist_2d / np.sum(hist_2d)
    
    # Compute entropy (filter zeros)
    p_xy_flat = p_xy.flatten()
    p_xy_valid = p_xy_flat[p_xy_flat > 0]
    
    H_xy = -np.sum(p_xy_valid * np.log2(p_xy_valid))
    
    return float(H_xy)


def compute_conditional_entropy(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    n_bins: int = 20
) -> float:
    """
    Compute conditional entropy H(Y|X).
    
    Conditional entropy measures how much uncertainty remains in Y
    after knowing X.
    
    H(Y|X) = H(X, Y) - H(X)
    
    Parameters
    ----------
    x : NDArray[np.float64]
        Conditioning variable.
    y : NDArray[np.float64]
        Variable whose conditional entropy is computed.
    n_bins : int, optional
        Number of bins for discretization. Default is 20.
    
    Returns
    -------
    float
        Conditional entropy H(Y|X) in bits.
    
    Examples
    --------
    >>> x = np.random.randn(1000)
    >>> y = x + 0.1 * np.random.randn(1000)  # Y depends on X
    >>> H_y_given_x = compute_conditional_entropy(x, y)
    >>> # H_y_given_x will be small since Y ≈ X
    """
    H_xy = compute_joint_entropy(x, y, n_bins)
    H_x, _ = compute_entropy_continuous(x, n_bins)
    
    return H_xy - H_x


def compute_mutual_information(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    n_bins: int = 20
) -> float:
    """
    Compute mutual information I(X; Y) between two signals.
    
    MI measures how much information is shared between X and Y.
    It captures both linear and non-linear dependencies.
    
    I(X; Y) = H(X) + H(Y) - H(X, Y)
    
    Parameters
    ----------
    x : NDArray[np.float64]
        First signal.
    y : NDArray[np.float64]
        Second signal.
    n_bins : int, optional
        Number of bins for discretization. Default is 20.
    
    Returns
    -------
    float
        Mutual information I(X; Y) in bits.
    
    Examples
    --------
    >>> x = np.random.randn(1000)
    >>> y = x**2  # Non-linear relationship
    >>> mi = compute_mutual_information(x, y)
    >>> corr = np.corrcoef(x, y)[0, 1]
    >>> # MI will be high even though correlation ≈ 0
    """
    H_x, _ = compute_entropy_continuous(x, n_bins)
    H_y, _ = compute_entropy_continuous(y, n_bins)
    H_xy = compute_joint_entropy(x, y, n_bins)
    
    mi = H_x + H_y - H_xy
    
    # MI should be non-negative (numerical errors can make it slightly negative)
    return float(max(0, mi))


def compute_normalized_mi(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    n_bins: int = 20,
    method: str = "arithmetic"
) -> float:
    """
    Compute normalized mutual information (range 0-1).
    
    Parameters
    ----------
    x : NDArray[np.float64]
        First signal.
    y : NDArray[np.float64]
        Second signal.
    n_bins : int, optional
        Number of bins for discretization. Default is 20.
    method : str, optional
        Normalization method:
        - "arithmetic": NMI = 2 * I(X;Y) / (H(X) + H(Y))
        - "geometric": NMI = I(X;Y) / sqrt(H(X) * H(Y))
        - "min": NMI = I(X;Y) / min(H(X), H(Y))
        - "max": NMI = I(X;Y) / max(H(X), H(Y))
        Default is "arithmetic".
    
    Returns
    -------
    float
        Normalized MI in range [0, 1].
    
    Examples
    --------
    >>> x = np.random.randn(1000)
    >>> y = x.copy()  # Perfect dependence
    >>> nmi = compute_normalized_mi(x, y)
    >>> # nmi ≈ 1.0
    """
    H_x, _ = compute_entropy_continuous(x, n_bins)
    H_y, _ = compute_entropy_continuous(y, n_bins)
    mi = compute_mutual_information(x, y, n_bins)
    
    if method == "arithmetic":
        denom = (H_x + H_y) / 2
    elif method == "geometric":
        denom = np.sqrt(H_x * H_y)
    elif method == "min":
        denom = min(H_x, H_y)
    elif method == "max":
        denom = max(H_x, H_y)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    if denom <= 0:
        return 0.0
    
    return float(min(1.0, mi / denom))


def mi_significance_test(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    n_bins: int = 20,
    n_surrogates: int = 100,
    alpha: float = 0.05
) -> dict:
    """
    Test MI significance using surrogate data.
    
    Creates shuffled surrogates to build a null distribution
    of MI values under the hypothesis of independence.
    
    Parameters
    ----------
    x : NDArray[np.float64]
        First signal.
    y : NDArray[np.float64]
        Second signal.
    n_bins : int, optional
        Number of bins for MI estimation. Default is 20.
    n_surrogates : int, optional
        Number of shuffled surrogates. Default is 100.
    alpha : float, optional
        Significance level. Default is 0.05.
    
    Returns
    -------
    dict
        Dictionary with:
        - 'mi_observed': Observed MI value
        - 'mi_surrogates': Array of surrogate MI values
        - 'p_value': Proportion of surrogates >= observed MI
        - 'significant': Boolean, whether MI is significant
        - 'threshold': MI threshold at (1-alpha) percentile
    
    Examples
    --------
    >>> x = np.random.randn(1000)
    >>> y = 0.5 * x + np.random.randn(1000)
    >>> result = mi_significance_test(x, y)
    >>> print(f"Significant: {result['significant']}")
    """
    # Observed MI
    mi_observed = compute_mutual_information(x, y, n_bins)
    
    # Generate surrogate distribution
    mi_surrogates = np.zeros(n_surrogates)
    
    for i in range(n_surrogates):
        # Shuffle one signal to break any dependency
        y_shuffled = np.random.permutation(y)
        mi_surrogates[i] = compute_mutual_information(x, y_shuffled, n_bins)
    
    # Compute p-value (proportion of surrogates >= observed)
    p_value = np.mean(mi_surrogates >= mi_observed)
    
    # Significance threshold
    threshold = np.percentile(mi_surrogates, 100 * (1 - alpha))
    
    return {
        'mi_observed': mi_observed,
        'mi_surrogates': mi_surrogates,
        'p_value': p_value,
        'significant': p_value < alpha,
        'threshold': threshold
    }


def compute_mi_sliding_window(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    window_size: int,
    step_size: int,
    n_bins: int = 20
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Compute MI in sliding windows for time-varying analysis.
    
    Parameters
    ----------
    x : NDArray[np.float64]
        First signal.
    y : NDArray[np.float64]
        Second signal.
    window_size : int
        Window size in samples.
    step_size : int
        Step size between windows in samples.
    n_bins : int, optional
        Number of bins for MI estimation. Default is 20.
    
    Returns
    -------
    centers : NDArray[np.float64]
        Window center positions in samples.
    mi_values : NDArray[np.float64]
        MI value for each window.
    
    Examples
    --------
    >>> fs = 256
    >>> t = np.arange(0, 10, 1/fs)
    >>> x = np.random.randn(len(t))
    >>> y = np.random.randn(len(t))
    >>> y[fs*3:fs*7] += 0.5 * x[fs*3:fs*7]  # Coupling in middle
    >>> centers, mi_time = compute_mi_sliding_window(x, y, fs*2, fs//2)
    """
    n_samples = len(x)
    n_windows = (n_samples - window_size) // step_size + 1
    
    centers = np.zeros(n_windows)
    mi_values = np.zeros(n_windows)
    
    for i in range(n_windows):
        start = i * step_size
        end = start + window_size
        centers[i] = (start + end) / 2
        
        mi_values[i] = compute_mutual_information(x[start:end], y[start:end], n_bins)
    
    return centers, mi_values


def compute_time_lagged_mi(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    max_lag: int,
    n_bins: int = 20
) -> Tuple[NDArray[np.int64], NDArray[np.float64]]:
    """
    Compute MI at different time lags for directionality analysis.
    
    Parameters
    ----------
    x : NDArray[np.float64]
        First signal.
    y : NDArray[np.float64]
        Second signal.
    max_lag : int
        Maximum lag in samples (both positive and negative).
    n_bins : int, optional
        Number of bins for MI estimation. Default is 20.
    
    Returns
    -------
    lags : NDArray[np.int64]
        Array of lag values (negative = X leads, positive = Y leads).
    mi_values : NDArray[np.float64]
        MI values at each lag.
    
    Examples
    --------
    >>> x = np.random.randn(1000)
    >>> y = np.zeros(1000)
    >>> y[10:] = x[:-10]  # Y follows X with 10 sample delay
    >>> lags, mi_lagged = compute_time_lagged_mi(x, y, max_lag=50)
    >>> peak_lag = lags[np.argmax(mi_lagged)]
    >>> # peak_lag should be around -10 (X leads)
    """
    lags = np.arange(-max_lag, max_lag + 1)
    mi_values = np.zeros(len(lags))
    
    for i, lag in enumerate(lags):
        if lag < 0:
            x_shifted = x[:lag]  # X leads
            y_shifted = y[-lag:]
        elif lag > 0:
            x_shifted = x[lag:]  # Y leads
            y_shifted = y[:-lag]
        else:
            x_shifted = x
            y_shifted = y
        
        mi_values[i] = compute_mutual_information(x_shifted, y_shifted, n_bins)
    
    return lags, mi_values


def compute_mi_matrix(
    signals: NDArray[np.float64],
    n_bins: int = 20,
    normalize: bool = True
) -> NDArray[np.float64]:
    """
    Compute MI connectivity matrix for multiple signals.
    
    Parameters
    ----------
    signals : NDArray[np.float64]
        2D array of shape (n_channels, n_samples).
    n_bins : int, optional
        Number of bins for MI estimation. Default is 20.
    normalize : bool, optional
        If True, normalize MI to [0, 1] range. Default is True.
    
    Returns
    -------
    mi_matrix : NDArray[np.float64]
        Symmetric MI matrix of shape (n_channels, n_channels).
    
    Examples
    --------
    >>> signals = np.random.randn(5, 1000)  # 5 channels
    >>> mi_matrix = compute_mi_matrix(signals)
    >>> # mi_matrix[i, j] is MI between channel i and j
    """
    n_channels = signals.shape[0]
    mi_matrix = np.zeros((n_channels, n_channels))
    
    for i in range(n_channels):
        for j in range(i + 1, n_channels):
            if normalize:
                mi = compute_normalized_mi(signals[i], signals[j], n_bins)
            else:
                mi = compute_mutual_information(signals[i], signals[j], n_bins)
            
            mi_matrix[i, j] = mi
            mi_matrix[j, i] = mi
    
    return mi_matrix


# =============================================================================
# TRANSFER ENTROPY FUNCTIONS
# =============================================================================


def create_embedding_vectors(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    k: int = 1,
    l: int = 1,
    tau: int = 1
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Create embedded state vectors for transfer entropy computation.
    
    Parameters
    ----------
    x : NDArray[np.float64]
        Source signal.
    y : NDArray[np.float64]
        Target signal.
    k : int, optional
        History length for target Y. Default is 1.
    l : int, optional
        History length for source X. Default is 1.
    tau : int, optional
        Embedding delay in samples. Default is 1.
    
    Returns
    -------
    y_future : NDArray[np.float64]
        Future values of Y, shape (n_valid,).
    y_past : NDArray[np.float64]
        Past values of Y, shape (n_valid, k).
    x_past : NDArray[np.float64]
        Past values of X, shape (n_valid, l).
    
    Examples
    --------
    >>> x = np.random.randn(1000)
    >>> y = np.random.randn(1000)
    >>> y_fut, y_past, x_past = create_embedding_vectors(x, y, k=2, l=2, tau=5)
    """
    n = len(x)
    
    # Determine valid range
    max_history = max(k, l) * tau
    start_idx = max_history
    
    n_valid = n - start_idx
    
    # Initialize arrays
    y_future = np.zeros(n_valid)
    y_past = np.zeros((n_valid, k))
    x_past = np.zeros((n_valid, l))
    
    for i in range(n_valid):
        t = start_idx + i
        y_future[i] = y[t]
        
        # Y past: [y(t-tau), y(t-2*tau), ..., y(t-k*tau)]
        for j in range(k):
            y_past[i, j] = y[t - (j + 1) * tau]
        
        # X past: [x(t-tau), x(t-2*tau), ..., x(t-l*tau)]
        for j in range(l):
            x_past[i, j] = x[t - (j + 1) * tau]
    
    return y_future, y_past, x_past


def compute_transfer_entropy(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    k: int = 1,
    l: int = 1,
    tau: int = 1,
    n_bins: int = 8
) -> float:
    """
    Compute transfer entropy from X to Y.
    
    Transfer entropy measures the directed information flow from X to Y,
    quantifying how much knowing X's past helps predict Y's future,
    beyond what Y's past already tells us.
    
    TE_{X→Y} = I(Y_t; X_past | Y_past)
             = H(Y_t | Y_past) - H(Y_t | Y_past, X_past)
    
    Parameters
    ----------
    x : NDArray[np.float64]
        Source signal.
    y : NDArray[np.float64]
        Target signal.
    k : int, optional
        History length for target Y. Default is 1.
    l : int, optional
        History length for source X. Default is 1.
    tau : int, optional
        Embedding delay in samples. Default is 1.
    n_bins : int, optional
        Number of bins for discretization. Default is 8.
    
    Returns
    -------
    float
        Transfer entropy from X to Y in bits.
    
    Examples
    --------
    >>> x = np.random.randn(1000)
    >>> y = np.zeros(1000)
    >>> y[10:] = 0.5 * x[:-10] + 0.5 * np.random.randn(990)  # Y follows X
    >>> te_xy = compute_transfer_entropy(x, y, k=1, l=1, tau=10)
    >>> te_yx = compute_transfer_entropy(y, x, k=1, l=1, tau=10)
    >>> # te_xy > te_yx (X drives Y)
    """
    # Create embedding vectors
    y_future, y_past, x_past = create_embedding_vectors(x, y, k, l, tau)
    
    n_samples = len(y_future)
    
    # Discretize all variables
    def discretize(arr: NDArray[np.float64]) -> NDArray[np.int64]:
        """Discretize array into bins."""
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        
        result = np.zeros(arr.shape, dtype=np.int64)
        for col in range(arr.shape[1]):
            percentiles = np.linspace(0, 100, n_bins + 1)
            bin_edges = np.percentile(arr[:, col], percentiles)
            result[:, col] = np.digitize(arr[:, col], bin_edges[1:-1])
        
        return result
    
    y_fut_d = discretize(y_future).flatten()
    y_past_d = discretize(y_past)
    x_past_d = discretize(x_past)
    
    # Convert multi-dimensional indices to single index
    def to_single_index(arr: NDArray[np.int64]) -> NDArray[np.int64]:
        """Convert multi-column discrete array to single index."""
        if arr.ndim == 1:
            return arr
        result = np.zeros(len(arr), dtype=np.int64)
        multiplier = 1
        for col in range(arr.shape[1] - 1, -1, -1):
            result += arr[:, col] * multiplier
            multiplier *= n_bins
        return result
    
    y_past_idx = to_single_index(y_past_d)
    x_past_idx = to_single_index(x_past_d)
    
    # Combined index for (y_past, x_past)
    yx_past_idx = y_past_idx * (n_bins ** l) + x_past_idx
    
    # Compute conditional entropies
    def entropy_from_joint(idx1: NDArray[np.int64], idx2: NDArray[np.int64]) -> float:
        """Compute H(idx1 | idx2) = H(idx1, idx2) - H(idx2)."""
        joint = np.ravel_multi_index(
            (idx1, idx2), 
            (int(idx1.max()) + 1, int(idx2.max()) + 1)
        )
        _, joint_counts = np.unique(joint, return_counts=True)
        p_joint = joint_counts / n_samples
        H_joint = -np.sum(p_joint * np.log2(p_joint + 1e-12))
        
        _, marginal_counts = np.unique(idx2, return_counts=True)
        p_marginal = marginal_counts / n_samples
        H_marginal = -np.sum(p_marginal * np.log2(p_marginal + 1e-12))
        
        return H_joint - H_marginal
    
    # H(Y_t | Y_past)
    H_y_given_ypast = entropy_from_joint(y_fut_d, y_past_idx)
    
    # H(Y_t | Y_past, X_past)
    H_y_given_yxpast = entropy_from_joint(y_fut_d, yx_past_idx)
    
    # Transfer Entropy
    te = H_y_given_ypast - H_y_given_yxpast
    
    return max(0.0, float(te))


def compute_net_transfer_entropy(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    k: int = 1,
    l: int = 1,
    tau: int = 1,
    n_bins: int = 8
) -> dict:
    """
    Compute transfer entropy in both directions and net flow.
    
    Net TE = TE_{X→Y} - TE_{Y→X}
    - Positive: X dominates (more information flows X→Y)
    - Negative: Y dominates (more information flows Y→X)
    - Zero: Balanced bidirectional coupling
    
    Parameters
    ----------
    x : NDArray[np.float64]
        First signal.
    y : NDArray[np.float64]
        Second signal.
    k : int, optional
        History length. Default is 1.
    l : int, optional
        History length. Default is 1.
    tau : int, optional
        Embedding delay. Default is 1.
    n_bins : int, optional
        Number of bins. Default is 8.
    
    Returns
    -------
    dict
        Dictionary with:
        - 'te_x_to_y': TE from X to Y
        - 'te_y_to_x': TE from Y to X
        - 'net_te': Net transfer entropy (positive = X dominates)
        - 'dominant': 'X', 'Y', or 'balanced'
    
    Examples
    --------
    >>> x = np.random.randn(1000)
    >>> y = np.zeros(1000)
    >>> y[5:] = 0.5 * x[:-5] + np.random.randn(995)
    >>> result = compute_net_transfer_entropy(x, y, tau=5)
    >>> print(f"Dominant: {result['dominant']}")  # Should be 'X'
    """
    te_x_to_y = compute_transfer_entropy(x, y, k, l, tau, n_bins)
    te_y_to_x = compute_transfer_entropy(y, x, k, l, tau, n_bins)
    
    net_te = te_x_to_y - te_y_to_x
    
    # Determine dominant direction (with small threshold for "balanced")
    threshold = 0.01
    if net_te > threshold:
        dominant = "X"
    elif net_te < -threshold:
        dominant = "Y"
    else:
        dominant = "balanced"
    
    return {
        "te_x_to_y": te_x_to_y,
        "te_y_to_x": te_y_to_x,
        "net_te": net_te,
        "dominant": dominant
    }


def compute_te_surrogate(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    k: int = 1,
    l: int = 1,
    tau: int = 1,
    n_bins: int = 8,
    seed: Optional[int] = None
) -> float:
    """
    Compute TE with shuffled source signal (null hypothesis).
    
    Parameters
    ----------
    x : NDArray[np.float64]
        Source signal (will be shuffled).
    y : NDArray[np.float64]
        Target signal (kept intact).
    k : int, optional
        History length for target. Default is 1.
    l : int, optional
        History length for source. Default is 1.
    tau : int, optional
        Embedding delay. Default is 1.
    n_bins : int, optional
        Number of bins. Default is 8.
    seed : int, optional
        Random seed.
    
    Returns
    -------
    float
        TE computed on shuffled source (null TE).
    """
    rng = np.random.RandomState(seed)
    x_shuffled = rng.permutation(x)
    return compute_transfer_entropy(x_shuffled, y, k, l, tau, n_bins)


def te_significance_test(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    k: int = 1,
    l: int = 1,
    tau: int = 1,
    n_bins: int = 8,
    n_surrogates: int = 200,
    seed: Optional[int] = None
) -> dict:
    """
    Test significance of transfer entropy using surrogate distribution.
    
    Parameters
    ----------
    x : NDArray[np.float64]
        Source signal.
    y : NDArray[np.float64]
        Target signal.
    k : int, optional
        History length for target. Default is 1.
    l : int, optional
        History length for source. Default is 1.
    tau : int, optional
        Embedding delay. Default is 1.
    n_bins : int, optional
        Number of bins. Default is 8.
    n_surrogates : int, optional
        Number of surrogates. Default is 200.
    seed : int, optional
        Random seed.
    
    Returns
    -------
    dict
        Contains: te_observed, te_effective, null_mean, null_std, p_value
    """
    te_observed = compute_transfer_entropy(x, y, k, l, tau, n_bins)
    
    rng = np.random.RandomState(seed)
    null_te = np.array([
        compute_te_surrogate(x, y, k, l, tau, n_bins, seed=rng.randint(100000))
        for _ in range(n_surrogates)
    ])
    
    null_mean = float(np.mean(null_te))
    null_std = float(np.std(null_te))
    te_effective = te_observed - null_mean
    p_value = float(np.mean(null_te >= te_observed))
    
    return {
        "te_observed": te_observed,
        "te_effective": te_effective,
        "null_mean": null_mean,
        "null_std": null_std,
        "null_distribution": null_te,
        "p_value": p_value
    }


def compute_te_matrix(
    data: NDArray[np.float64],
    k: int = 1,
    l: int = 1,
    tau: int = 1,
    n_bins: int = 8
) -> NDArray[np.float64]:
    """
    Compute directed TE matrix for multi-channel data.
    
    Unlike MI matrix, the TE matrix is NOT symmetric:
    TE[i, j] = TE from channel i to channel j
    
    Parameters
    ----------
    data : NDArray[np.float64]
        Multi-channel data of shape (n_channels, n_samples).
    k : int, optional
        History length for target. Default is 1.
    l : int, optional
        History length for source. Default is 1.
    tau : int, optional
        Embedding delay. Default is 1.
    n_bins : int, optional
        Number of bins. Default is 8.
    
    Returns
    -------
    NDArray[np.float64]
        TE matrix of shape (n_channels, n_channels).
        Entry [i, j] = TE from channel i to channel j.
        Diagonal is NaN.
    
    Examples
    --------
    >>> data = np.random.randn(5, 1000)
    >>> te_matrix = compute_te_matrix(data)
    >>> # Note: te_matrix[i,j] != te_matrix[j,i] in general
    """
    n_channels = data.shape[0]
    te_matrix = np.full((n_channels, n_channels), np.nan)
    
    for i in range(n_channels):
        for j in range(n_channels):
            if i != j:
                te_matrix[i, j] = compute_transfer_entropy(
                    data[i], data[j], k, l, tau, n_bins
                )
    
    return te_matrix


def compute_net_te_matrix(te_matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Compute net TE matrix from directed TE matrix.
    
    Net[i, j] = TE[i, j] - TE[j, i]
    
    The net TE matrix is antisymmetric: Net[i, j] = -Net[j, i]
    
    Parameters
    ----------
    te_matrix : NDArray[np.float64]
        Directed TE matrix from compute_te_matrix.
    
    Returns
    -------
    NDArray[np.float64]
        Net TE matrix (antisymmetric).
    
    Examples
    --------
    >>> te_matrix = compute_te_matrix(data)
    >>> net_te = compute_net_te_matrix(te_matrix)
    >>> # net_te[i,j] > 0 means i→j dominates
    """
    return te_matrix - te_matrix.T


def compute_te_hyperscanning(
    data_p1: NDArray[np.float64],
    data_p2: NDArray[np.float64],
    k: int = 1,
    l: int = 1,
    tau: int = 1,
    n_bins: int = 8
) -> dict:
    """
    Compute inter-brain TE for hyperscanning analysis.
    
    Computes TE between all channel pairs across two participants
    to determine who leads the interaction.
    
    Parameters
    ----------
    data_p1 : NDArray[np.float64]
        Participant 1 data, shape (n_channels, n_samples).
    data_p2 : NDArray[np.float64]
        Participant 2 data, shape (n_channels, n_samples).
    k : int, optional
        History length. Default is 1.
    l : int, optional
        History length. Default is 1.
    tau : int, optional
        Embedding delay. Default is 1.
    n_bins : int, optional
        Number of bins. Default is 8.
    
    Returns
    -------
    dict
        Contains:
        - 'te_p1_to_p2': TE matrix from P1 to P2
        - 'te_p2_to_p1': TE matrix from P2 to P1
        - 'mean_p1_to_p2': Mean TE from P1 to P2
        - 'mean_p2_to_p1': Mean TE from P2 to P1
        - 'net_te': Net TE (positive = P1 leads)
        - 'leader': 'P1', 'P2', or 'balanced'
    
    Examples
    --------
    >>> data_p1 = np.random.randn(3, 1000)  # 3 channels
    >>> data_p2 = np.random.randn(3, 1000)
    >>> result = compute_te_hyperscanning(data_p1, data_p2)
    >>> print(f"Leader: {result['leader']}")
    """
    n_ch1 = data_p1.shape[0]
    n_ch2 = data_p2.shape[0]
    
    te_p1_to_p2 = np.zeros((n_ch1, n_ch2))
    te_p2_to_p1 = np.zeros((n_ch2, n_ch1))
    
    for i in range(n_ch1):
        for j in range(n_ch2):
            te_p1_to_p2[i, j] = compute_transfer_entropy(
                data_p1[i], data_p2[j], k, l, tau, n_bins
            )
            te_p2_to_p1[j, i] = compute_transfer_entropy(
                data_p2[j], data_p1[i], k, l, tau, n_bins
            )
    
    mean_p1_to_p2 = float(np.mean(te_p1_to_p2))
    mean_p2_to_p1 = float(np.mean(te_p2_to_p1))
    net_te = mean_p1_to_p2 - mean_p2_to_p1
    
    threshold = 0.01
    if net_te > threshold:
        leader = "P1"
    elif net_te < -threshold:
        leader = "P2"
    else:
        leader = "balanced"
    
    return {
        "te_p1_to_p2": te_p1_to_p2,
        "te_p2_to_p1": te_p2_to_p1,
        "mean_p1_to_p2": mean_p1_to_p2,
        "mean_p2_to_p1": mean_p2_to_p1,
        "net_te": net_te,
        "leader": leader
    }
