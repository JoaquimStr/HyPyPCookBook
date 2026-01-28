# ============================================================================
# Statistical Significance for Connectivity Analysis
# ============================================================================
#
# This module provides functions for testing statistical significance of
# connectivity metrics using surrogate data methods, multiple comparisons
# correction, and effect size computation.
#
# ============================================================================

import numpy as np
from numpy.typing import NDArray
from scipy import stats
from scipy.fft import fft, ifft
from scipy.signal import hilbert
from typing import Any, Callable, Dict, List, Optional, Tuple


# ============================================================================
# Surrogate Data Generation
# ============================================================================

def phase_shuffle(signal: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Create a phase-shuffled surrogate of a signal.
    
    Preserves the power spectrum while randomizing phase relationships.
    This is the gold standard for testing phase-based connectivity metrics.
    
    Parameters
    ----------
    signal : NDArray[np.floating]
        Input signal (1D array).
    
    Returns
    -------
    NDArray[np.floating]
        Phase-shuffled surrogate signal with identical power spectrum.
    
    Examples
    --------
    >>> signal = np.sin(2 * np.pi * 10 * np.arange(256) / 256)
    >>> surrogate = phase_shuffle(signal)
    >>> # Power spectra are identical
    >>> np.allclose(np.abs(np.fft.fft(signal)), np.abs(np.fft.fft(surrogate)))
    True
    """
    n = len(signal)
    
    # FFT
    spectrum = fft(signal)
    
    # Get magnitude
    magnitude = np.abs(spectrum)
    
    # Generate random phases (symmetric for real output)
    random_phases = np.random.uniform(0, 2 * np.pi, n // 2 + 1)
    
    # Build symmetric phase array for real signal
    if n % 2 == 0:  # Even length
        new_phases = np.concatenate([
            [0],  # DC component (no phase)
            random_phases[1:-1],
            [0],  # Nyquist (no phase)
            -random_phases[-2:0:-1]  # Negative frequencies
        ])
    else:  # Odd length
        new_phases = np.concatenate([
            [0],  # DC component
            random_phases[1:],
            -random_phases[-1:0:-1]  # Negative frequencies
        ])
    
    # Reconstruct spectrum with new phases
    surrogate_spectrum = magnitude * np.exp(1j * new_phases)
    
    # Inverse FFT
    surrogate = np.real(ifft(surrogate_spectrum))
    
    return surrogate


def time_shift(signal: NDArray[np.floating],
               min_shift: Optional[int] = None,
               max_shift: Optional[int] = None) -> NDArray[np.floating]:
    """
    Create a time-shifted surrogate of a signal.
    
    A simpler and faster alternative to phase shuffling. Uses circular
    shifting to break temporal alignment between signals.
    
    Parameters
    ----------
    signal : NDArray[np.floating]
        Input signal (1D array).
    min_shift : int, optional
        Minimum shift in samples. Default: 10% of signal length.
    max_shift : int, optional
        Maximum shift in samples. Default: 90% of signal length.
    
    Returns
    -------
    NDArray[np.floating]
        Time-shifted surrogate signal (circular shift).
    
    Notes
    -----
    Time shifting is less rigorous than phase shuffling but much faster.
    Use for quick exploratory analyses or very long signals.
    """
    n = len(signal)
    
    if min_shift is None:
        min_shift = n // 10
    if max_shift is None:
        max_shift = 9 * n // 10
    
    # Random shift
    shift = np.random.randint(min_shift, max_shift)
    
    # Circular shift (wraps around)
    surrogate = np.roll(signal, shift)
    
    return surrogate


def generate_surrogates(signal: NDArray[np.floating],
                        n_surrogates: int = 1000,
                        method: str = 'phase_shuffle') -> NDArray[np.floating]:
    """
    Generate multiple surrogate signals.
    
    Parameters
    ----------
    signal : NDArray[np.floating]
        Input signal (1D array).
    n_surrogates : int
        Number of surrogates to generate. Default: 1000.
    method : str
        Surrogate method: 'phase_shuffle' or 'time_shift'.
    
    Returns
    -------
    NDArray[np.floating]
        Array of shape (n_surrogates, len(signal)) containing surrogates.
    
    Raises
    ------
    ValueError
        If method is not recognized.
    """
    if method == 'phase_shuffle':
        surrogate_func = phase_shuffle
    elif method == 'time_shift':
        surrogate_func = time_shift
    else:
        raise ValueError(f"Unknown method: {method}. Use 'phase_shuffle' or 'time_shift'.")
    
    surrogates = np.zeros((n_surrogates, len(signal)))
    for i in range(n_surrogates):
        surrogates[i] = surrogate_func(signal)
    
    return surrogates


# ============================================================================
# Null Distribution and P-Value Computation
# ============================================================================

def build_null_distribution(signal1: NDArray[np.floating],
                            signal2: NDArray[np.floating],
                            connectivity_func: Callable,
                            n_surrogates: int = 1000,
                            method: str = 'phase_shuffle',
                            shuffle_which: str = 'second') -> NDArray[np.floating]:
    """
    Build null distribution of connectivity using surrogate data.
    
    Parameters
    ----------
    signal1 : NDArray[np.floating]
        First signal.
    signal2 : NDArray[np.floating]
        Second signal.
    connectivity_func : Callable
        Function that takes two signals and returns a connectivity value.
        Signature: connectivity_func(signal1, signal2) -> float
    n_surrogates : int
        Number of surrogates to generate.
    method : str
        Surrogate method: 'phase_shuffle' or 'time_shift'.
    shuffle_which : str
        Which signal to shuffle: 'first', 'second', or 'both'.
    
    Returns
    -------
    NDArray[np.floating]
        Array of connectivity values under the null hypothesis.
    """
    null_values = np.zeros(n_surrogates)
    
    for i in range(n_surrogates):
        if shuffle_which == 'first':
            if method == 'phase_shuffle':
                surr1 = phase_shuffle(signal1)
            else:
                surr1 = time_shift(signal1)
            null_values[i] = connectivity_func(surr1, signal2)
        elif shuffle_which == 'second':
            if method == 'phase_shuffle':
                surr2 = phase_shuffle(signal2)
            else:
                surr2 = time_shift(signal2)
            null_values[i] = connectivity_func(signal1, surr2)
        else:  # both
            if method == 'phase_shuffle':
                surr1 = phase_shuffle(signal1)
                surr2 = phase_shuffle(signal2)
            else:
                surr1 = time_shift(signal1)
                surr2 = time_shift(signal2)
            null_values[i] = connectivity_func(surr1, surr2)
    
    return null_values


def compute_pvalue(observed: float,
                   null_distribution: NDArray[np.floating],
                   alternative: str = 'greater') -> float:
    """
    Compute p-value from null distribution.
    
    Parameters
    ----------
    observed : float
        Observed connectivity value.
    null_distribution : NDArray[np.floating]
        Null distribution values from surrogate testing.
    alternative : str
        Type of alternative hypothesis:
        - 'greater': test if observed > null (typical for connectivity)
        - 'less': test if observed < null
        - 'two-sided': test if observed differs from null
    
    Returns
    -------
    float
        P-value (probability of observing value this extreme under H0).
    
    Notes
    -----
    Uses the formula (k + 1) / (n + 1) to avoid p-values of exactly 0.
    """
    n = len(null_distribution)
    
    if alternative == 'greater':
        p = (np.sum(null_distribution >= observed) + 1) / (n + 1)
    elif alternative == 'less':
        p = (np.sum(null_distribution <= observed) + 1) / (n + 1)
    else:  # two-sided
        mean_null = np.mean(null_distribution)
        deviation = np.abs(observed - mean_null)
        p = (np.sum(np.abs(null_distribution - mean_null) >= deviation) + 1) / (n + 1)
    
    return float(p)


# ============================================================================
# Multiple Comparisons Correction
# ============================================================================

def bonferroni_correction(pvalues: NDArray[np.floating],
                          alpha: float = 0.05) -> Tuple[NDArray[np.bool_], float]:
    """
    Apply Bonferroni correction to p-values.
    
    The most conservative correction method. Controls the family-wise
    error rate (FWER) - the probability of making ANY false positive.
    
    Parameters
    ----------
    pvalues : NDArray[np.floating]
        Array of p-values from multiple tests.
    alpha : float
        Desired family-wise error rate.
    
    Returns
    -------
    Tuple[NDArray[np.bool_], float]
        Boolean mask of significant tests, and corrected alpha threshold.
    
    Notes
    -----
    Bonferroni is very conservative and may miss true effects (low power).
    Use when false positives are very costly or when testing few hypotheses.
    """
    n_tests = len(pvalues)
    alpha_corrected = alpha / n_tests
    significant = pvalues < alpha_corrected
    
    return significant, alpha_corrected


def fdr_correction(pvalues: NDArray[np.floating],
                   alpha: float = 0.05) -> Tuple[NDArray[np.bool_], NDArray[np.floating]]:
    """
    Apply FDR correction using Benjamini-Hochberg procedure.
    
    Controls the false discovery rate (FDR) - the expected proportion
    of false positives among all rejected hypotheses.
    
    Parameters
    ----------
    pvalues : NDArray[np.floating]
        Array of p-values from multiple tests.
    alpha : float
        Desired false discovery rate.
    
    Returns
    -------
    Tuple[NDArray[np.bool_], NDArray[np.floating]]
        Boolean mask of significant tests, and adjusted p-values.
    
    Notes
    -----
    FDR is less conservative than Bonferroni and has higher power.
    Preferred for exploratory analyses with many tests.
    
    References
    ----------
    Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery
    rate: a practical and powerful approach to multiple testing.
    """
    n = len(pvalues)
    
    # Sort p-values and keep track of original order
    sorted_idx = np.argsort(pvalues)
    sorted_pvals = pvalues[sorted_idx]
    
    # Compute BH threshold for each rank
    ranks = np.arange(1, n + 1)
    bh_threshold = ranks / n * alpha
    
    # Find the largest p-value below its threshold
    below_threshold = sorted_pvals <= bh_threshold
    if np.any(below_threshold):
        max_below = np.max(np.where(below_threshold)[0])
        reject_sorted = np.arange(n) <= max_below
    else:
        reject_sorted = np.zeros(n, dtype=bool)
    
    # Map back to original order
    reject = np.zeros(n, dtype=bool)
    reject[sorted_idx] = reject_sorted
    
    # Compute adjusted p-values
    adjusted = np.zeros(n)
    adjusted[sorted_idx] = np.minimum.accumulate(
        (sorted_pvals * n / ranks)[::-1]
    )[::-1]
    adjusted = np.minimum(adjusted, 1.0)
    
    return reject, adjusted


# ============================================================================
# Permutation Testing
# ============================================================================

def permutation_test(group1: NDArray[np.floating],
                     group2: NDArray[np.floating],
                     n_permutations: int = 1000,
                     statistic: str = 'mean_diff') -> Tuple[float, float, NDArray[np.floating]]:
    """
    Perform a permutation test comparing two groups.
    
    Non-parametric test that makes no assumptions about the underlying
    distribution. Tests whether the observed difference could have
    occurred by chance under random group assignment.
    
    Parameters
    ----------
    group1 : NDArray[np.floating]
        Connectivity values for group 1.
    group2 : NDArray[np.floating]
        Connectivity values for group 2.
    n_permutations : int
        Number of permutations to perform.
    statistic : str
        Test statistic: 'mean_diff' or 't_stat'.
    
    Returns
    -------
    Tuple[float, float, NDArray[np.floating]]
        Observed statistic, two-sided p-value, and null distribution.
    
    Examples
    --------
    >>> group1 = np.random.normal(0.3, 0.1, 20)
    >>> group2 = np.random.normal(0.5, 0.1, 20)
    >>> stat, pval, null = permutation_test(group1, group2)
    """
    n1, n2 = len(group1), len(group2)
    pooled = np.concatenate([group1, group2])
    
    # Compute observed statistic
    if statistic == 'mean_diff':
        observed = np.mean(group1) - np.mean(group2)
    else:
        observed = stats.ttest_ind(group1, group2)[0]
    
    # Generate null distribution
    null_stats = np.zeros(n_permutations)
    
    for i in range(n_permutations):
        # Shuffle and split
        np.random.shuffle(pooled)
        perm_g1 = pooled[:n1]
        perm_g2 = pooled[n1:]
        
        if statistic == 'mean_diff':
            null_stats[i] = np.mean(perm_g1) - np.mean(perm_g2)
        else:
            null_stats[i] = stats.ttest_ind(perm_g1, perm_g2)[0]
    
    # Two-sided p-value
    p_value = np.mean(np.abs(null_stats) >= np.abs(observed))
    
    return float(observed), float(p_value), null_stats


# ============================================================================
# Bootstrap Confidence Intervals
# ============================================================================

def bootstrap_ci(data: NDArray[np.floating],
                 statistic: Callable = np.mean,
                 n_bootstrap: int = 1000,
                 ci: float = 0.95) -> Tuple[float, float, float, NDArray[np.floating]]:
    """
    Compute bootstrap confidence interval for a statistic.
    
    Parameters
    ----------
    data : NDArray[np.floating]
        Input data array.
    statistic : Callable
        Function to compute the statistic. Default: np.mean.
    n_bootstrap : int
        Number of bootstrap resamples.
    ci : float
        Confidence level (e.g., 0.95 for 95% CI).
    
    Returns
    -------
    Tuple[float, float, float, NDArray[np.floating]]
        Point estimate, CI lower bound, CI upper bound, and bootstrap distribution.
    
    Notes
    -----
    Uses the percentile method for confidence interval estimation.
    """
    n = len(data)
    point_estimate = float(statistic(data))
    
    # Bootstrap resampling
    bootstrap_stats = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        resample = np.random.choice(data, size=n, replace=True)
        bootstrap_stats[i] = statistic(resample)
    
    # Percentile method for CI
    alpha = (1 - ci) / 2
    ci_lower = float(np.percentile(bootstrap_stats, alpha * 100))
    ci_upper = float(np.percentile(bootstrap_stats, (1 - alpha) * 100))
    
    return point_estimate, ci_lower, ci_upper, bootstrap_stats


# ============================================================================
# Effect Size
# ============================================================================

def cohens_d(group1: NDArray[np.floating],
             group2: NDArray[np.floating]) -> float:
    """
    Compute Cohen's d effect size.
    
    Measures the standardized difference between two group means.
    
    Parameters
    ----------
    group1 : NDArray[np.floating]
        First group values.
    group2 : NDArray[np.floating]
        Second group values.
    
    Returns
    -------
    float
        Cohen's d effect size.
    
    Notes
    -----
    Interpretation guidelines (Cohen, 1988):
    - |d| < 0.2: negligible
    - |d| ~ 0.2: small
    - |d| ~ 0.5: medium
    - |d| ~ 0.8: large
    - |d| > 1.0: very large
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    if pooled_std == 0:
        return 0.0
    
    return float((np.mean(group1) - np.mean(group2)) / pooled_std)


def hedges_g(group1: NDArray[np.floating],
             group2: NDArray[np.floating]) -> float:
    """
    Compute Hedge's g effect size.
    
    A bias-corrected version of Cohen's d, better for small samples.
    
    Parameters
    ----------
    group1 : NDArray[np.floating]
        First group values.
    group2 : NDArray[np.floating]
        Second group values.
    
    Returns
    -------
    float
        Hedge's g effect size.
    """
    d = cohens_d(group1, group2)
    n = len(group1) + len(group2)
    
    # Correction factor for small samples
    correction = 1 - (3 / (4 * n - 9))
    
    return d * correction


# ============================================================================
# Complete Pipeline
# ============================================================================

def significance_test_pair(signal1: NDArray[np.floating],
                           signal2: NDArray[np.floating],
                           connectivity_func: Callable,
                           n_surrogates: int = 1000,
                           method: str = 'phase_shuffle',
                           alpha: float = 0.05) -> Dict[str, Any]:
    """
    Complete significance test for a single pair of signals.
    
    Parameters
    ----------
    signal1 : NDArray[np.floating]
        First signal.
    signal2 : NDArray[np.floating]
        Second signal.
    connectivity_func : Callable
        Function that computes connectivity between two signals.
    n_surrogates : int
        Number of surrogates for null distribution.
    method : str
        Surrogate method: 'phase_shuffle' or 'time_shift'.
    alpha : float
        Significance level.
    
    Returns
    -------
    Dict[str, Any]
        Dictionary with keys:
        - 'observed': observed connectivity value
        - 'null_mean': mean of null distribution
        - 'null_std': std of null distribution
        - 'pvalue': p-value
        - 'significant': whether result is significant at alpha
        - 'null_distribution': full null distribution array
    """
    # Compute observed connectivity
    observed = connectivity_func(signal1, signal2)
    
    # Build null distribution
    null_dist = build_null_distribution(
        signal1, signal2, connectivity_func,
        n_surrogates=n_surrogates, method=method
    )
    
    # Compute p-value
    pvalue = compute_pvalue(observed, null_dist, alternative='greater')
    
    return {
        'observed': float(observed),
        'null_mean': float(np.mean(null_dist)),
        'null_std': float(np.std(null_dist)),
        'pvalue': pvalue,
        'significant': pvalue < alpha,
        'null_distribution': null_dist
    }


def significance_test_matrix(signals: List[NDArray[np.floating]],
                             connectivity_func: Callable,
                             n_surrogates: int = 500,
                             method: str = 'phase_shuffle',
                             alpha: float = 0.05,
                             correction: str = 'fdr') -> Dict[str, Any]:
    """
    Complete significance testing pipeline for multiple channel pairs.
    
    Parameters
    ----------
    signals : List[NDArray[np.floating]]
        List of signals (one per channel).
    connectivity_func : Callable
        Function that computes connectivity between two signals.
    n_surrogates : int
        Number of surrogates per pair.
    method : str
        Surrogate method: 'phase_shuffle' or 'time_shift'.
    alpha : float
        Significance level.
    correction : str
        Multiple comparisons correction: 'bonferroni', 'fdr', or 'none'.
    
    Returns
    -------
    Dict[str, Any]
        Dictionary with keys:
        - 'connectivity_matrix': matrix of connectivity values
        - 'pvalue_matrix': matrix of p-values
        - 'significant_matrix': boolean matrix of significant pairs
        - 'n_significant': number of significant pairs
        - 'correction': correction method used
    """
    n_channels = len(signals)
    
    # Initialize matrices
    conn_matrix = np.zeros((n_channels, n_channels))
    pvalue_matrix = np.ones((n_channels, n_channels))
    
    # Collect p-values for all pairs
    pvalues_flat = []
    pairs_list = []
    
    for i in range(n_channels):
        for j in range(i + 1, n_channels):
            # Test this pair
            result = significance_test_pair(
                signals[i], signals[j], connectivity_func,
                n_surrogates=n_surrogates, method=method, alpha=alpha
            )
            
            # Store in matrices
            conn_matrix[i, j] = result['observed']
            conn_matrix[j, i] = result['observed']
            pvalue_matrix[i, j] = result['pvalue']
            pvalue_matrix[j, i] = result['pvalue']
            
            pvalues_flat.append(result['pvalue'])
            pairs_list.append((i, j))
    
    pvalues_flat = np.array(pvalues_flat)
    
    # Apply correction
    if correction == 'bonferroni':
        significant_flat, _ = bonferroni_correction(pvalues_flat, alpha)
    elif correction == 'fdr':
        significant_flat, _ = fdr_correction(pvalues_flat, alpha)
    else:
        significant_flat = pvalues_flat < alpha
    
    # Build significance matrix
    sig_matrix = np.zeros((n_channels, n_channels), dtype=bool)
    for k, (i, j) in enumerate(pairs_list):
        sig_matrix[i, j] = significant_flat[k]
        sig_matrix[j, i] = significant_flat[k]
    
    return {
        'connectivity_matrix': conn_matrix,
        'pvalue_matrix': pvalue_matrix,
        'significant_matrix': sig_matrix,
        'n_significant': int(np.sum(significant_flat)),
        'correction': correction
    }
