"""Symbolic-sequence complexity measures: Lempel-Ziv complexity and Shannon block entropy.

Pure Python/numpy (numba for the phrase-count scan). All functions here are written from
the published definitions cited in their docstrings.
"""
import numpy as np
from numba import njit
from scipy.stats import rankdata


@njit(cache=True)
def _lempel_ziv_phrase_count_scan(s):
    n = s.shape[0]
    count = 0
    p = 0  # start of the current component
    while p < n:
        # longest prefix of s[p:] that is a substring of the preceding text, where a copy
        # source starts at some j < p and may run on into the component being built
        longest = 0
        for j in range(p):
            m = 0
            while p + m < n and s[j + m] == s[p + m]:
                m += 1
            if m > longest:
                longest = m
        count += 1
        # the component is that prefix plus the first symbol that breaks the copy; if the
        # sequence ends first, the last (non-exhaustive) component runs to the end
        p += longest + 1
    return count


def _lempel_ziv_phrase_count(symbols: np.ndarray) -> int:
    """Number of components in the exhaustive history of a symbol sequence (Lempel-Ziv 1976).

    The sequence is parsed left to right. Each component is the shortest substring starting
    at the current position that cannot be copied from the text before its last symbol (the
    copy may overlap the component being built). A final component that is still
    reproducible when the sequence ends is counted. For example ``0001101001000101`` parses
    as ``0 . 001 . 10 . 100 . 1000 . 101`` and has complexity 6.

    Parameters
    ----------
    symbols : array-like of int
        The symbol sequence (any integer or hashable-as-integer labels).

    Returns
    -------
    int
        The Lempel-Ziv complexity ``c`` (0 for an empty sequence, 1 for a single symbol).

    Notes
    -----
    A. Lempel and J. Ziv, "On the complexity of finite sequences", IEEE Trans. Inf. Theory
    22(1):75-81 (1976). The quadratic scan (cf. Kaspar and Schuster, Phys. Rev. A 36:842,
    1987) is compiled with numba; it is intended for sequences up to about 10^4 symbols.
    """
    s = np.ascontiguousarray(np.asarray(symbols).ravel())
    if s.size == 0:
        return 0
    if s.dtype.kind not in 'iu':
        s = np.unique(s, return_inverse=True)[1].astype(np.int64)
    return int(_lempel_ziv_phrase_count_scan(s.astype(np.int64)))


def _equiprobable_symbols(x: np.ndarray, n_bins: int) -> np.ndarray:
    """Coarse-grain ``x`` into ``n_bins`` roughly equiprobable symbols 1..n_bins by rank.

    With R the average rank (1..N) of each sample, the symbol is
    ``floor(R * n_bins / (N + 1)) + 1``; tied values share their average rank, hence a
    symbol. Symbols present are not guaranteed to cover all of 1..n_bins when many values
    are tied.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    ranks = rankdata(x, method='average')
    return (np.floor(ranks * n_bins / (x.size + 1)).astype(np.int64) + 1)


def _normalized_lz_complexity(labels: np.ndarray) -> float:
    """Lempel-Ziv complexity of a sequence of small non-negative integer labels, normalized
    by its value for a random sequence.

    With ``c`` the complexity (see :func:`_lempel_ziv_phrase_count`), ``N`` the length and
    ``k = max(labels) + 1`` the alphabet size inferred from the largest label,
    returns ``c * log(N) / (N * log(k))`` (Kaspar and Schuster 1987: the complexity of a
    random k-ary sequence grows as ``N / log_k(N)``). ``k = 1`` gives ``inf`` for ``N > 1``;
    a single symbol gives 0; an empty input gives NaN.
    """
    labels = np.asarray(labels).ravel()
    n = labels.size
    if n == 0:
        return np.nan
    if n == 1:
        return 0.0
    c = _lempel_ziv_phrase_count(labels)
    k = int(np.floor(labels.max())) + 1
    if k <= 1:
        return np.inf
    return float(c * np.log(n) / (n * np.log(k)))


def _shannon_block_entropy(y: np.ndarray, num_bins: int, depth: int) -> float:
    """Shannon entropy (nats) of the overlapping words of ``depth`` symbols in an
    equal-frequency coarse-graining of ``y`` into ``num_bins`` symbols (not divided by depth).

    Symbols come from the ranks of the samples: the sorted series is cut into ``num_bins``
    consecutive groups of (nearly) equal size, group ``g`` (``g = 0..num_bins-1``) holding the
    ranks ``floor(g N / num_bins)`` to ``floor((g + 1) N / num_bins) - 1`` (0-based). A sample
    gets the symbol of the group in which its value starts, so equal values always share a
    symbol (a tie that straddles a group boundary goes to the upper group). The entropy is
    ``-sum p ln p`` over the empirical frequencies ``p`` of the ``N - depth + 1`` overlapping
    words. The cut at rank ``floor(g N / num_bins)`` is the convention that reproduces hctsa's
    values.

    Returns NaN if ``y`` contains NaN or is shorter than ``depth``, and 0 if there is only
    one symbol or one word.

    Notes
    -----
    C. E. Shannon, Bell Syst. Tech. J. 27:379 (1948); P. Grassberger, Int. J. Theor. Phys.
    25:907 (1986); M. Small, Applied Nonlinear Time Series Analysis (World Scientific, 2005).
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    n_bins, depth = int(num_bins), int(depth)
    if n_bins < 1 or depth < 1:
        raise ValueError("num_bins and depth must be positive integers")
    n = y.size
    if np.isnan(y).any() or depth > n:
        return np.nan
    n_words = n - depth + 1
    if n_bins == 1 or n_words <= 1:
        return 0.0

    sorted_y = np.sort(y)
    cut_values = sorted_y[(np.arange(1, n_bins) * n) // n_bins]  # first value of groups 1..n_bins-1
    symbols = np.searchsorted(cut_values, y, side='right')  # number of cut values <= sample

    # each row is one word: `depth` consecutive symbols
    words = np.lib.stride_tricks.sliding_window_view(symbols, depth)
    if n_bins ** depth < 2 ** 62:
        weights = n_bins ** np.arange(depth - 1, -1, -1, dtype=np.int64)
        counts = np.unique(words.astype(np.int64) @ weights, return_counts=True)[1]
    else:
        counts = np.unique(words, axis=0, return_counts=True)[1]
    p = counts / n_words
    return float(-np.sum(p * np.log(p)))
