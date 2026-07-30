"""Null-aware event-recovery metrics for the ICTAI fidelity audit."""
from __future__ import annotations

from math import comb

import numpy as np
import pandas as pd

from crop_yield_xai.audit_rules import top_k_recovery


def spearman(observed: np.ndarray, predicted: np.ndarray) -> float:
    left, right = pd.Series(observed).rank(), pd.Series(predicted).rank()
    if left.nunique() < 2 or right.nunique() < 2:
        return 0.0
    return float(left.corr(right, method="pearson"))


def kendall(observed: np.ndarray, predicted: np.ndarray) -> float:
    return float(pd.Series(observed).corr(pd.Series(predicted), method="kendall"))


def hypergeometric_tail_probability(population: int, successes: int, draws: int, observed: int) -> float:
    """P[X >= observed] when two top-k sets overlap under random ranking."""
    lower, upper = max(0, draws + successes - population), min(draws, successes)
    numerator = sum(comb(successes, x) * comb(population - successes, draws - x) for x in range(max(observed, lower), upper + 1))
    return numerator / comb(population, draws)


def _year_bootstrap_indices(years: np.ndarray, n_boot: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    unique = np.unique(years)
    by_year = {year: np.flatnonzero(years == year) for year in unique}
    return [np.concatenate([by_year[year] for year in rng.choice(unique, size=len(unique), replace=True)]) for _ in range(n_boot)]


def topk_null_audit(observed: np.ndarray, predicted: np.ndarray, years: np.ndarray, k: int, n_boot: int, n_permutations: int, seed: int) -> dict[str, float]:
    if not (len(observed) == len(predicted) == len(years)):
        raise ValueError("observed, predicted, and years must have identical length")
    n = len(observed)
    if k <= 0 or k > n:
        raise ValueError("k must be between 1 and tail size")
    recovery = top_k_recovery(observed, predicted, k)
    expected = k / n
    overlap = int(round(recovery * k))
    rng = np.random.default_rng(seed)
    permutation = np.array([top_k_recovery(observed, rng.permutation(predicted), k) for _ in range(n_permutations)])
    lifts = []
    for indices in _year_bootstrap_indices(years, n_boot, seed + 1):
        sampled_k = min(k, len(indices))
        sampled_recovery = top_k_recovery(observed[indices], predicted[indices], sampled_k)
        lifts.append(sampled_recovery / (sampled_k / len(indices)))
    return {
        "n": n,
        "k": k,
        "overlap": overlap,
        "recovery": recovery,
        "random_expectation": expected,
        "lift": recovery / expected,
        "hypergeometric_pvalue": hypergeometric_tail_probability(n, k, k, overlap),
        "permutation_pvalue": float((1 + np.sum(permutation >= recovery)) / (1 + n_permutations)),
        "lift_ci95_low": float(np.quantile(lifts, 0.025)),
        "lift_ci95_high": float(np.quantile(lifts, 0.975)),
        "n_boot": n_boot,
        "n_permutations": n_permutations,
    }


def rank_null_audit(observed: np.ndarray, predicted: np.ndarray, years: np.ndarray, n_boot: int, n_permutations: int, seed: int) -> dict[str, float]:
    if not (len(observed) == len(predicted) == len(years)):
        raise ValueError("observed, predicted, and years must have identical length")
    observed_rho = spearman(observed, predicted)
    observed_tau = kendall(observed, predicted)
    rng = np.random.default_rng(seed)
    permutations = []
    for _ in range(n_permutations):
        shuffled = predicted.copy()
        for year in np.unique(years):
            indices = np.flatnonzero(years == year)
            shuffled[indices] = rng.permutation(shuffled[indices])
        permutations.append(spearman(observed, shuffled))
    bootstrap = [spearman(observed[indices], predicted[indices]) for indices in _year_bootstrap_indices(years, n_boot, seed + 3)]
    return {
        "n": len(observed),
        "spearman": observed_rho,
        "kendall_tau": observed_tau,
        "spearman_ci95_low": float(np.quantile(bootstrap, 0.025)),
        "spearman_ci95_high": float(np.quantile(bootstrap, 0.975)),
        "permutation_pvalue": float((1 + np.sum(np.asarray(permutations) >= observed_rho)) / (1 + n_permutations)),
        "n_boot": n_boot,
        "n_permutations": n_permutations,
    }
