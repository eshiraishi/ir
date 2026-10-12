"""Pre-registered statistical tests (docs/preregistration.md).

Per dataset, systems are compared *paired* on the same test sentences:
  * approximate randomization (sentence-level swaps) for the p-value of the micro-F1 difference,
  * paired bootstrap for its 95% CI,
  * exact McNemar on sentence-level exact match (the paired chi-squared test).
Across datasets: Wilcoxon signed-rank, Friedman + Nemenyi critical difference, Page's L trend
test. Holm-Bonferroni within each declared family. All resampling is seeded via trice.seeds.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats

from trice.eval.metrics import f1_of_sums
from trice.seeds import rng


@dataclass(frozen=True)
class PairedResult:
    delta: float  # F1(a) - F1(b)
    p_value: float  # approximate randomization, two-sided
    ci_low: float  # paired bootstrap percentile CI of delta
    ci_high: float
    se: float  # bootstrap standard error of delta
    n: int


def _check(a: np.ndarray, b: np.ndarray) -> None:
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 3:
        raise ValueError("expected two aligned (n_sentences, 3) count arrays")


def approximate_randomization(
    a: np.ndarray, b: np.ndarray, n_resamples: int = 10_000, seed: tuple = ("ar",), chunk: int = 500
) -> float:
    """Two-sided p-value for H0: systems a and b are exchangeable on every sentence."""
    _check(a, b)
    sa, sb = a.sum(0).astype(float), b.sum(0).astype(float)
    observed = abs(f1_of_sums(sa) - f1_of_sums(sb))
    d = (b - a).astype(float)  # swapping sentence i moves d_i into a and out of b
    gen = rng(*seed)
    hits, done = 0, 0
    while done < n_resamples:
        m = min(chunk, n_resamples - done)
        shift = (gen.random((m, len(a))) < 0.5).astype(float) @ d
        diff = np.abs(f1_of_sums(sa + shift) - f1_of_sums(sb - shift))
        hits += int(np.sum(diff >= observed - 1e-12))
        done += m
    return (hits + 1) / (n_resamples + 1)


def paired_bootstrap(
    a: np.ndarray,
    b: np.ndarray,
    n_resamples: int = 10_000,
    seed: tuple = ("bootstrap",),
    chunk: int = 250,
) -> np.ndarray:
    """Bootstrap distribution of F1(a) - F1(b), resampling sentences jointly."""
    _check(a, b)
    gen = rng(*seed)
    n, out = len(a), []
    while sum(len(x) for x in out) < n_resamples:
        m = min(chunk, n_resamples - sum(len(x) for x in out))
        idx = gen.integers(0, n, size=(m, n))
        out.append(f1_of_sums(a[idx].sum(axis=1)) - f1_of_sums(b[idx].sum(axis=1)))
    return np.concatenate(out)


def compare(
    a: np.ndarray,
    b: np.ndarray,
    *,
    seed: tuple = (),
    n_resamples: int = 10_000,
    alpha: float = 0.05,
) -> PairedResult:
    delta = float(f1_of_sums(a.sum(0)) - f1_of_sums(b.sum(0)))
    p = approximate_randomization(a, b, n_resamples, ("ar", *seed))
    boot = paired_bootstrap(a, b, n_resamples, ("bootstrap", *seed))
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    return PairedResult(delta, p, float(lo), float(hi), float(boot.std(ddof=1)), len(a))


def bootstrap_se(counts: np.ndarray, n_resamples: int = 2_000, seed: tuple = ("se",)) -> float:
    """Bootstrap standard error of micro-F1 for one system."""
    gen = rng(*seed)
    idx = gen.integers(0, len(counts), size=(n_resamples, len(counts)))
    return float(f1_of_sums(counts[idx].sum(axis=1)).std(ddof=1))


def mcnemar_exact(a_correct: np.ndarray, b_correct: np.ndarray) -> float:
    """Exact (binomial) McNemar test on paired binary outcomes, two-sided."""
    a_correct, b_correct = np.asarray(a_correct, bool), np.asarray(b_correct, bool)
    only_a = int(np.sum(a_correct & ~b_correct))
    only_b = int(np.sum(~a_correct & b_correct))
    if only_a + only_b == 0:
        return 1.0
    return float(stats.binomtest(only_a, only_a + only_b, 0.5).pvalue)


def holm(p_values: Sequence[float]) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (step-down, monotone, capped at 1)."""
    p = np.asarray(p_values, float)
    order = np.argsort(p, kind="stable")
    m = len(p)
    adjusted = np.maximum.accumulate((m - np.arange(m)) * p[order])
    out = np.empty(m)
    out[order] = np.minimum(adjusted, 1.0)
    return out


def min_detectable_effect(se: float, alpha: float = 0.05, power: float = 0.8) -> float:
    return float((stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)) * se)


def wilcoxon(deltas: Sequence[float]) -> float:
    """Two-sided Wilcoxon signed-rank p-value over per-dataset differences."""
    d = np.asarray(deltas, float)
    if np.all(d == 0):
        return 1.0
    return float(stats.wilcoxon(d, zero_method="wilcox", alternative="two-sided").pvalue)


@dataclass(frozen=True)
class FriedmanResult:
    statistic: float
    p_value: float
    mean_ranks: dict[str, float]  # rank 1 = best (highest score)
    critical_difference: float  # Nemenyi CD at alpha


def friedman_nemenyi(
    scores: np.ndarray, names: Sequence[str], alpha: float = 0.05
) -> FriedmanResult:
    """``scores`` is (blocks, treatments), higher is better (Demsar 2006)."""
    n_blocks, k = scores.shape
    stat, p = stats.friedmanchisquare(*scores.T)
    ranks = stats.rankdata(-scores, axis=1).mean(axis=0)
    q = stats.studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)
    cd = q * np.sqrt(k * (k + 1) / (6 * n_blocks))
    return FriedmanResult(
        float(stat), float(p), dict(zip(names, map(float, ranks), strict=True)), float(cd)
    )


@dataclass(frozen=True)
class TrendResult:
    p_increasing: float
    p_decreasing: float
    p_two_sided: float  # Bonferroni x2 over the two one-sided tests


def page_trend(data: np.ndarray) -> TrendResult:
    """Page's L test; ``data`` is (blocks, treatments) with treatments in increasing size order."""
    up = float(stats.page_trend_test(data).pvalue)
    down = float(stats.page_trend_test(data[:, ::-1]).pvalue)
    return TrendResult(up, down, min(1.0, 2 * min(up, down)))


def kendall_tau(x: Sequence[float], y: Sequence[float]) -> tuple[float, float]:
    res = stats.kendalltau(x, y)
    return float(res.statistic), float(res.pvalue)
