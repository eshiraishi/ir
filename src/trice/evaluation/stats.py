"""Pre-registered tests as thin wrappers over scipy, statsmodels and scikit-posthocs.

Paired comparisons resample *sentences*: the two systems' per-sentence counts are stacked and the
statistic is computed from row indices, so scipy's paired permutation test (exact when the number
of sentences is small) and paired bootstrap apply directly to micro-F1.
"""

import numpy as np
import pandas as pd
import scikit_posthocs as sp
from pydantic import BaseModel
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests

from trice.evaluation.scoring import Evaluator
from trice.schema import derive_seed


class Comparison(BaseModel):
    delta: float  # F1(a) - F1(b)
    p_value: float  # paired permutation test, two-sided
    ci_low: float  # paired bootstrap percentile CI
    ci_high: float
    se: float
    mcnemar_p: float  # exact McNemar on sentence-level exact match


class Friedman(BaseModel):
    statistic: float
    p_value: float
    mean_ranks: dict[str, float]  # 1 = best
    nemenyi: dict[str, dict[str, float]]


class Trend(BaseModel):
    p_increasing: float
    p_decreasing: float
    p_two_sided: float  # Bonferroni x2


class Stats(BaseModel):
    resamples: int
    alpha: float

    def compare(self, a: np.ndarray, b: np.ndarray, *seed: object) -> Comparison:
        n, stacked = len(a), np.vstack([a, b])

        def delta(x: np.ndarray, y: np.ndarray, axis: int = -1) -> np.ndarray:
            return Evaluator.f1(stacked[x].sum(axis=-2)) - Evaluator.f1(stacked[y].sum(axis=-2))

        x, y = np.arange(n), np.arange(n, 2 * n)
        test = stats.permutation_test(
            (x, y),
            delta,
            permutation_type="samples",
            vectorized=True,
            n_resamples=self.resamples,
            rng=derive_seed("permutation", *seed),
        )
        boot = stats.bootstrap(
            (x, y),
            delta,
            paired=True,
            vectorized=True,
            method="percentile",
            n_resamples=self.resamples,
            confidence_level=1 - self.alpha,
            rng=derive_seed("bootstrap", *seed),
        )
        exact_a, exact_b = (a[:, 1:].sum(1) == 0), (b[:, 1:].sum(1) == 0)
        table = [
            [np.sum(exact_a & exact_b), np.sum(exact_a & ~exact_b)],
            [np.sum(~exact_a & exact_b), np.sum(~exact_a & ~exact_b)],
        ]
        return Comparison(
            delta=float(test.statistic),
            p_value=float(test.pvalue),
            ci_low=float(boot.confidence_interval.low),
            ci_high=float(boot.confidence_interval.high),
            se=float(boot.standard_error),
            mcnemar_p=float(mcnemar(table, exact=True).pvalue),
        )

    def standard_error(self, counts: np.ndarray, *seed: object) -> float:
        """Bootstrap SE of one system's micro-F1 (for the 1-SE robustness rule)."""
        return float(
            stats.bootstrap(
                (np.arange(len(counts)),),
                lambda i, axis=-1: Evaluator.f1(counts[i].sum(axis=-2)),
                vectorized=True,
                method="percentile",
                n_resamples=self.resamples,
                rng=derive_seed("se", *seed),
            ).standard_error
        )

    @staticmethod
    def holm(p_values: list[float]) -> list[float]:
        return multipletests(p_values, method="holm")[1].tolist() if p_values else []

    @staticmethod
    def wilcoxon(deltas: list[float]) -> float:
        return 1.0 if not any(deltas) else float(stats.wilcoxon(deltas).pvalue)

    @staticmethod
    def friedman(scores: pd.DataFrame) -> Friedman:
        """``scores``: rows = blocks (dataset x model), columns = treatments, higher is better."""
        result = stats.friedmanchisquare(*scores.T.to_numpy())
        return Friedman(
            statistic=float(result.statistic),
            p_value=float(result.pvalue),
            mean_ranks=scores.rank(axis=1, ascending=False).mean().to_dict(),
            nemenyi=sp.posthoc_nemenyi_friedman(scores).to_dict(),
        )

    @staticmethod
    def page(data: np.ndarray) -> Trend:
        """Page's L; ``data`` is (blocks, treatments) with treatments in increasing size."""
        up, down = stats.page_trend_test(data).pvalue, stats.page_trend_test(data[:, ::-1]).pvalue
        return Trend(p_increasing=up, p_decreasing=down, p_two_sided=min(1.0, 2 * min(up, down)))
