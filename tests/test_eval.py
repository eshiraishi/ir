import itertools

import numpy as np
import pytest
from scipy import stats as sps

from trice.eval import stats
from trice.eval.metrics import (
    counts_matrix,
    exact_match,
    f1_of_sums,
    micro,
    per_type,
    prf,
    sentence_counts,
)
from trice.select_k import select_k


def test_sentence_counts_are_multiset_based():
    gold = [("Paris", "location"), ("Paris", "location"), ("Ann", "person")]
    pred = [("Paris", "location"), ("Ann", "location"), ("Bob", "person")]
    assert sentence_counts(pred, gold) == (1, 2, 2)
    assert prf(1, 2, 2).f1 == pytest.approx(2 / 6)


def test_micro_per_type_and_exact_match():
    preds = [[("A", "x")], [("B", "y"), ("C", "x")], []]
    golds = [[("A", "x")], [("B", "y")], [("D", "x")]]
    c = counts_matrix(preds, golds)
    assert c.tolist() == [[1, 0, 0], [1, 1, 0], [0, 0, 1]]
    assert micro(c).f1 == pytest.approx(2 * 2 / (2 * 2 + 1 + 1))
    pt = per_type(preds, golds)
    assert (pt["x"].tp, pt["x"].fp, pt["x"].fn) == (1, 1, 1)
    assert (pt["y"].tp, pt["y"].fp, pt["y"].fn) == (1, 0, 0)
    assert exact_match(c).tolist() == [True, False, False]
    assert f1_of_sums(np.array([0, 0, 0])) == 0.0


def _random_counts(n, p_tp, seed):
    g = np.random.default_rng(seed)
    gold = g.integers(1, 6, n)
    tp = g.binomial(gold, p_tp)
    return np.stack([tp, g.integers(0, 3, n), gold - tp], axis=1)


def test_approximate_randomization_matches_exact_permutation():
    a, b = _random_counts(10, 0.8, 1), _random_counts(10, 0.5, 2)
    observed = abs(f1_of_sums(a.sum(0)) - f1_of_sums(b.sum(0)))
    hits = 0
    for mask in itertools.product([0, 1], repeat=10):
        m = np.array(mask, bool)[:, None]
        hits += (
            abs(f1_of_sums(np.where(m, b, a).sum(0)) - f1_of_sums(np.where(m, a, b).sum(0)))
            >= observed - 1e-12
        )
    exact = hits / 2**10
    approx = stats.approximate_randomization(a, b, n_resamples=20_000)
    assert approx == pytest.approx(exact, abs=0.015)


def test_compare_identical_and_clearly_different():
    a = _random_counts(400, 0.7, 3)
    same = stats.compare(a, a.copy(), n_resamples=2_000)
    assert same.delta == 0 and same.p_value == 1.0
    better = stats.compare(
        _random_counts(400, 0.9, 4), _random_counts(400, 0.5, 5), n_resamples=2_000
    )
    assert (
        better.p_value < 0.001
        and better.ci_low > 0
        and better.ci_low < better.delta < better.ci_high
    )


def test_resampling_is_deterministic():
    a, b = _random_counts(50, 0.7, 6), _random_counts(50, 0.6, 7)
    assert stats.compare(a, b, seed=("x",), n_resamples=500) == stats.compare(
        a, b, seed=("x",), n_resamples=500
    )


def test_mcnemar_and_holm():
    a = np.array([1, 1, 1, 1, 1, 1, 1, 0, 1, 1], bool)
    b = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 1], bool)
    assert stats.mcnemar_exact(a, b) == pytest.approx(sps.binomtest(6, 6, 0.5).pvalue)
    assert stats.mcnemar_exact(a, a) == 1.0
    assert stats.holm([0.01, 0.04, 0.03, 0.005]).tolist() == pytest.approx([0.03, 0.06, 0.06, 0.02])


def test_friedman_nemenyi_cd_matches_demsar_table():
    g = np.random.default_rng(0)
    scores = g.random((20, 4)) + np.array([0.0, 0.1, 0.2, 0.3])
    res = stats.friedman_nemenyi(scores, ["a", "b", "c", "d"])
    # Demsar (2006) Table 5: q_0.05 = 2.569 for k = 4.
    assert res.critical_difference == pytest.approx(2.569 * np.sqrt(4 * 5 / (6 * 20)), rel=1e-3)
    assert res.mean_ranks["d"] < res.mean_ranks["a"]


def test_page_trend_direction():
    data = np.array([[1, 2, 3, 4], [1.5, 2.1, 2.9, 4.2], [0.9, 2.5, 3.3, 3.9], [1, 1.9, 3.1, 4.4]])
    res = stats.page_trend(data)
    assert res.p_increasing < 0.01 < res.p_decreasing
    assert res.p_two_sided == pytest.approx(2 * res.p_increasing)


def _counts_for_f1(f1, n=200):
    tp = round(10 * f1)
    return np.tile([tp, 10 - tp, 10 - tp], (n, 1))


def test_select_k_knee_flat_and_decreasing():
    ks = [0, 1, 2, 4, 8, 16, 32, 64]
    rising = dict(
        zip(ks, [_counts_for_f1(f) for f in [0.2, 0.4, 0.5, 0.6, 0.7, 0.7, 0.7, 0.7]], strict=True)
    )
    sel = select_k(rising)
    assert sel.rule == "kneedle" and sel.k_star in (4, 8)
    flat = {k: _counts_for_f1(0.6) for k in ks}
    assert (select_k(flat).k_star, select_k(flat).rule) == (0, "isotonic-max")
    falling = dict(
        zip(ks, [_counts_for_f1(f) for f in [0.7, 0.6, 0.6, 0.5, 0.5, 0.4, 0.4, 0.3]], strict=True)
    )
    assert select_k(falling).k_star == 0
