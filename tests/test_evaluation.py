import itertools

import numpy as np
import pandas as pd
import pytest
from pydantic import BaseModel

from trice.evaluation.inference import Inferrer
from trice.evaluation.scoring import Evaluator
from trice.schema import PathTemplate


def counts(n: int, p_tp: float, seed: int) -> np.ndarray:
    g = np.random.default_rng(seed)
    gold = g.integers(1, 6, n)
    tp = g.binomial(gold, p_tp)
    return np.stack([tp, g.integers(0, 3, n), gold - tp], axis=1)


def test_counts_are_strict_entity_matches():
    gold = [["B-PER", "I-PER", "O", "B-LOC"], ["O", "B-ORG"]]
    pred = [["B-PER", "O", "O", "B-LOC"], ["O", "B-ORG"]]
    c = Evaluator().count_matches(pred, gold)
    assert c.tolist() == [[1, 1, 1], [1, 0, 0]]
    assert Evaluator.compute_f1(c.sum(0)) == pytest.approx(2 * 2 / (2 * 2 + 1 + 1))


def test_permutation_test_is_exact_for_small_samples():
    a, b = counts(10, 0.8, 1), counts(10, 0.5, 2)
    observed = abs(Evaluator.compute_f1(a.sum(0)) - Evaluator.compute_f1(b.sum(0)))
    hits = 0
    for mask in itertools.product([False, True], repeat=10):
        m = np.array(mask)[:, None]
        hits += (
            abs(Evaluator.compute_f1(np.where(m, b, a).sum(0)) - Evaluator.compute_f1(np.where(m, a, b).sum(0)))
            >= observed - 1e-12
        )
    assert Inferrer(resamples=10_000, alpha=0.05).compare_pair(a, b, "x").p_value == pytest.approx(hits / 2**10)


def test_compare_detects_differences_and_is_seeded():
    inferrer = Inferrer(resamples=2_000, alpha=0.05)
    a, b = counts(300, 0.9, 3), counts(300, 0.5, 4)
    result = inferrer.compare_pair(a, b, "y")
    assert result.p_value < 0.001 and result.ci_low < result.delta < result.ci_high and result.ci_low > 0
    assert inferrer.compare_pair(a, b, "y") == result
    assert inferrer.compare_pair(a, a.copy(), "y").delta == 0


def test_omnibus_and_trend_tests():
    inferrer = Inferrer(resamples=1_000, alpha=0.05)
    scores = pd.DataFrame(np.random.default_rng(0).random((20, 4)) + np.array([0, 0.5, 1.0, 1.5]), columns=list("abcd"))
    friedman = inferrer.test_friedman(scores)
    assert friedman.p_value < 0.05 and friedman.mean_ranks["d"] < friedman.mean_ranks["a"]
    trend = inferrer.test_trend(
        np.array([[1, 2, 3, 4], [1.5, 2.1, 2.9, 4.2], [0.9, 2.5, 3.3, 3.9], [1, 1.9, 3.1, 4.4]])
    )
    assert trend.p_increasing < 0.01 < trend.p_decreasing


class Row(BaseModel):
    model: str
    f1: float


class Point(BaseModel):
    k: int
    f1: float
    curve: str


def test_reporter_writes_tables_and_tikz(app, tmp_path):
    reporter = app.analyzer.reporter.model_copy(update={"tables": PathTemplate(f"{tmp_path}/{{name}}.{{suffix}}")})
    reporter.write_table("t", [Row(model="m_1", f1=0.12345)], "Caption & more")
    assert "0.1235" in (tmp_path / "t.md").read_text()
    assert r"m\_1" in (tmp_path / "t.tex").read_text()
    points = [
        Point(k=k, f1=f, curve=c)
        for c, fs in (("a", [0.1, 0.3, 0.4]), ("b", [0.2, 0.3, 0.35]))
        for k, f in enumerate(fs)
    ]
    reporter.plot_lines("curves", points, x="k", y="f1", hue="curve")
    assert "\\begin{tikzpicture}" in (tmp_path / "curves.tex").read_text()
    reporter.plot_critical_difference(
        "cd",
        Inferrer(resamples=10, alpha=0.05).test_friedman(
            pd.DataFrame(np.random.default_rng(1).random((12, 3)) + np.array([0, 0.2, 0.4]), columns=["x", "y", "z"])
        ),
    )
    assert (tmp_path / "cd.pdf").exists()
