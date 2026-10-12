"""Entity-level scoring: multisets of (normalized span, label) per sentence, micro-averaged.

Per-sentence (tp, fp, fn) counts are kept as arrays because every paired statistical test
(approximate randomization, bootstrap) resamples sentences, not entities.
"""

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from trice.data.schema import Sentence
from trice.prompt.labels import LabelSet
from trice.prompt.parse import normalize_ws

Pair = tuple[str, str]


def gold_pairs(sentence: Sentence, labels: LabelSet) -> list[Pair]:
    return [(normalize_ws(e.text), labels.verbal(e.label)) for e in sentence.entities]


def sentence_counts(pred: Iterable[Pair], gold: Iterable[Pair]) -> tuple[int, int, int]:
    p, g = Counter(pred), Counter(gold)
    tp = sum((p & g).values())
    return tp, sum(p.values()) - tp, sum(g.values()) - tp


def counts_matrix(preds: Sequence[Iterable[Pair]], golds: Sequence[Iterable[Pair]]) -> np.ndarray:
    """``(n_sentences, 3)`` int array of per-sentence (tp, fp, fn)."""
    if len(preds) != len(golds):
        raise ValueError("predictions and gold must align sentence by sentence")
    return np.array(
        [sentence_counts(p, g) for p, g in zip(preds, golds, strict=True)], dtype=np.int64
    ).reshape(-1, 3)


@dataclass(frozen=True)
class PRF:
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int


def prf(tp: int, fp: int, fn: int) -> PRF:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0
    return PRF(p, r, f, int(tp), int(fp), int(fn))


def micro(counts: np.ndarray) -> PRF:
    tp, fp, fn = counts.sum(axis=0)
    return prf(int(tp), int(fp), int(fn))


def f1_of_sums(sums: np.ndarray) -> np.ndarray:
    """Vectorized micro-F1 for an ``(..., 3)`` array of summed (tp, fp, fn)."""
    tp, fp, fn = sums[..., 0], sums[..., 1], sums[..., 2]
    denom = 2 * tp + fp + fn
    return np.divide(2 * tp, denom, out=np.zeros(np.shape(tp), dtype=float), where=denom > 0)


def per_type(preds: Sequence[Iterable[Pair]], golds: Sequence[Iterable[Pair]]) -> dict[str, PRF]:
    totals: dict[str, list[int]] = {}
    for p, g in zip(preds, golds, strict=True):
        pc, gc = Counter(p), Counter(g)
        for label in {lab for _, lab in pc} | {lab for _, lab in gc}:
            pl = Counter({k: v for k, v in pc.items() if k[1] == label})
            gl = Counter({k: v for k, v in gc.items() if k[1] == label})
            tp = sum((pl & gl).values())
            t = totals.setdefault(label, [0, 0, 0])
            t[0] += tp
            t[1] += sum(pl.values()) - tp
            t[2] += sum(gl.values()) - tp
    return {label: prf(*t) for label, t in sorted(totals.items())}


def exact_match(counts: np.ndarray) -> np.ndarray:
    """Per-sentence boolean: every gold mention found and nothing spurious."""
    return (counts[:, 1] == 0) & (counts[:, 2] == 0)
