"""Entity-level scoring with seqeval (conlleval semantics, IOB2 tags over raw labels).

Per-sentence (tp, fp, fn) are kept because every paired test resamples sentences.
"""

import evaluate
import numpy as np
from pydantic import BaseModel
from seqeval.metrics.sequence_labeling import get_entities


class Evaluator(BaseModel):
    def counts(self, predictions: list[list[str]], references: list[list[str]]) -> np.ndarray:
        """(n_sentences, 3) array of per-sentence (tp, fp, fn)."""
        rows = []
        for pred, gold in zip(predictions, references, strict=True):
            p, g = set(get_entities(pred)), set(get_entities(gold))
            rows.append((len(p & g), len(p - g), len(g - p)))
        return np.array(rows, dtype=np.int64).reshape(-1, 3)

    @staticmethod
    def f1(sums: np.ndarray) -> np.ndarray:
        """Micro-F1 of summed (tp, fp, fn), vectorized over leading axes."""
        tp, fp, fn = sums[..., 0], sums[..., 1], sums[..., 2]
        denom = 2 * tp + fp + fn
        return np.divide(2 * tp, denom, out=np.zeros(np.shape(tp), float), where=denom > 0)

    def report(self, predictions: list[list[str]], references: list[list[str]]) -> dict:
        """Per-type and overall precision/recall/F1 from the `evaluate` seqeval metric."""
        metric = evaluate.load("seqeval")
        # ty misreads evaluate's decorated EvaluationModule.compute(self, *, predictions, references).
        return metric.compute(predictions=predictions, references=references) or {}  # ty: ignore[missing-argument]
