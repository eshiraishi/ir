"""Entity-level scoring with seqeval (conlleval semantics, IOB2 tags over raw labels).

Per-sentence (tp, fp, fn) are kept because every paired test resamples sentences.
"""

import evaluate
import numpy as np
from pydantic import BaseModel
from seqeval.metrics.sequence_labeling import get_entities


class Evaluator(BaseModel):
    def count_matches(self, predictions: list[list[str]], references: list[list[str]]) -> np.ndarray:
        """(n_sentences, 3) array of per-sentence (tp, fp, fn)."""
        rows = []
        for pred, gold in zip(predictions, references, strict=True):
            predicted = set(get_entities(pred))
            expected = set(get_entities(gold))
            rows.append((len(predicted & expected), len(predicted - expected), len(expected - predicted)))
        return np.array(rows, dtype=np.int64).reshape(-1, 3)

    @staticmethod
    def compute_f1(sums: np.ndarray) -> np.ndarray:
        """Micro-F1 of summed (tp, fp, fn), vectorized over leading axes."""
        tp = sums[..., 0]
        fp = sums[..., 1]
        fn = sums[..., 2]
        denom = 2 * tp + fp + fn
        return np.divide(2 * tp, denom, out=np.zeros(np.shape(tp), float), where=denom > 0)

    def report(self, predictions: list[list[str]], references: list[list[str]]) -> dict:
        """Per-type and overall precision/recall/F1 from the `evaluate` seqeval metric."""
        metric = evaluate.load("seqeval")
        # ty misreads evaluate's decorated EvaluationModule.compute(self, *, predictions, references).
        result = metric.compute(predictions=predictions, references=references)  # ty: ignore[missing-argument]
        if result is None:
            raise RuntimeError("seqeval returned no result")
        return result
