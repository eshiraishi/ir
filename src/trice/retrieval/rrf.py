"""Ranking utilities: deterministic ordering and Reciprocal Rank Fusion."""

import numpy as np

RRF_K = 60  # Cormack, Clarke & Buettcher (2009); fixed, never tuned.


def order(scores: np.ndarray, ids: np.ndarray) -> np.ndarray:
    """Indices sorting by descending score, ties broken by ascending id (fully deterministic)."""
    return np.lexsort((ids, -scores))


def ranks(scores: np.ndarray, ids: np.ndarray) -> np.ndarray:
    """1-based rank of every item under :func:`order`."""
    r = np.empty(len(scores), dtype=np.int64)
    r[order(scores, ids)] = np.arange(1, len(scores) + 1)
    return r


def rrf_scores(rank_lists: list[np.ndarray], k: int = RRF_K) -> np.ndarray:
    """Fuse full rank vectors (same item order) with RRF: sum_i 1 / (k + rank_i)."""
    return np.sum([1.0 / (k + r) for r in rank_lists], axis=0)
