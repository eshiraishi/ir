"""Demonstration retrieval: first-stage scorers, exact RRF, reranking, and demo selection.

Pure ranking logic; the scorers themselves (models, APIs) live in core/encoders.py.
"""

import operator
from typing import Protocol, runtime_checkable

import numpy as np
from pydantic import BaseModel, ConfigDict

from trice.schema import Condition, Example, Ranking, derive_seed


@runtime_checkable
class Scorer(Protocol):
    def score(self, queries: list[str], pool: list[str]) -> np.ndarray:
        """Relevance of every pool text to every query text: an (n_queries, n_pool) array."""
        ...


@runtime_checkable
class Reranker(Protocol):
    def score_pairs(self, query: str, docs: list[str]) -> np.ndarray: ...


class Stage(BaseModel):
    """A first-stage scorer and the name its rankings are stored under."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    scorer: Scorer


class Retriever(BaseModel):
    """Stages ``sparse``/``dense``/... (full-pool scores), ``rrf`` (exact RRF over full-pool ranks)
    and ``full`` (reranker over the top fused candidates). A single first stage is stored as
    ``full``. Leave-one-out queries never see themselves or pool sentences with identical text."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    stages: list[Stage]
    reranker: Reranker | None = None
    rrf_k: int
    rerank_depth: int
    store_depth: int

    @staticmethod
    def order(scores: np.ndarray, ids: np.ndarray) -> np.ndarray:
        """Descending score, ties broken by ascending id."""
        return np.lexsort((ids, -scores))

    def rank(self, queries: list[Example], pool: list[Example], loo: bool) -> list[Ranking]:
        pool = sorted(pool, key=operator.attrgetter("id"))
        ids = np.array([e.id for e in pool])
        texts = [e.text for e in pool]
        matrices = [(s.name, s.scorer.score([q.text for q in queries], texts)) for s in self.stages]
        rankings: list[Ranking] = []
        for qi, query in enumerate(queries):
            keep = np.array([not (loo and (e.id == query.id or e.text == query.text)) for e in pool])
            kept = np.flatnonzero(keep)
            stages = []
            rrf = np.zeros(len(kept))
            for name, matrix in matrices:
                order = self.order(matrix[qi][kept], ids[kept])
                stages.append(Ranking(query_id=query.id, stage=name, ids=ids[kept][order][: self.store_depth].tolist()))
                ranks = np.empty(len(kept))
                ranks[order] = np.arange(1, len(kept) + 1)
                rrf += 1.0 / (self.rrf_k + ranks)
            if len(matrices) == 1:
                stages = [stages[0].model_copy(update={"stage": "full"})]
            else:
                fused = self.order(rrf, ids[kept])
                stages.append(
                    Ranking(query_id=query.id, stage="rrf", ids=ids[kept][fused][: self.store_depth].tolist())
                )
                if self.reranker:
                    cand = kept[fused[: self.rerank_depth]]
                    scores = self.reranker.score_pairs(query.text, [texts[i] for i in cand])
                    reranked = ids[cand][self.order(scores, ids[cand])][: self.store_depth].tolist()
                    stages.append(Ranking(query_id=query.id, stage="full", ids=reranked))
            rankings += stages
        return rankings


def select_demonstrations(
    condition: Condition, k: int, dataset: str, query: Example, pool: list[Example], ranked: list[str]
) -> list[Example]:
    """The k demonstrations for one query, shuffled with the per-query seed so their order never
    carries rank or dataset order."""
    if condition.kind == "zero_shot" or k == 0:
        return []
    by_id = {e.id: e for e in pool}
    if condition.kind == "random":
        ids = sorted(e.id for e in pool if e.id != query.id and e.text != query.text)
        rng = np.random.default_rng(derive_seed("random", condition.seed, dataset, query.id))
        chosen = rng.choice(ids, size=k, replace=False).tolist()
    else:
        chosen = ranked[:k]
        if len(chosen) < k:
            raise ValueError(f"{dataset}/{query.id}: only {len(chosen)} candidates for k={k}")
    rng = np.random.default_rng(derive_seed("order", dataset, query.id))
    return [by_id[i] for i in rng.permutation(sorted(chosen)).tolist()]
