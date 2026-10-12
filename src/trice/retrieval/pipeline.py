"""Compute and store rankings for one (stack, dataset, query set).

Stages for a TRICE stack: ``sparse`` and ``dense`` (full-pool scores), ``rrf`` (exact RRF over
full-pool ranks, k=60), ``full`` (the reranker over the top ``rerank_depth`` fused candidates).
v1 stacks have only ``full``. For leave-one-out queries, the query itself and every pool
sentence with identical text are excluded before ranking.

Rankings are written once to ``data/rankings/<stack>/<dataset>/<queries>.jsonl`` and only read
afterwards; their SHA-256 goes into ``data/rankings/manifest.json``.
"""

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Literal

import numpy as np

from trice.config import StackSpec, repo_root, retrieval_config
from trice.data.build import load_split
from trice.data.schema import Sentence
from trice.retrieval.rrf import order, ranks, rrf_scores
from trice.seeds import sample

QuerySet = Literal["test", "loo"]
Ranked = list[tuple[str, float]]


def rankings_dir() -> Path:
    return repo_root() / "data" / "rankings"


def loo_queries(dataset: str, n: int) -> list[Sentence]:
    """Leave-one-out query set: the whole pool if it has at most ``n`` sentences (all CrossNER
    domains), else a seeded sample of ``n``."""
    pool = load_split(dataset, "train")
    if len(pool) <= n:
        return sorted(pool, key=lambda s: s.id)
    return sorted(sample(pool, n, "loo", dataset, key=lambda s: s.id), key=lambda s: s.id)


def query_set(dataset: str, queries: QuerySet, loo_n: int) -> list[Sentence]:
    return load_split(dataset, "test") if queries == "test" else loo_queries(dataset, loo_n)


def exclusion_mask(query: Sentence, pool: Sequence[Sentence], loo: bool) -> np.ndarray:
    if not loo:
        return np.zeros(len(pool), bool)
    return np.array([p.id == query.id or p.text == query.text for p in pool])


@dataclass
class Scorers:
    """Callables producing (n_queries, n_pool) score matrices, plus a pairwise reranker."""

    first_stage: dict[str, Callable[[Sequence[str], Sequence[str]], np.ndarray]]
    rerank: Callable[[str, Sequence[str]], np.ndarray] | None = None


def rank_queries(
    queries: Sequence[Sentence],
    pool: Sequence[Sentence],
    scorers: Scorers,
    loo: bool,
    rrf_k: int,
    rerank_depth: int,
    store_depth: int,
) -> list[dict[str, Ranked]]:
    """Return, per query, ``{stage: [(pool_id, score), ...]}`` truncated to ``store_depth``."""
    pool = sorted(pool, key=lambda s: s.id)
    ids = np.array([p.id for p in pool])
    pool_texts = [p.text for p in pool]
    q_texts = [q.text for q in queries]
    matrices = {name: fn(q_texts, pool_texts) for name, fn in scorers.first_stage.items()}

    out = []
    for qi, query in enumerate(queries):
        keep = ~exclusion_mask(query, pool, loo)
        kept_pos = np.flatnonzero(keep)
        kept_ids = ids[kept_pos]
        stages: dict[str, Ranked] = {}
        stage_ranks = []
        for name, m in matrices.items():
            s = m[qi][kept_pos]
            idx = order(s, kept_ids)[:store_depth]
            stages[name] = [(str(kept_ids[i]), float(s[i])) for i in idx]
            stage_ranks.append(ranks(s, kept_ids))
        if len(matrices) == 1:  # v1 stacks: a single stage, stored as "full"
            stages = {"full": next(iter(stages.values()))}
        else:
            fused = rrf_scores(stage_ranks, rrf_k)
            idx = order(fused, kept_ids)
            stages["rrf"] = [(str(kept_ids[i]), float(fused[i])) for i in idx[:store_depth]]
            if scorers.rerank is not None:
                cand = idx[:rerank_depth]
                rs = np.asarray(
                    scorers.rerank(query.text, [pool_texts[kept_pos[i]] for i in cand]), float
                )
                cand_ids = kept_ids[cand]
                ridx = order(rs, cand_ids)[:store_depth]
                stages["full"] = [(str(cand_ids[i]), float(rs[i])) for i in ridx]
        out.append(stages)
    return out


def _need[T](value: T | None, what: str, stack: StackSpec) -> T:
    if value is None:
        raise ValueError(f"stack {stack.name}: missing {what}")
    return value


def build_scorers(stack: StackSpec, cache_path: Path | None = None) -> Scorers:
    from trice.llm.cache import ResponseCache
    from trice.retrieval import encoders as enc

    cfg = retrieval_config()
    if stack.kind == "v1_bm25":
        k1, b = _need(stack.bm25_k1, "bm25_k1", stack), _need(stack.bm25_b, "bm25_b", stack)
        return Scorers({"bm25": enc.BM25V1(k1, b, stack.lemmatize).score})
    dense_ref = _need(stack.dense, "dense", stack)
    if stack.kind == "v1_dense":
        return Scorers({"dense": enc.DenseScorer(dense_ref, stack.query_instruction).score})

    sparse_ref = _need(stack.sparse, "sparse", stack)
    rerank_ref = _need(stack.reranker, "reranker", stack)
    api_cache = ResponseCache(cache_path or repo_root() / "cache" / "retrieval_api.sqlite")
    dense = (
        enc.OpenRouterEmbeddings(dense_ref, api_cache).score
        if dense_ref.backend == "openrouter"
        else enc.DenseScorer(dense_ref).score
    )
    reranker = (
        enc.OpenRouterReranker(rerank_ref, api_cache).score_pairs
        if rerank_ref.backend == "openrouter"
        else enc.QwenReranker(rerank_ref, cfg.reranker_instruction).score_pairs
    )
    return Scorers({"sparse": enc.SparseScorer(sparse_ref).score, "dense": dense}, rerank=reranker)


def _path(stack: str, dataset: str, queries: QuerySet) -> Path:
    return rankings_dir() / stack / dataset / f"{queries}.jsonl"


def write_rankings(
    stack: str, dataset: str, queries: QuerySet, qs: Sequence[Sentence], ranked
) -> str:
    path = _path(stack, dataset, queries)
    path.parent.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256()
    with path.open("wb") as f:
        for q, stages in zip(qs, ranked, strict=True):
            line = (
                json.dumps({"query_id": q.id, "stages": stages}, separators=(",", ":")) + "\n"
            ).encode()
            h.update(line)
            f.write(line)
    digest = h.hexdigest()
    manifest_path = rankings_dir() / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest[f"{stack}/{dataset}/{queries}"] = {"sha256": digest, "queries": len(qs)}
    manifest_path.write_text(json.dumps(dict(sorted(manifest.items())), indent=2) + "\n")
    return digest


def rank(stack_name: str, dataset: str, queries: QuerySet, loo_n: int = 200) -> str:
    cfg = retrieval_config()
    stack = cfg.stacks[stack_name]
    qs = query_set(dataset, queries, loo_n)
    pool = load_split(dataset, "train")
    ranked = rank_queries(
        qs,
        pool,
        build_scorers(stack),
        queries == "loo",
        cfg.rrf_k,
        cfg.rerank_depth,
        cfg.store_depth,
    )
    return write_rankings(stack_name, dataset, queries, qs, ranked)


@cache
def load_rankings(stack: str, dataset: str, queries: QuerySet) -> dict[str, dict[str, Ranked]]:
    path = _path(stack, dataset, queries)
    out = {}
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            out[row["query_id"]] = {k: [tuple(x) for x in v] for k, v in row["stages"].items()}
    return out
