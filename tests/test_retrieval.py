import numpy as np
import pytest

from trice.data.bio import to_sentence
from trice.retrieval.demos import Condition, select_demos
from trice.retrieval.pipeline import Scorers, rank_queries
from trice.retrieval.rrf import order, ranks, rrf_scores


def _s(i, words, tag="O"):
    return to_sentence(f"d/train/{i:06d}", words, [tag] * len(words))


POOL = [_s(i, [f"w{i}", "x"]) for i in range(6)] + [_s(6, ["w0", "x"])]  # 6 duplicates 0's text


def test_order_and_ranks_break_ties_by_id():
    scores = np.array([0.5, 0.9, 0.5, 0.1])
    ids = np.array(["c", "a", "b", "d"])
    assert order(scores, ids).tolist() == [1, 2, 0, 3]
    assert ranks(scores, ids).tolist() == [3, 1, 2, 4]


def test_rrf_matches_reference():
    r1, r2 = np.array([1, 2, 3]), np.array([3, 1, 2])
    expected = [1 / 61 + 1 / 63, 1 / 62 + 1 / 61, 1 / 63 + 1 / 62]
    assert rrf_scores([r1, r2]).tolist() == pytest.approx(expected)


def _fake_scorers(rerank=True):
    # Sparse prefers low ids, dense prefers high ids; the reranker prefers even ids.
    def sparse(qs, ps):
        return np.tile(-np.arange(len(ps), dtype=float), (len(qs), 1))

    def dense(qs, ps):
        return np.tile(np.arange(len(ps), dtype=float), (len(qs), 1))

    def rr(q, docs):
        return np.array([1.0 if int(d.split()[0][1:]) % 2 == 0 else 0.0 for d in docs])

    return Scorers({"sparse": sparse, "dense": dense}, rerank=rr if rerank else None)


def test_rank_queries_loo_excludes_self_and_duplicates():
    out = rank_queries(
        [POOL[0]], POOL, _fake_scorers(), loo=True, rrf_k=60, rerank_depth=10, store_depth=10
    )
    (stages,) = out
    for name in ("sparse", "dense", "rrf", "full"):
        got = {pid for pid, _ in stages[name]}
        assert "d/train/000000" not in got and "d/train/000006" not in got, name
        assert len(got) == 5
    assert [pid for pid, _ in stages["sparse"]][:2] == ["d/train/000001", "d/train/000002"]
    reranked = [pid for pid, _ in stages["full"]]
    assert reranked[:2] == ["d/train/000002", "d/train/000004"]  # even ids first, ties by id


def test_rank_queries_test_mode_keeps_everything_and_truncates():
    q = _s(99, ["zz"])
    (stages,) = rank_queries(
        [q], POOL, _fake_scorers(), loo=False, rrf_k=60, rerank_depth=3, store_depth=2
    )
    assert len(stages["sparse"]) == 2 and len(stages["full"]) == 2
    single = rank_queries(
        [q], POOL, Scorers({"bm25": _fake_scorers().first_stage["sparse"]}), False, 60, 3, 4
    )
    assert list(single[0]) == ["full"]


def test_condition_parse_roundtrip():
    for text in ("zero_shot", "random:2", "small/full", "v1_bm25/full"):
        assert str(Condition.parse(text)) == text
    with pytest.raises(ValueError, match="bad condition"):
        Condition.parse("small")


def test_random_demos_are_deterministic_and_exclude_query():
    by_id = {p.id: p for p in POOL}
    cond = Condition.parse("random:0")
    a = select_demos(cond, 3, "d", "loo", POOL[0], POOL, by_id)
    b = select_demos(cond, 3, "d", "loo", POOL[0], list(reversed(POOL)), by_id)
    assert a == b and len(a) == 3
    assert all(d.text != POOL[0].text for d in a)
    assert select_demos(Condition.parse("random:1"), 3, "d", "loo", POOL[0], POOL, by_id) != a
    assert select_demos(Condition.parse("zero_shot"), 3, "d", "loo", POOL[0], POOL, by_id) == []
