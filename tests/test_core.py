import numpy as np
import pytest
from conftest import example
from pydantic import BaseModel

from trice.core.retrieval import Retriever, Stage, select_demonstrations
from trice.core.selection import select_k
from trice.schema import Condition, Ranking

S1 = example(0, ["John", "Smith", "visited", "New", "York", "."], ["B-PER", "I-PER", "O", "B-LOC", "I-LOC", "O"])
S2 = example(1, ["Nothing", "here", "."])
S3 = example(2, ["Python", "beats", "Python", "?"], ["B-programlang", "O", "B-programlang", "O"])


def test_labels_resolve_any_alias(labels):
    for alias in ("Programming Language", "programlang", "programming_language"):
        assert labels.resolve(alias) == "programlang"
    assert labels.resolve("organisation") is None


@pytest.mark.parametrize("fmt", ["P0", "P1", "P2", "P3"])
@pytest.mark.parametrize("sentence", [S1, S2, S3])
def test_render_then_parse_roundtrip(app, labels, fmt, sentence):
    format = app.formats[fmt]
    parsed = format.parse_output(labels, format.render_answer(labels, sentence), sentence)
    assert parsed.tags == sentence.tags
    assert parsed.invalid == 0 and not parsed.malformed


def test_p0_reproduces_v1_listing(app, labels):
    prompt = app.formats["P0"].render_prompt(labels, [S1], example(9, ["Mary", "went", "to", "Paris", "."]))
    assert prompt.stop == ["\nExample "]
    assert prompt.messages[0].content == (
        "Defn: An entity is a location, person, or programming language.\n"
        "Example 1: John Smith visited New York .\nAnswer:\n"
        "1. John Smith (person)\n2. New York (location)\n"
        "Example 2: Mary went to Paris .\nAnswer:"
    )


def test_chat_formats(app, labels):
    assert app.formats["P3"].render_answer(labels, S1) == (
        "<person>John Smith</person> visited <location>New York</location> ."
    )
    assert app.formats["P2"].render_answer(labels, S2) == '{"entities": []}'
    for fmt in ("P1", "P2", "P3"):
        prompt = app.formats[fmt].render_prompt(labels, [S1, S2], S3)
        assert [m.role for m in prompt.messages] == ["system", "user", "assistant", "user", "assistant", "user"]
        assert (prompt.response_format is not None) == (fmt == "P2")


def test_parse_counts_invalid_mentions(app, labels):
    out = "1. John Smith (person)\n2. Boston (location)\n3. New York (city)\nchatter"
    parsed = app.formats["P1"].parse_output(labels, out, S1)
    assert parsed.tags == ["B-PER", "I-PER", "O", "O", "O", "O"]
    assert (parsed.candidates, parsed.invalid) == (4, 3)
    assert app.formats["P2"].parse_output(labels, "not json", S1).malformed


@pytest.mark.parametrize("dataset", ["crossner_ai", "crossner_music", "conll2003", "wnut17", "ncbi_disease"])
def test_roundtrip_on_real_data(app, dataset):
    """P3 carries positions and is exact; list/JSON formats can only miss repeated surface forms."""
    if not app.config.layout.canonical.fill(dataset=dataset).exists():
        pytest.skip("run `trice data` first")
    labels, examples = app.corpus.load_labels(dataset), app.corpus.load(dataset, "train")[:1500]
    for fmt, format in app.formats.items():
        wrong = sum(format.parse_output(labels, format.render_answer(labels, e), e).tags != e.tags for e in examples)
        assert wrong == 0 if fmt == "P3" else wrong <= 3, (fmt, wrong)


class Fixed(BaseModel):
    sign: float

    def score(self, queries: list[str], pool: list[str]) -> np.ndarray:
        return np.tile(self.sign * np.arange(len(pool), dtype=float), (len(queries), 1))


class EvenFirst(BaseModel):
    def score_pairs(self, query: str, docs: list[str]) -> np.ndarray:
        return np.array([float(int(d.split()[0][1:]) % 2 == 0) for d in docs])


POOL = [example(i, [f"w{i}", "x"]) for i in range(6)] + [example(6, ["w0", "x"])]


def stage_ids(rankings: list[Ranking], stage: str) -> list[str]:
    return next(r.ids for r in rankings if r.stage == stage)


def test_retriever_loo_rrf_and_rerank():
    retriever = Retriever(
        stages=[Stage(name="sparse", scorer=Fixed(sign=-1)), Stage(name="dense", scorer=Fixed(sign=1))],
        reranker=EvenFirst(),
        rrf_k=60,
        rerank_depth=10,
        store_depth=10,
    )
    rankings = retriever.rank([POOL[0]], POOL, loo=True)
    assert [r.stage for r in rankings] == ["sparse", "dense", "rrf", "full"]
    for ranking in rankings:
        assert set(ranking.ids) == {f"d/train/{i:06d}" for i in range(1, 6)}  # self and duplicate excluded
    assert stage_ids(rankings, "sparse")[:2] == ["d/train/000001", "d/train/000002"]
    assert stage_ids(rankings, "full")[:2] == ["d/train/000002", "d/train/000004"]  # even first, ties by id
    single = Retriever(stages=[Stage(name="bm25", scorer=Fixed(sign=1))], rrf_k=60, rerank_depth=3, store_depth=2).rank(
        [S2], POOL, loo=False
    )
    assert [r.stage for r in single] == ["full"] and len(single[0].ids) == 2


def test_rrf_matches_formula():
    retriever = Retriever(
        stages=[Stage(name="a", scorer=Fixed(sign=-1)), Stage(name="b", scorer=Fixed(sign=1))],
        rrf_k=60,
        rerank_depth=3,
        store_depth=7,
    )
    rankings = retriever.rank([S2], POOL, loo=False)
    ranks_a, ranks_b = np.arange(1, 8), np.arange(7, 0, -1)
    rrf = 1 / (60 + ranks_a) + 1 / (60 + ranks_b)
    ids = [p.id for p in POOL]
    assert stage_ids(rankings, "rrf") == [ids[i] for i in np.lexsort((ids, -rrf))]


def test_demonstrations_are_seeded_and_order_free():
    random0 = Condition.parse("random:0")
    a = select_demonstrations(random0, 3, "d", POOL[0], POOL, None)
    assert a == select_demonstrations(random0, 3, "d", POOL[0], list(reversed(POOL)), None)
    assert all(d.text != POOL[0].text for d in a)
    assert a != select_demonstrations(Condition.parse("random:1"), 3, "d", POOL[0], POOL, None)
    ranked = [p.id for p in POOL[1:5]]
    shuffled = select_demonstrations(Condition.parse("small/full"), 3, "d", POOL[0], POOL, ranked)
    reversed_rank = select_demonstrations(Condition.parse("small/full"), 3, "d", POOL[0], POOL, ranked[:3][::-1])
    assert shuffled == reversed_rank  # order depends on the set, never on rank
    assert select_demonstrations(Condition.parse("zero_shot"), 3, "d", POOL[0], POOL, None) == []
    with pytest.raises(ValueError, match="bad condition"):
        Condition.parse("small")


def test_select_k():
    ks = [0, 1, 2, 4, 8, 16, 32, 64]
    rising = select_k(dict(zip(ks, [0.2, 0.4, 0.5, 0.6, 0.7, 0.7, 0.7, 0.7], strict=True)))
    assert rising.rule == "kneedle" and rising.k_star in (4, 8)
    assert select_k(dict.fromkeys(ks, 0.6)).model_dump(include={"k_star", "rule"}) == {
        "k_star": 0,
        "rule": "isotonic-max",
    }
    assert select_k(dict(zip(ks, [0.7, 0.6, 0.6, 0.5, 0.5, 0.4, 0.4, 0.3], strict=True))).k_star == 0
