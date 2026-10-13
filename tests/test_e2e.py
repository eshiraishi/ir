"""End to end through compose(), offline: S0 format choice -> LOO curves -> k* -> S1 ->
tables and TikZ figures -> byte-identical replay -> budget. The LLM is an oracle that answers
correctly only when given at least two demonstrations, so k* has a known knee."""

import tomllib
from pathlib import Path

import numpy as np
import pytest
import tomli_w
from datasets import Dataset, DatasetDict

from trice.budget import TokenCounter
from trice.compose import compose
from trice.config import Config
from trice.core.formats import Format
from trice.core.retrieval import Retriever, Stage
from trice.llm import ChatRequest, Endpoint, EndpointTable
from trice.schema import Completion, Example, FormatName, LabelMap

NAMES = ["Alice", "Bruno", "Carla", "Diego", "Elena", "Fabio", "Gina", "Hugo", "Iris", "Joao"]
CITIES = ["Paris", "Lima", "Oslo", "Quito", "Rome", "Seoul", "Tokyo", "Accra"]
LABELS = {"PER": "person", "LOC": "location"}


def sentences(name: str, split: str, n: int, offset: int) -> list[Example]:
    out = []
    for i in range(n):
        if i % 5 == 4:
            tokens, tags = ["nothing", "happened", f"day{i + offset}", "."], ["O"] * 4
        else:
            tokens = [NAMES[(i + offset) % 10], "visited", CITIES[(3 * i + offset) % 8], f"in{i + offset}", "."]
            tags = ["B-PER", "O", "B-LOC", "O", "O"]
        out.append(Example(id=f"{name}/{split}/{i:06d}", tokens=tokens, tags=tags))
    return out


class Overlap:
    def score(self, queries: list[str], pool: list[str]) -> np.ndarray:
        return np.array([[len(set(q.split()) & set(p.split())) for p in pool] for q in queries], float)


class OverlapReranker:
    def score_pairs(self, query: str, docs: list[str]) -> np.ndarray:
        return Overlap().score([query], docs)[0]


class Oracle:
    """Plays the network: answers correctly only when given at least two demonstrations."""

    def __init__(self) -> None:
        self.formats: dict[FormatName, Format] = {}
        self.gold: dict[str, Example] = {}

    def send(self, request: ChatRequest) -> Completion:
        messages = request.messages
        if messages[0].role == "system":
            fmt = "P2" if request.response_format else "P3" if "repeating" in messages[0].content else "P1"
            query, demos = messages[-1].content, (len(messages) - 2) // 2
        else:
            content = messages[0].content
            fmt, demos = "P0", content.count("\nExample ") - 1
            query = content.rsplit("\nExample ", 1)[1].split(": ", 1)[1].removesuffix("\nAnswer:")
        example = self.gold[query]
        shown = example if demos >= 2 else example.model_copy(update={"tags": ["O"] * len(example.tokens)})
        return Completion(
            text=self.formats[fmt].render_answer(LabelMap(LABELS), shown), finish_reason="stop", provider="P"
        )


class WordTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[str]:
        return text.split()

    def apply_chat_template(self, messages: list[dict], **kwargs: object) -> list[str]:
        return [t for m in messages for t in ["<s>", m["role"], *m["content"].split(), "</s>"]] + ["<a>"]


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    config = tomllib.loads((Path(__file__).parents[1] / "trice.toml").read_text())
    spec = {"source": "none", "train": {"path": "x", "size": 30}, "test": {"path": "y", "size": 12}, "labels": LABELS}
    config["sources"] = {"none": {"url": "http://unused"}}
    config["datasets"] = {"toy_a": spec, "toy_b": spec}
    model = {"openrouter": "x", "hf_repo": "x", "ladder": "toy"}
    config["models"] = {f"m{i}": {**model, "params_b": i, "active_params_b": i, "reference": i == 1} for i in (1, 2, 3)}
    config["studies"] = {k: v for k, v in config["studies"].items() if k in ("S0", "S1-loo", "S1")}
    config["stats"]["resamples"] = 500
    (tmp_path / "trice.toml").write_text(tomli_w.dumps(config))
    monkeypatch.chdir(tmp_path)
    layout = Config().layout
    endpoints = [
        Endpoint(name=m, backend="openrouter", model="x", provider="P", prompt_price=1e-7, completion_price=2e-7)
        for m in config["models"]
    ]
    layout.endpoints.prepare().write_text(EndpointTable(endpoints).model_dump_json())
    for offset, name in enumerate(("toy_a", "toy_b")):
        splits = {"train": sentences(name, "train", 30, 10 * offset), "test": sentences(name, "test", 12, 100 + offset)}
        DatasetDict({s: Dataset.from_list([e.model_dump() for e in rows]) for s, rows in splits.items()}).save_to_disk(
            layout.canonical.fill(dataset=name)
        )
    return tmp_path


def test_full_pipeline(project: Path) -> None:
    oracle = Oracle()
    retriever = Retriever(
        stages=[Stage(name="sparse", scorer=Overlap()), Stage(name="dense", scorer=Overlap())],
        reranker=OverlapReranker(),
        rrf_k=60,
        rerank_depth=200,
        store_depth=200,
    )
    with compose(Config(), oracle) as app:
        runner = app.runner
        oracle.formats = app.formats
        oracle.gold = {
            e.text: e
            for name in ("toy_a", "toy_b")
            for split in ("train", "test")
            for e in app.corpus.load(name, split)
        }
        for name in ("toy_a", "toy_b"):
            for queries in ("test", "loo"):
                runner.store_rankings("small", retriever, name, queries)

        for run in runner.expand_plan("S0"):
            runner.execute(run, app.llms[run.model])
        assert runner.select_format() == "P0"  # every format is perfect; ties go to P0

        loo = runner.expand_plan("S1-loo")
        assert {r.k for r in loo} == {0, 1, 2, 4, 8, 16}  # grid capped by the 30-sentence pool
        for run in loo:
            runner.execute(run, app.llms[run.model])
        assert runner.select_k().find("m1", "toy_a").selection.k_star == 2

        for run in runner.expand_plan("S1"):
            runner.execute(run, app.llms[run.model])
        assert app.analyzer.run_all() == ["S1", "S5"]
        tables = app.config.layout.tables
        assert tables.fill(name="S1", suffix="tex").exists()
        assert "\\begin{tikzpicture}" in tables.fill(name="S5-scaling", suffix="tex").read_text()
        assert '"delta": 1.0' in tables.fill(name="S1-vs-zero_shot", suffix="json").read_text()

        layout = app.config.layout
        runs = layout.runs.match(slug="**/*")
        before = [p.read_bytes() for p in runs]
        assert not layout.runs_partial.match(slug="**/*")
        for p in runs:
            p.unlink()
        for study in ("S0", "S1-loo", "S1"):
            for run in runner.expand_plan(study):
                runner.execute(run, app.llms[run.model].model_copy(update={"mode": "replay"}))
        assert [p.read_bytes() for p in layout.runs.match(slug="**/*")] == before

        budget = app.budget.model_copy(
            update={"counters": {m: TokenCounter(tokenizer=WordTokenizer()) for m in app.config.models}}
        )
        payload = budget.estimate(["S1"])
        distinct = {
            (r.model, p.model_dump_json()) for r in runner.expand_plan("S1") for _, p in runner.render_prompts(r)
        }
        assert payload.total.calls == len(distinct) and payload.max_error == 0.0  # repeats are priced once
        assert payload.total.cost > 0  # priced with the pinned endpoints
        assert "| S1 |" in budget.render(payload)


def test_failed_run_leaves_no_partial_result(project: Path) -> None:
    class Failing:
        def complete_many(self, prompts: list, audit_tag: str = "") -> list:
            raise RuntimeError("network down")

    with compose(Config(), Oracle()) as app:
        run = app.runner.expand_plan("S0", assume_format="P0")[0].model_copy(update={"condition": "zero_shot", "k": 0})
        with pytest.raises(RuntimeError):
            app.runner.execute(run, Failing())
        assert not app.config.layout.runs.match(slug="**/*")
