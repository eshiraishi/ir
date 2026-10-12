"""End to end through the container, offline: S0 format choice -> LOO curves -> k* -> S1 ->
tables and TikZ figures -> byte-identical replay -> budget. The LLM is an oracle that answers
correctly only when given at least two demonstrations, so k* has a known knee."""

import tomllib
from pathlib import Path

import numpy as np
import pytest
import tomli_w
from datasets import Dataset, DatasetDict

from trice.budget import TokenCounter
from trice.container import Container
from trice.core.prompting import Labels, Prompter
from trice.core.retrieval import Reranker, Retriever, Scorer
from trice.llm import LLM, Endpoint, Mode
from trice.schema import Example

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


class Overlap(Scorer):
    def score(self, queries: list[str], pool: list[str]) -> np.ndarray:
        return np.array([[len(set(q.split()) & set(p.split())) for p in pool] for q in queries], float)


class OverlapReranker(Reranker):
    def score_pairs(self, query: str, docs: list[str]) -> np.ndarray:
        return Overlap().score([query], docs)[0]


class Oracle(LLM):
    prompter: Prompter
    gold: dict[str, Example]

    def send(self, request: dict) -> dict:
        messages = request["messages"]
        if messages[0]["role"] == "system":
            system = messages[0]["content"]
            fmt = "P2" if "response_format" in request else "P3" if "repeating" in system else "P1"
            query, demos = messages[-1]["content"], (len(messages) - 2) // 2
        else:
            content = messages[0]["content"]
            fmt, demos = "P0", content.count("\nExample ") - 1
            query = content.rsplit("\nExample ", 1)[1].split(": ", 1)[1].removesuffix("\nAnswer:")
        example = self.gold[query]
        shown = example if demos >= 2 else example.model_copy(update={"tags": ["O"] * len(example.tokens)})
        text = self.prompter.answer(fmt, shown, Labels(raw_to_verbal=LABELS))
        return {"choices": [{"message": {"content": text}, "finish_reason": "stop"}], "provider": "P"}


class OfflineContainer(Container):
    def llm(self, model: str, mode: Mode = "readwrite") -> LLM:
        gold = {
            e.text: e
            for name in ("toy_a", "toy_b")
            for split in ("train", "test")
            for e in self.corpus.load(name, split)
        }
        return Oracle(
            endpoint=Endpoint(
                backend="openrouter", model=model, provider="P", prompt_price=1e-7, completion_price=2e-7
            ),
            client=self.openrouter,
            cache=self.cache,
            settings=self.settings.llm,
            mode=mode,
            prompter=self.experiment.prompter,
            gold=gold,
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
    for offset, name in enumerate(("toy_a", "toy_b")):
        splits = {"train": sentences(name, "train", 30, 10 * offset), "test": sentences(name, "test", 12, 100 + offset)}
        DatasetDict({s: Dataset.from_list([e.model_dump() for e in rows]) for s, rows in splits.items()}).save_to_disk(
            tmp_path / "data" / "canonical" / name
        )
    return tmp_path


def test_full_pipeline(project: Path) -> None:
    c = OfflineContainer()
    retriever = Retriever(
        stages={"sparse": Overlap(), "dense": Overlap()},
        reranker=OverlapReranker(),
        rrf_k=60,
        rerank_depth=200,
        store_depth=200,
    )
    for name in ("toy_a", "toy_b"):
        for queries in ("test", "loo"):
            c.experiment.rank("small", retriever, name, queries)

    for run in c.experiment.plan("S0"):
        c.experiment.execute(run, c.llm(run.model))
    assert c.experiment.select_format() == "P0"  # every format is perfect; ties go to P0

    loo = c.experiment.plan("S1-loo")
    assert {r.k for r in loo} == {0, 1, 2, 4, 8, 16}  # grid capped by the 30-sentence pool
    for run in loo:
        c.experiment.execute(run, c.llm(run.model))
    assert c.experiment.select_k()["m1"]["toy_a"]["k_star"] == 2

    for run in c.experiment.plan("S1"):
        c.experiment.execute(run, c.llm(run.model))
    assert c.analysis.run() == ["S1", "S5"]
    tables = project / "results" / "tables"
    assert (tables / "S1.tex").exists() and "\\begin{tikzpicture}" in (tables / "S5-scaling.tex").read_text()
    vs_zero = (tables / "S1-vs-zero_shot.json").read_text()
    assert '"delta": 1.0' in vs_zero

    runs = sorted((project / "results" / "runs").rglob("*.jsonl"))
    before = [p.read_bytes() for p in runs]
    for p in runs:
        p.unlink()
    replay = OfflineContainer()
    for study in ("S0", "S1-loo", "S1"):
        for run in replay.experiment.plan(study):
            replay.experiment.execute(run, replay.llm(run.model, mode="replay"))
    assert [p.read_bytes() for p in sorted((project / "results" / "runs").rglob("*.jsonl"))] == before

    budget = c.budget.model_copy(
        update={"counters": {m: TokenCounter(tokenizer=WordTokenizer()) for m in c.settings.models}}
    )
    payload = budget.estimate(["S1"])
    distinct = {(r.model, p.model_dump_json()) for r in c.experiment.plan("S1") for _, p in c.experiment.prompts(r)}
    assert payload["total"]["calls"] == len(distinct) and payload["max_error"] == 0.0  # repeats are priced once
    assert "| S1 |" in budget.render(payload)
