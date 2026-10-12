"""End-to-end run on a synthetic dataset with a mock OpenRouter: S0 format selection -> LOO curves
-> k* -> S1 -> analysis -> replay determinism -> budget. No network, no model weights."""

import hashlib
import json
import shutil
from pathlib import Path

import httpx
import pytest
import yaml

from trice import analysis, budget, config, runner
from trice.data import build
from trice.data.bio import to_sentence
from trice.data.schema import write_jsonl
from trice.llm.cache import ResponseCache
from trice.llm.client import Backend, ChatClient
from trice.prompt.formats import render_answer
from trice.prompt.labels import LabelSet
from trice.retrieval import pipeline
from trice.studies import expand

REPO = Path(__file__).resolve().parents[1]
NAMES = ["Alice", "Bruno", "Carla", "Diego", "Elena", "Fabio", "Gina", "Hugo", "Iris", "Joao"]
CITIES = ["Paris", "Lima", "Oslo", "Quito", "Rome", "Seoul", "Tokyo", "Accra"]
LABELS = {"PER": "person", "LOC": "location"}


def _sentences(split: str, n: int, offset: int):
    out = []
    for i in range(n):
        name, city = NAMES[(i + offset) % len(NAMES)], CITIES[(i * 3 + offset) % len(CITIES)]
        if i % 5 == 4:
            words, tags = ["nothing", "happened", f"day{i + offset}", "."], ["O"] * 4
        else:
            words = [name, "visited", city, f"in{i + offset}", "."]
            tags = ["B-PER", "O", "B-LOC", "O", "O"]
        out.append(to_sentence(f"toy/{split}/{i:06d}", words, tags))
    return out


def _clear_caches():
    for fn in (
        config.datasets_config,
        config.models_config,
        config.retrieval_config,
        config.studies_config,
        build._load,
        runner._pool,
        pipeline.load_rankings,
    ):
        fn.cache_clear()


@pytest.fixture
def root(tmp_path, monkeypatch):
    (tmp_path / "configs").mkdir()
    shutil.copy(REPO / "configs" / "retrieval.yaml", tmp_path / "configs")
    datasets = {
        "sources": {"none": {"kind": "http"}},
        "datasets": {
            "toy": {
                "source": "none",
                "format": "token_tag",
                "files": {"train": {"url": "x"}, "test": {"url": "y"}},
                "expected": {"train": 30, "test": 12},
                "labels": LABELS,
            }
        },
    }
    models = {
        "models": {
            "m1": {
                "family": "f",
                "ladder": "gemma3",
                "params_b": 1,
                "active_params_b": 1,
                "openrouter": "x/m1",
                "hf_repo": "x/m1",
                "reference": True,
            }
        }
    }
    studies = yaml.safe_load((REPO / "configs" / "studies.yaml").read_text())
    studies["studies"] = {
        k: v for k, v in studies["studies"].items() if k in ("S0", "S1-loo", "S1")
    }
    for name, obj in (("datasets", datasets), ("models", models), ("studies", studies)):
        (tmp_path / "configs" / f"{name}.yaml").write_text(yaml.safe_dump(obj))
    (tmp_path / "configs" / "endpoints.lock.json").write_text(
        json.dumps(
            {
                "m1": {
                    "backend": "openrouter",
                    "model": "x/m1",
                    "provider": "P",
                    "quantization": "bf16",
                    "prompt_price": 1e-7,
                    "completion_price": 2e-7,
                }
            }
        )
    )
    monkeypatch.setenv("TRICE_ROOT", str(tmp_path))
    _clear_caches()

    train, test = _sentences("train", 30, 0), _sentences("test", 12, 100)
    ddir = tmp_path / "data" / "canonical" / "toy"
    write_jsonl(ddir / "train.jsonl", train)
    write_jsonl(ddir / "test.jsonl", test)

    # Stored rankings for small/full: fake scorers (lexical overlap), real ranking code.
    def overlap(qs, ps):
        import numpy as np

        return np.array([[len(set(q.split()) & set(p.split())) for p in ps] for q in qs], float)

    scorers = pipeline.Scorers(
        {"sparse": overlap, "dense": overlap}, rerank=lambda q, d: overlap([q], d)[0]
    )
    for qs_name in ("test", "loo"):
        qs = pipeline.query_set("toy", qs_name, 200)
        ranked = pipeline.rank_queries(qs, train, scorers, qs_name == "loo", 60, 200, 200)
        pipeline.write_rankings("small", "toy", qs_name, qs, ranked)
    yield tmp_path
    _clear_caches()


def _oracle(all_sentences):
    """Mock model: answers with the gold entities only when given >= 2 demonstrations."""
    by_text = {s.text: s for s in all_sentences}
    labels = LabelSet(LABELS)

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        msgs = body["messages"]
        if msgs[0]["role"] == "system":
            sys = msgs[0]["content"]
            fmt = "P2" if "response_format" in body else ("P3" if "repeating" in sys else "P1")
            query, n_demos = msgs[-1]["content"], (len(msgs) - 2) // 2
        else:
            content = msgs[0]["content"]
            fmt, n_demos = "P0", content.count("\nExample ") - 1
            query = content.rsplit("\nExample ", 1)[1].split(": ", 1)[1].rsplit("\nAnswer:", 1)[0]
        s = by_text[query]
        shown = s if n_demos >= 2 else s.model_copy(update={"entities": ()})
        text = render_answer(fmt, shown, labels)
        return httpx.Response(
            200,
            json={
                "id": "g",
                "provider": "P",
                "model": body["model"],
                "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )

    return handler


def _client(root, transport, mode="readwrite"):
    return ChatClient(
        Backend("openrouter", "http://mock"),
        ResponseCache(root / "cache" / "llm.sqlite"),
        mode=mode,
        transport=transport,
    )


def _digest(paths):
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(p.read_bytes())
    return h.hexdigest()


def test_full_pipeline(root):
    from trice.llm.endpoints import load_pins

    pin = load_pins()["m1"]
    sentences = build.load_split("toy", "train") + build.load_split("toy", "test")
    transport = httpx.MockTransport(_oracle(sentences))
    client = _client(root, transport)

    for run in expand("S0"):
        runner.execute(run, client, pin, workers=4)
    frozen = analysis.select_format()
    assert frozen["format"] == "P0"  # all formats tie at F1=1; ties go to the earliest format
    assert set(frozen["mean_f1"].values()) == {1.0}

    loo_runs = list(expand("S1-loo"))
    assert {r.k for r in loo_runs} == {0, 1, 2, 4, 8, 16}  # grid capped by the 30-sentence pool
    for run in loo_runs:
        runner.execute(run, client, pin, workers=4)
    k_star = analysis.select_all_k()["m1"]["toy"]
    assert k_star["k_star"] == 2 and k_star["rule"] == "kneedle"

    s1 = list(expand("S1"))
    assert {(r.condition, r.k) for r in s1} == {
        ("zero_shot", 0),
        ("random:0", 2),
        ("small/full", 2),
    }
    for run in s1:
        runner.execute(run, client, pin, workers=4)
    table = analysis.table_s1()
    f1 = {c["condition"]: c["f1"] for c in table["cells"]}
    assert f1 == {"small/full": 1.0, "random:0": 1.0, "zero_shot": 0.0}
    (vs_zero,) = table["comparisons"]["zero_shot"]
    assert vs_zero["delta"] == 1.0 and vs_zero["p_value"] < 0.01 and vs_zero["mcnemar_p"] < 0.01

    # Replay: deleting predictions and re-running from the cache reproduces them byte for byte,
    # without any network access.
    files = list((root / "results" / "runs").rglob("*.jsonl"))
    before = _digest(files)
    for f in files:
        f.unlink()

    def no_network(request):
        raise AssertionError("replay must not hit the network")

    replay = _client(root, httpx.MockTransport(no_network), mode="replay")
    for run in [*expand("S0"), *expand("S1-loo"), *expand("S1")]:
        runner.execute(run, replay, pin, workers=4)
    assert _digest(list((root / "results" / "runs").rglob("*.jsonl"))) == before

    # Budget: whitespace tokenizer with an additive chat template -> exact self-check.
    class FakeTok:
        def encode(self, text, add_special_tokens=False):
            return text.split()

        def apply_chat_template(self, messages, **_):
            ids = []
            for m in messages:
                ids += ["<s>", m["role"], *m["content"].split(), "</s>"]
            return [*ids, "<assistant>"]

    report = budget.estimate(["S1"], lambda _: budget.TokenCounter(FakeTok()), load_pins())
    total = report["total"]
    assert total["calls"] == 3 * 12  # three conditions x 12 test queries, no duplicate prompts
    assert total["input_tokens"] > 0
    assert total["cost_usd"] == pytest.approx(
        total["input_tokens"] * 1e-7 + total["output_tokens"] * 2e-7
    )
    assert report["self_check"]["max_rel_error"] == 0.0
