"""Execute runs: demonstrations -> prompt -> (cached) model call -> parse -> predictions file."""

import json
from dataclasses import asdict
from functools import cache
from pathlib import Path

from trice.config import studies_config
from trice.data.build import label_map, load_split
from trice.data.schema import Sentence
from trice.llm.client import ChatClient, EndpointPin, build_body
from trice.prompt.formats import Prompt, build_prompt
from trice.prompt.labels import LabelSet
from trice.prompt.parse import parse
from trice.retrieval.demos import Condition, select_demos
from trice.retrieval.pipeline import query_set
from trice.studies import Run, results_dir


def predictions_path(run: Run) -> Path:
    return results_dir() / "runs" / f"{run.slug}.jsonl"


@cache
def _pool(dataset: str) -> tuple[tuple[Sentence, ...], dict[str, Sentence]]:
    pool = tuple(load_split(dataset, "train"))
    return pool, {s.id: s for s in pool}


def queries_for(run: Run) -> list[Sentence]:
    return query_set(run.dataset, run.queries, studies_config().loo_queries)


def prompts_for(run: Run) -> list[tuple[Sentence, Prompt]]:
    """Every (query, prompt) of a run. Shared by the runner and the budget."""
    pool, by_id = _pool(run.dataset)
    labels = LabelSet(label_map(run.dataset))
    cond = Condition.parse(run.condition)
    out = []
    for q in queries_for(run):
        demos = select_demos(cond, run.k, run.dataset, run.queries, q, pool, by_id)
        out.append((q, build_prompt(run.fmt, labels, demos, q.text)))
    return out


def execute(run: Run, client: ChatClient, pin: EndpointPin, workers: int = 16) -> Path:
    """Run every query of ``run`` and write one prediction line per query."""
    pairs = prompts_for(run)
    labels = LabelSet(label_map(run.dataset))
    completions = client.complete_many([build_body(pin, p) for _, p in pairs], workers=workers)
    path = predictions_path(run)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for (q, _), c in zip(pairs, completions, strict=True):
            parsed = parse(run.fmt, c.text, q.text, labels)
            row = {
                "query_id": q.id,
                "output": c.text,
                "entities": [list(e) for e in parsed.entities],
                "n_candidates": parsed.n_candidates,
                "n_invalid": parsed.n_invalid,
                "malformed": parsed.malformed,
                "text_mismatch": parsed.text_mismatch,
                "finish_reason": c.finish_reason,
                "prompt_tokens": c.prompt_tokens,
                "completion_tokens": c.completion_tokens,
                "reasoning_tokens": c.reasoning_tokens,
                "provider": c.provider,
                "response_id": c.id,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    meta = {"run": asdict(run), "pin": asdict(pin), "queries": len(pairs)}
    path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return path


def load_predictions(run: Run) -> list[dict]:
    with predictions_path(run).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]
