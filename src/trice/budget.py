"""Dry-run token budget: build every planned prompt, count it with each model's own chat template
and tokenizer, estimate outputs from the gold answers, and price it with the pinned endpoints.

Counting is additive per message (template overhead per role sequence + cached content counts),
because demonstrations repeat across thousands of prompts; a sample is re-rendered in full to
report the error of that shortcut."""

import hashlib
import json
import random
from collections import defaultdict
from typing import Any

from jinja2 import Environment
from pydantic import BaseModel, ConfigDict, Field
from transformers import AutoTokenizer

from trice.llm import Endpoint
from trice.runner import Runner
from trice.schema import Condition, FormatName, Run


class TokenCounter(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    tokenizer: Any
    texts: dict[str, int] = Field(default_factory=dict)
    overheads: dict[tuple[str, ...], int] = Field(default_factory=dict)

    def count_text(self, text: str) -> int:
        if text not in self.texts:
            self.texts[text] = len(self.tokenizer.encode(text, add_special_tokens=False))
        return self.texts[text]

    def count_exact(self, messages: list[dict]) -> int:
        return len(
            self.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_dict=False, enable_thinking=False
            )
        )

    def count_messages(self, messages: list[dict]) -> int:
        roles = tuple(m["role"] for m in messages)
        if roles not in self.overheads:
            self.overheads[roles] = self.count_exact([{"role": r, "content": ""} for r in roles])
        return self.overheads[roles] + sum(self.count_text(m["content"]) for m in messages)


class Budget(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    runner: Runner
    env: Environment
    endpoints: dict[str, Endpoint]
    counters: dict[str, TokenCounter] = Field(default_factory=dict)

    def load_counter(self, model: str) -> TokenCounter:
        if model not in self.counters:
            spec, endpoint = self.runner.config.models[model], self.endpoints.get(model)
            revision = endpoint.hf_revision if endpoint else None
            self.counters[model] = TokenCounter(
                tokenizer=AutoTokenizer.from_pretrained(spec.hf_repo, revision=revision)
            )
        return self.counters[model]

    def derive_proxy(self, run: Run) -> Run:
        """A retrieval condition without stored rankings yet (e.g. the API-backed large stack) is
        priced with small/full, else random demonstrations: same k and format, same length."""
        if Condition.parse(run.condition).kind != "retrieval":
            return run
        for candidate in (run.condition, "small/full"):
            stack = Condition.parse(candidate).stack
            if (self.runner.config.layout.rankings / stack / run.dataset / run.queries).exists():
                return run.model_copy(update={"condition": candidate})
        return run.model_copy(update={"condition": "random:0"})

    def estimate(
        self,
        studies: list[str],
        assume_k: int | None = None,
        assume_format: FormatName | None = None,
        models: list[str] | None = None,
        datasets: list[str] | None = None,
        checks: int = 200,
    ) -> dict:
        """Tokens and cost per study. Identical requests are counted once (the replay cache serves
        repeats), and prompts are built once per (dataset, queries, condition, format, k)."""
        stats = self.runner.config.stats
        groups: dict[tuple, list[tuple[str, Run]]] = defaultdict(list)
        for study in studies:
            for run in self.runner.expand_plan(study, assume_k, assume_format, models, datasets):
                groups[(run.dataset, run.queries, run.condition, run.fmt, run.k)].append((study, run))
        seen: set[str] = set()
        totals: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        sample, rnd, calls, proxies = [], random.Random(0), 0, set()
        for members in groups.values():
            first = members[0][1]
            priced = self.derive_proxy(first)
            if priced.condition != first.condition:
                proxies.add(f"{first.condition}->{priced.condition}")
            labels = self.runner.corpus.load_labels(first.dataset)
            pairs = self.runner.render_prompts(priced)
            for study, run in members:
                counter, endpoint = (
                    self.load_counter(run.model),
                    self.endpoints.get(run.model, Endpoint(backend="local", model=run.model)),
                )
                for query, prompt in pairs:
                    key = hashlib.sha256(
                        json.dumps([run.model, prompt.model_dump()], sort_keys=True).encode()
                    ).hexdigest()
                    if key in seen:
                        continue
                    seen.add(key)
                    n_in = counter.count_messages(prompt.messages)
                    answer = self.runner.formats[run.fmt].render_answer(labels, query)
                    n_out = round(counter.count_text(answer) * stats.output_factor + stats.output_extra)
                    row = totals[study]
                    row["calls"] += 1
                    row["input_tokens"] += n_in
                    row["output_tokens"] += n_out
                    row["cost"] += n_in * endpoint.prompt_price + n_out * endpoint.completion_price
                    calls += 1
                    if len(sample) < checks:
                        sample.append((run.model, prompt.messages))
                    elif (j := rnd.randrange(calls)) < checks:
                        sample[j] = (run.model, prompt.messages)
        errors = [
            abs(self.load_counter(m).count_messages(msgs) - self.load_counter(m).count_exact(msgs))
            / self.load_counter(m).count_exact(msgs)
            for m, msgs in sample
        ]
        rows = [{"study": s, **{k: int(v) if k != "cost" else v for k, v in t.items()}} for s, t in totals.items()]
        keys = ("calls", "input_tokens", "output_tokens", "cost")
        total = {k: sum(t[k] for t in totals.values()) for k in keys}
        return {
            "assume_k": assume_k,
            "assume_format": assume_format,
            "studies": rows,
            "total": total,
            "max_error": max(errors, default=0.0),
            "checked": len(errors),
            "proxies": sorted(proxies),
        }

    def render(self, payload: dict, contingency: float = 1.5) -> str:
        return self.env.get_template("budget.md.j2").render(contingency=contingency, **payload)
