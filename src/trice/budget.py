"""Dry-run token budget: build every planned prompt, count it with each model's own chat template
and tokenizer, estimate outputs from the gold answers, and price it with the pinned endpoints.

Counting is additive per message (template overhead per role sequence + cached content counts),
because demonstrations repeat across thousands of prompts; a sample is re-rendered in full to
report the error of that shortcut."""

import hashlib
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from jinja2 import Environment
from pydantic import BaseModel, ConfigDict, Field
from transformers import AutoTokenizer

from trice.llm import Endpoint
from trice.runner import Runner
from trice.schema import Condition, FormatName, Message, Run


class Cost(BaseModel):
    study: str
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0  # USD at the pinned endpoints' prices


class BudgetReport(BaseModel):
    assume_k: int | None
    assume_format: FormatName | None
    studies: list[Cost]
    total: Cost
    max_error: float  # of the additive counting shortcut, over `checked` fully rendered prompts
    checked: int
    proxies: list[str]  # conditions priced with a stand-in because their rankings do not exist yet


class TokenCounter(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    tokenizer: Any
    texts: dict[str, int] = Field(default_factory=dict)
    overheads: dict[tuple[str, ...], int] = Field(default_factory=dict)

    def count_text(self, text: str) -> int:
        if text not in self.texts:
            self.texts[text] = len(self.tokenizer.encode(text, add_special_tokens=False))
        return self.texts[text]

    def count_exact(self, messages: list[Message]) -> int:
        return len(
            self.tokenizer.apply_chat_template(
                [m.model_dump() for m in messages],
                tokenize=True,
                add_generation_prompt=True,
                return_dict=False,
                enable_thinking=False,
            )
        )

    def count_messages(self, messages: list[Message]) -> int:
        roles = tuple(m.role for m in messages)
        if roles not in self.overheads:
            self.overheads[roles] = self.count_exact([Message(role=r, content="") for r in roles])
        return self.overheads[roles] + sum(self.count_text(m.content) for m in messages)


class Budget(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    runner: Runner
    env: Environment
    endpoints: list[Endpoint]
    counters: dict[str, TokenCounter] = Field(default_factory=dict)

    def find_endpoint(self, model: str) -> Endpoint:
        for endpoint in self.endpoints:
            if endpoint.name == model:
                return endpoint
        raise KeyError(f"{model} has no pinned endpoint (run `trice pin`)")

    def load_counter(self, model: str) -> TokenCounter:
        if model not in self.counters:
            tokenizer = AutoTokenizer.from_pretrained(
                self.runner.config.models[model].hf_repo, revision=self.find_endpoint(model).hf_revision
            )
            self.counters[model] = TokenCounter(tokenizer=tokenizer)
        return self.counters[model]

    def estimate(
        self,
        studies: list[str],
        assume_k: int | None = None,
        assume_format: FormatName | None = None,
        models: list[str] | None = None,
        datasets: list[str] | None = None,
        checks: int = 200,
    ) -> BudgetReport:
        """Tokens and cost per study. Identical requests are counted once (the replay cache serves
        repeats), and prompts are built once per (dataset, queries, condition, format, k)."""
        stats = self.runner.config.stats
        rankings = self.runner.config.layout.rankings
        groups: dict[tuple, list[tuple[str, Run]]] = defaultdict(list)
        for study in studies:
            for run in self.runner.expand_plan(study, assume_k, assume_format, models, datasets):
                groups[(run.dataset, run.queries, run.condition, run.fmt, run.k)].append((study, run))
        seen: set[str] = set()
        totals = {study: Cost(study=study) for study in studies}
        sample: list[tuple[str, list[Message]]] = []
        sampler = random.Random(0)
        calls = 0
        proxies: set[str] = set()
        for members in groups.values():
            first = members[0][1]
            # A retrieval condition without stored rankings yet (e.g. the API-backed large stack) is
            # priced with small/full, else random demonstrations: same k and format, same length.
            stored = first.condition
            if Condition.parse(first.condition).kind == "retrieval":
                own = rankings.format(
                    stack=Condition.parse(first.condition).stack, dataset=first.dataset, queries=first.queries
                )
                small = rankings.format(stack="small", dataset=first.dataset, queries=first.queries)
                if Path(own).exists():
                    stored = first.condition
                elif Path(small).exists():
                    stored = "small/full"
                else:
                    stored = "random:0"
            if stored != first.condition:
                proxies.add(f"{first.condition}->{stored}")
            labels = self.runner.corpus.load_labels(first.dataset)
            pairs = self.runner.render_prompts(first.model_copy(update={"condition": stored}))
            for study, run in members:
                counter = self.load_counter(run.model)
                endpoint = self.find_endpoint(run.model)
                for query, prompt in pairs:
                    key = hashlib.sha256(f"{run.model}|{prompt.model_dump_json()}".encode()).hexdigest()
                    if key in seen:
                        continue
                    seen.add(key)
                    tokens_in = counter.count_messages(prompt.messages)
                    answer = self.runner.formats[run.fmt].render_answer(labels, query)
                    tokens_out = round(counter.count_text(answer) * stats.output_factor + stats.output_extra)
                    totals[study].calls += 1
                    totals[study].input_tokens += tokens_in
                    totals[study].output_tokens += tokens_out
                    totals[study].cost += tokens_in * endpoint.prompt_price + tokens_out * endpoint.completion_price
                    calls += 1
                    if len(sample) < checks:
                        sample.append((run.model, prompt.messages))
                    else:
                        slot = sampler.randrange(calls)
                        if slot < checks:
                            sample[slot] = (run.model, prompt.messages)
        errors = []
        for model, messages in sample:
            exact = self.load_counter(model).count_exact(messages)
            errors.append(abs(self.load_counter(model).count_messages(messages) - exact) / exact)
        total = Cost(study="total")
        for cost in totals.values():
            total.calls += cost.calls
            total.input_tokens += cost.input_tokens
            total.output_tokens += cost.output_tokens
            total.cost += cost.cost
        return BudgetReport(
            assume_k=assume_k,
            assume_format=assume_format,
            studies=list(totals.values()),
            total=total,
            max_error=max(errors) if len(errors) > 0 else 0.0,
            checked=len(errors),
            proxies=sorted(proxies),
        )

    def render(self, report: BudgetReport, contingency: float = 1.5) -> str:
        return self.env.get_template("budget.md.j2").render(report=report, contingency=contingency)
