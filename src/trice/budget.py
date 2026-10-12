"""Dry-run token budget (M5): build every planned prompt, count it with each model's own chat
template and tokenizer, estimate outputs from the gold answers, and price it. No model calls.

Counting is additive per message (chat-template overhead per role sequence + tokens of each
content string, cached) because demonstrations repeat across thousands of prompts. A self-check
re-renders a sample of prompts in full and reports the relative error of the additive count.
"""

import hashlib
import json
import random
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field, replace

from trice.config import models_config, retrieval_config
from trice.data.build import label_map, load_split
from trice.llm.cache import canonical_json
from trice.llm.client import EndpointPin
from trice.prompt.formats import Format, render_answer
from trice.prompt.labels import LabelSet
from trice.retrieval.demos import Condition
from trice.retrieval.pipeline import rankings_dir
from trice.runner import prompts_for
from trice.studies import Run, expand, results_dir

OUTPUT_FACTOR, OUTPUT_EXTRA = 1.25, 8


class TokenCounter:
    """Token counts for one model. ``tokenizer`` is a HF tokenizer (or any object with
    ``encode(text, add_special_tokens=False)`` and ``apply_chat_template``)."""

    def __init__(self, tokenizer) -> None:
        self.tok = tokenizer
        self._text: dict[str, int] = {}
        self._overhead: dict[tuple[str, ...], int] = {}

    @classmethod
    def for_model(cls, name: str) -> TokenCounter:
        """The model's own tokenizer at the revision pinned by `trice endpoints` (if any)."""
        from transformers import AutoTokenizer

        from trice.llm.endpoints import lock_path

        spec = models_config()[name]
        revision = spec.hf_revision
        if revision is None and lock_path().exists():
            revision = json.loads(lock_path().read_text()).get(name, {}).get("hf_revision")
        return cls(AutoTokenizer.from_pretrained(spec.hf_repo, revision=revision))

    def text(self, s: str) -> int:
        n = self._text.get(s)
        if n is None:
            n = self._text[s] = len(self.tok.encode(s, add_special_tokens=False))
        return n

    def _render(self, messages: list[dict]) -> list[int]:
        ids = self.tok.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=False,
            enable_thinking=False,
        )
        return list(ids)

    def messages(self, messages: list[dict]) -> int:
        roles = tuple(m["role"] for m in messages)
        if roles not in self._overhead:
            self._overhead[roles] = len(self._render([{"role": r, "content": ""} for r in roles]))
        return self._overhead[roles] + sum(self.text(m["content"]) for m in messages)

    def exact(self, messages: list[dict]) -> int:
        return len(self._render(messages))


@dataclass
class Tally:
    runs: int = 0
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    proxies: set[str] = field(default_factory=set)


def _has_rankings(cond: Condition, run: Run) -> bool:
    if cond.stack is None:
        return False
    return (rankings_dir() / cond.stack / run.dataset / f"{run.queries}.jsonl").exists()


def proxied(run: Run) -> tuple[Run, str | None]:
    """Use a stand-in condition when a retrieval stack has no rankings yet (e.g. the API-backed
    large stack): same k and format, so prompt lengths are representative."""
    cond = Condition.parse(run.condition)
    if cond.kind != "retrieval" or _has_rankings(cond, run):
        return run, None
    if _has_rankings(Condition.parse("small/full"), run):
        return replace(run, condition="small/full"), f"{run.condition}->small/full"
    return replace(run, condition="random:0"), f"{run.condition}->random:0"


def estimate(
    studies: Iterable[str],
    counter_for: Callable[[str], TokenCounter],
    pins: dict[str, EndpointPin] | None = None,
    *,
    assume_k: int | None = None,
    assume_format: Format | None = None,
    models: list[str] | None = None,
    datasets: list[str] | None = None,
    self_check: int = 200,
) -> dict:
    """Tokens and cost per (study, model). Identical requests are counted once (the replay
    cache would serve repeats for free), in study order."""
    seen: set[bytes] = set()
    tallies: dict[tuple[str, str], Tally] = defaultdict(Tally)
    sample: list[tuple[str, list[dict]]] = []  # reservoir of fully re-rendered prompts
    rnd, seen_calls = random.Random(0), 0
    counters: dict[str, TokenCounter] = {}

    # Prompts depend on (dataset, query set, condition, format, k) but not on the model, so each
    # group is built once and counted with every model's tokenizer. Groups keep first-seen
    # (study) order, so a request shared by two studies is attributed to the first.
    groups: dict[tuple, list[tuple[str, Run]]] = defaultdict(list)
    for study in studies:
        for run in expand(
            study, assume_k=assume_k, assume_format=assume_format, models=models, datasets=datasets
        ):
            groups[(run.dataset, run.queries, run.condition, run.fmt, run.k)].append((study, run))

    for members in groups.values():
        first = members[0][1]
        eff, proxy = proxied(first)
        labels = LabelSet(label_map(first.dataset))
        pairs = prompts_for(eff)
        for study, run in members:
            counter = counters.setdefault(run.model, counter_for(run.model))
            pin = (pins or {}).get(run.model)
            t = tallies[(study, run.model)]
            t.runs += 1
            if proxy:
                t.proxies.add(proxy)
            for q, prompt in pairs:
                key = hashlib.sha256(
                    canonical_json(
                        [run.model, prompt.messages, prompt.response_format, prompt.stop]
                    ).encode()
                ).digest()
                if key in seen:
                    continue
                seen.add(key)
                n_in = counter.messages(prompt.messages)
                n_out = round(
                    counter.text(render_answer(run.fmt, q, labels)) * OUTPUT_FACTOR + OUTPUT_EXTRA
                )
                t.calls += 1
                t.input_tokens += n_in
                t.output_tokens += n_out
                if pin:
                    t.cost_usd += n_in * pin.prompt_price + n_out * pin.completion_price
                seen_calls += 1
                if len(sample) < self_check:
                    sample.append((run.model, prompt.messages))
                elif self_check and (j := rnd.randrange(seen_calls)) < self_check:
                    sample[j] = (run.model, prompt.messages)

    errors = [
        abs(counters[m].messages(msgs) - counters[m].exact(msgs)) / max(1, counters[m].exact(msgs))
        for m, msgs in sample
    ]
    rows = [
        {
            "study": s,
            "model": m,
            **{k: v for k, v in asdict(t).items() if k != "proxies"},
            "proxies": sorted(t.proxies),
        }
        for (s, m), t in sorted(tallies.items())
    ]
    ts = list(tallies.values())
    return {
        "assume_k": assume_k,
        "assume_format": assume_format,
        "rows": rows,
        "total": {
            "calls": sum(t.calls for t in ts),
            "input_tokens": sum(t.input_tokens for t in ts),
            "output_tokens": sum(t.output_tokens for t in ts),
            "cost_usd": sum(t.cost_usd for t in ts),
        },
        "self_check": {
            "prompts": len(errors),
            "max_rel_error": max(errors, default=0.0),
            "mean_rel_error": sum(errors) / len(errors) if errors else 0.0,
        },
    }


def retrieval_api_estimate(
    counter: TokenCounter, datasets: Iterable[str], embed_price: float, rerank_price: float
) -> dict:
    """Large-stack API tokens (S2 only): embeddings for every pool and test sentence, and one
    reranker pass per (test query, candidate) over min(|pool|, rerank_depth) candidates."""
    cfg = retrieval_config()
    instr = counter.text(cfg.reranker_instruction) + 40  # instruction + reranker template
    embed = rerank = 0
    for ds in datasets:
        pool, test = load_split(ds, "train"), load_split(ds, "test")
        pool_tok = [counter.text(s.text) for s in pool]
        embed += sum(pool_tok) + sum(counter.text(s.text) for s in test)
        depth = min(len(pool), cfg.rerank_depth)
        mean_doc = sum(pool_tok) / len(pool_tok)
        rerank += sum(depth * (counter.text(q.text) + mean_doc + instr) for q in test)
    return {
        "embedding_tokens": int(embed),
        "rerank_tokens": int(rerank),
        "cost_usd": embed * embed_price + rerank * rerank_price,
    }


def write_report(name: str, payload: dict) -> str:
    path = results_dir() / "budget" / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return str(path)


def markdown(payload: dict, contingency: float = 1.5) -> str:
    by_study: dict[str, list[int | float]] = defaultdict(lambda: [0, 0, 0.0])
    for r in payload["rows"]:
        acc = by_study[r["study"]]
        acc[0] += r["input_tokens"]
        acc[1] += r["output_tokens"]
        acc[2] += r["cost_usd"]
    lines = [
        f"Budget (k={payload['assume_k'] or 'k*'}, format={payload['assume_format'] or 'frozen'})",
        "",
        "| Study | Input tokens | Output tokens | Cost (USD) |",
        "|---|---:|---:|---:|",
    ]
    for s, (i, o, c) in by_study.items():
        lines.append(f"| {s} | {i:,} | {o:,} | {c:,.2f} |")
    t = payload["total"]
    lines += [
        f"| **Total** | **{t['input_tokens']:,}** | **{t['output_tokens']:,}** "
        f"| **{t['cost_usd']:,.2f}** |",
        f"| Ask (x{contingency}) | {int(t['input_tokens'] * contingency):,} | "
        f"{int(t['output_tokens'] * contingency):,} | {t['cost_usd'] * contingency:,.2f} |",
        "",
        "Additive-count self-check: max relative error "
        f"{payload['self_check']['max_rel_error']:.4%} "
        f"over {payload['self_check']['prompts']} fully rendered prompts.",
    ]
    proxies = sorted({p for r in payload["rows"] for p in r["proxies"]})
    if proxies:
        lines.append(f"Proxied conditions (no rankings yet): {', '.join(proxies)}.")
    return "\n".join(lines)
