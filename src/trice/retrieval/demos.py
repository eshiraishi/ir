"""Turn a condition into the demonstrations for one query.

Conditions: ``zero_shot``; ``random:<seed>`` (a seeded per-query sample of the pool);
``<stack>/<stage>`` (top-k of a stored ranking). Whatever the source, the chosen set is then
shuffled with the per-query seed ``("order", dataset, query_id)``: demonstration order never
carries rank or dataset order (position bias is spread evenly instead of built in).
"""

from collections.abc import Sequence
from dataclasses import dataclass

from trice.data.schema import Sentence
from trice.retrieval.pipeline import QuerySet, load_rankings
from trice.seeds import sample, shuffled


@dataclass(frozen=True)
class Condition:
    kind: str  # "zero_shot" | "random" | "retrieval"
    seed: int | None = None
    stack: str | None = None
    stage: str | None = None

    @classmethod
    def parse(cls, text: str) -> Condition:
        if text == "zero_shot":
            return cls("zero_shot")
        if text.startswith("random:"):
            return cls("random", seed=int(text.split(":", 1)[1]))
        stack, _, stage = text.partition("/")
        if not stage:
            raise ValueError(f"bad condition {text!r}")
        return cls("retrieval", stack=stack, stage=stage)

    def __str__(self) -> str:
        if self.kind == "random":
            return f"random:{self.seed}"
        if self.kind == "retrieval":
            return f"{self.stack}/{self.stage}"
        return self.kind


def select_demos(
    condition: Condition,
    k: int,
    dataset: str,
    queries: QuerySet,
    query: Sentence,
    pool: Sequence[Sentence],
    pool_by_id: dict[str, Sentence],
) -> list[Sentence]:
    if condition.kind == "zero_shot" or k == 0:
        return []
    if condition.kind == "random":
        candidates = [p for p in pool if p.id != query.id and p.text != query.text]
        chosen = sample(
            candidates, k, "random", condition.seed, dataset, query.id, key=lambda s: s.id
        )
    else:
        if condition.stack is None or condition.stage is None:
            raise ValueError(f"{condition}: retrieval condition needs a stack and a stage")
        ranked = load_rankings(condition.stack, dataset, queries)[query.id][condition.stage]
        if len(ranked) < k:
            raise ValueError(f"{condition}: only {len(ranked)} stored candidates for k={k}")
        chosen = [pool_by_id[pid] for pid, _ in ranked[:k]]
    return shuffled(chosen, "order", dataset, query.id, key=lambda s: s.id)
