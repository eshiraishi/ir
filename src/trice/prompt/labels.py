"""Label verbalizations shared by every prompt format."""

import re
from dataclasses import dataclass
from functools import cached_property


def tag_name(verbalization: str) -> str:
    """Inline-tag name of a verbalization ("programming language" -> "programming_language")."""
    return re.sub(r"[^a-z0-9]+", "_", verbalization.lower()).strip("_")


@dataclass(frozen=True)
class LabelSet:
    """Raw dataset labels and the verbalizations shown to models (one-to-one)."""

    raw_to_verbal: dict[str, str]

    def __post_init__(self) -> None:
        verbals = list(self.raw_to_verbal.values())
        if len(set(verbals)) != len(verbals):
            raise ValueError("verbalizations must be unique")
        tags = [tag_name(v) for v in verbals]
        if len(set(tags)) != len(tags) or not all(tags):
            raise ValueError("verbalizations must map to unique, non-empty tag names")

    @cached_property
    def verbalizations(self) -> tuple[str, ...]:
        """Sorted, so every prompt lists labels in the same order."""
        return tuple(sorted(self.raw_to_verbal.values()))

    @cached_property
    def _lookup(self) -> dict[str, str]:
        table: dict[str, str] = {}
        for raw, verbal in self.raw_to_verbal.items():
            for alias in (verbal, raw, tag_name(verbal)):
                table[alias.casefold()] = verbal
        return table

    def verbal(self, raw: str) -> str:
        return self.raw_to_verbal[raw]

    def resolve(self, label: str) -> str | None:
        """Map a model-produced label (verbalization, raw label or tag, any case) to a
        verbalization, or ``None`` if it is not a known label."""
        return self._lookup.get(label.strip().casefold())
