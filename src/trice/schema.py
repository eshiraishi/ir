"""Data shapes shared by every part of TRICE."""

import hashlib
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, RootModel
from seqeval.metrics.sequence_labeling import get_entities

Split = Literal["train", "test"]
QuerySet = Literal["test", "loo"]
FormatName = Literal["P0", "P1", "P2", "P3"]
Backend = Literal["local", "openrouter"]
Mode = Literal["readwrite", "replay", "bypass"]
Span = tuple[str, int, int]  # (raw label, first token, last token + 1)


def derive_seed(*parts: object) -> int:
    """A 32-bit seed that depends only on what it is for, never on call order."""
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


class ModelRef(BaseModel):
    id: str
    revision: str | None = None
    backend: Backend = "local"


class Runtime(BaseModel):
    """Where and how local retrieval models run."""

    device: str
    threads: int
    batch_size: int


class Example(BaseModel):
    """One sentence with IOB2 tags over raw dataset labels."""

    model_config = ConfigDict(frozen=True)

    id: str
    tokens: list[str]
    tags: list[str]

    @property
    def text(self) -> str:
        return " ".join(self.tokens)

    @property
    def spans(self) -> list[Span]:
        return [(label, start, end + 1) for label, start, end in get_entities(self.tags)]


class LabelMap(RootModel[dict[str, str]]):
    """Raw dataset labels and the verbalizations shown to models (one-to-one)."""

    model_config = ConfigDict(frozen=True)

    @property
    def types(self) -> list[str]:
        return sorted(self.root.values())

    @staticmethod
    def tag(verbal: str) -> str:
        return re.sub(r"[^a-z0-9]+", "_", verbal.lower()).strip("_")

    def resolve(self, label: str) -> str | None:
        """Raw label for a model-produced verbalization, raw label or tag (any case)."""
        key = label.strip().casefold()
        for raw, verbal in self.root.items():
            if key in (raw.casefold(), verbal.casefold(), self.tag(verbal)):
                return raw
        return None

    def describe(self, example: Example) -> dict:
        """What the templates need to show one example and its gold answer."""
        mentions, segments, pos = [], [], 0
        for raw, start, end in example.spans:
            text, verbal = " ".join(example.tokens[start:end]), self.root[raw]
            mentions.append({"text": text, "type": verbal})
            segments += [{"text": t, "tag": ""} for t in example.tokens[pos:start]]
            segments.append({"text": text, "tag": self.tag(verbal)})
            pos = end
        segments += [{"text": t, "tag": ""} for t in example.tokens[pos:]]
        return {"text": example.text, "mentions": mentions, "segments": segments}


class Condition(BaseModel):
    """``zero_shot``, ``random:<seed>`` or ``<stack>/<stage>``."""

    kind: Literal["zero_shot", "random", "retrieval"]
    seed: int = 0
    stack: str = ""
    stage: str = ""

    @classmethod
    def parse(cls, text: str) -> Condition:
        if text == "zero_shot":
            return cls(kind="zero_shot")
        if text.startswith("random:"):
            return cls(kind="random", seed=int(text.removeprefix("random:")))
        stack, _, stage = text.partition("/")
        if not stage:
            raise ValueError(f"bad condition {text!r}")
        return cls(kind="retrieval", stack=stack, stage=stage)


class Run(BaseModel):
    """One (model, dataset, query set, condition, format, k) cell: one request per query."""

    model_config = ConfigDict(frozen=True)

    model: str
    dataset: str
    queries: QuerySet
    condition: str
    fmt: FormatName
    k: int

    @property
    def slug(self) -> str:
        cond = self.condition.replace("/", "-").replace(":", "")
        return f"{self.model}/{self.dataset}/{self.queries}/{cond}/{self.fmt}/k{self.k}"


class Prompt(BaseModel):
    messages: list[dict[str, str]]
    response_format: dict | None = None
    stop: list[str] | None = None


class Completion(BaseModel):
    text: str
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    reasoning_tokens: int | None = None
    provider: str | None = None


class Parsed(BaseModel):
    """A model output aligned to the query tokens."""

    tags: list[str]
    candidates: int
    invalid: int
    malformed: bool


class Prediction(Parsed):
    query_id: str
    output: str
    completion: Completion
