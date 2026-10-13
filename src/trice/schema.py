"""Data shapes shared by every part of TRICE."""

import hashlib
import re
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, RootModel
from seqeval.metrics.sequence_labeling import get_entities

Split = Literal["train", "test"]
QuerySet = Literal["test", "loo"]
FormatName = Literal["P0", "P1", "P2", "P3"]
Backend = Literal["local", "openrouter"]
Mode = Literal["readwrite", "replay", "bypass"]
Span = tuple[str, int, int]  # (raw label, first token, last token + 1)


def ensure_parent(path: str) -> str:
    """The same path, once its directory exists (for writers that do not create it)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return path


def write_text(path: str, text: str) -> None:
    Path(ensure_parent(path)).write_text(text)


def derive_seed(*parts: object) -> int:
    """A 32-bit seed that depends only on what it is for, never on call order."""
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


class LocalRef(BaseModel):
    """A Hugging Face model run here, pinned to a commit."""

    backend: Literal["local"]
    id: str
    revision: str


class HostedRef(BaseModel):
    """A model served by OpenRouter."""

    backend: Literal["openrouter"]
    id: str


ModelRef = Annotated[LocalRef | HostedRef, Field(discriminator="backend")]


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


class Entity(BaseModel):
    text: str
    type: str


class Segment(BaseModel):
    text: str
    tag: str  # "" for a plain token


class Description(BaseModel):
    """What the templates need to show one example and its gold answer."""

    text: str
    mentions: list[Entity]
    segments: list[Segment]


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

    def describe(self, example: Example) -> Description:
        mentions = []
        segments = []
        pos = 0
        for raw, start, end in example.spans:
            text = " ".join(example.tokens[start:end])
            verbal = self.root[raw]
            mentions.append(Entity(text=text, type=verbal))
            for token in example.tokens[pos:start]:
                segments.append(Segment(text=token, tag=""))
            segments.append(Segment(text=text, tag=self.tag(verbal)))
            pos = end
        for token in example.tokens[pos:]:
            segments.append(Segment(text=token, tag=""))
        return Description(text=example.text, mentions=mentions, segments=segments)


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


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class Ranking(BaseModel):
    """Pool ids for one query at one retrieval stage, best first."""

    query_id: str
    stage: str
    ids: list[str]


class Prompt(BaseModel):
    messages: list[Message]
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


class Prediction(BaseModel):
    query_id: str
    output: str
    completion: Completion
    parsed: Parsed
