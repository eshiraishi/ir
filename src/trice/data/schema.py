"""Canonical sentence/entity representation shared by every stage."""

import hashlib
import json
from collections.abc import Iterable, Iterator
from pathlib import Path

from pydantic import BaseModel, ConfigDict, model_validator


class Entity(BaseModel):
    """A gold mention. Character offsets index :attr:`Sentence.text`; token offsets are
    half-open indices into :attr:`Sentence.tokens`. ``label`` is the *raw* dataset label."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    start: int
    end: int
    token_start: int
    token_end: int
    label: str
    text: str


class Sentence(BaseModel):
    """One example. ``text`` is always ``" ".join(tokens)``, so offsets are reproducible."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    tokens: tuple[str, ...]
    text: str
    entities: tuple[Entity, ...]

    @model_validator(mode="after")
    def _check_offsets(self) -> Sentence:
        if self.text != " ".join(self.tokens):
            raise ValueError(f"{self.id}: text is not the space-joined tokens")
        for e in self.entities:
            if self.text[e.start : e.end] != e.text:
                raise ValueError(f"{self.id}: entity offsets do not match its text: {e}")
            if " ".join(self.tokens[e.token_start : e.token_end]) != e.text:
                raise ValueError(f"{self.id}: entity token span does not match its text: {e}")
        return self


def write_jsonl(path: Path, sentences: Iterable[Sentence]) -> str:
    """Write sentences as JSONL and return the SHA-256 of the written bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256()
    with path.open("wb") as f:
        for s in sentences:
            line = (json.dumps(s.model_dump(mode="json"), ensure_ascii=False) + "\n").encode()
            h.update(line)
            f.write(line)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[Sentence]:
    return list(iter_jsonl(path))


def iter_jsonl(path: Path) -> Iterator[Sentence]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield Sentence.model_validate_json(line)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
