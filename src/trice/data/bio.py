"""Reading column-format NER files and converting BIO tags to entity spans."""

from collections.abc import Iterator
from typing import Literal

from trice.data.schema import Entity, Sentence

ColumnFormat = Literal["token_tag", "tag_token"]


def read_columns(content: str, fmt: ColumnFormat) -> Iterator[tuple[list[str], list[str]]]:
    """Yield ``(tokens, tags)`` per blank-line-separated sentence.

    ``token_tag`` lines are ``token<TAB>tag`` (CrossNER, WNUT-17, NCBI-disease); ``tag_token`` lines
    are ``tag<TAB>token`` (MIT SLS). ``-DOCSTART-`` lines are skipped.
    """
    tokens: list[str] = []
    tags: list[str] = []
    for lineno, raw in enumerate(content.splitlines(), start=1):
        line = raw.rstrip("\r\n")
        if not line.strip():
            if tokens:
                yield tokens, tags
                tokens, tags = [], []
            continue
        parts = line.split("\t")
        if len(parts) != 2:
            parts = line.split()
        if len(parts) != 2:
            raise ValueError(f"line {lineno}: expected 2 columns, got {line!r}")
        token, tag = parts if fmt == "token_tag" else parts[::-1]
        if token == "-DOCSTART-":
            continue
        tokens.append(token)
        tags.append(tag)
    if tokens:
        yield tokens, tags


def bio_to_token_spans(tags: list[str]) -> list[tuple[int, int, str]]:
    """Convert BIO tags to half-open token spans ``(start, end, label)``.

    IOB1-tolerant: an ``I-X`` that does not continue an ``X`` span starts a new one.
    """
    spans: list[tuple[int, int, str]] = []
    current: tuple[int, str] | None = None  # (start, label) of the open span
    for i, tag in enumerate([*tags, "O"]):
        if tag != "O" and not tag.startswith(("B-", "I-")):
            raise ValueError(f"unsupported tag {tag!r}")
        if tag == "O" or tag.startswith("B-") or current is None or tag[2:] != current[1]:
            if current is not None:
                spans.append((current[0], i, current[1]))
                current = None
            if tag != "O":
                current = (i, tag[2:])
    return spans


def to_sentence(sentence_id: str, tokens: list[str], tags: list[str]) -> Sentence:
    """Build a canonical :class:`Sentence` with character and token offsets."""
    if any(not t or any(c.isspace() for c in t) for t in tokens):
        raise ValueError(f"{sentence_id}: empty token or token containing whitespace")
    offsets, pos = [], 0
    for tok in tokens:
        offsets.append(pos)
        pos += len(tok) + 1
    text = " ".join(tokens)
    entities = []
    for ts, te, label in bio_to_token_spans(tags):
        start = offsets[ts]
        end = offsets[te - 1] + len(tokens[te - 1])
        entities.append(
            Entity(
                start=start,
                end=end,
                token_start=ts,
                token_end=te,
                label=label,
                text=text[start:end],
            )
        )
    return Sentence(id=sentence_id, tokens=tuple(tokens), text=text, entities=tuple(entities))
