"""Parse model outputs into (span, label) pairs.

A candidate is kept only if its label resolves to a known verbalization and its span occurs in the
query text (after whitespace normalization). Dropped candidates are counted, never silently lost:
the invalid-output rate is itself a scaling metric (S5).
"""

import json
import re
from dataclasses import dataclass

from trice.prompt.formats import NONE_ANSWER, Format
from trice.prompt.labels import LabelSet

_WS = re.compile(r"\s+")
_LIST_ITEM = re.compile(r"^\s*\d+\s*[.)]\s*(?P<span>.+)\((?P<label>[^()]*)\)\s*$")
_TAG = re.compile(r"<(?P<close>/?)(?P<name>[A-Za-z0-9_\-]+)>")


def normalize_ws(text: str) -> str:
    return _WS.sub(" ", text).strip()


@dataclass(frozen=True)
class Parsed:
    entities: tuple[tuple[str, str], ...]  # (normalized span, verbalization), in output order
    n_candidates: int  # mentions the output proposed
    n_invalid: int  # candidates dropped: unknown label, span not in text, broken item
    malformed: bool  # output not in the expected format at all (e.g. invalid JSON)
    text_mismatch: bool = False  # P3 only: de-tagged output differs from the input


def _validate(
    candidates: list[tuple[str, str]], query_text: str, labels: LabelSet
) -> tuple[tuple[tuple[str, str], ...], int]:
    haystack = normalize_ws(query_text)
    kept: list[tuple[str, str]] = []
    invalid = 0
    for span, label in candidates:
        span_n = normalize_ws(span)
        verbal = labels.resolve(label)
        if verbal is None or not span_n or span_n not in haystack:
            invalid += 1
        else:
            kept.append((span_n, verbal))
    return tuple(kept), invalid


def _parse_list(output: str) -> tuple[list[tuple[str, str]], int, bool]:
    lines = [ln for ln in output.strip().splitlines() if ln.strip()]
    if not lines or (
        len(lines) == 1 and lines[0].strip().rstrip(".").lower() == NONE_ANSWER.lower()
    ):
        return [], 0, False
    candidates, broken = [], 0
    for line in lines:
        m = _LIST_ITEM.match(line)
        if m:
            candidates.append((m["span"].strip(), m["label"]))
        else:
            broken += 1
    return candidates, broken, not candidates and broken > 0


def _parse_json(output: str) -> tuple[list[tuple[str, str]], int, bool]:
    try:
        obj = json.loads(output)
        items = obj["entities"]
        if not isinstance(items, list):
            raise TypeError
    except json.JSONDecodeError, KeyError, TypeError:
        return [], 0, True
    candidates, broken = [], 0
    for item in items:
        if (
            isinstance(item, dict)
            and isinstance(item.get("text"), str)
            and isinstance(item.get("type"), str)
        ):
            candidates.append((item["text"], item["type"]))
        else:
            broken += 1
    return candidates, broken, False


def _parse_inline(output: str, query_text: str) -> tuple[list[tuple[str, str]], int, bool]:
    """Walk the tags; return spans with their tag names and whether the de-tagged text matches."""
    candidates: list[tuple[str, str]] = []
    plain: list[str] = []
    open_tag: tuple[str, int] | None = None  # (name, start index in plain text)
    broken, pos = 0, 0
    for m in _TAG.finditer(output):
        plain.append(output[pos : m.start()])
        pos = m.end()
        name = m["name"]
        if not m["close"]:
            if open_tag is not None:  # nested or unclosed tag: drop the outer one
                broken += 1
            open_tag = (name, len("".join(plain)))
        elif open_tag is not None and open_tag[0] == name:
            candidates.append(("".join(plain)[open_tag[1] :], name))
            open_tag = None
        else:  # stray closing tag
            broken += 1
    plain.append(output[pos:])
    if open_tag is not None:
        broken += 1
    mismatch = normalize_ws("".join(plain)) != normalize_ws(query_text)
    return candidates, broken, mismatch


def parse(fmt: Format, output: str, query_text: str, labels: LabelSet) -> Parsed:
    if fmt in ("P0", "P1"):
        candidates, broken, malformed = _parse_list(output)
        mismatch = False
    elif fmt == "P2":
        candidates, broken, malformed = _parse_json(output)
        mismatch = False
    elif fmt == "P3":
        candidates, broken, mismatch = _parse_inline(output, query_text)
        malformed = False
    else:
        raise ValueError(fmt)
    kept, invalid = _validate(candidates, query_text, labels)
    return Parsed(
        entities=kept,
        n_candidates=len(candidates) + broken,
        n_invalid=invalid + broken,
        malformed=malformed,
        text_mismatch=mismatch,
    )
