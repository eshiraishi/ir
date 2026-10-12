"""The four candidate prompt formats (S0 selects one on training data; S3 reports all on test).

P0  v1 Listing 1.1: one user message, PromptNER-style, numbered ``span (label)`` list.
P1  chat template: system message + demos as user/assistant turns, same list output.
P2  chat template + structured generation: strict JSON Schema with a label enum.
P3  chat template + inline tags: the model rewrites the text with ``<label>...</label>``.
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from trice.data.schema import Sentence
from trice.prompt.labels import LabelSet, tag_name

Format = Literal["P0", "P1", "P2", "P3"]
FORMATS: tuple[Format, ...] = ("P0", "P1", "P2", "P3")
NONE_ANSWER = "None"


@dataclass(frozen=True)
class Prompt:
    messages: list[dict[str, str]]
    response_format: dict | None = None
    stop: list[str] | None = None


def _gold(sentence: Sentence, labels: LabelSet) -> list[tuple[str, str]]:
    return [
        (e.text, labels.verbal(e.label)) for e in sorted(sentence.entities, key=lambda e: e.start)
    ]


def render_answer(fmt: Format, sentence: Sentence, labels: LabelSet) -> str:
    """Gold answer for ``sentence`` in format ``fmt`` (demonstrations, output-size estimates)."""
    gold = _gold(sentence, labels)
    if fmt in ("P0", "P1"):
        if not gold:
            return NONE_ANSWER
        return "\n".join(f"{i}. {span} ({label})" for i, (span, label) in enumerate(gold, start=1))
    if fmt == "P2":
        return json.dumps(
            {"entities": [{"text": s, "type": t} for s, t in gold]}, ensure_ascii=False
        )
    if fmt == "P3":
        out, pos = [], 0
        for e in sorted(sentence.entities, key=lambda e: e.start):
            tag = tag_name(labels.verbal(e.label))
            out += [sentence.text[pos : e.start], f"<{tag}>", e.text, f"</{tag}>"]
            pos = e.end
        out.append(sentence.text[pos:])
        return "".join(out)
    raise ValueError(fmt)


def _definition(labels: LabelSet) -> str:
    names = list(labels.verbalizations)
    listed = names[0] if len(names) == 1 else ", ".join(names[:-1]) + ", or " + names[-1]
    return f"Defn: An entity is a {listed}."


def _system(fmt: Format, labels: LabelSet) -> str:
    types = ", ".join(labels.verbalizations)
    head = (
        "You are an expert annotator for named entity recognition.\n"
        f"Entity types: {types}.\n"
        "Find every mention of these entity types in the user's text and copy each mention "
        "exactly as it appears in the text.\n"
    )
    if fmt == "P1":
        return head + (
            "Answer with a numbered list in order of appearance, one mention per line, written as "
            f"`N. mention (entity type)`. If there are no mentions, answer `{NONE_ANSWER}`."
        )
    if fmt == "P2":
        return head + (
            'Answer with a JSON object {"entities": [{"text": mention, "type": entity type}, ...]} '
            "listing mentions in order of appearance; use an empty list if there are none."
        )
    if fmt == "P3":
        tags = ", ".join(f"<{tag_name(v)}>" for v in labels.verbalizations)
        return head + (
            "Answer by repeating the text exactly, wrapping each mention in the tag of its entity "
            f"type, like <tag>mention</tag>. Tags: {tags}. Change nothing else."
        )
    raise ValueError(fmt)


def json_schema(labels: LabelSet) -> dict:
    """Strict schema for P2: entity types restricted to the label set."""
    return {
        "type": "object",
        "properties": {
            "entities": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "type": {"type": "string", "enum": list(labels.verbalizations)},
                    },
                    "required": ["text", "type"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["entities"],
        "additionalProperties": False,
    }


def build_prompt(
    fmt: Format, labels: LabelSet, demos: Sequence[Sentence], query_text: str
) -> Prompt:
    """Assemble the request for one query. ``demos`` are used in the order given (callers pass
    them already shuffled with the per-query seed)."""
    if fmt == "P0":
        lines = [_definition(labels)]
        for i, d in enumerate(demos, start=1):
            lines += [f"Example {i}: {d.text}", "Answer:", render_answer("P0", d, labels)]
        lines += [f"Example {len(demos) + 1}: {query_text}", "Answer:"]
        return Prompt(messages=[{"role": "user", "content": "\n".join(lines)}], stop=["\nExample "])

    messages = [{"role": "system", "content": _system(fmt, labels)}]
    for d in demos:
        messages.append({"role": "user", "content": d.text})
        messages.append({"role": "assistant", "content": render_answer(fmt, d, labels)})
    messages.append({"role": "user", "content": query_text})
    response_format = None
    if fmt == "P2":
        response_format = {
            "type": "json_schema",
            "json_schema": {"name": "entities", "strict": True, "schema": json_schema(labels)},
        }
    return Prompt(messages=messages, response_format=response_format)
