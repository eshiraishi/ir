"""Prompt formats P0-P3: each renders prompts and gold answers (text lives in templates/) and parses
model outputs back onto the query tokens.

P0  v1 Listing 1.1: one user message, numbered ``span (label)`` list.
P1  chat template: system message + demonstrations as user/assistant turns, same list output.
P2  chat template + strict JSON Schema (structured generation) with a label enum.
P3  chat template + inline tags: the text repeated with ``<label>...</label>``.
"""

import re
from typing import Literal, Protocol, runtime_checkable

from jinja2 import Environment
from pydantic import BaseModel, ConfigDict, ValidationError

from trice.schema import Example, FormatName, LabelMap, Message, Parsed, Prompt

Candidates = tuple[
    list[tuple[str, str, int]], int, bool
]  # (span, label, token position or -1), broken items, malformed


class JsonEntity(BaseModel):
    text: str
    type: str


class JsonAnswer(BaseModel):
    """The P2 output; items that are not {text, type} objects are kept to be counted as invalid."""

    entities: list[JsonEntity | object]


@runtime_checkable
class Format(Protocol):
    @property
    def name(self) -> FormatName: ...

    def render_prompt(self, labels: LabelMap, demos: list[Example], query: Example) -> Prompt: ...

    def render_answer(self, labels: LabelMap, example: Example) -> str: ...

    def parse_output(self, labels: LabelMap, output: str, query: Example) -> Parsed: ...


def render_answer(env: Environment, name: FormatName, labels: LabelMap, example: Example) -> str:
    return env.get_template("answer_only.j2").render(fmt=name, **labels.describe(example))


def render_chat(
    env: Environment, name: FormatName, labels: LabelMap, demos: list[Example], query: Example
) -> list[Message]:
    types = labels.types
    system = env.get_template("system.j2").render(fmt=name, types=types, tags=[labels.tag(t) for t in types])
    messages = [Message(role="system", content=system)]
    for demo in demos:
        messages += [
            Message(role="user", content=demo.text),
            Message(role="assistant", content=render_answer(env, name, labels, demo)),
        ]
    return [*messages, Message(role="user", content=query.text)]


def align(candidates: Candidates, labels: LabelMap, query: Example) -> Parsed:
    """Align each proposed mention to the query tokens: at its own position when the output gives
    one (P3), else at the first free occurrence. Mentions with an unknown label or no free
    occurrence count as invalid."""
    pairs, broken, malformed = candidates
    tags, invalid = ["O"] * len(query.tokens), broken
    for text, label, position in pairs:
        raw, words = labels.resolve(label), text.split()
        starts = [
            i
            for i in range(len(query.tokens) - len(words) + 1)
            if words and query.tokens[i : i + len(words)] == words and set(tags[i : i + len(words)]) == {"O"}
        ]
        if raw is None or not starts:
            invalid += 1
            continue
        first = position if position in starts else starts[0]
        tags[first : first + len(words)] = [f"B-{raw}"] + [f"I-{raw}"] * (len(words) - 1)
    return Parsed(tags=tags, candidates=len(pairs) + broken, invalid=invalid, malformed=malformed)


def read_list(output: str) -> Candidates:
    lines = [line for line in output.strip().splitlines() if line.strip()]
    if [line.strip().rstrip(".").lower() for line in lines] in ([], ["none"]):
        return [], 0, False
    items = [re.match(r"^\s*\d+\s*[.)]\s*(.+)\(([^()]*)\)\s*$", line) for line in lines]
    pairs = [(m[1].strip(), m[2], -1) for m in items if m]
    return pairs, len(items) - len(pairs), not pairs


class ListingFormat(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: Literal["P0"] = "P0"
    env: Environment

    def render_prompt(self, labels: LabelMap, demos: list[Example], query: Example) -> Prompt:
        content = self.env.get_template("p0.j2").render(
            types=labels.types, demos=[labels.describe(d) for d in demos], query=query.text
        )
        return Prompt(messages=[Message(role="user", content=content)], stop=["\nExample "])

    def render_answer(self, labels: LabelMap, example: Example) -> str:
        return render_answer(self.env, self.name, labels, example)

    def parse_output(self, labels: LabelMap, output: str, query: Example) -> Parsed:
        return align(read_list(output), labels, query)


class ListFormat(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: Literal["P1"] = "P1"
    env: Environment

    def render_prompt(self, labels: LabelMap, demos: list[Example], query: Example) -> Prompt:
        return Prompt(messages=render_chat(self.env, self.name, labels, demos, query))

    def render_answer(self, labels: LabelMap, example: Example) -> str:
        return render_answer(self.env, self.name, labels, example)

    def parse_output(self, labels: LabelMap, output: str, query: Example) -> Parsed:
        return align(read_list(output), labels, query)


class JsonFormat(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: Literal["P2"] = "P2"
    env: Environment

    def render_prompt(self, labels: LabelMap, demos: list[Example], query: Example) -> Prompt:
        entity = {
            "type": "object",
            "required": ["text", "type"],
            "additionalProperties": False,
            "properties": {"text": {"type": "string"}, "type": {"type": "string", "enum": labels.types}},
        }
        schema = {
            "type": "object",
            "required": ["entities"],
            "additionalProperties": False,
            "properties": {"entities": {"type": "array", "items": entity}},
        }
        return Prompt(
            messages=render_chat(self.env, self.name, labels, demos, query),
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "entities", "strict": True, "schema": schema},
            },
        )

    def render_answer(self, labels: LabelMap, example: Example) -> str:
        return render_answer(self.env, self.name, labels, example)

    def parse_output(self, labels: LabelMap, output: str, query: Example) -> Parsed:
        try:
            items = JsonAnswer.model_validate_json(output).entities
        except ValidationError:
            return align(([], 0, True), labels, query)
        pairs = [(i.text, i.type, -1) for i in items if isinstance(i, JsonEntity)]
        return align((pairs, len(items) - len(pairs), False), labels, query)


class InlineFormat(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: Literal["P3"] = "P3"
    env: Environment

    def render_prompt(self, labels: LabelMap, demos: list[Example], query: Example) -> Prompt:
        return Prompt(messages=render_chat(self.env, self.name, labels, demos, query))

    def render_answer(self, labels: LabelMap, example: Example) -> str:
        return render_answer(self.env, self.name, labels, example)

    def parse_output(self, labels: LabelMap, output: str, query: Example) -> Parsed:
        tag = r"</?[A-Za-z0-9_\-]+>"
        pairs = [
            (m[2], m[1], len(re.sub(tag, "", output[: m.start()]).split()))
            for m in re.finditer(r"<([A-Za-z0-9_\-]+)>(.*?)</\1>", output)
        ]
        return align((pairs, 0, False), labels, query)
