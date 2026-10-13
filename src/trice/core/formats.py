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


class Candidate(BaseModel):
    text: str
    label: str
    position: int  # first token in the query when the output says so (P3), else -1


class Candidates(BaseModel):
    found: list[Candidate]
    broken: int  # items that could not be read at all
    malformed: bool  # nothing could be read


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
    description = labels.describe(example)
    return env.get_template("answer_only.j2").render(
        fmt=name, mentions=description.mentions, segments=description.segments
    )


def render_chat(
    env: Environment, name: FormatName, labels: LabelMap, demos: list[Example], query: Example
) -> list[Message]:
    types = labels.types
    system = env.get_template("system.j2").render(fmt=name, types=types, tags=[labels.tag(t) for t in types])
    messages = [Message(role="system", content=system)]
    for demo in demos:
        messages.append(Message(role="user", content=demo.text))
        messages.append(Message(role="assistant", content=render_answer(env, name, labels, demo)))
    messages.append(Message(role="user", content=query.text))
    return messages


def align(candidates: Candidates, labels: LabelMap, query: Example) -> Parsed:
    """Align each proposed mention to the query tokens: at its own position when the output gives
    one (P3), else at the first free occurrence. Mentions with an unknown label or no free
    occurrence count as invalid."""
    tags = ["O"] * len(query.tokens)
    invalid = candidates.broken
    for candidate in candidates.found:
        raw = labels.resolve(candidate.label)
        words = candidate.text.split()
        starts = []
        for i in range(len(query.tokens) - len(words) + 1):
            free = all(tag == "O" for tag in tags[i : i + len(words)])
            if len(words) > 0 and query.tokens[i : i + len(words)] == words and free:
                starts.append(i)
        if raw is None or len(starts) == 0:
            invalid += 1
            continue
        first = candidate.position if candidate.position in starts else starts[0]
        tags[first : first + len(words)] = [f"B-{raw}"] + [f"I-{raw}"] * (len(words) - 1)
    return Parsed(
        tags=tags, candidates=len(candidates.found) + candidates.broken, invalid=invalid, malformed=candidates.malformed
    )


def read_list(output: str) -> Candidates:
    lines = [line for line in output.strip().splitlines() if line.strip() != ""]
    stripped = [line.strip().rstrip(".").lower() for line in lines]
    if stripped == [] or stripped == ["none"]:
        return Candidates(found=[], broken=0, malformed=False)
    found = []
    for line in lines:
        match = re.match(r"^\s*\d+\s*[.)]\s*(.+)\(([^()]*)\)\s*$", line)
        if match is not None:
            found.append(Candidate(text=match[1].strip(), label=match[2], position=-1))
    return Candidates(found=found, broken=len(lines) - len(found), malformed=len(found) == 0)


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
            return align(Candidates(found=[], broken=0, malformed=True), labels, query)
        found = []
        for item in items:
            if isinstance(item, JsonEntity):
                found.append(Candidate(text=item.text, label=item.type, position=-1))
        return align(Candidates(found=found, broken=len(items) - len(found), malformed=False), labels, query)


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
        found = []
        for match in re.finditer(r"<([A-Za-z0-9_\-]+)>(.*?)</\1>", output):
            position = len(re.sub(tag, "", output[: match.start()]).split())
            found.append(Candidate(text=match[2], label=match[1], position=position))
        return align(Candidates(found=found, broken=0, malformed=False), labels, query)
