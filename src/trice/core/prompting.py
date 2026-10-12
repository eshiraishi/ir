"""Prompt formats P0-P3 (text lives in templates/) and parsing outputs back onto query tokens.

P0  v1 Listing 1.1: one user message, numbered ``span (label)`` list.
P1  chat template: system message + demonstrations as user/assistant turns, same list output.
P2  chat template + strict JSON Schema (structured generation) with a label enum.
P3  chat template + inline tags: the text repeated with ``<label>...</label>``.
"""

import json
import re

from jinja2 import Environment
from pydantic import BaseModel, ConfigDict

from trice.schema import Example, Parsed, Prompt
from trice.settings import Format


class Labels(BaseModel):
    """Raw dataset labels and the verbalizations shown to models (one-to-one)."""

    model_config = ConfigDict(frozen=True)

    raw_to_verbal: dict[str, str]

    @property
    def types(self) -> list[str]:
        return sorted(self.raw_to_verbal.values())

    @staticmethod
    def tag(verbal: str) -> str:
        return re.sub(r"[^a-z0-9]+", "_", verbal.lower()).strip("_")

    def resolve(self, label: str) -> str | None:
        """Raw label for a model-produced verbalization, raw label or tag (any case)."""
        key = label.strip().casefold()
        for raw, verbal in self.raw_to_verbal.items():
            if key in (raw.casefold(), verbal.casefold(), self.tag(verbal)):
                return raw
        return None


class Prompter(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    env: Environment

    def view(self, example: Example, labels: Labels) -> dict:
        """What the templates need to show one example and its gold answer."""
        mentions, segments, pos = [], [], 0
        for raw, start, end in example.spans:
            text, verbal = " ".join(example.tokens[start:end]), labels.raw_to_verbal[raw]
            mentions.append({"text": text, "type": verbal})
            segments += [{"text": t, "tag": ""} for t in example.tokens[pos:start]]
            segments.append({"text": text, "tag": labels.tag(verbal)})
            pos = end
        segments += [{"text": t, "tag": ""} for t in example.tokens[pos:]]
        return {"text": example.text, "mentions": mentions, "segments": segments}

    def answer(self, fmt: Format, example: Example, labels: Labels) -> str:
        view = self.view(example, labels)
        return self.env.get_template("answer_only.j2").render(fmt=fmt, **view)

    def prompt(self, fmt: Format, labels: Labels, demos: list[Example], query: Example) -> Prompt:
        types = labels.types
        if fmt == "P0":
            content = self.env.get_template("p0.j2").render(
                types=types, demos=[self.view(d, labels) for d in demos], query=query.text
            )
            return Prompt(messages=[{"role": "user", "content": content}], stop=["\nExample "])
        system = self.env.get_template("system.j2").render(fmt=fmt, types=types, tags=[labels.tag(t) for t in types])
        messages = [{"role": "system", "content": system}]
        for d in demos:
            messages += [
                {"role": "user", "content": d.text},
                {"role": "assistant", "content": self.answer(fmt, d, labels)},
            ]
        messages.append({"role": "user", "content": query.text})
        schema = {
            "type": "object",
            "required": ["entities"],
            "additionalProperties": False,
            "properties": {
                "entities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["text", "type"],
                        "additionalProperties": False,
                        "properties": {"text": {"type": "string"}, "type": {"type": "string", "enum": types}},
                    },
                }
            },
        }
        response_format = {"type": "json_schema", "json_schema": {"name": "entities", "strict": True, "schema": schema}}
        return Prompt(messages=messages, response_format=response_format if fmt == "P2" else None)

    def candidates(self, fmt: Format, output: str) -> tuple[list[tuple[str, str, int]], int, bool]:
        """(span, label, token position or -1) proposed by the output, unparseable items, and
        whether the output is malformed. Only inline tags (P3) carry positions."""
        if fmt == "P2":
            try:
                items = json.loads(output)["entities"]
            except json.JSONDecodeError, KeyError, TypeError:
                return [], 0, True
            pairs = [(i["text"], i["type"], -1) for i in items if isinstance(i, dict) and {"text", "type"} <= i.keys()]
            return pairs, len(items) - len(pairs), False
        if fmt == "P3":
            tag = r"</?[A-Za-z0-9_\-]+>"
            return (
                [
                    (m[2], m[1], len(re.sub(tag, "", output[: m.start()]).split()))
                    for m in re.finditer(r"<([A-Za-z0-9_\-]+)>(.*?)</\1>", output)
                ],
                0,
                False,
            )
        lines = [line for line in output.strip().splitlines() if line.strip()]
        if [line.strip().rstrip(".").lower() for line in lines] in ([], ["none"]):
            return [], 0, False
        items = [re.match(r"^\s*\d+\s*[.)]\s*(.+)\(([^()]*)\)\s*$", line) for line in lines]
        pairs = [(m[1].strip(), m[2], -1) for m in items if m]
        return pairs, len(items) - len(pairs), not pairs

    def parse(self, fmt: Format, output: str, query: Example, labels: Labels) -> Parsed:
        """Align each proposed mention to the query tokens: at its own position when the output
        gives one (P3), else at the first free occurrence. Mentions with an unknown label or no
        free occurrence count as invalid."""
        pairs, broken, malformed = self.candidates(fmt, output)
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
