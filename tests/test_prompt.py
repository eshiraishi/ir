import json
from collections import Counter

import pytest

from trice.data.bio import to_sentence
from trice.data.build import canonical_dir, label_map, load_split
from trice.eval.metrics import gold_pairs
from trice.prompt.formats import FORMATS, build_prompt, json_schema, render_answer
from trice.prompt.labels import LabelSet, tag_name
from trice.prompt.parse import parse

LABELS = LabelSet({"PER": "person", "LOC": "location", "programlang": "programming language"})
S1 = to_sentence(
    "t/0",
    ["John", "Smith", "visited", "New", "York", "."],
    ["B-PER", "I-PER", "O", "B-LOC", "I-LOC", "O"],
)
S2 = to_sentence("t/1", ["Nothing", "here", "."], ["O", "O", "O"])
S3 = to_sentence(
    "t/2", ["Python", "beats", "Python", "?"], ["B-programlang", "O", "B-programlang", "O"]
)


def test_labelset_resolve_and_tags():
    assert LABELS.verbalizations == ("location", "person", "programming language")
    assert tag_name("programming language") == "programming_language"
    for alias in ("Programming Language", "programlang", "programming_language"):
        assert LABELS.resolve(alias) == "programming language"
    assert LABELS.resolve("organisation") is None
    with pytest.raises(ValueError, match="unique"):
        LabelSet({"a": "x", "b": "x"})


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("sentence", [S1, S2, S3])
def test_render_then_parse_roundtrip(fmt, sentence):
    parsed = parse(fmt, render_answer(fmt, sentence, LABELS), sentence.text, LABELS)
    assert Counter(parsed.entities) == Counter(gold_pairs(sentence, LABELS))
    assert parsed.n_invalid == 0 and not parsed.malformed and not parsed.text_mismatch


def test_render_formats():
    assert render_answer("P1", S1, LABELS) == "1. John Smith (person)\n2. New York (location)"
    assert render_answer("P1", S2, LABELS) == "None"
    assert json.loads(render_answer("P2", S2, LABELS)) == {"entities": []}
    assert (
        render_answer("P3", S1, LABELS)
        == "<person>John Smith</person> visited <location>New York</location> ."
    )


def test_p0_matches_v1_listing():
    p = build_prompt("P0", LABELS, [S1], "Mary went to Paris .")
    assert p.stop == ["\nExample "]
    assert p.messages == [
        {
            "role": "user",
            "content": "Defn: An entity is a location, person, or programming language.\n"
            "Example 1: John Smith visited New York .\nAnswer:\n"
            "1. John Smith (person)\n2. New York (location)\n"
            "Example 2: Mary went to Paris .\nAnswer:",
        }
    ]


def test_chat_formats_structure():
    for fmt in ("P1", "P2", "P3"):
        p = build_prompt(fmt, LABELS, [S1, S2], "Mary went to Paris .")
        roles = [m["role"] for m in p.messages]
        assert roles == ["system", "user", "assistant", "user", "assistant", "user"]
        assert p.messages[-1]["content"] == "Mary went to Paris ."
    p2 = build_prompt("P2", LABELS, [], "x")
    assert p2.response_format["json_schema"]["schema"] == json_schema(LABELS)
    assert p2.response_format["json_schema"]["strict"] is True


def test_parse_rejects_invalid_candidates():
    text = "John Smith visited New York ."
    out = "1. John Smith (person)\n2. Boston (location)\n3. New York (city)\nsome chatter"
    p = parse("P1", out, text, LABELS)
    assert p.entities == (("John Smith", "person"),)
    assert (p.n_candidates, p.n_invalid, p.malformed) == (4, 3, False)

    p = parse("P2", "not json", text, LABELS)
    assert p.malformed and p.entities == ()

    p = parse("P3", "<person>John Smith</person> visited <city>New York</city> !", text, LABELS)
    assert p.entities == (("John Smith", "person"),)
    assert p.n_invalid == 1 and p.text_mismatch


def test_parse_list_span_with_parentheses_and_none():
    text = "Python ( programming language ) rocks"
    p = parse("P1", "1. Python ( programming language ) (programming language)", text, LABELS)
    assert p.entities == (("Python ( programming language )", "programming language"),)
    assert parse("P0", "None.", text, LABELS).n_candidates == 0


REAL = [
    d
    for d in ("crossner_ai", "crossner_music", "conll2003", "wnut17", "ncbi_disease")
    if (canonical_dir() / d / "train.jsonl").exists()
]


@pytest.mark.skipif(not REAL, reason="canonical data not built (run `trice data build`)")
@pytest.mark.parametrize("dataset", REAL)
def test_roundtrip_on_real_data(dataset):
    labels = LabelSet(label_map(dataset))
    for s in load_split(dataset, "train")[:2000]:
        for fmt in FORMATS:
            parsed = parse(fmt, render_answer(fmt, s, labels), s.text, labels)
            assert Counter(parsed.entities) == Counter(gold_pairs(s, labels)), (fmt, s.id)
            assert parsed.n_invalid == 0, (fmt, s.id)
