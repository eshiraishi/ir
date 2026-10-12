import pytest

from trice.config import DatasetSpec
from trice.data.bio import bio_to_token_spans, read_columns, to_sentence
from trice.data.build import parse_dataset
from trice.data.schema import Sentence, read_jsonl, write_jsonl


def test_bio_spans_basic_and_iob1():
    tags = ["B-PER", "I-PER", "O", "I-LOC", "I-LOC", "B-LOC", "I-ORG"]
    assert bio_to_token_spans(tags) == [(0, 2, "PER"), (3, 5, "LOC"), (5, 6, "LOC"), (6, 7, "ORG")]


def test_bio_rejects_unknown_scheme():
    with pytest.raises(ValueError, match="unsupported tag"):
        bio_to_token_spans(["S-PER"])


def test_read_columns_both_orders_and_docstart():
    tt = "-DOCSTART-\tO\n\nJohn\tB-PER\nruns\tO\n\nParis\tB-LOC\n"
    assert list(read_columns(tt, "token_tag")) == [
        (["John", "runs"], ["B-PER", "O"]),
        (["Paris"], ["B-LOC"]),
    ]
    tg = "B-ACTOR\ttom\nI-ACTOR\thanks\n"
    assert list(read_columns(tg, "tag_token")) == [(["tom", "hanks"], ["B-ACTOR", "I-ACTOR"])]


def test_to_sentence_offsets_roundtrip(tmp_path):
    s = to_sentence(
        "d/test/000000", ["New", "York", "is", "big", "."], ["B-LOC", "I-LOC", "O", "O", "O"]
    )
    assert s.text == "New York is big ."
    (e,) = s.entities
    assert (e.start, e.end, e.token_start, e.token_end, e.text) == (0, 8, 0, 2, "New York")
    path = tmp_path / "x.jsonl"
    digest = write_jsonl(path, [s])
    assert read_jsonl(path) == [s]
    assert write_jsonl(tmp_path / "y.jsonl", [s]) == digest


def test_sentence_validator_rejects_bad_offsets():
    with pytest.raises(ValueError, match="offsets"):
        Sentence(
            id="x",
            tokens=("a", "b"),
            text="a b",
            entities=(
                {"start": 0, "end": 2, "token_start": 0, "token_end": 1, "label": "L", "text": "a"},
            ),
        )


def _spec(**kw) -> DatasetSpec:
    base = dict(
        name="toy",
        source="s",
        format="token_tag",
        files={"train": {"path": "a"}, "test": {"path": "b"}},
        expected={"train": 1, "test": 1},
        labels={"PER": "person"},
    )
    return DatasetSpec.model_validate({**base, **kw})


def test_parse_dataset_validates_labels_and_counts():
    ok = {"train": "Ann\tB-PER\n", "test": "Bob\tB-PER\n"}
    assert len(parse_dataset(_spec(), ok)["test"]) == 1
    with pytest.raises(ValueError, match="labels missing"):
        parse_dataset(_spec(), {"train": "Ann\tB-ORG\n", "test": "Bob\tB-PER\n"})
    with pytest.raises(ValueError, match="expected 1 sentences"):
        parse_dataset(_spec(), {"train": "Ann\tB-PER\n\nBo\tO\n", "test": "Bob\tB-PER\n"})
