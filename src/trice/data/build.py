"""Build canonical JSONL splits, label files and the data manifest; load splits and pools."""

import json
from functools import cache
from pathlib import Path

from trice.config import DatasetSpec, datasets_config, repo_root
from trice.data.bio import read_columns, to_sentence
from trice.data.fetch import fetch_dataset
from trice.data.schema import Sentence, read_jsonl, write_jsonl
from trice.seeds import shuffled

SPLITS = ("train", "test")


def canonical_dir() -> Path:
    return repo_root() / "data" / "canonical"


def manifest_path() -> Path:
    return repo_root() / "data" / "manifest.json"


def parse_dataset(ds: DatasetSpec, contents: dict[str, str]) -> dict[str, list[Sentence]]:
    """Parse raw file contents into canonical sentences, validating labels and split sizes."""
    out: dict[str, list[Sentence]] = {}
    for split in SPLITS:
        sentences = [
            to_sentence(f"{ds.name}/{split}/{i:06d}", tokens, tags)
            for i, (tokens, tags) in enumerate(read_columns(contents[split], ds.format))
        ]
        unknown = {e.label for s in sentences for e in s.entities} - ds.labels.keys()
        if unknown:
            raise ValueError(f"{ds.name}/{split}: labels missing from config: {sorted(unknown)}")
        if len(sentences) != ds.expected[split]:
            raise ValueError(
                f"{ds.name}/{split}: expected {ds.expected[split]} sentences, got {len(sentences)}"
            )
        out[split] = sentences
    return out


def build_dataset(ds: DatasetSpec) -> dict:
    files = fetch_dataset(ds)
    contents = {split: path.read_text(encoding="utf-8") for split, (path, _) in files.items()}
    splits = parse_dataset(ds, contents)
    ddir = canonical_dir() / ds.name
    entry: dict = {"labels": ds.labels, "splits": {}}
    for split, sentences in splits.items():
        digest = write_jsonl(ddir / f"{split}.jsonl", sentences)
        entry["splits"][split] = {
            "sentences": len(sentences),
            "entities": sum(len(s.entities) for s in sentences),
            "raw_sha256": files[split][1],
            "canonical_sha256": digest,
        }
    labels = {"labels": ds.labels, "verbalizations": sorted(set(ds.labels.values()))}
    (ddir / "labels.json").write_text(json.dumps(labels, indent=2, ensure_ascii=False) + "\n")
    return entry


def build_all(names: list[str] | None = None) -> dict:
    cfg = datasets_config()
    path = manifest_path()
    manifest = json.loads(path.read_text()) if path.exists() else {}
    for name in names or list(cfg.datasets):
        manifest[name] = build_dataset(cfg.datasets[name])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(sorted(manifest.items())), indent=2) + "\n")
    return manifest


# --------------------------------------------------------------------------- loading


def base_name(dataset: str) -> str:
    """``conll2003@50`` -> ``conll2003``."""
    return dataset.split("@", 1)[0]


def pool_size(dataset: str) -> int | None:
    return int(dataset.split("@", 1)[1]) if "@" in dataset else None


@cache
def _load(dataset: str, split: str) -> tuple[Sentence, ...]:
    return tuple(read_jsonl(canonical_dir() / dataset / f"{split}.jsonl"))


def load_split(dataset: str, split: str) -> list[Sentence]:
    """Load a canonical split. ``<name>@<n>`` restricts the *train* pool to the first ``n``
    sentences of one seeded permutation, so pools are nested (S6); the test split is unchanged."""
    base, n = base_name(dataset), pool_size(dataset)
    sentences = list(_load(base, split))
    if split == "train" and n is not None:
        sentences = shuffled(sentences, "pool", base, key=lambda s: s.id)[:n]
    return sentences


def label_map(dataset: str) -> dict[str, str]:
    return datasets_config().datasets[base_name(dataset)].labels
