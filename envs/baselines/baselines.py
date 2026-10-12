"""Specialist NER baselines writing {id, entities: [{text, label}]} JSONL for `trice baselines`.

Zero-shot: GLiNER v2.5 / NuNER-Zero (gliner) and GLiNER2 / GLiNER2.5 (gliner2), library-default
threshold. Few-shot: GLiNER fine-tuned on each training pool with library-default hyperparameters,
no validation set, seeds 0/1/2.

    uv run python baselines.py predict gliner2 fastino/gliner2-large-v1 <sha> gliner2-large
    uv run python baselines.py finetune urchade/gliner_large-v2.5 <sha> gliner-v2.5-large --seed 0
"""

import json
import tempfile
import tomllib
from pathlib import Path
from typing import Literal

import torch
import typer
from datasets import load_from_disk
from gliner import GLiNER
from gliner2 import GLiNER2
from transformers import set_seed

app = typer.Typer(no_args_is_help=True)


def datasets_and_labels(root: Path, names: list[str] | None) -> dict[str, dict[str, str]]:
    config = tomllib.loads((root / "trice.toml").read_text())
    return {n: s["labels"] for n, s in config["datasets"].items() if not names or n in names}


def write(root: Path, tag: str, dataset: str, ids: list[str], entities: list[list[dict]]) -> None:
    out = root / "results" / "baselines" / tag / f"{dataset}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "".join(
            json.dumps({"id": i, "entities": e}, ensure_ascii=False) + "\n" for i, e in zip(ids, entities, strict=True)
        )
    )


@app.command()
def predict(
    library: Literal["gliner", "gliner2"],
    model: str,
    revision: str,
    tag: str,
    datasets: list[str] | None = None,
    root: Path = Path("../.."),
) -> None:
    set_seed(0, deterministic=True)
    for name, labels in datasets_and_labels(root, datasets).items():
        test = load_from_disk(root / "data" / "canonical" / name)["test"]
        texts, types = [" ".join(t) for t in test["tokens"]], sorted(labels.values())
        if library == "gliner":
            preds = GLiNER.from_pretrained(model, revision=revision).eval().inference(texts, types, flat_ner=True)
            entities = [[{"text": e["text"], "label": e["label"]} for e in p] for p in preds]
        else:
            results = GLiNER2.from_pretrained(model, revision=revision).batch_extract_entities(texts, types)
            entities = [[{"text": t, "label": lab} for lab, ts in r["entities"].items() for t in ts] for r in results]
        write(root, tag, name, test["id"], entities)


@app.command()
def finetune(
    model: str, revision: str, tag: str, seed: int, datasets: list[str] | None = None, root: Path = Path("../..")
) -> None:
    for name, labels in datasets_and_labels(root, datasets).items():
        set_seed(seed, deterministic=True)
        data = load_from_disk(root / "data" / "canonical" / name)
        train = [
            {"tokenized_text": row["tokens"], "ner": [[s, e, labels[lab]] for lab, s, e in spans(row["tags"])]}
            for row in data["train"]
        ]
        gliner = GLiNER.from_pretrained(model, revision=revision)
        with tempfile.TemporaryDirectory() as tmp:
            gliner.train_model(
                train_dataset=train, eval_dataset=None, output_dir=tmp, seed=seed, data_seed=seed, save_strategy="no"
            )
        texts = [" ".join(t) for t in data["test"]["tokens"]]
        with torch.no_grad():
            preds = gliner.eval().inference(texts, sorted(labels.values()), flat_ner=True)
        write(
            root,
            f"{tag}-ft-s{seed}",
            name,
            data["test"]["id"],
            [[{"text": e["text"], "label": e["label"]} for e in p] for p in preds],
        )


def spans(tags: list[str]) -> list[tuple[str, int, int]]:
    """(label, first, last) token spans of IOB2 tags (GLiNER uses inclusive ends)."""
    out: list[tuple[str, int, int]] = []
    for i, tag in enumerate(tags):
        if tag.startswith("B-") or (tag.startswith("I-") and (not out or out[-1][2] != i - 1 or out[-1][0] != tag[2:])):
            out.append((tag[2:], i, i))
        elif tag.startswith("I-"):
            out[-1] = (out[-1][0], out[-1][1], i)
    return out


if __name__ == "__main__":
    app()
