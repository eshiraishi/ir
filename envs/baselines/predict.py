"""Zero-shot specialist baselines: GLiNER v2.5, NuNER-Zero (gliner lib) and GLiNER2 / GLiNER2.5.

Reads the canonical test split written by `trice data build`, predicts with the library-default
threshold (never tuned) and writes `results/baselines/<tag>/<dataset>.jsonl` with one line per
sentence: {"id": ..., "entities": [{"text": ..., "label": <verbalization>}]}. Scoring happens in
the core package (`trice score-baselines`) so every system shares one scorer.

    uv run python predict.py --library gliner --model urchade/gliner_large-v2.5 \
        --revision <sha> --tag gliner-v2.5-large --root ../.. [--datasets ...]
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch


def set_determinism(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def load_split(root: Path, dataset: str, split: str) -> tuple[list[dict], list[str]]:
    ddir = root / "data" / "canonical" / dataset
    rows = [json.loads(line) for line in (ddir / f"{split}.jsonl").read_text().splitlines()]
    labels = json.loads((ddir / "labels.json").read_text())["verbalizations"]
    return rows, labels


def predict_gliner(model_id: str, revision: str, rows, labels, batch_size: int) -> list[list[dict]]:
    from gliner import GLiNER

    model = GLiNER.from_pretrained(model_id, revision=revision).eval()
    out = []
    for i in range(0, len(rows), batch_size):
        texts = [r["text"] for r in rows[i : i + batch_size]]
        preds = model.inference(texts, labels, flat_ner=True, batch_size=batch_size)
        out.extend([{"text": e["text"], "label": e["label"]} for e in p] for p in preds)
    return out


def predict_gliner2(
    model_id: str, revision: str, rows, labels, batch_size: int
) -> list[list[dict]]:
    from gliner2 import GLiNER2

    model = GLiNER2.from_pretrained(model_id, revision=revision)
    results = model.batch_extract_entities(
        [r["text"] for r in rows], labels, batch_size=batch_size, include_spans=False
    )
    out = []
    for res in results:
        ents = res.get("entities", {})
        out.append([{"text": t, "label": lab} for lab, spans in ents.items() for t in spans])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--library", choices=["gliner", "gliner2"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", required=True, help="pinned HF commit")
    ap.add_argument("--tag", required=True, help="output folder name")
    ap.add_argument("--root", type=Path, default=Path("../.."))
    ap.add_argument("--datasets", nargs="*")
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    set_determinism()
    canonical = args.root / "data" / "canonical"
    datasets = args.datasets or sorted(p.name for p in canonical.iterdir() if p.is_dir())
    predict = predict_gliner if args.library == "gliner" else predict_gliner2
    for ds in datasets:
        rows, labels = load_split(args.root, ds, "test")
        preds = predict(args.model, args.revision, rows, labels, args.batch_size)
        out = args.root / "results" / "baselines" / args.tag / f"{ds}.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as f:
            for row, ents in zip(rows, preds, strict=True):
                f.write(json.dumps({"id": row["id"], "entities": ents}, ensure_ascii=False) + "\n")
        print(f"{args.tag} {ds}: {len(rows)} sentences -> {out}")


if __name__ == "__main__":
    main()
