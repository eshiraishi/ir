"""Few-shot specialist baseline: fine-tune a GLiNER checkpoint on each training pool.

Pre-registered rule: library-default hyperparameters (`GLiNER.create_training_args` defaults),
no validation set, no early stopping, seeds 0/1/2. The final checkpoint predicts the test split
with the default threshold and writes `results/baselines/<tag>-ft-s<seed>/<dataset>.jsonl`.

    uv run python finetune.py --model urchade/gliner_large-v2.5 --revision <sha> \
        --tag gliner-v2.5-large --seed 0 --root ../.. [--datasets ...]
"""

import argparse
import json
import random
import tempfile
from pathlib import Path

import numpy as np
import torch
from predict import load_split


def to_gliner_example(row: dict, verbal: dict[str, str]) -> dict:
    # GLiNER expects token-level spans with an inclusive end index.
    ner = [[e["token_start"], e["token_end"] - 1, verbal[e["label"]]] for e in row["entities"]]
    return {"tokenized_text": row["tokens"], "ner": ner}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--root", type=Path, default=Path("../.."))
    ap.add_argument("--datasets", nargs="*")
    args = ap.parse_args()

    from gliner import GLiNER

    canonical = args.root / "data" / "canonical"
    datasets = args.datasets or sorted(p.name for p in canonical.iterdir() if p.is_dir())
    for ds in datasets:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        train, labels = load_split(args.root, ds, "train")
        label_map = json.loads((canonical / ds / "labels.json").read_text())["labels"]
        test, _ = load_split(args.root, ds, "test")
        model = GLiNER.from_pretrained(args.model, revision=args.revision)
        with tempfile.TemporaryDirectory() as tmp:
            model.train_model(
                train_dataset=[to_gliner_example(r, label_map) for r in train],
                eval_dataset=None,
                output_dir=tmp,
                seed=args.seed,
                data_seed=args.seed,
                full_determinism=True,
                save_strategy="no",
            )
        model.eval()
        preds = model.inference([r["text"] for r in test], labels, flat_ner=True)
        out = args.root / "results" / "baselines" / f"{args.tag}-ft-s{args.seed}" / f"{ds}.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as f:
            for row, ents in zip(test, preds, strict=True):
                ents = [{"text": e["text"], "label": e["label"]} for e in ents]
                f.write(json.dumps({"id": row["id"], "entities": ents}, ensure_ascii=False) + "\n")
        print(f"{args.tag} seed={args.seed} {ds}: trained on {len(train)}, wrote {out}")


if __name__ == "__main__":
    main()
