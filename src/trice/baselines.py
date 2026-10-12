"""Score the specialist baselines written by envs/baselines with TRICE's own scorer."""

import json

from trice.data.build import label_map, load_split
from trice.eval.metrics import counts_matrix, f1_of_sums, gold_pairs
from trice.prompt.labels import LabelSet
from trice.prompt.parse import normalize_ws
from trice.studies import results_dir


def score_file(tag: str, dataset: str) -> float:
    labels = LabelSet(label_map(dataset))
    test = load_split(dataset, "test")
    rows = [
        json.loads(x)
        for x in (results_dir() / "baselines" / tag / f"{dataset}.jsonl").read_text().splitlines()
    ]
    if [r["id"] for r in rows] != [s.id for s in test]:
        raise ValueError(f"{tag}/{dataset}: predictions do not align with the test split")
    preds = []
    for r in rows:
        pairs = []
        for e in r["entities"]:
            verbal = labels.resolve(e["label"])
            if verbal is not None:
                pairs.append((normalize_ws(e["text"]), verbal))
        preds.append(pairs)
    counts = counts_matrix(preds, [gold_pairs(s, labels) for s in test])
    return float(f1_of_sums(counts.sum(0)))


def score_all() -> dict[str, dict[str, float]]:
    root = results_dir() / "baselines"
    out = (
        {
            tag.name: {f.stem: score_file(tag.name, f.stem) for f in sorted(tag.glob("*.jsonl"))}
            for tag in sorted(p for p in root.iterdir() if p.is_dir())
        }
        if root.exists()
        else {}
    )
    path = results_dir() / "tables" / "baselines.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2) + "\n")
    return out
