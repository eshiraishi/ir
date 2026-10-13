"""Run the configured baselines on the canonical datasets and write {id, entities} JSONL."""

import json
import tempfile
from pathlib import Path

from datasets import load_from_disk
from pydantic import BaseModel
from seqeval.metrics.sequence_labeling import get_entities
from transformers import set_seed

from baselines.config import Config
from baselines.extractors import Gliner2Extractor, GlinerExtractor, Mention


class Benchmark(BaseModel):
    config: Config

    def write_predictions(self, tag: str, dataset: str, ids: list[str], mentions: list[list[Mention]]) -> None:
        out = self.config.layout.baselines / tag / f"{dataset}.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            "".join(
                json.dumps({"id": i, "entities": [m.model_dump() for m in found]}, ensure_ascii=False) + "\n"
                for i, found in zip(ids, mentions, strict=True)
            )
        )

    def predict(self, tag: str, datasets: list[str] | None) -> None:
        spec = self.config.baselines[tag]
        extractor = {"gliner": GlinerExtractor, "gliner2": Gliner2Extractor}[spec.library](spec=spec)
        set_seed(0, deterministic=True)
        for name, dataset in self.config.datasets.items():
            if not datasets or name in datasets:
                test = load_from_disk(self.config.layout.canonical / name)["test"]
                found = extractor.extract([" ".join(t) for t in test["tokens"]], sorted(dataset.labels.values()))
                self.write_predictions(tag, name, test["id"], found)

    def finetune(self, tag: str, seed: int, datasets: list[str] | None) -> None:
        spec = self.config.baselines[tag]
        for name, dataset in self.config.datasets.items():
            if not datasets or name in datasets:
                set_seed(seed, deterministic=True)
                data = load_from_disk(self.config.layout.canonical / name)
                extractor = GlinerExtractor(spec=spec)
                with tempfile.TemporaryDirectory() as output:
                    extractor.train(
                        [
                            {
                                "tokenized_text": row["tokens"],
                                "ner": [[s, e, dataset.labels[label]] for label, s, e in get_entities(row["tags"])],
                            }
                            for row in data["train"]
                        ],
                        seed,
                        Path(output),
                    )
                found = extractor.extract(
                    [" ".join(t) for t in data["test"]["tokens"]], sorted(dataset.labels.values())
                )
                self.write_predictions(f"{tag}-ft-s{seed}", name, data["test"]["id"], found)
