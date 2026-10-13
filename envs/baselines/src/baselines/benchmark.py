"""Run the configured baselines on the canonical datasets and write one JSONL of predictions per system."""

from datasets import load_from_disk
from pydantic import BaseModel
from seqeval.metrics.sequence_labeling import get_entities
from transformers import set_seed

from baselines.config import Config
from baselines.extractors import Gliner2Extractor, GlinerExtractor, Mention, Prediction, TrainingRow


class Benchmark(BaseModel):
    config: Config

    def write_predictions(self, system: str, dataset: str, ids: list[str], mentions: list[list[Mention]]) -> None:
        self.config.layout.baselines.prepare(system=system, dataset=dataset).write_text(
            "".join(
                Prediction(id=i, entities=found).model_dump_json() + "\n"
                for i, found in zip(ids, mentions, strict=True)
            )
        )

    def predict(self, tag: str, datasets: list[str] | None) -> None:
        spec = self.config.baselines[tag]
        match spec.library:
            case "gliner":
                extractor = GlinerExtractor(spec=spec)
            case "gliner2":
                extractor = Gliner2Extractor(spec=spec)
        set_seed(0, deterministic=True)
        for name, dataset in self.config.datasets.items():
            if not datasets or name in datasets:
                test = load_from_disk(self.config.layout.canonical.fill(dataset=name))["test"]
                found = extractor.extract([" ".join(t) for t in test["tokens"]], sorted(dataset.labels.values()))
                self.write_predictions(tag, name, test["id"], found)

    def finetune(self, tag: str, seed: int, datasets: list[str] | None) -> None:
        system = f"{tag}-ft-s{seed}"
        for name, dataset in self.config.datasets.items():
            if not datasets or name in datasets:
                set_seed(seed, deterministic=True)
                data = load_from_disk(self.config.layout.canonical.fill(dataset=name))
                extractor = GlinerExtractor(spec=self.config.baselines[tag])
                extractor.train(
                    [
                        TrainingRow(
                            tokenized_text=row["tokens"],
                            ner=[(s, e, dataset.labels[label]) for label, s, e in get_entities(row["tags"])],
                        )
                        for row in data["train"]
                    ],
                    seed,
                    self.config.layout.training.fill(system=system, dataset=name),
                )
                found = extractor.extract(
                    [" ".join(t) for t in data["test"]["tokens"]], sorted(dataset.labels.values())
                )
                self.write_predictions(system, name, data["test"]["id"], found)
