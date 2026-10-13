"""Specialist NER baselines (GLiNER family, NuNER-Zero). They read the same trice.toml as the core
(datasets, [layout], [baselines]) and write one JSONL of predictions per system for `trice baselines`.

    docker compose -f envs/compose.yaml run --rm baselines predict gliner2-large
    docker compose -f envs/compose.yaml run --rm baselines finetune nuner-zero -d crossner_ai
"""

from functools import cached_property
from pathlib import Path
from typing import Annotated, Literal

import torch
import typer
from datasets import load_from_disk
from gliner import GLiNER
from gliner2 import GLiNER2
from pydantic import BaseModel, ConfigDict
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict, TomlConfigSettingsSource
from seqeval.metrics.sequence_labeling import get_entities
from transformers import set_seed


class Layout(BaseModel):
    canonical: str
    baselines: str
    training: str


class DatasetSpec(BaseModel):
    labels: dict[str, str]


class BaselineSpec(BaseModel):
    library: Literal["gliner", "gliner2"]
    model: str
    revision: str
    finetune: bool = False  # few-shot fine-tuning on each training pool (gliner only)
    seeds: list[int] = [0, 1, 2]


class Config(BaseSettings):
    model_config = SettingsConfigDict(toml_file="trice.toml", extra="ignore")

    layout: Layout
    datasets: dict[str, DatasetSpec]
    baselines: dict[str, BaselineSpec]

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return init_settings, TomlConfigSettingsSource(settings_cls)


class Mention(BaseModel):
    text: str
    label: str


class Prediction(BaseModel):
    """One line of the output (read by `trice baselines`)."""

    id: str
    entities: list[Mention]


class TrainingRow(BaseModel):
    """GLiNER's training format: tokens and [first, last (inclusive), type] spans."""

    tokenized_text: list[str]
    ner: list[tuple[int, int, str]]


class GlinerExtractor(BaseModel):
    """GLiNER v2.5 and NuNER-Zero; library-default threshold and training hyperparameters."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    spec: BaselineSpec

    @cached_property
    def model(self) -> GLiNER:
        return GLiNER.from_pretrained(self.spec.model, revision=self.spec.revision)

    def train(self, rows: list[TrainingRow], seed: int, output: str) -> None:
        self.model.train_model(
            train_dataset=[row.model_dump() for row in rows],
            eval_dataset=None,
            output_dir=output,
            seed=seed,
            data_seed=seed,
            save_strategy="no",
        )

    def extract(self, texts: list[str], types: list[str]) -> list[list[Mention]]:
        self.model.eval()
        with torch.no_grad():
            predicted = self.model.inference(texts, types, flat_ner=True)
        return [[Mention(text=e["text"], label=e["label"]) for e in found] for found in predicted]


class Gliner2Extractor(BaseModel):
    """GLiNER2 and GLiNER2.5; library-default threshold."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    spec: BaselineSpec

    @cached_property
    def model(self) -> GLiNER2:
        return GLiNER2.from_pretrained(self.spec.model, revision=self.spec.revision)

    def extract(self, texts: list[str], types: list[str]) -> list[list[Mention]]:
        results = self.model.batch_extract_entities(texts, types)
        mentions = []
        for result in results:
            found = []
            for label, texts_found in result["entities"].items():
                for text in texts_found:
                    found.append(Mention(text=text, label=label))
            mentions.append(found)
        return mentions


class Benchmark(BaseModel):
    config: Config

    def write_predictions(self, system: str, dataset: str, ids: list[str], mentions: list[list[Mention]]) -> None:
        path = Path(self.config.layout.baselines.format(system=system, dataset=dataset))
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            Prediction(id=i, entities=found).model_dump_json() + "\n" for i, found in zip(ids, mentions, strict=True)
        ]
        path.write_text("".join(lines))

    def predict(self, tag: str, datasets: list[str] | None) -> None:
        spec = self.config.baselines[tag]
        match spec.library:
            case "gliner":
                extractor = GlinerExtractor(spec=spec)
            case "gliner2":
                extractor = Gliner2Extractor(spec=spec)
        set_seed(0, deterministic=True)
        for name, dataset in self.config.datasets.items():
            if datasets is None or name in datasets:
                test = load_from_disk(self.config.layout.canonical.format(dataset=name))["test"]
                texts = [" ".join(tokens) for tokens in test["tokens"]]
                self.write_predictions(tag, name, test["id"], extractor.extract(texts, sorted(dataset.labels.values())))

    def finetune(self, tag: str, seed: int, datasets: list[str] | None) -> None:
        system = f"{tag}-ft-s{seed}"
        for name, dataset in self.config.datasets.items():
            if datasets is None or name in datasets:
                set_seed(seed, deterministic=True)
                data = load_from_disk(self.config.layout.canonical.format(dataset=name))
                rows = []
                for row in data["train"]:
                    spans = [(first, last, dataset.labels[label]) for label, first, last in get_entities(row["tags"])]
                    rows.append(TrainingRow(tokenized_text=row["tokens"], ner=spans))
                extractor = GlinerExtractor(spec=self.config.baselines[tag])
                extractor.train(rows, seed, self.config.layout.training.format(system=system, dataset=name))
                texts = [" ".join(tokens) for tokens in data["test"]["tokens"]]
                found = extractor.extract(texts, sorted(dataset.labels.values()))
                self.write_predictions(system, name, data["test"]["id"], found)


app = typer.Typer(no_args_is_help=True, add_completion=False)
Datasets = Annotated[list[str] | None, typer.Option("--dataset", "-d")]


@app.command()
def predict(tag: str, datasets: Datasets = None) -> None:
    """Zero-shot predictions on the test split of every dataset."""
    Benchmark(config=Config()).predict(tag, datasets)


@app.command()
def finetune(tag: str, datasets: Datasets = None) -> None:
    """Fine-tune on each training pool (library-default hyperparameters, no validation set) per seed."""
    benchmark = Benchmark(config=Config())
    for seed in benchmark.config.baselines[tag].seeds:
        benchmark.finetune(tag, seed, datasets)


if __name__ == "__main__":
    app()
