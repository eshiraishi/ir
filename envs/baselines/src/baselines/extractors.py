"""One extractor per library: texts and entity types in, mentions out."""

from functools import cached_property
from pathlib import Path
from typing import Protocol, runtime_checkable

import torch
from gliner import GLiNER
from gliner2 import GLiNER2
from pydantic import BaseModel, ConfigDict

from baselines.config import BaselineSpec


class Mention(BaseModel):
    text: str
    label: str


@runtime_checkable
class Extractor(Protocol):
    def extract(self, texts: list[str], types: list[str]) -> list[list[Mention]]: ...


class GlinerExtractor(BaseModel):
    """GLiNER v2.5 and NuNER-Zero; library-default threshold and training hyperparameters."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    spec: BaselineSpec

    @cached_property
    def model(self) -> GLiNER:
        return GLiNER.from_pretrained(self.spec.model, revision=self.spec.revision)

    def train(self, rows: list[dict], seed: int, output: Path) -> None:
        self.model.train_model(
            train_dataset=rows, eval_dataset=None, output_dir=output, seed=seed, data_seed=seed, save_strategy="no"
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
        return [
            [Mention(text=text, label=label) for label, found in result["entities"].items() for text in found]
            for result in self.model.batch_extract_entities(texts, types)
        ]
