"""Typed access to the YAML configs under ``configs/``."""

import os
from functools import cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def repo_root() -> Path:
    """Repository root: ``$TRICE_ROOT`` or the nearest ancestor of the cwd holding ``configs/``."""
    if env := os.environ.get("TRICE_ROOT"):
        return Path(env).resolve()
    here = Path.cwd().resolve()
    for p in (here, *here.parents):
        if (p / "configs" / "datasets.yaml").exists():
            return p
    raise FileNotFoundError("could not find configs/datasets.yaml; set TRICE_ROOT")


def _load(name: str) -> dict:
    with (repo_root() / "configs" / name).open(encoding="utf-8") as f:
        return yaml.safe_load(f)


# --------------------------------------------------------------------------- datasets


class SourceSpec(_Strict):
    kind: Literal["git", "http"]
    url: str | None = None
    commit: str | None = None


class FileSpec(_Strict):
    path: str | None = None
    url: str | None = None
    sha256: str | None = None


class DatasetSpec(_Strict):
    name: str
    source: str
    format: Literal["token_tag", "tag_token"]
    files: dict[Literal["train", "test"], FileSpec]
    expected: dict[Literal["train", "test"], int]
    labels: dict[str, str]


class DatasetsConfig(_Strict):
    sources: dict[str, SourceSpec]
    datasets: dict[str, DatasetSpec]


@cache
def datasets_config() -> DatasetsConfig:
    raw = _load("datasets.yaml")
    for name, spec in raw["datasets"].items():
        spec["name"] = name
    return DatasetsConfig.model_validate(raw)


# --------------------------------------------------------------------------- models


class LLMSpec(_Strict):
    name: str
    family: str
    ladder: Literal["qwen3.5-dense", "qwen3.5-moe", "gemma3"]
    params_b: float
    active_params_b: float
    openrouter: str
    hf_repo: str
    hf_revision: str | None = None
    thinking: bool = False
    reference: bool = False


@cache
def models_config() -> dict[str, LLMSpec]:
    raw = _load("models.yaml")["models"]
    return {name: LLMSpec.model_validate({"name": name, **spec}) for name, spec in raw.items()}


# --------------------------------------------------------------------------- retrieval


class ModelRef(_Strict):
    id: str
    revision: str | None = None
    backend: Literal["local", "openrouter"] = "local"


class StackSpec(_Strict):
    name: str
    kind: Literal["trice", "v1_bm25", "v1_dense"]
    sparse: ModelRef | None = None
    dense: ModelRef | None = None
    reranker: ModelRef | None = None
    query_instruction: str | None = None
    lemmatize: bool = False
    bm25_k1: float | None = None
    bm25_b: float | None = None


class RetrievalConfig(_Strict):
    rrf_k: int
    rerank_depth: int
    store_depth: int
    reranker_instruction: str
    stacks: dict[str, StackSpec]


@cache
def retrieval_config() -> RetrievalConfig:
    raw = _load("retrieval.yaml")
    for name, spec in raw["stacks"].items():
        spec["name"] = name
    return RetrievalConfig.model_validate(raw)


# --------------------------------------------------------------------------- studies


class StudySpec(_Strict):
    name: str
    description: str
    queries: Literal["test", "loo"]
    models: Literal["all", "reference"] | list[str]
    datasets: Literal["all"] | list[str]
    conditions: list[str]
    k: Literal["k_star", "grid"] | int
    formats: Literal["frozen", "others"] | list[Literal["P0", "P1", "P2", "P3"]]


class StudiesConfig(_Strict):
    k_grid: list[int]
    loo_queries: int
    pool_sizes: list[int]
    studies: dict[str, StudySpec]


@cache
def studies_config() -> StudiesConfig:
    raw = _load("studies.yaml")
    for name, spec in raw["studies"].items():
        spec["name"] = name
    return StudiesConfig.model_validate(raw)
