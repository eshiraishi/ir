"""All configuration: trice.toml (+ secrets from .env), parsed once by the container."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, SecretStr
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict, TomlConfigSettingsSource

Split = Literal["train", "test"]
QuerySet = Literal["test", "loo"]
Format = Literal["P0", "P1", "P2", "P3"]
Backend = Literal["local", "openrouter"]


class Paths(BaseModel):
    data: Path
    results: Path
    cache: Path


class LLMSettings(BaseModel):
    openrouter_url: str
    local_url: str
    temperature: float
    top_p: float
    seed: int
    max_tokens: int
    max_retries: int
    timeout: float
    workers: int
    quantization_preference: list[str]
    required_parameters: list[str]


class ModelRef(BaseModel):
    id: str
    revision: str | None = None
    backend: Backend = "local"


class StackSpec(BaseModel):
    kind: Literal["trice", "bm25", "dense"]
    sparse: ModelRef | None = None
    dense: ModelRef | None = None
    reranker: ModelRef | None = None
    query_instruction: str | None = None
    bm25_k1: float = 1.2
    bm25_b: float = 0.75


class RetrievalSettings(BaseModel):
    rrf_k: int
    rerank_depth: int
    store_depth: int
    device: str
    threads: int
    batch_size: int
    reranker_instruction: str
    stacks: dict[str, StackSpec]


class SelectionSettings(BaseModel):
    k_grid: list[int]
    loo_queries: int
    pool_sizes: list[int]
    format_k: int


class StatsSettings(BaseModel):
    resamples: int
    alpha: float
    audit_prompts: int
    probe_lines: int
    output_factor: float
    output_extra: int


class SourceSpec(BaseModel):
    url: str
    commit: str | None = None  # git source when set, plain HTTP otherwise


class FileSpec(BaseModel):
    path: str
    size: int
    sha256: str | None = None


class DatasetSpec(BaseModel):
    source: str
    columns: Literal["token_tag", "tag_token"] = "token_tag"
    train: FileSpec
    test: FileSpec
    labels: dict[str, str]


class ModelSpec(BaseModel):
    openrouter: str
    hf_repo: str
    ladder: str
    params_b: float
    active_params_b: float
    thinking: bool = False
    reference: bool = False


class StudySpec(BaseModel):
    queries: QuerySet = "test"
    models: Literal["all", "reference"] | list[str] = "all"
    datasets: Literal["all"] | list[str] = "all"
    conditions: list[str]
    k: Literal["k_star", "grid"] | int = "k_star"
    formats: Literal["frozen", "others"] | list[Format] = "frozen"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(toml_file="trice.toml", env_file=".env", extra="ignore")

    openrouter_api_key: SecretStr | None = None
    hf_token: SecretStr | None = None
    paths: Paths
    llm: LLMSettings
    retrieval: RetrievalSettings
    selection: SelectionSettings
    stats: StatsSettings
    sources: dict[str, SourceSpec]
    datasets: dict[str, DatasetSpec]
    models: dict[str, ModelSpec]
    studies: dict[str, StudySpec]

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return init_settings, env_settings, dotenv_settings, TomlConfigSettingsSource(settings_cls)
