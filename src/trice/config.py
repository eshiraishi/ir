"""All configuration: trice.toml (+ secrets from .env), parsed once by compose()."""

from typing import Annotated, Literal

from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict, TomlConfigSettingsSource

from trice.schema import FormatName, LabelMap, ModelRef, PathTemplate, QuerySet, Runtime, Split


class Layout(BaseModel):
    raw: PathTemplate
    canonical: PathTemplate
    rankings: PathTemplate
    cache: PathTemplate
    endpoints: PathTemplate
    frozen_format: PathTemplate
    k_star: PathTemplate
    runs: PathTemplate
    runs_partial: PathTemplate
    audit: PathTemplate
    contamination: PathTemplate
    baselines: PathTemplate
    training: PathTemplate
    budget: PathTemplate
    tables: PathTemplate


class LLMConfig(BaseModel):
    openrouter_url: str
    local_url: str
    temperature: float
    top_p: float
    seed: int
    max_tokens: int
    max_retries: int
    timeout: float
    workers: int
    allow_spend: bool
    quantization_preference: list[str]
    required_parameters: list[str]


class TriceStack(BaseModel):
    kind: Literal["trice"]
    sparse: ModelRef
    dense: ModelRef
    reranker: ModelRef


class BM25Stack(BaseModel):
    kind: Literal["bm25"]
    k1: float = 1.2
    b: float = 0.75


class DenseStack(BaseModel):
    kind: Literal["dense"]
    dense: ModelRef
    query_instruction: str | None = None


StackSpec = Annotated[TriceStack | BM25Stack | DenseStack, Field(discriminator="kind")]


class RetrievalConfig(BaseModel):
    rrf_k: int
    rerank_depth: int
    store_depth: int
    runtime: Runtime
    reranker_instruction: str
    stacks: dict[str, StackSpec]


class SelectionConfig(BaseModel):
    k_grid: list[int]
    loo_queries: int
    pool_sizes: list[int]
    format_k: int


class StatsConfig(BaseModel):
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
    labels: LabelMap

    def select(self, split: Split) -> FileSpec:
        match split:
            case "train":
                return self.train
            case "test":
                return self.test


class BaselineSpec(BaseModel):
    library: Literal["gliner", "gliner2"]
    model: str
    revision: str
    finetune: bool = False  # few-shot fine-tuning on each training pool (gliner only)
    seeds: list[int] = [0, 1, 2]


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
    formats: Literal["frozen", "others"] | list[FormatName] = "frozen"


class Config(BaseSettings):
    model_config = SettingsConfigDict(toml_file="trice.toml", env_file=".env", extra="ignore")

    openrouter_api_key: SecretStr | None = None
    hf_token: SecretStr | None = None
    layout: Layout
    llm: LLMConfig
    retrieval: RetrievalConfig
    selection: SelectionConfig
    stats: StatsConfig
    sources: dict[str, SourceSpec]
    datasets: dict[str, DatasetSpec]
    models: dict[str, ModelSpec]
    baselines: dict[str, BaselineSpec]
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
