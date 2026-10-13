"""The baselines read the same trice.toml as the core: its datasets, layout and [baselines]."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, RootModel
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict, TomlConfigSettingsSource


class PathTemplate(RootModel[str]):
    """A relative path with ``{placeholders}`` (the same type as the core's, which this image cannot import)."""

    model_config = ConfigDict(frozen=True)

    def fill(self, **fields: object) -> Path:
        return Path(self.root.format(**fields))

    def prepare(self, **fields: object) -> Path:
        """The path to write to, with its directory created."""
        path = self.fill(**fields)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path


class Layout(BaseModel):
    canonical: PathTemplate
    baselines: PathTemplate
    training: PathTemplate


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
