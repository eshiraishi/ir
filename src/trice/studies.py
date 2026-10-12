"""Expand study definitions (configs/studies.yaml) into concrete runs.

A :class:`Run` is one (model, dataset, query set, condition, format, k) cell; the runner sends one
request per query of its query set. The budget expands exactly the same runs, substituting
assumed values for k* and the frozen format when those are not decided yet.
"""

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from trice.config import datasets_config, models_config, repo_root, studies_config
from trice.prompt.formats import FORMATS, Format
from trice.retrieval.pipeline import QuerySet


@dataclass(frozen=True, order=True)
class Run:
    model: str
    dataset: str
    queries: QuerySet
    condition: str
    fmt: Format
    k: int

    @property
    def slug(self) -> str:
        cond = self.condition.replace("/", "-").replace(":", "")
        return f"{self.model}/{self.dataset}/{self.queries}/{cond}/{self.fmt}/k{self.k}"


def results_dir() -> Path:
    return repo_root() / "results"


def frozen_format() -> Format | None:
    path = results_dir() / "frozen_format.json"
    return json.loads(path.read_text())["format"] if path.exists() else None


def k_star_table() -> dict[str, dict[str, int]]:
    path = results_dir() / "k_star.json"
    if not path.exists():
        return {}
    return {
        m: {d: v["k_star"] for d, v in ds.items()} for m, ds in json.loads(path.read_text()).items()
    }


def _models(sel) -> list[str]:
    models = models_config()
    if sel == "all":
        return list(models)
    if sel == "reference":
        return [n for n, m in models.items() if m.reference]
    unknown = set(sel) - models.keys()
    if unknown:
        raise ValueError(f"unknown models {sorted(unknown)}")
    return list(sel)


def _datasets(sel) -> list[str]:
    return list(datasets_config().datasets) if sel == "all" else list(sel)


def max_k(dataset: str) -> int:
    """Largest k a leave-one-out query can be given: the pool size minus the query itself."""
    base, _, n = dataset.partition("@")
    size = int(n) if n else datasets_config().datasets[base].expected["train"]
    return size - 1


def expand(
    study: str,
    *,
    assume_k: int | None = None,
    assume_format: Format | None = None,
    models: list[str] | None = None,
    datasets: list[str] | None = None,
) -> Iterator[Run]:
    """Yield the runs of ``study``. ``k_star``/``frozen`` resolve from results/ unless the
    corresponding ``assume_*`` override is given (budgeting before those decisions exist)."""
    cfg = studies_config()
    spec = cfg.studies[study]
    frozen = assume_format or frozen_format()
    if spec.formats == "frozen" or spec.formats == "others":
        if frozen is None:
            raise RuntimeError(f"{study}: no frozen format yet (run S0 and `trice select-format`)")
        fmts = [frozen] if spec.formats == "frozen" else [f for f in FORMATS if f != frozen]
    else:
        fmts = list(spec.formats)
    k_table = {} if assume_k is not None else k_star_table()

    for model in _models(spec.models):
        if models and model not in models:
            continue
        for dataset in _datasets(spec.datasets):
            if datasets and dataset not in datasets:
                continue
            if spec.k == "grid":
                ks = [k for k in cfg.k_grid if k <= max_k(dataset)]
            elif spec.k == "k_star":
                if assume_k is not None:
                    ks = [assume_k]
                elif dataset in k_table.get(model, {}):
                    ks = [k_table[model][dataset]]
                else:
                    raise RuntimeError(
                        f"{study}: no k* for {model}/{dataset} (run `trice select-k`)"
                    )
            else:
                ks = [spec.k]
            for condition in spec.conditions:
                for fmt in fmts:
                    for k in ks:
                        yield Run(
                            model,
                            dataset,
                            spec.queries,
                            condition,
                            fmt,
                            0 if condition == "zero_shot" else k,
                        )
