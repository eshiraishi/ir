"""`baselines predict <tag>` (zero-shot) and `baselines finetune <tag>` (few-shot, seeds from trice.toml).

Tags are the [baselines.<tag>] entries of trice.toml; run from the repository root."""

from typing import Annotated

import typer

from baselines.benchmark import Benchmark
from baselines.config import Config

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
