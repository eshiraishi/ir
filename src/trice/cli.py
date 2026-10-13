"""`trice` command line: every command opens the composed app and calls its components.

Order of use: data -> rank -> pin -> smoke -> budget -> run S0 -> select-format -> run S1-loo
-> select-k -> run S1..S6 -> audit / probe -> analyze."""

import json
from typing import Annotated, Literal

import typer
from huggingface_hub import HfApi

from trice.compose import App, compose
from trice.config import Config
from trice.schema import FormatName, write_text

app = typer.Typer(no_args_is_help=True, add_completion=False)
Datasets = Annotated[list[str] | None, typer.Option("--dataset", "-d")]
Models = Annotated[list[str] | None, typer.Option("--model", "-m")]


@app.callback()
def main(ctx: typer.Context) -> None:
    ctx.obj = ctx.with_resource(compose(Config()))


@app.command()
def data(ctx: typer.Context, datasets: Datasets = None) -> None:
    """Fetch pinned files (hash-checked) and build the HF datasets."""
    trice: App = ctx.obj
    for name in datasets or list(trice.config.datasets):
        built = trice.corpus.build(name)
        typer.echo(f"{name}: train={built['train'].num_rows} test={built['test'].num_rows}")


@app.command()
def rank(
    ctx: typer.Context,
    stack: str = "small",
    queries: Literal["test", "loo", "both"] = "both",
    datasets: Datasets = None,
    pools: bool = False,
) -> None:
    """Compute and store rankings once; later stages only read them."""
    trice: App = ctx.obj
    names = (datasets or list(trice.config.datasets)) + (
        [f"conll2003@{n}" for n in trice.config.selection.pool_sizes] if pools else []
    )
    for name in names:
        for qs in ("test", "loo") if queries == "both" else (queries,):
            digest = trice.runner.store_rankings(stack, trice.retrievers[stack], name, qs)
            typer.echo(f"{stack}/{name}/{qs}: {digest}")


@app.command()
def pin(ctx: typer.Context, models: Models = None) -> None:
    """Pin one OpenRouter provider per model (or local vLLM) and the tokenizer revision."""
    trice: App = ctx.obj
    pins = {name: endpoint.model_dump() for name, endpoint in trice.endpoints.items()}
    for name, spec in trice.config.models.items():
        if not models or name in models:
            pins[name] = trice.pinner.pin(spec, HfApi().model_info(spec.hf_repo).sha).model_dump()
            typer.echo(f"{name}: {pins[name]['backend']} {pins[name]['provider']} {pins[name]['quantization']}")
    write_text(trice.config.layout.endpoints, json.dumps(pins, indent=2) + "\n")


@app.command()
def smoke(ctx: typer.Context, model: str) -> None:
    """Check one model end to end; exits non-zero if any check fails."""
    trice: App = ctx.obj
    checks = trice.runner.run_smoke(trice.llms[model].model_copy(update={"mode": "bypass"}))
    typer.echo(json.dumps(checks, indent=2))
    raise typer.Exit(0 if all(all(v.values()) for v in checks.values()) else 1)


@app.command()
def budget(
    ctx: typer.Context,
    k: Annotated[list[int], typer.Option(help="Assumed k* (repeat for bounds).")] = [16, 64],  # noqa: B006
    fmt: Annotated[FormatName, typer.Option("--format", help="Assumed frozen format.")] = "P3",
    studies: Annotated[list[str] | None, typer.Option("--study")] = None,
) -> None:
    """Dry run: count every planned prompt with each model's tokenizer and price it. No calls."""
    trice: App = ctx.obj
    for assumed in k:
        payload = trice.budget.estimate(studies or list(trice.config.studies), assumed, fmt)
        write_text(trice.config.layout.budget / f"k{assumed}-{fmt}.json", json.dumps(payload, indent=2) + "\n")
        typer.echo(trice.budget.render(payload))


@app.command()
def run(
    ctx: typer.Context,
    study: str,
    models: Models = None,
    datasets: Datasets = None,
    mode: Literal["readwrite", "replay"] = "readwrite",
) -> None:
    """Execute a study (cached: re-running is free and byte-identical)."""
    trice: App = ctx.obj
    runs = trice.runner.expand_plan(study, models=models, datasets=datasets)
    for i, r in enumerate(runs, start=1):
        trice.runner.execute(r, trice.llms[r.model].model_copy(update={"mode": mode}))
        typer.echo(f"[{i}/{len(runs)}] {r.slug}")


@app.command("select-format")
def select_format(ctx: typer.Context) -> None:
    """S0 decision: freeze the format with the best train-LOO F1."""
    typer.echo(ctx.obj.runner.select_format())


@app.command("select-k")
def select_k(ctx: typer.Context, study: str = "S1-loo") -> None:
    """Choose k* per (model, dataset) from the train-LOO curves."""
    for model, by_dataset in ctx.obj.runner.select_k(study).items():
        typer.echo(f"{model}: " + ", ".join(f"{d}={v['k_star']}" for d, v in by_dataset.items()))


@app.command()
def audit(ctx: typer.Context, model: str, tag: str) -> None:
    """S7: resend the fixed audit prompts live (repeat with a new tag on different days)."""
    trice: App = ctx.obj
    llm = trice.llms[model].model_copy(update={"mode": "bypass"})
    typer.echo(json.dumps(trice.runner.audit(model, llm, tag), indent=2))


@app.command()
def probe(ctx: typer.Context, model: str) -> None:
    """S8: memorization probe on the opening lines of every raw test file."""
    trice: App = ctx.obj
    typer.echo(json.dumps(trice.runner.probe(model, trice.llms[model]), indent=2))


@app.command()
def analyze(ctx: typer.Context) -> None:
    """Write every pre-registered table and figure whose runs are complete."""
    typer.echo(", ".join(ctx.obj.analyzer.run_all()) or "nothing complete yet")


@app.command()
def baselines(ctx: typer.Context) -> None:
    """Score the specialist baselines written by envs/baselines."""
    for row in ctx.obj.analyzer.score_baselines():
        typer.echo(f"{row['system']} {row['dataset']}: {row['f1']:.4f}")
