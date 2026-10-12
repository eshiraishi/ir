"""`trice` command line: each command asks the container for components and calls them.

Order of use: data -> rank -> pin -> smoke -> budget -> run S0 -> select-format -> run S1-loo
-> select-k -> run S1..S6 -> audit / probe -> analyze."""

import json
from typing import Annotated, Literal

import typer
from huggingface_hub import HfApi

from trice.container import Container
from trice.settings import Format

app = typer.Typer(no_args_is_help=True, add_completion=False)
Datasets = Annotated[list[str] | None, typer.Option("--dataset", "-d")]
Models = Annotated[list[str] | None, typer.Option("--model", "-m")]


@app.callback()
def main(ctx: typer.Context) -> None:
    ctx.obj = Container()


@app.command()
def data(ctx: typer.Context, datasets: Datasets = None) -> None:
    """Fetch pinned files (hash-checked) and build the HF datasets."""
    c: Container = ctx.obj
    for name in datasets or list(c.settings.datasets):
        built = c.corpus.build(name)
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
    c: Container = ctx.obj
    retriever = c.retriever(stack)
    names = (datasets or list(c.settings.datasets)) + (
        [f"conll2003@{n}" for n in c.settings.selection.pool_sizes] if pools else []
    )
    for name in names:
        for qs in ("test", "loo") if queries == "both" else (queries,):
            typer.echo(f"{stack}/{name}/{qs}: {c.experiment.rank(stack, retriever, name, qs)}")


@app.command()
def pin(ctx: typer.Context, models: Models = None) -> None:
    """Pin one OpenRouter provider per model (or local vLLM) and the tokenizer revision."""
    c: Container = ctx.obj
    pins = {n: e.model_dump() for n, e in c.endpoints.items()}
    for name, spec in c.settings.models.items():
        if not models or name in models:
            pins[name] = c.pinner.pin(spec, HfApi().model_info(spec.hf_repo).sha).model_dump()
            typer.echo(f"{name}: {pins[name]['backend']} {pins[name]['provider']} {pins[name]['quantization']}")
    (c.settings.paths.results / "endpoints.json").write_text(json.dumps(pins, indent=2) + "\n")


@app.command()
def smoke(ctx: typer.Context, model: str) -> None:
    """Check one model end to end; exits non-zero if any check fails."""
    checks = ctx.obj.experiment.smoke(ctx.obj.llm(model, mode="bypass"))
    typer.echo(json.dumps(checks, indent=2))
    raise typer.Exit(0 if all(all(v.values()) for v in checks.values()) else 1)


@app.command()
def budget(
    ctx: typer.Context,
    k: Annotated[list[int], typer.Option(help="Assumed k* (repeat for bounds).")] = [16, 64],  # noqa: B006
    fmt: Annotated[Format, typer.Option("--format", help="Assumed frozen format.")] = "P3",
    studies: Annotated[list[str] | None, typer.Option("--study")] = None,
) -> None:
    """Dry run: count every planned prompt with each model's tokenizer and price it. No calls."""
    c: Container = ctx.obj
    for assumed in k:
        payload = c.budget.estimate(studies or list(c.settings.studies), assumed, fmt)
        (c.settings.paths.results / "budget").mkdir(parents=True, exist_ok=True)
        (c.settings.paths.results / "budget" / f"k{assumed}-{fmt}.json").write_text(
            json.dumps(payload, indent=2) + "\n"
        )
        typer.echo(c.budget.render(payload))


@app.command()
def run(
    ctx: typer.Context,
    study: str,
    models: Models = None,
    datasets: Datasets = None,
    mode: Literal["readwrite", "replay"] = "readwrite",
) -> None:
    """Execute a study (cached: re-running is free and byte-identical)."""
    c: Container = ctx.obj
    runs = c.experiment.plan(study, models=models, datasets=datasets)
    for i, r in enumerate(runs, start=1):
        typer.echo(f"[{i}/{len(runs)}] {c.experiment.execute(r, c.llm(r.model, mode))}")


@app.command("select-format")
def select_format(ctx: typer.Context) -> None:
    """S0 decision: freeze the format with the best train-LOO F1."""
    typer.echo(ctx.obj.experiment.select_format())


@app.command("select-k")
def select_k(ctx: typer.Context, study: str = "S1-loo") -> None:
    """Choose k* per (model, dataset) from the train-LOO curves."""
    for model, by_dataset in ctx.obj.experiment.select_k(study).items():
        typer.echo(f"{model}: " + ", ".join(f"{d}={v['k_star']}" for d, v in by_dataset.items()))


@app.command()
def audit(ctx: typer.Context, model: str, tag: str) -> None:
    """S7: resend the fixed audit prompts live (repeat with a new tag on different days)."""
    typer.echo(json.dumps(ctx.obj.experiment.audit(model, ctx.obj.llm(model, mode="bypass"), tag), indent=2))


@app.command()
def probe(ctx: typer.Context, model: str) -> None:
    """S8: memorization probe on the opening lines of every raw test file."""
    typer.echo(json.dumps(ctx.obj.experiment.probe(model, ctx.obj.llm(model)), indent=2))


@app.command()
def analyze(ctx: typer.Context) -> None:
    """Write every pre-registered table and figure whose runs are complete."""
    typer.echo(", ".join(ctx.obj.analysis.run()) or "nothing complete yet")


@app.command()
def baselines(ctx: typer.Context) -> None:
    """Score the specialist baselines written by envs/baselines."""
    for row in ctx.obj.analysis.baselines():
        typer.echo(f"{row['system']} {row['dataset']}: {row['f1']:.4f}")
