"""`trice` command line: every command opens the composed app and calls its components.

Order of use: data -> rank -> pin -> smoke -> budget -> run S0 -> select-format -> run S1-loo
-> select-k -> run S1..S6 -> audit / probe -> analyze."""

from typing import Annotated, Literal

import typer
from huggingface_hub import HfApi

from trice.compose import App, compose
from trice.config import Config
from trice.llm import Endpoint, EndpointTable
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
    for name in list(trice.config.datasets) if datasets is None else datasets:
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
    names = list(trice.config.datasets) if datasets is None else datasets
    if pools:
        for size in trice.config.selection.pool_sizes:
            names.append(f"conll2003@{size}")
    for name in names:
        if queries in ["test", "both"]:
            digest = trice.runner.store_rankings(stack, trice.retrievers[stack], name, "test")
            typer.echo(f"{stack}/{name}/test: {digest}")
        if queries in ["loo", "both"]:
            digest = trice.runner.store_rankings(stack, trice.retrievers[stack], name, "loo")
            typer.echo(f"{stack}/{name}/loo: {digest}")


@app.command()
def pin(ctx: typer.Context, models: Models = None) -> None:
    """Pin one OpenRouter provider per model (or local vLLM) and the tokenizer revision."""
    trice: App = ctx.obj
    pinned: list[Endpoint] = []
    for name, spec in trice.config.models.items():
        if models is None or name in models:
            revision = HfApi().model_info(spec.hf_repo).sha
            if revision is None:
                raise ValueError(f"{spec.hf_repo}: Hugging Face returned no commit")
            pinned.append(trice.pinner.pin(name, spec, revision))
            typer.echo(f"{name}: {pinned[-1].backend} {pinned[-1].provider} {pinned[-1].quantization}")
    kept = [e for e in trice.endpoints if e.name not in {p.name for p in pinned}]
    write_text(trice.config.layout.endpoints, EndpointTable(kept + pinned).model_dump_json(indent=2))


@app.command()
def smoke(ctx: typer.Context, model: str) -> None:
    """Check one model end to end; exits non-zero if any check fails."""
    trice: App = ctx.obj
    checks = trice.runner.run_smoke(trice.llms[model].model_copy(update={"mode": "bypass"}))
    typer.echo("[" + ",".join(c.model_dump_json() for c in checks) + "]")
    raise typer.Exit(0 if all(c.answered and c.parses and c.no_thinking and c.pinned for c in checks) else 1)


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
        report = trice.budget.estimate(list(trice.config.studies) if studies is None else studies, assumed, fmt)
        write_text(trice.config.layout.budget.format(k=assumed, format=fmt), report.model_dump_json(indent=2))
        typer.echo(trice.budget.render(report))


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
    for choice in ctx.obj.runner.select_k(study).root:
        typer.echo(f"{choice.model} {choice.dataset}: k*={choice.selection.k_star} ({choice.selection.rule})")


@app.command()
def audit(ctx: typer.Context, model: str, tag: str) -> None:
    """S7: resend the fixed audit prompts live (repeat with a new tag on different days)."""
    trice: App = ctx.obj
    audited = trice.runner.audit(model, trice.llms[model].model_copy(update={"mode": "bypass"}), tag)
    typer.echo(audited.model_dump_json(indent=2))


@app.command()
def probe(ctx: typer.Context, model: str) -> None:
    """S8: memorization probe on the opening lines of every raw test file."""
    trice: App = ctx.obj
    for score in trice.runner.probe(model, trice.llms[model]):
        typer.echo(f"{score.dataset}: {score.share:.3f}")


@app.command()
def analyze(ctx: typer.Context) -> None:
    """Write every pre-registered table and figure whose runs are complete."""
    done = ctx.obj.analyzer.run_all()
    typer.echo(", ".join(done) if len(done) > 0 else "nothing complete yet")


@app.command()
def baselines(ctx: typer.Context) -> None:
    """Score the specialist baselines written by envs/baselines."""
    for row in ctx.obj.analyzer.score_baselines():
        typer.echo(f"{row.system} {row.dataset}: {row.f1:.4f}")
