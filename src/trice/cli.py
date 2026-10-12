"""`trice` command line. Stages follow the milestones in the README:
data -> rank -> endpoints/smoke -> budget -> run S0 -> select-format -> run S1-loo -> select-k
-> run S1..S6 -> audit/contamination -> analyze."""

import json
import logging
from typing import Annotated, Literal

import typer

from trice.config import datasets_config, models_config, retrieval_config, studies_config

app = typer.Typer(no_args_is_help=True, add_completion=False)
data_app = typer.Typer(no_args_is_help=True)
app.add_typer(data_app, name="data", help="Fetch pinned sources and build canonical splits.")

Datasets = Annotated[
    list[str] | None, typer.Option("--dataset", "-d", help="Restrict to datasets.")
]
Models = Annotated[list[str] | None, typer.Option("--model", "-m", help="Restrict to models.")]


@app.callback()
def _main(verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING, format="%(levelname)s %(message)s"
    )


@data_app.command("build")
def data_build(datasets: Datasets = None) -> None:
    """Fetch (verifying SHA-256) and write data/canonical/* plus data/manifest.json."""
    from trice.data.build import build_all

    manifest = build_all(datasets)
    for name in datasets or list(datasets_config().datasets):
        s = manifest[name]["splits"]
        typer.echo(
            f"{name:22s} train={s['train']['sentences']:6d} test={s['test']['sentences']:5d}"
        )


@app.command()
def rank(
    stack: Annotated[
        str, typer.Option(help="Retrieval stack from configs/retrieval.yaml.")
    ] = "small",
    queries: Annotated[Literal["test", "loo", "both"], typer.Option(help="Query set.")] = "both",
    datasets: Datasets = None,
    pools: Annotated[bool, typer.Option(help="Also rank the S6 nested CoNLL pools.")] = False,
) -> None:
    """Compute and store rankings (run once; later stages only read them)."""
    from trice.retrieval.pipeline import rank as do_rank

    if stack not in retrieval_config().stacks:
        raise typer.BadParameter(f"unknown stack {stack}")
    names = datasets or list(datasets_config().datasets)
    if pools:
        names += [f"conll2003@{n}" for n in studies_config().pool_sizes]
    for ds in names:
        query_sets: tuple[Literal["test", "loo"], ...] = (
            ("test", "loo") if queries == "both" else (queries,)
        )
        for qs in query_sets:
            digest = do_rank(stack, ds, qs, studies_config().loo_queries)
            typer.echo(f"{stack}/{ds}/{qs}: {digest[:16]}")


@app.command()
def endpoints(models: Models = None) -> None:
    """Pin one OpenRouter provider per model (or local vLLM) and the HF tokenizer revision."""
    from huggingface_hub import HfApi

    from trice.llm.endpoints import fetch_endpoints, lock_path, pin_for

    path = lock_path()
    lock = json.loads(path.read_text()) if path.exists() else {}
    api = HfApi()
    for name, spec in models_config().items():
        if models and name not in models:
            continue
        try:
            eps = fetch_endpoints(spec.openrouter)
        except Exception as exc:  # model not served on OpenRouter
            typer.echo(f"{name}: no OpenRouter endpoints ({exc}); pinning local")
            eps = []
        entry = pin_for(spec, eps)
        entry["hf_revision"] = spec.hf_revision or api.model_info(spec.hf_repo).sha
        lock[name] = entry
        typer.echo(
            f"{name}: {entry['backend']} {entry.get('provider') or ''} "
            f"{entry.get('quantization') or ''}"
        )
    path.write_text(json.dumps(dict(sorted(lock.items())), indent=2) + "\n")


def _client(
    model: str,
    mode: Literal["readwrite", "replay", "bypass"] = "readwrite",
    local_url: str | None = None,
):
    from trice.config import repo_root
    from trice.llm.cache import ResponseCache
    from trice.llm.client import Backend, ChatClient
    from trice.llm.endpoints import load_pins

    pin = load_pins()[model]
    backend = (
        Backend.openrouter()
        if pin.backend == "openrouter"
        else Backend.local(local_url or "http://127.0.0.1:8000/v1")
    )
    cache = ResponseCache(repo_root() / "cache" / "llm.sqlite")
    return ChatClient(backend, cache, mode=mode), pin


@app.command()
def smoke(
    model: Annotated[str, typer.Option("--model", "-m")], local_url: str | None = None
) -> None:
    """Check one model end to end: every format parses, thinking is off, the pin is honoured."""
    from trice.data.build import label_map, load_split
    from trice.llm.client import build_body
    from trice.prompt.formats import FORMATS, build_prompt
    from trice.prompt.labels import LabelSet
    from trice.prompt.parse import parse

    client, pin = _client(model, mode="bypass", local_url=local_url)
    labels = LabelSet(label_map("crossner_ai"))
    pool, query = load_split("crossner_ai", "train")[:4], load_split("crossner_ai", "test")[0]
    ok = True
    for fmt in FORMATS:
        c = client.complete(
            build_body(pin, build_prompt(fmt, labels, pool, query.text)), audit_tag="smoke"
        )
        p = parse(fmt, c.text, query.text, labels)
        checks = {
            "non_empty": bool(c.text.strip()),
            "parses": not p.malformed,
            "no_thinking": not c.reasoning_tokens,
            "provider_pinned": pin.backend == "local" or c.provider == pin.provider,
        }
        ok &= all(checks.values())
        typer.echo(
            f"{fmt}: {checks} entities={len(p.entities)} invalid={p.n_invalid} "
            f"provider={c.provider}"
        )
    raise typer.Exit(0 if ok else 1)


@app.command()
def budget(
    k: Annotated[list[int], typer.Option(help="Assumed k* (repeat for bounds).")] = [16, 64],  # noqa: B006
    fmt: Annotated[
        Literal["P0", "P1", "P2", "P3"],
        typer.Option("--format", help="Assumed frozen format (P3 is the costliest)."),
    ] = "P3",
    studies: Annotated[list[str] | None, typer.Option("--study")] = None,
    embed_price: Annotated[
        float, typer.Option(help="USD/token, large-stack embeddings.")
    ] = 0.01e-6,
    rerank_price: Annotated[float, typer.Option(help="USD/token, large-stack reranker.")] = 0.20e-6,
) -> None:
    """Dry run: count every planned prompt with each model's tokenizer and price it. No calls."""
    from trice.budget import TokenCounter, estimate, markdown, retrieval_api_estimate, write_report
    from trice.llm.endpoints import load_pins, lock_path

    pins = load_pins() if lock_path().exists() else None
    names = studies or list(studies_config().studies)
    for kk in k:
        payload = estimate(names, TokenCounter.for_model, pins, assume_k=kk, assume_format=fmt)
        if "S2" in names:
            counter = TokenCounter.for_model(
                next(n for n, m in models_config().items() if m.reference)
            )
            payload["retrieval_api"] = retrieval_api_estimate(
                counter, datasets_config().datasets, embed_price, rerank_price
            )
        path = write_report(f"k{kk}-{fmt}", payload)
        typer.echo(markdown(payload))
        if "retrieval_api" in payload:
            r = payload["retrieval_api"]
            typer.echo(
                f"Retrieval API (S2 large stack): {r['embedding_tokens']:,} embedding + "
                f"{r['rerank_tokens']:,} rerank tokens, ~${r['cost_usd']:,.2f}"
            )
        typer.echo(f"-> {path}\n")


@app.command()
def run(
    study: Annotated[str, typer.Option("--study", "-s")],
    models: Models = None,
    datasets: Datasets = None,
    workers: int = 16,
    mode: Annotated[
        Literal["readwrite", "replay"], typer.Option(help="replay = cache only, no calls.")
    ] = "readwrite",
    local_url: str | None = None,
) -> None:
    """Execute a study's runs (cached; re-running is free and byte-identical)."""
    from trice.runner import execute
    from trice.studies import expand

    runs = list(expand(study, models=models, datasets=datasets))
    clients: dict = {}
    for i, r in enumerate(runs, start=1):
        if r.model not in clients:
            clients[r.model] = _client(r.model, mode=mode, local_url=local_url)
        client, pin = clients[r.model]
        path = execute(r, client, pin, workers=workers)
        typer.echo(f"[{i}/{len(runs)}] {r.slug} -> {path.name}")


@app.command("select-format")
def select_format_cmd() -> None:
    """S0 decision: freeze the format with the best train-LOO F1 (results/frozen_format.json)."""
    from trice.analysis import select_format

    res = select_format()
    typer.echo(json.dumps({"format": res["format"], "mean_f1": res["mean_f1"]}, indent=2))


@app.command("select-k")
def select_k_cmd(study: str = "S1-loo") -> None:
    """Choose k* per (model, dataset) from the LOO curves (results/k_star.json)."""
    from trice.analysis import select_all_k

    for model, by_ds in select_all_k(study).items():
        typer.echo(f"{model}: " + ", ".join(f"{d}={v['k_star']}" for d, v in by_ds.items()))


@app.command()
def audit(
    model: Annotated[str, typer.Option("--model", "-m")], tag: str, local_url: str | None = None
) -> None:
    """S7: resend the fixed audit prompts live under ``tag`` (repeat on different days)."""
    from trice.audit import audit_summary, run_audit

    client, pin = _client(model, mode="bypass", local_url=local_url)
    typer.echo(run_audit(model, client, pin, tag))
    typer.echo(json.dumps(audit_summary(model), indent=2))


@app.command()
def contamination(
    model: Annotated[str, typer.Option("--model", "-m")], local_url: str | None = None
) -> None:
    """S8: memorization probe on the opening lines of every raw test file."""
    from trice.audit import contamination_probe

    client, pin = _client(model, mode="readwrite", local_url=local_url)
    typer.echo(json.dumps(contamination_probe(model, client, pin), indent=2))


@app.command()
def analyze() -> None:
    """Write every pre-registered table whose runs are complete (results/tables/)."""
    from trice.analysis import analyze as do_analyze

    for name, path in do_analyze().items():
        typer.echo(f"{name}: {path}")


@app.command("score-baselines")
def score_baselines() -> None:
    """Score specialist baselines (envs/baselines output) with the shared scorer."""
    from trice.baselines import score_all

    for tag, by_ds in score_all().items():
        typer.echo(f"{tag}: " + ", ".join(f"{d}={v:.4f}" for d, v in by_ds.items()))
