"""From predictions to decisions (S0 format, k*) and pre-registered result tables (S1-S5)."""

import json
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from trice.config import models_config, studies_config
from trice.data.build import label_map
from trice.eval import stats
from trice.eval.metrics import counts_matrix, exact_match, f1_of_sums, gold_pairs
from trice.prompt.formats import FORMATS
from trice.prompt.labels import LabelSet
from trice.runner import load_predictions, predictions_path, queries_for
from trice.select_k import select_k
from trice.studies import Run, expand, frozen_format, k_star_table, results_dir


def run_counts(run: Run) -> np.ndarray:
    """(n_queries, 3) counts aligned with the run's query set (order checked)."""
    queries = queries_for(run)
    preds = load_predictions(run)
    if [p["query_id"] for p in preds] != [q.id for q in queries]:
        raise ValueError(f"{run.slug}: predictions do not align with the query set")
    labels = LabelSet(label_map(run.dataset))
    return counts_matrix(
        [[tuple(e) for e in p["entities"]] for p in preds], [gold_pairs(q, labels) for q in queries]
    )


def f1(run: Run) -> float:
    return float(f1_of_sums(run_counts(run).sum(0)))


def invalid_rate(run: Run) -> float:
    preds = load_predictions(run)
    cand = sum(p["n_candidates"] for p in preds)
    return sum(p["n_invalid"] for p in preds) / cand if cand else 0.0


def _write(name: str, payload: dict | list) -> Path:
    path = results_dir() / "tables" / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


# --------------------------------------------------------------------------- decisions (train only)


def select_format(k: int = 8) -> dict:
    """S0: format with the highest mean LOO micro-F1 over datasets x reference models."""
    table: dict[str, dict[str, float]] = defaultdict(dict)
    for run in expand("S0"):
        table[run.fmt][f"{run.model}/{run.dataset}"] = f1(run)
    means = {fmt: float(np.mean(list(v.values()))) for fmt, v in table.items()}
    winner = max(FORMATS, key=lambda f: (means.get(f, -1.0), -FORMATS.index(f)))
    payload = {"format": winner, "k": k, "mean_f1": means, "per_cell": table}
    (results_dir() / "frozen_format.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def select_all_k(study: str = "S1-loo") -> dict:
    """k* per (model, dataset) from the LOO curves; written to results/k_star.json."""
    grouped: dict[tuple[str, str], dict[int, np.ndarray]] = defaultdict(dict)
    for run in expand(study):
        grouped[(run.model, run.dataset)][run.k] = run_counts(run)
    path = results_dir() / "k_star.json"
    out = json.loads(path.read_text()) if path.exists() else {}
    for (model, dataset), by_k in sorted(grouped.items()):
        out.setdefault(model, {})[dataset] = asdict(select_k(by_k, seed=(model, dataset)))
    path.write_text(json.dumps(out, indent=2) + "\n")
    return out


# --------------------------------------------------------------------------- tables


def _pair(a: Run, b: Run, seed: tuple) -> dict:
    ca, cb = run_counts(a), run_counts(b)
    res = stats.compare(ca, cb, seed=seed)
    return {**asdict(res), "mcnemar_p": stats.mcnemar_exact(exact_match(ca), exact_match(cb))}


def _holm_inplace(rows: Sequence[dict], key: str = "p_value") -> None:
    adjusted = stats.holm([r[key] for r in rows])
    for r, p in zip(rows, adjusted, strict=True):
        r["p_holm"] = float(p)


def table_s1(
    main: str = "small/full", baselines: Sequence[str] = ("random:0", "zero_shot")
) -> dict:
    runs = {(r.model, r.dataset, r.condition): r for r in expand("S1")}
    cells, comparisons = [], {b: [] for b in baselines}
    for (model, dataset, cond), run in sorted(runs.items()):
        cells.append(
            {
                "model": model,
                "dataset": dataset,
                "condition": cond,
                "k": run.k,
                "f1": f1(run),
                "invalid_rate": invalid_rate(run),
            }
        )
        if cond == main:
            for b in baselines:
                row = _pair(run, runs[(model, dataset, b)], seed=(model, dataset, b))
                comparisons[b].append({"model": model, "dataset": dataset, **row})
    per_model = {}
    for b, rows in comparisons.items():
        _holm_inplace(rows)
        by_model = defaultdict(list)
        for r in rows:
            by_model[r["model"]].append(r["delta"])
        per_model[b] = {
            m: {"wilcoxon_p": stats.wilcoxon(d), "mean_delta": float(np.mean(d))}
            for m, d in by_model.items()
        }
    payload = {"cells": cells, "comparisons": comparisons, "across_datasets": per_model}
    _write("S1", payload)
    return payload


def table_friedman(study: str, by: str) -> dict:
    """S2 (by="condition") and S3 (by="fmt"): Friedman/Nemenyi over the varied factor with blocks =
    datasets x reference models, plus Holm-corrected paired tests of TRICE against each
    alternative. Random seeds enter the Friedman test as one averaged treatment, but every seed
    is tested separately in the paired comparisons (a claim must hold against all of them)."""
    alts = list(expand(study))
    ref_models = {r.model for r in alts}
    trice = {
        (r.model, r.dataset): r
        for r in expand("S1")
        if r.condition == "small/full" and r.model in ref_models
    }
    if by == "condition":
        alts += [r for r in expand("S1") if r.condition == "random:0" and r.model in ref_models]

    def treatment(r: Run) -> str:
        name = getattr(r, by)
        return "random" if name.startswith("random:") else name

    blocks: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in [*trice.values(), *alts]:
        blocks[(r.model, r.dataset)][treatment(r)].append(f1(r))
    names = sorted({n for b in blocks.values() for n in b})
    scores = np.array([[float(np.mean(b[n])) for n in names] for b in blocks.values()])
    fr = stats.friedman_nemenyi(scores, names)

    pairs = []
    for r in sorted(alts):
        ref = trice[(r.model, r.dataset)]
        pairs.append(
            {
                "model": r.model,
                "dataset": r.dataset,
                "vs": getattr(r, by),
                **_pair(ref, r, (study, r.slug)),
            }
        )
    _holm_inplace(pairs)
    payload = {"friedman": asdict(fr), "names": names, "pairs": pairs}
    _write(study, payload)
    return payload


def table_s4() -> dict:
    """Regret of the train-chosen k* against the oracle (best-on-test) k, per (model, dataset),
    and the rank agreement between the LOO curve and the test curve."""
    grid: dict[tuple[str, str], dict[int, float]] = defaultdict(dict)
    for r in expand("S4"):
        grid[(r.model, r.dataset)][r.k] = f1(r)
    loo = json.loads((results_dir() / "k_star.json").read_text())
    rows = []
    for (model, dataset), curve in sorted(grid.items()):
        sel = loo[model][dataset]
        k_star = sel["k_star"]
        k_oracle = max(curve, key=lambda k: (curve[k], -k))
        common = [k for k in sel["ks"] if k in curve]
        rho = spearmanr([sel["f1"][sel["ks"].index(k)] for k in common], [curve[k] for k in common])
        rows.append(
            {
                "model": model,
                "dataset": dataset,
                "k_star": k_star,
                "k_oracle": k_oracle,
                "f1_k_star": curve[k_star],
                "f1_oracle": curve[k_oracle],
                "regret": curve[k_oracle] - curve[k_star],
                "spearman_loo_vs_test": float(rho.statistic),
            }
        )
    payload = {"rows": rows, "mean_regret": float(np.mean([r["regret"] for r in rows]))}
    _write("S4", payload)
    return payload


def table_s5() -> dict:
    """Scaling: Page's L trend per ladder on F1(full) and on the retrieval gain over random."""
    models = models_config()
    s1 = {(r.model, r.dataset, r.condition): r for r in expand("S1")}
    datasets = sorted({d for _, d, _ in s1})
    out = {}
    for ladder in sorted({m.ladder for m in models.values()}):
        names = sorted(
            (n for n, m in models.items() if m.ladder == ladder), key=lambda n: models[n].params_b
        )
        if len(names) < 2:
            continue
        full = np.array([[f1(s1[(m, d, "small/full")]) for m in names] for d in datasets])
        rand = np.array([[f1(s1[(m, d, "random:0")]) for m in names] for d in datasets])
        sizes = [models[n].params_b for n in names]
        out[ladder] = {
            "models": names,
            "f1_full": asdict(stats.page_trend(full)),
            "gain_vs_random": asdict(stats.page_trend(full - rand)),
            "kendall_tau_gain": {
                d: stats.kendall_tau(sizes, (full - rand)[i])[0] for i, d in enumerate(datasets)
            },
            "invalid_rate": {
                m: float(np.mean([invalid_rate(s1[(m, d, "small/full")]) for d in datasets]))
                for m in names
            },
            "k_star": {m: k_star_table().get(m, {}) for m in names},
        }
    _write("S5", out)
    return out


def available(study: str) -> bool:
    try:
        return all(predictions_path(r).exists() for r in expand(study))
    except RuntimeError:
        return False


def analyze() -> dict[str, Path]:
    """Write every table whose runs are complete."""
    written = {}
    if available("S1"):
        table_s1()
        written["S1"] = results_dir() / "tables" / "S1.json"
    if available("S2") and available("S1"):
        table_friedman("S2", "condition")
        written["S2"] = results_dir() / "tables" / "S2.json"
    if available("S3") and available("S1"):
        table_friedman("S3", "fmt")
        written["S3"] = results_dir() / "tables" / "S3.json"
    if available("S4"):
        table_s4()
        written["S4"] = results_dir() / "tables" / "S4.json"
    if available("S1"):
        table_s5()
        written["S5"] = results_dir() / "tables" / "S5.json"
    return written


__all__ = ["analyze", "frozen_format", "select_all_k", "select_format", "studies_config"]
