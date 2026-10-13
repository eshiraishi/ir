"""Running studies: plan runs, store rankings, execute runs, and make the train-only decisions.
Wires core/ (retrieval, formats, selection), the LLM and the scorer together."""

import hashlib
import json
from collections import defaultdict
from collections.abc import Generator
from contextlib import contextmanager
from typing import TextIO

import numpy as np
from datasets import Dataset
from jinja2 import Environment
from pydantic import BaseModel, ConfigDict

from trice.config import Config
from trice.core import selection
from trice.core.formats import Format
from trice.core.retrieval import Ranking, Retriever, select_demonstrations
from trice.data import Corpus
from trice.evaluation.inference import Inferrer
from trice.evaluation.scoring import Evaluator
from trice.llm import LLM, Completer
from trice.schema import Condition, Example, FormatName, Prediction, Prompt, QuerySet, Run, derive_seed, write_text


class Runner(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    config: Config
    corpus: Corpus
    formats: dict[FormatName, Format]
    env: Environment
    evaluator: Evaluator
    inferrer: Inferrer

    # ----------------------------------------------------------------------- planning

    def load_frozen_format(self) -> FormatName | None:
        path = self.config.layout.frozen_format
        return json.loads(path.read_text())["format"] if path.exists() else None

    def load_k_star(self) -> dict[str, dict[str, int]]:
        path = self.config.layout.k_star
        table = json.loads(path.read_text()) if path.exists() else {}
        return {m: {d: v["k_star"] for d, v in ds.items()} for m, ds in table.items()}

    def resolve_formats(self, which: list[FormatName] | str, assume: FormatName | None, study: str) -> list[FormatName]:
        if isinstance(which, list):
            return which
        frozen = assume or self.load_frozen_format()
        if frozen is None:
            raise RuntimeError(f"{study}: no frozen format yet (run S0, then `trice select-format`)")
        return [frozen] if which == "frozen" else [f for f in self.formats if f != frozen]

    def expand_plan(
        self,
        study: str,
        assume_k: int | None = None,
        assume_format: FormatName | None = None,
        models: list[str] | None = None,
        datasets: list[str] | None = None,
    ) -> list[Run]:
        """Runs of a study. ``assume_*`` stand in for k* and the frozen format before they exist."""
        spec, k_star = self.config.studies[study], self.load_k_star()
        reference = [n for n, m in self.config.models.items() if m.reference]
        chosen = {"all": list(self.config.models), "reference": reference}.get(str(spec.models), spec.models)
        names = list(self.config.datasets) if spec.datasets == "all" else spec.datasets
        runs = []
        for model in [m for m in chosen if not models or m in models]:
            for dataset in [d for d in names if not datasets or d in datasets]:
                if spec.k == "grid":
                    ks = [k for k in self.config.selection.k_grid if k < self.corpus.count_pool(dataset)]
                else:
                    ks = (
                        [assume_k if assume_k is not None else k_star[model][dataset]]
                        if spec.k == "k_star"
                        else [spec.k]
                    )
                runs += [
                    Run(
                        model=model,
                        dataset=dataset,
                        queries=spec.queries,
                        condition=c,
                        fmt=f,
                        k=0 if c == "zero_shot" else k,
                    )
                    for c in spec.conditions
                    for f in self.resolve_formats(spec.formats, assume_format, study)
                    for k in ks
                ]
        return runs

    def select_queries(self, dataset: str, queries: QuerySet) -> list[Example]:
        """Test split, or the LOO query set: the whole pool when small (CrossNER), else a seeded sample."""
        if queries == "test":
            return self.corpus.load(dataset, "test")
        pool = sorted(self.corpus.load(dataset, "train"), key=lambda e: e.id)
        n = self.config.selection.loo_queries
        if len(pool) <= n:
            return pool
        rng = np.random.default_rng(derive_seed("loo", dataset))
        return [pool[i] for i in sorted(rng.choice(len(pool), n, replace=False))]

    # ----------------------------------------------------------------------- rankings

    def store_rankings(self, stack: str, retriever: Retriever, dataset: str, queries: QuerySet) -> str:
        """Compute once and store as a Dataset; later stages only read it. Returns the SHA-256 of the rankings."""
        qs = self.select_queries(dataset, queries)
        ranked = retriever.rank(qs, self.corpus.load(dataset, "train"), loo=queries == "loo")
        rows = [
            {"query_id": q.id, "stage": s, "ids": ids} for q, r in zip(qs, ranked, strict=True) for s, ids in r.items()
        ]
        Dataset.from_list(rows).save_to_disk(self.config.layout.rankings / stack / dataset / queries)
        return hashlib.sha256(json.dumps(rows).encode()).hexdigest()

    def load_rankings(self, stack: str, dataset: str, queries: QuerySet) -> dict[str, Ranking]:
        out: dict[str, Ranking] = defaultdict(dict)
        for row in Dataset.load_from_disk(self.config.layout.rankings / stack / dataset / queries):
            out[row["query_id"]][row["stage"]] = row["ids"]
        return out

    # ----------------------------------------------------------------------- execution

    def render_prompts(self, run: Run) -> list[tuple[Example, Prompt]]:
        condition, pool = Condition.parse(run.condition), self.corpus.load(run.dataset, "train")
        labels, fmt = self.corpus.load_labels(run.dataset), self.formats[run.fmt]
        ranked = self.load_rankings(condition.stack, run.dataset, run.queries) if condition.kind == "retrieval" else {}
        return [
            (
                q,
                fmt.render_prompt(
                    labels, select_demonstrations(condition, run.k, run.dataset, q, pool, ranked.get(q.id)), q
                ),
            )
            for q in self.select_queries(run.dataset, run.queries)
        ]

    @contextmanager
    def recording(self, run: Run) -> Generator[TextIO]:
        """Predictions of a run are written to a partial file that replaces the real one only on
        success, so a crash never leaves a truncated run that looks complete."""
        path = self.config.layout.runs / f"{run.slug}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(".partial")
        with partial.open("w") as handle:
            yield handle
        partial.replace(path)

    def execute(self, run: Run, llm: Completer) -> None:
        pairs, labels, fmt = self.render_prompts(run), self.corpus.load_labels(run.dataset), self.formats[run.fmt]
        completions = llm.complete_many([p for _, p in pairs])
        with self.recording(run) as handle:
            for (query, _), completion in zip(pairs, completions, strict=True):
                parsed = fmt.parse_output(labels, completion.text, query)
                prediction = Prediction(
                    query_id=query.id, output=completion.text, completion=completion, **parsed.model_dump()
                )
                handle.write(prediction.model_dump_json() + "\n")

    def load_predictions(self, run: Run) -> list[Prediction]:
        path = self.config.layout.runs / f"{run.slug}.jsonl"
        return [Prediction.model_validate_json(line) for line in path.read_text().splitlines()]

    def count_matches(self, run: Run) -> np.ndarray:
        preds, gold = self.load_predictions(run), self.select_queries(run.dataset, run.queries)
        if [p.query_id for p in preds] != [q.id for q in gold]:
            raise ValueError(f"{run.slug}: predictions do not align with the query set")
        return self.evaluator.count_matches([p.tags for p in preds], [q.tags for q in gold])

    def score_f1(self, run: Run) -> float:
        return float(self.evaluator.compute_f1(self.count_matches(run).sum(0)))

    # ----------------------------------------------------------------------- train-only decisions

    def select_format(self) -> FormatName:
        """S0: best mean LOO F1 over datasets x reference models, frozen for every model."""
        by_format: dict[FormatName, list[float]] = defaultdict(list)
        for run in self.expand_plan("S0"):
            by_format[run.fmt].append(self.score_f1(run))
        means = {fmt: float(np.mean(v)) for fmt, v in by_format.items()}
        winner = selection.select_format(means)
        write_text(self.config.layout.frozen_format, json.dumps({"format": winner, "mean_f1": means}, indent=2))
        return winner

    def select_k(self, study: str = "S1-loo") -> dict:
        """k* per (model, dataset) from the LOO curves, with the 1-SE rule for robustness."""
        curves: dict[tuple[str, str], dict[int, Run]] = defaultdict(dict)
        for run in self.expand_plan(study):
            curves[(run.model, run.dataset)][run.k] = run
        path = self.config.layout.k_star
        table = json.loads(path.read_text()) if path.exists() else {}
        for (model, dataset), runs in sorted(curves.items()):
            choice = selection.select_k({k: self.score_f1(r) for k, r in runs.items()})
            best = runs[choice.ks[int(np.argmax(choice.f1))]]
            error = self.inferrer.bootstrap_error(self.count_matches(best), model, dataset)
            one_se = next(k for k, f in zip(choice.ks, choice.f1, strict=True) if f >= max(choice.f1) - error)
            table.setdefault(model, {})[dataset] = {**choice.model_dump(), "k_one_se": one_se}
        write_text(path, json.dumps(table, indent=2))
        return table

    # ----------------------------------------------------------------------- checks: smoke, S7 audit, S8 probe

    def run_smoke(self, llm: LLM) -> dict[str, dict[str, bool]]:
        """Every format parses, thinking is off and the pinned provider answered."""
        labels = self.corpus.load_labels("crossner_ai")
        demos = self.corpus.load("crossner_ai", "train")[:4]
        query = self.corpus.load("crossner_ai", "test")[0]
        checks = {}
        for name, fmt in self.formats.items():
            c = llm.complete(fmt.render_prompt(labels, demos, query), audit_tag="smoke")
            checks[name] = {
                "answered": bool(c.text.strip()),
                "parses": not fmt.parse_output(labels, c.text, query).malformed,
                "no_thinking": not c.reasoning_tokens,
                "pinned": llm.endpoint.backend == "local" or c.provider == llm.endpoint.provider,
            }
        return checks

    def audit(self, model: str, llm: Completer, tag: str) -> dict:
        """S7: resend a fixed sample of TRICE prompts with the cache bypassed; compare repetitions."""
        pairs = [
            (r, q, p)
            for r in self.expand_plan("S1", models=[model])
            if r.condition == "small/full"
            for q, p in self.render_prompts(r)
        ]
        rng = np.random.default_rng(derive_seed("audit", model))
        rows = []
        for i in sorted(rng.choice(len(pairs), self.config.stats.audit_prompts, replace=False)):
            run, query, prompt = pairs[i]
            output = llm.complete(prompt, audit_tag=tag).text
            tags = self.formats[run.fmt].parse_output(self.corpus.load_labels(run.dataset), output, query).tags
            rows.append({"key": f"{run.slug}/{query.id}", "output": output, "tags": tags, "gold": query.tags})
        path = write_text(
            self.config.layout.audit / model / f"{tag}.jsonl", "".join(json.dumps(r) + "\n" for r in rows)
        )
        reps = [[json.loads(x) for x in p.read_text().splitlines()] for p in sorted(path.parent.glob("*.jsonl"))]
        outputs: dict[str, set[str]] = defaultdict(set)
        for r in (r for rep in reps for r in rep):
            outputs[r["key"]].add(r["output"])
        f1 = [
            float(
                self.evaluator.compute_f1(
                    self.evaluator.count_matches([r["tags"] for r in rep], [r["gold"] for r in rep]).sum(0)
                )
            )
            for rep in reps
        ]
        return {
            "repetitions": len(reps),
            "exact_agreement": float(np.mean([len(v) == 1 for v in outputs.values()])),
            "f1": f1,
        }

    def probe(self, model: str, llm: Completer) -> dict:
        """S8: share of the next lines of each raw test file the model reproduces verbatim."""
        n, out = self.config.stats.probe_lines, {}
        for name in self.config.datasets:
            raw = self.corpus.fetch(name)["test"].read_text(encoding="utf-8").splitlines()
            content = self.env.get_template("probe.j2").render(lines=raw[:n])
            got = llm.complete(Prompt(messages=[{"role": "user", "content": content}])).text.strip("\n").splitlines()
            out[name] = sum(a.strip() == b.strip() for a, b in zip(got, raw[n : 2 * n], strict=False)) / n
        write_text(self.config.layout.contamination / f"{model}.json", json.dumps(out, indent=2))
        return out
