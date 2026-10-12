"""Running studies: plan runs, store rankings, execute runs, and make the train-only decisions.
Wires core/ (retrieval, prompting, selection), the LLM and the scorer together."""

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from datasets import Dataset
from pydantic import BaseModel

from trice.core.prompting import Labels, Prompter
from trice.core.retrieval import Ranking, Retriever, demonstrations
from trice.core.selection import select_format, select_k
from trice.data import Corpus
from trice.evaluation.scoring import Evaluator
from trice.evaluation.stats import Stats
from trice.llm import LLM
from trice.schema import Condition, Example, Prediction, Prompt, Run, derive_seed
from trice.settings import Format, QuerySet, Settings


class Experiment(BaseModel):
    settings: Settings
    corpus: Corpus
    prompter: Prompter
    evaluator: Evaluator
    stats: Stats

    def result(self, name: str) -> Path:
        path = self.settings.paths.results / name
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def labels(self, dataset: str) -> Labels:
        return Labels(raw_to_verbal=self.corpus.labels(dataset))

    # ----------------------------------------------------------------------- planning

    def frozen_format(self) -> Format | None:
        path = self.result("frozen_format.json")
        return json.loads(path.read_text())["format"] if path.exists() else None

    def k_star(self) -> dict[str, dict[str, int]]:
        path = self.result("k_star.json")
        table = json.loads(path.read_text()) if path.exists() else {}
        return {m: {d: v["k_star"] for d, v in ds.items()} for m, ds in table.items()}

    def formats(self, which: list[Format] | str, assume: Format | None, study: str) -> list[Format]:
        if isinstance(which, list):
            return which
        frozen = assume or self.frozen_format()
        if frozen is None:
            raise RuntimeError(f"{study}: no frozen format yet (run S0, then `trice select-format`)")
        all_formats: list[Format] = ["P0", "P1", "P2", "P3"]
        return [frozen] if which == "frozen" else [f for f in all_formats if f != frozen]

    def plan(
        self,
        study: str,
        assume_k: int | None = None,
        assume_format: Format | None = None,
        models: list[str] | None = None,
        datasets: list[str] | None = None,
    ) -> list[Run]:
        """Runs of a study. ``assume_*`` stand in for k* and the frozen format before they exist."""
        spec, k_star = self.settings.studies[study], self.k_star()
        reference = [n for n, m in self.settings.models.items() if m.reference]
        chosen = {"all": list(self.settings.models), "reference": reference}.get(str(spec.models), spec.models)
        names = list(self.settings.datasets) if spec.datasets == "all" else spec.datasets
        runs = []
        for model in [m for m in chosen if not models or m in models]:
            for dataset in [d for d in names if not datasets or d in datasets]:
                if spec.k == "grid":
                    ks = [k for k in self.settings.selection.k_grid if k < self.corpus.pool_size(dataset)]
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
                    for f in self.formats(spec.formats, assume_format, study)
                    for k in ks
                ]
        return runs

    def queries(self, dataset: str, queries: QuerySet) -> list[Example]:
        """Test split, or the LOO query set: the whole pool when small (CrossNER), else a seeded sample."""
        if queries == "test":
            return self.corpus.load(dataset, "test")
        pool, n = sorted(self.corpus.load(dataset, "train"), key=lambda e: e.id), self.settings.selection.loo_queries
        if len(pool) <= n:
            return pool
        return [
            pool[i]
            for i in sorted(np.random.default_rng(derive_seed("loo", dataset)).choice(len(pool), n, replace=False))
        ]

    # ----------------------------------------------------------------------- rankings

    def rankings_path(self, stack: str, dataset: str, queries: QuerySet) -> Path:
        return self.settings.paths.data / "rankings" / stack / dataset / queries

    def rank(self, stack: str, retriever: Retriever, dataset: str, queries: QuerySet) -> str:
        """Compute once and store as a Dataset; later stages only read it. Returns the SHA-256 of the rankings."""
        qs = self.queries(dataset, queries)
        ranked = retriever.rank(qs, self.corpus.load(dataset, "train"), loo=queries == "loo")
        rows = [
            {"query_id": q.id, "stage": s, "ids": ids} for q, r in zip(qs, ranked, strict=True) for s, ids in r.items()
        ]
        Dataset.from_list(rows).save_to_disk(self.rankings_path(stack, dataset, queries))
        return hashlib.sha256(json.dumps(rows).encode()).hexdigest()

    def rankings(self, stack: str, dataset: str, queries: QuerySet) -> dict[str, Ranking]:
        out: dict[str, Ranking] = defaultdict(dict)
        for row in Dataset.load_from_disk(self.rankings_path(stack, dataset, queries)):
            out[row["query_id"]][row["stage"]] = row["ids"]
        return out

    # ----------------------------------------------------------------------- execution

    def prompts(self, run: Run) -> list[tuple[Example, Prompt]]:
        condition, pool, labels = (
            Condition.parse(run.condition),
            self.corpus.load(run.dataset, "train"),
            self.labels(run.dataset),
        )
        ranked = self.rankings(condition.stack, run.dataset, run.queries) if condition.kind == "retrieval" else {}
        return [
            (
                q,
                self.prompter.prompt(
                    run.fmt, labels, demonstrations(condition, run.k, run.dataset, q, pool, ranked.get(q.id)), q
                ),
            )
            for q in self.queries(run.dataset, run.queries)
        ]

    def predictions_path(self, run: Run) -> Path:
        return self.result(f"runs/{run.slug}.jsonl")

    def execute(self, run: Run, llm: LLM) -> Path:
        pairs, labels = self.prompts(run), self.labels(run.dataset)
        completions = llm.complete_many([p for _, p in pairs])
        predictions = [
            Prediction(
                query_id=q.id,
                output=c.text,
                completion=c,
                **self.prompter.parse(run.fmt, c.text, q, labels).model_dump(),
            )
            for (q, _), c in zip(pairs, completions, strict=True)
        ]
        path = self.predictions_path(run)
        path.write_text("".join(p.model_dump_json() + "\n" for p in predictions))
        return path

    def predictions(self, run: Run) -> list[Prediction]:
        return [Prediction.model_validate_json(line) for line in self.predictions_path(run).read_text().splitlines()]

    def counts(self, run: Run) -> np.ndarray:
        preds, gold = self.predictions(run), self.queries(run.dataset, run.queries)
        if [p.query_id for p in preds] != [q.id for q in gold]:
            raise ValueError(f"{run.slug}: predictions do not align with the query set")
        return self.evaluator.counts([p.tags for p in preds], [q.tags for q in gold])

    def f1(self, run: Run) -> float:
        return float(self.evaluator.f1(self.counts(run).sum(0)))

    # ----------------------------------------------------------------------- train-only decisions

    def select_format(self) -> Format:
        """S0: best mean LOO F1 over datasets x reference models, frozen for every model."""
        by_format: dict[Format, list[float]] = defaultdict(list)
        for run in self.plan("S0"):
            by_format[run.fmt].append(self.f1(run))
        means = {fmt: float(np.mean(v)) for fmt, v in by_format.items()}
        winner = select_format(means)
        self.result("frozen_format.json").write_text(json.dumps({"format": winner, "mean_f1": means}, indent=2))
        return winner

    def select_k(self, study: str = "S1-loo") -> dict:
        """k* per (model, dataset) from the LOO curves, with the 1-SE rule for robustness."""
        curves: dict[tuple[str, str], dict[int, Run]] = defaultdict(dict)
        for run in self.plan(study):
            curves[(run.model, run.dataset)][run.k] = run
        path = self.result("k_star.json")
        table = json.loads(path.read_text()) if path.exists() else {}
        for (model, dataset), runs in sorted(curves.items()):
            choice = select_k({k: self.f1(r) for k, r in runs.items()})
            se = self.stats.standard_error(self.counts(runs[choice.ks[int(np.argmax(choice.f1))]]), model, dataset)
            one_se = next(k for k, f in zip(choice.ks, choice.f1, strict=True) if f >= max(choice.f1) - se)
            table.setdefault(model, {})[dataset] = {**choice.model_dump(), "k_one_se": one_se}
        path.write_text(json.dumps(table, indent=2))
        return table

    # ----------------------------------------------------------------------- checks: smoke, S7 audit, S8 probe

    def smoke(self, llm: LLM) -> dict[str, dict[str, bool]]:
        """Every format parses, thinking is off and the pinned provider answered."""
        labels, demos = self.labels("crossner_ai"), self.corpus.load("crossner_ai", "train")[:4]
        query = self.corpus.load("crossner_ai", "test")[0]
        checks = {}
        for fmt in ("P0", "P1", "P2", "P3"):
            c = llm.complete(self.prompter.prompt(fmt, labels, demos, query), audit_tag="smoke")
            checks[fmt] = {
                "answered": bool(c.text.strip()),
                "parses": not self.prompter.parse(fmt, c.text, query, labels).malformed,
                "no_thinking": not c.reasoning_tokens,
                "pinned": llm.endpoint.backend == "local" or c.provider == llm.endpoint.provider,
            }
        return checks

    def audit(self, model: str, llm: LLM, tag: str) -> dict:
        """S7: resend a fixed sample of TRICE prompts with the cache bypassed; compare repetitions."""
        pairs = [
            (r, q, p)
            for r in self.plan("S1", models=[model])
            if r.condition == "small/full"
            for q, p in self.prompts(r)
        ]
        rng = np.random.default_rng(derive_seed("audit", model))
        rows = []
        for i in sorted(rng.choice(len(pairs), self.settings.stats.audit_prompts, replace=False)):
            run, query, prompt = pairs[i]
            output = llm.complete(prompt, audit_tag=tag).text
            tags = self.prompter.parse(run.fmt, output, query, self.labels(run.dataset)).tags
            rows.append({"key": f"{run.slug}/{query.id}", "output": output, "tags": tags, "gold": query.tags})
        path = self.result(f"audit/{model}/{tag}.jsonl")
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        reps = [[json.loads(x) for x in p.read_text().splitlines()] for p in sorted(path.parent.glob("*.jsonl"))]
        outputs: dict[str, set[str]] = defaultdict(set)
        for r in (r for rep in reps for r in rep):
            outputs[r["key"]].add(r["output"])
        f1 = [
            float(self.evaluator.f1(self.evaluator.counts([r["tags"] for r in rep], [r["gold"] for r in rep]).sum(0)))
            for rep in reps
        ]
        return {
            "repetitions": len(reps),
            "exact_agreement": float(np.mean([len(v) == 1 for v in outputs.values()])),
            "f1": f1,
        }

    def probe(self, model: str, llm: LLM) -> dict:
        """S8: share of the next lines of each raw test file the model reproduces verbatim."""
        n, out = self.settings.stats.probe_lines, {}
        for name in self.settings.datasets:
            raw = self.corpus.fetch(name)["test"].read_text(encoding="utf-8").splitlines()
            content = self.prompter.env.get_template("probe.j2").render(lines=raw[:n])
            got = llm.complete(Prompt(messages=[{"role": "user", "content": content}])).text.strip("\n").splitlines()
            out[name] = sum(a.strip() == b.strip() for a, b in zip(got, raw[n : 2 * n], strict=False)) / n
        self.result(f"contamination/{model}.json").write_text(json.dumps(out, indent=2))
        return out
