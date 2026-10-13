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
from trice.core.retrieval import Retriever, select_demonstrations
from trice.data import Corpus
from trice.evaluation.inference import Inferrer
from trice.evaluation.scoring import Evaluator
from trice.llm import LLM, Completer
from trice.results import AuditReport, AuditRow, FormatChoice, FormatScore, KChoice, KTable, ProbeScore, SmokeCheck
from trice.schema import (
    Condition,
    Example,
    FormatName,
    Message,
    Prediction,
    Prompt,
    QuerySet,
    Ranking,
    Run,
    derive_seed,
)


class Runner(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    config: Config
    corpus: Corpus
    formats: dict[FormatName, Format]
    env: Environment
    evaluator: Evaluator
    inferrer: Inferrer

    # ----------------------------------------------------------------------- planning

    def expand_plan(
        self,
        study: str,
        assume_k: int | None = None,
        assume_format: FormatName | None = None,
        models: list[str] | None = None,
        datasets: list[str] | None = None,
    ) -> list[Run]:
        """Runs of a study. ``assume_*`` stand in for k* and the frozen format before they exist."""
        spec, layout = self.config.studies[study], self.config.layout
        k_path, format_path = layout.k_star.fill(), layout.frozen_format.fill()
        choices = KTable.model_validate_json(k_path.read_text()) if k_path.exists() else KTable([])
        if isinstance(spec.formats, list):
            formats = spec.formats
        else:
            frozen = assume_format or (
                FormatChoice.model_validate_json(format_path.read_text()).winner if format_path.exists() else None
            )
            if frozen is None:
                raise RuntimeError(f"{study}: no frozen format yet (run S0, then `trice select-format`)")
            formats = [frozen] if spec.formats == "frozen" else [f for f in self.formats if f != frozen]
        match spec.models:
            case "all":
                chosen = list(self.config.models)
            case "reference":
                chosen = [n for n, m in self.config.models.items() if m.reference]
            case _:
                chosen = spec.models
        runs = []
        for model in [m for m in chosen if not models or m in models]:
            for dataset in [
                d
                for d in (self.config.datasets if spec.datasets == "all" else spec.datasets)
                if not datasets or d in datasets
            ]:
                match spec.k:
                    case "grid":
                        ks = [k for k in self.config.selection.k_grid if k < self.corpus.count_pool(dataset)]
                    case "k_star":
                        ks = [assume_k if assume_k is not None else choices.find(model, dataset).selection.k_star]
                    case _:
                        ks = [spec.k]
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
                    for f in formats
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
        ranked = retriever.rank(
            self.select_queries(dataset, queries), self.corpus.load(dataset, "train"), loo=queries == "loo"
        )
        rows = [r.model_dump() for r in ranked]
        Dataset.from_list(rows).save_to_disk(
            self.config.layout.rankings.fill(stack=stack, dataset=dataset, queries=queries)
        )
        return hashlib.sha256(json.dumps(rows).encode()).hexdigest()

    # ----------------------------------------------------------------------- execution

    def render_prompts(self, run: Run) -> list[tuple[Example, Prompt]]:
        condition, pool = Condition.parse(run.condition), self.corpus.load(run.dataset, "train")
        stored = (
            Dataset.load_from_disk(
                self.config.layout.rankings.fill(stack=condition.stack, dataset=run.dataset, queries=run.queries)
            )
            if condition.kind == "retrieval"
            else []
        )
        ranked = {r.query_id: r.ids for r in map(Ranking.model_validate, stored) if r.stage == condition.stage}
        return [
            (
                q,
                self.formats[run.fmt].render_prompt(
                    self.corpus.load_labels(run.dataset),
                    select_demonstrations(condition, run.k, run.dataset, q, pool, ranked.get(q.id)),
                    q,
                ),
            )
            for q in self.select_queries(run.dataset, run.queries)
        ]

    @contextmanager
    def recording(self, run: Run) -> Generator[TextIO]:
        """Predictions of a run are written to a partial file that replaces the real one only on
        success, so a crash never leaves a truncated run that looks complete."""
        layout = self.config.layout
        partial = layout.runs_partial.prepare(slug=run.slug)
        with partial.open("w") as handle:
            yield handle
        partial.replace(layout.runs.fill(slug=run.slug))

    def execute(self, run: Run, llm: Completer) -> None:
        pairs = self.render_prompts(run)
        completions = llm.complete_many([p for _, p in pairs])
        with self.recording(run) as handle:
            for (query, _), completion in zip(pairs, completions, strict=True):
                parsed = self.formats[run.fmt].parse_output(
                    self.corpus.load_labels(run.dataset), completion.text, query
                )
                handle.write(
                    Prediction(
                        query_id=query.id, output=completion.text, completion=completion, **parsed.model_dump()
                    ).model_dump_json()
                    + "\n"
                )

    def load_predictions(self, run: Run) -> list[Prediction]:
        lines = self.config.layout.runs.fill(slug=run.slug).read_text().splitlines()
        return [Prediction.model_validate_json(line) for line in lines]

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
        runs = self.expand_plan("S0")
        scores = [
            FormatScore(format=name, mean_f1=float(np.mean([self.score_f1(r) for r in runs if r.fmt == name])))
            for name in sorted({r.fmt for r in runs})
        ]
        choice = FormatChoice(winner=max(scores, key=lambda s: s.mean_f1).format, scores=scores)  # ties: earliest
        self.config.layout.frozen_format.prepare().write_text(choice.model_dump_json(indent=2))
        return choice.winner

    def select_k(self, study: str = "S1-loo") -> KTable:
        """k* per (model, dataset) from the LOO curves, with the 1-SE rule for robustness."""
        curves: dict[tuple[str, str], dict[int, Run]] = defaultdict(dict)
        for run in self.expand_plan(study):
            curves[(run.model, run.dataset)][run.k] = run
        path = self.config.layout.k_star.fill()
        earlier = KTable.model_validate_json(path.read_text()).root if path.exists() else []
        chosen = []
        for (model, dataset), runs in sorted(curves.items()):
            choice = selection.select_k({k: self.score_f1(r) for k, r in runs.items()})
            error = self.inferrer.bootstrap_error(
                self.count_matches(runs[choice.ks[int(np.argmax(choice.f1))]]), model, dataset
            )
            one_se = next(k for k, f in zip(choice.ks, choice.f1, strict=True) if f >= max(choice.f1) - error)
            chosen.append(KChoice(model=model, dataset=dataset, selection=choice, k_one_se=one_se))
        table = KTable([c for c in earlier if (c.model, c.dataset) not in curves] + chosen)
        path.write_text(table.model_dump_json(indent=2))
        return table

    # ----------------------------------------------------------------------- checks: smoke, S7 audit, S8 probe

    def run_smoke(self, llm: LLM) -> list[SmokeCheck]:
        """Every format parses, thinking is off and the pinned provider answered."""
        labels, query = self.corpus.load_labels("crossner_ai"), self.corpus.load("crossner_ai", "test")[0]
        demos, checks = self.corpus.load("crossner_ai", "train")[:4], []
        for name, fmt in self.formats.items():
            c = llm.complete(fmt.render_prompt(labels, demos, query), audit_tag="smoke")
            checks.append(
                SmokeCheck(
                    format=name,
                    answered=bool(c.text.strip()),
                    parses=not fmt.parse_output(labels, c.text, query).malformed,
                    no_thinking=not c.reasoning_tokens,
                    pinned=llm.endpoint.backend == "local" or c.provider == llm.endpoint.provider,
                )
            )
        return checks

    def audit(self, model: str, llm: Completer, tag: str) -> AuditReport:
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
            parsed = self.formats[run.fmt].parse_output(self.corpus.load_labels(run.dataset), output, query)
            rows.append(AuditRow(key=f"{run.slug}/{query.id}", output=output, tags=parsed.tags, gold=query.tags))
        layout = self.config.layout
        layout.audit.prepare(model=model, tag=tag).write_text("".join(r.model_dump_json() + "\n" for r in rows))
        reps = [
            [AuditRow.model_validate_json(x) for x in path.read_text().splitlines()]
            for path in layout.audit.match(model=model, tag="*")
        ]
        outputs: dict[str, set[str]] = defaultdict(set)
        for row in (row for rep in reps for row in rep):
            outputs[row.key].add(row.output)
        return AuditReport(
            repetitions=len(reps),
            exact_agreement=float(np.mean([len(v) == 1 for v in outputs.values()])),
            f1=[
                float(
                    self.evaluator.compute_f1(
                        self.evaluator.count_matches([r.tags for r in rep], [r.gold for r in rep]).sum(0)
                    )
                )
                for rep in reps
            ],
        )

    def probe(self, model: str, llm: Completer) -> list[ProbeScore]:
        """S8: share of the next lines of each raw test file the model reproduces verbatim."""
        n, scores = self.config.stats.probe_lines, []
        for name in self.config.datasets:
            raw = self.corpus.fetch(name, "test").read_text(encoding="utf-8").splitlines()
            content = self.env.get_template("probe.j2").render(lines=raw[:n])
            got = llm.complete(Prompt(messages=[Message(role="user", content=content)])).text.strip("\n").splitlines()
            share = sum(a.strip() == b.strip() for a, b in zip(got, raw[n : 2 * n], strict=False)) / n
            scores.append(ProbeScore(dataset=name, share=share))
        self.config.layout.contamination.prepare(model=model).write_text(
            "[" + ",".join(s.model_dump_json() for s in scores) + "]"
        )
        return scores
