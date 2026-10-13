"""Running studies: plan runs, store rankings, execute runs, and make the train-only decisions.
Wires core/ (retrieval, formats, selection), the LLM and the scorer together."""

import glob
import hashlib
import json
import operator
from collections import defaultdict
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
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
    ensure_parent,
    write_text,
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
        spec = self.config.studies[study]
        k_path = Path(self.config.layout.k_star)
        format_path = Path(self.config.layout.frozen_format)
        choices = KTable.model_validate_json(k_path.read_text()) if k_path.exists() else KTable([])
        if isinstance(spec.formats, list):
            formats = spec.formats
        else:
            if assume_format is not None:
                frozen = assume_format
            elif format_path.exists():
                frozen = FormatChoice.model_validate_json(format_path.read_text()).winner
            else:
                raise RuntimeError(f"{study}: no frozen format yet (run S0, then `trice select-format`)")
            formats = [frozen] if spec.formats == "frozen" else [f for f in self.formats if f != frozen]
        match spec.models:
            case "all":
                study_models = list(self.config.models)
            case "reference":
                study_models = [name for name, model in self.config.models.items() if model.reference]
            case _:
                study_models = spec.models
        study_datasets = list(self.config.datasets) if spec.datasets == "all" else spec.datasets
        if models is not None:
            study_models = [m for m in study_models if m in models]
        if datasets is not None:
            study_datasets = [d for d in study_datasets if d in datasets]
        runs = []
        for model in study_models:
            for dataset in study_datasets:
                match spec.k:
                    case "grid":
                        ks = [k for k in self.config.selection.k_grid if k < self.corpus.count_pool(dataset)]
                    case "k_star":
                        ks = [assume_k if assume_k is not None else choices.find(model, dataset).selection.k_star]
                    case _:
                        ks = [spec.k]
                for condition in spec.conditions:
                    for fmt in formats:
                        for k in ks:
                            k_used = 0 if condition == "zero_shot" else k
                            runs.append(
                                Run(
                                    model=model,
                                    dataset=dataset,
                                    queries=spec.queries,
                                    condition=condition,
                                    fmt=fmt,
                                    k=k_used,
                                )
                            )
        return runs

    def select_queries(self, dataset: str, queries: QuerySet) -> list[Example]:
        """Test split, or the LOO query set: the whole pool when small (CrossNER), else a seeded sample."""
        if queries == "test":
            return self.corpus.load(dataset, "test")
        pool = sorted(self.corpus.load(dataset, "train"), key=operator.attrgetter("id"))
        if len(pool) <= self.config.selection.loo_queries:
            return pool
        rng = np.random.default_rng(derive_seed("loo", dataset))
        chosen = rng.choice(len(pool), self.config.selection.loo_queries, replace=False)
        return [pool[i] for i in sorted(chosen)]

    # ----------------------------------------------------------------------- rankings

    def store_rankings(self, stack: str, retriever: Retriever, dataset: str, queries: QuerySet) -> str:
        """Compute once and store as a Dataset; later stages only read it. Returns the SHA-256 of the rankings."""
        ranked = retriever.rank(
            self.select_queries(dataset, queries), self.corpus.load(dataset, "train"), loo=queries == "loo"
        )
        rows = [r.model_dump() for r in ranked]
        path = self.config.layout.rankings.format(stack=stack, dataset=dataset, queries=queries)
        Dataset.from_list(rows).save_to_disk(path)
        return hashlib.sha256(json.dumps(rows).encode()).hexdigest()

    # ----------------------------------------------------------------------- execution

    def render_prompts(self, run: Run) -> list[tuple[Example, Prompt]]:
        condition = Condition.parse(run.condition)
        pool = self.corpus.load(run.dataset, "train")
        labels = self.corpus.load_labels(run.dataset)
        fmt = self.formats[run.fmt]
        ranked: dict[str, list[str]] = {}
        if condition.kind == "retrieval":
            path = self.config.layout.rankings.format(stack=condition.stack, dataset=run.dataset, queries=run.queries)
            for row in Dataset.load_from_disk(path):
                ranking = Ranking.model_validate(row)
                if ranking.stage == condition.stage:
                    ranked[ranking.query_id] = ranking.ids
        prompts = []
        for query in self.select_queries(run.dataset, run.queries):
            ids = ranked[query.id] if condition.kind == "retrieval" else []
            demos = select_demonstrations(condition, run.k, run.dataset, query, pool, ids)
            prompts.append((query, fmt.render_prompt(labels, demos, query)))
        return prompts

    @contextmanager
    def recording(self, run: Run) -> Generator[TextIO]:
        """Predictions of a run are written to a partial file that replaces the real one only on
        success, so a crash never leaves a truncated run that looks complete."""
        partial = Path(ensure_parent(self.config.layout.runs_partial.format(slug=run.slug)))
        with partial.open("w") as handle:
            yield handle
        partial.replace(self.config.layout.runs.format(slug=run.slug))

    def execute(self, run: Run, llm: Completer) -> None:
        pairs = self.render_prompts(run)
        completions = llm.complete_many([prompt for _, prompt in pairs])
        labels = self.corpus.load_labels(run.dataset)
        fmt = self.formats[run.fmt]
        with self.recording(run) as handle:
            for (query, _), completion in zip(pairs, completions, strict=True):
                parsed = fmt.parse_output(labels, completion.text, query)
                prediction = Prediction(query_id=query.id, output=completion.text, completion=completion, parsed=parsed)
                handle.write(prediction.model_dump_json() + "\n")

    def load_predictions(self, run: Run) -> list[Prediction]:
        lines = Path(self.config.layout.runs.format(slug=run.slug)).read_text().splitlines()
        return [Prediction.model_validate_json(line) for line in lines]

    def count_matches(self, run: Run) -> np.ndarray:
        predictions = self.load_predictions(run)
        gold = self.select_queries(run.dataset, run.queries)
        if [p.query_id for p in predictions] != [q.id for q in gold]:
            raise ValueError(f"{run.slug}: predictions do not align with the query set")
        return self.evaluator.count_matches([p.parsed.tags for p in predictions], [q.tags for q in gold])

    def score_f1(self, run: Run) -> float:
        return float(self.evaluator.compute_f1(self.count_matches(run).sum(0)))

    # ----------------------------------------------------------------------- train-only decisions

    def select_format(self) -> FormatName:
        """S0: best mean LOO F1 over datasets x reference models, frozen for every model."""
        runs = self.expand_plan("S0")
        scores = []
        for name in sorted({r.fmt for r in runs}):
            f1 = [self.score_f1(r) for r in runs if r.fmt == name]
            scores.append(FormatScore(format=name, mean_f1=float(np.mean(f1))))
        winner = scores[0]
        for score in scores[1:]:
            if score.mean_f1 > winner.mean_f1:  # ties go to the earliest format
                winner = score
        write_text(
            self.config.layout.frozen_format,
            FormatChoice(winner=winner.format, scores=scores).model_dump_json(indent=2),
        )
        return winner.format

    def select_k(self, study: str = "S1-loo") -> KTable:
        """k* per (model, dataset) from the LOO curves, with the 1-SE rule for robustness."""
        curves: dict[tuple[str, str], dict[int, Run]] = defaultdict(dict)
        for run in self.expand_plan(study):
            curves[(run.model, run.dataset)][run.k] = run
        path = Path(self.config.layout.k_star)
        earlier = KTable.model_validate_json(path.read_text()).root if path.exists() else []
        chosen = []
        for (model, dataset), runs in sorted(curves.items()):
            curve = selection.select_k({k: self.score_f1(r) for k, r in runs.items()})
            best_run = runs[curve.ks[int(np.argmax(curve.f1))]]
            error = self.inferrer.bootstrap_error(self.count_matches(best_run), model, dataset)
            threshold = max(curve.f1) - error
            within = [k for k, f1 in zip(curve.ks, curve.f1, strict=True) if f1 >= threshold]
            chosen.append(KChoice(model=model, dataset=dataset, selection=curve, k_one_se=within[0]))
        table = KTable([c for c in earlier if (c.model, c.dataset) not in curves] + chosen)
        write_text(self.config.layout.k_star, table.model_dump_json(indent=2))
        return table

    # ----------------------------------------------------------------------- checks: smoke, S7 audit, S8 probe

    def run_smoke(self, llm: LLM) -> list[SmokeCheck]:
        """Every format parses, thinking is off and the pinned provider answered."""
        labels = self.corpus.load_labels("crossner_ai")
        query = self.corpus.load("crossner_ai", "test")[0]
        demos = self.corpus.load("crossner_ai", "train")[:4]
        checks = []
        for name, fmt in self.formats.items():
            reply = llm.complete(fmt.render_prompt(labels, demos, query), audit_tag="smoke")
            checks.append(
                SmokeCheck(
                    format=name,
                    answered=reply.text.strip() != "",
                    parses=not fmt.parse_output(labels, reply.text, query).malformed,
                    no_thinking=reply.reasoning_tokens is None or reply.reasoning_tokens == 0,
                    pinned=llm.endpoint.backend == "local" or reply.provider == llm.endpoint.provider,
                )
            )
        return checks

    def audit(self, model: str, llm: Completer, tag: str) -> AuditReport:
        """S7: resend a fixed sample of TRICE prompts with the cache bypassed; compare repetitions."""
        pairs = []
        for run in self.expand_plan("S1", models=[model]):
            if run.condition == "small/full":
                for query, prompt in self.render_prompts(run):
                    pairs.append((run, query, prompt))
        rng = np.random.default_rng(derive_seed("audit", model))
        rows = []
        for i in sorted(rng.choice(len(pairs), self.config.stats.audit_prompts, replace=False)):
            run, query, prompt = pairs[i]
            output = llm.complete(prompt, audit_tag=tag).text
            parsed = self.formats[run.fmt].parse_output(self.corpus.load_labels(run.dataset), output, query)
            rows.append(AuditRow(key=f"{run.slug}/{query.id}", output=output, tags=parsed.tags, gold=query.tags))
        layout = self.config.layout
        write_text(layout.audit.format(model=model, tag=tag), "".join(r.model_dump_json() + "\n" for r in rows))
        repetitions = []
        for path in sorted(glob.glob(layout.audit.format(model=model, tag="*"))):
            repetitions.append([AuditRow.model_validate_json(x) for x in Path(path).read_text().splitlines()])
        outputs: dict[str, set[str]] = defaultdict(set)
        f1 = []
        for repetition in repetitions:
            for row in repetition:
                outputs[row.key].add(row.output)
            counts = self.evaluator.count_matches([r.tags for r in repetition], [r.gold for r in repetition])
            f1.append(float(self.evaluator.compute_f1(counts.sum(0))))
        agreement = float(np.mean([len(distinct) == 1 for distinct in outputs.values()]))
        return AuditReport(repetitions=len(repetitions), exact_agreement=agreement, f1=f1)

    def probe(self, model: str, llm: Completer) -> list[ProbeScore]:
        """S8: share of the next lines of each raw test file the model reproduces verbatim."""
        lines = self.config.stats.probe_lines
        scores = []
        for name in self.config.datasets:
            raw = self.corpus.fetch(name, "test").read_text(encoding="utf-8").splitlines()
            content = self.env.get_template("probe.j2").render(lines=raw[:lines])
            reply = llm.complete(Prompt(messages=[Message(role="user", content=content)]))
            got = reply.text.strip("\n").splitlines()
            expected = raw[lines : 2 * lines]
            share = sum(a.strip() == b.strip() for a, b in zip(got, expected, strict=False)) / lines
            scores.append(ProbeScore(dataset=name, share=share))
        write_text(
            self.config.layout.contamination.format(model=model),
            "[" + ",".join(s.model_dump_json() for s in scores) + "]",
        )
        return scores
