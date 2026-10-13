"""Pre-registered tables and figures (S1-S5) and the specialist baselines, from stored predictions."""

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel
from scipy import stats
from statsmodels.stats.multitest import multipletests

from trice.core.formats import JsonAnswer, JsonEntity
from trice.evaluation.report import Reporter
from trice.results import ExternalPrediction, KTable
from trice.rows import (
    BaselineRow,
    CellRow,
    CurvePoint,
    ModelDeltaRow,
    PairedRow,
    RankRow,
    RegretRow,
    ScalingPoint,
    ScalingRow,
)
from trice.runner import Runner
from trice.schema import Run


class Analyzer(BaseModel):
    runner: Runner
    reporter: Reporter

    def is_complete(self, study: str) -> bool:
        for run in self.runner.expand_plan(study):
            if not Path(self.runner.config.layout.runs.format(slug=run.slug)).exists():
                return False
        return True

    def test_pairs(self, pairs: list[tuple[Run, Run, str]]) -> list[PairedRow]:
        """Paired tests of each (a, b, label of b) with Holm correction across the family."""
        e = self.runner
        comparisons = []
        for a, b, _ in pairs:
            comparisons.append(e.inferrer.compare_pair(e.count_matches(a), e.count_matches(b), a.slug, b.slug))
        holm = multipletests([c.p_value for c in comparisons], method="holm")[1] if len(pairs) > 0 else []
        rows = []
        for (a, _, versus), c, p_holm in zip(pairs, comparisons, holm, strict=True):
            rows.append(
                PairedRow(
                    model=a.model,
                    dataset=a.dataset,
                    versus=versus,
                    delta=c.delta,
                    p_value=c.p_value,
                    ci_low=c.ci_low,
                    ci_high=c.ci_high,
                    se=c.se,
                    mcnemar_p=c.mcnemar_p,
                    p_holm=float(p_holm),
                )
            )
        return rows

    def tabulate_s1(self) -> None:
        e = self.runner
        runs = {(r.model, r.dataset, r.condition): r for r in e.expand_plan("S1")}
        cells = []
        for (model, dataset, condition), run in sorted(runs.items()):
            predictions = e.load_predictions(run)
            candidates = sum(p.parsed.candidates for p in predictions)
            invalid = sum(p.parsed.invalid for p in predictions)
            cells.append(
                CellRow(
                    model=model,
                    dataset=dataset,
                    condition=condition,
                    k=run.k,
                    f1=e.score_f1(run),
                    invalid=invalid / candidates if candidates > 0 else 0.0,
                )
            )
        self.reporter.write_table("S1", cells, "Micro-F1 per model, dataset and condition (test).")
        self.tabulate_versus(runs, "random:0")
        self.tabulate_versus(runs, "zero_shot")

    def tabulate_versus(self, runs: dict[tuple[str, str, str], Run], baseline: str) -> None:
        """TRICE against one baseline condition: paired tests per cell, Wilcoxon per model."""
        pairs = []
        for (model, dataset, condition), run in sorted(runs.items()):
            if condition == "small/full":
                pairs.append((run, runs[(model, dataset, baseline)], baseline))
        rows = self.test_pairs(pairs)
        name = f"S1-vs-{baseline.replace(':', '')}"
        self.reporter.write_table(name, rows, f"TRICE vs {baseline}: paired tests, Holm.")
        deltas: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            deltas[row.model].append(row.delta)
        per_model = []
        for model, values in deltas.items():
            wilcoxon_p = 1.0 if not any(values) else float(stats.wilcoxon(values).pvalue)
            per_model.append(ModelDeltaRow(model=model, mean_delta=float(np.mean(values)), wilcoxon_p=wilcoxon_p))
        self.reporter.write_table(f"{name}-models", per_model, f"TRICE vs {baseline} across datasets (Wilcoxon).")

    def tabulate_friedman(self, study: str, factor: str) -> None:
        """S2 (factor="condition") and S3 (factor="fmt"): Friedman/Nemenyi with random seeds averaged
        into one treatment, plus Holm-corrected paired tests of TRICE against each alternative."""
        e = self.runner
        alternatives = e.expand_plan(study)
        reference = {r.model for r in alternatives}
        s1 = [r for r in e.expand_plan("S1") if r.model in reference]
        trice = {(r.model, r.dataset): r for r in s1 if r.condition == "small/full"}
        if factor == "condition":
            alternatives += [r for r in s1 if r.condition == "random:0"]
        scores: dict[tuple[str, str, str], list[float]] = defaultdict(list)
        for run in list(trice.values()) + alternatives:
            name = self.name_treatment(run, factor)
            treatment = "random" if name.startswith("random") else name
            scores[(run.model, run.dataset, treatment)].append(e.score_f1(run))
        records = [(model, dataset, t, float(np.mean(f1))) for (model, dataset, t), f1 in scores.items()]
        frame = pd.DataFrame(records, columns=["model", "dataset", "treatment", "f1"])
        friedman = e.inferrer.test_friedman(frame.pivot(index=["model", "dataset"], columns="treatment", values="f1"))
        pairs = []
        for run in alternatives:
            pairs.append((trice[(run.model, run.dataset)], run, self.name_treatment(run, factor)))
        self.reporter.write_table(
            study, self.test_pairs(pairs), f"{study}: TRICE vs each alternative, paired tests, Holm."
        )
        ranks = []
        for treatment, rank in friedman.mean_ranks.items():
            ranks.append(
                RankRow(
                    study=study,
                    treatment=str(treatment),
                    mean_rank=float(rank),
                    statistic=friedman.statistic,
                    p_value=friedman.p_value,
                )
            )
        self.reporter.write_table(f"{study}-friedman", ranks, f"{study}: Friedman test.")
        self.reporter.plot_critical_difference(f"{study}-cd", friedman)

    def name_treatment(self, run: Run, factor: str) -> str:
        match factor:
            case "condition":
                return run.condition
            case "fmt":
                return run.fmt
            case _:
                raise ValueError(f"unknown factor {factor!r}")

    def tabulate_s4(self) -> None:
        """Regret of the train-chosen k* against the oracle test k, and LOO vs test curves."""
        e = self.runner
        loo = KTable.model_validate_json(Path(e.config.layout.k_star).read_text())
        curves: dict[tuple[str, str], dict[int, float]] = defaultdict(dict)
        for run in e.expand_plan("S4"):
            curves[(run.model, run.dataset)][run.k] = e.score_f1(run)
        rows = []
        points = []
        for (model, dataset), curve in sorted(curves.items()):
            selection = loo.find(model, dataset).selection
            ks = sorted(curve)
            oracle = ks[0]
            for k in ks:
                if curve[k] > curve[oracle]:  # ties go to the smaller k
                    oracle = k
            regret = curve[oracle] - curve[selection.k_star]
            rows.append(
                RegretRow(model=model, dataset=dataset, k_star=selection.k_star, k_oracle=oracle, regret=regret)
            )
            for k in ks:
                points.append(CurvePoint(dataset=dataset, k=k, f1=curve[k], curve="test"))
            for k, f1 in zip(selection.ks, selection.f1, strict=True):
                points.append(CurvePoint(dataset=dataset, k=k, f1=f1, curve="train LOO"))
        self.reporter.write_table("S4", rows, "Regret of k* (train LOO) against the oracle test k.")
        self.reporter.plot_lines("S4-curves", points, x="k", y="f1", hue="curve", col="dataset")

    def tabulate_s5(self) -> None:
        """Scaling: Page's L per ladder on F1 and on the gain over random demonstrations."""
        e = self.runner
        models = e.config.models
        params = {name: model.params_b for name, model in models.items()}
        runs = {(r.model, r.dataset, r.condition): r for r in e.expand_plan("S1")}
        datasets = sorted({dataset for _, dataset, _ in runs})
        rows = []
        points = []
        for ladder in sorted({model.ladder for model in models.values()}):
            names = sorted([name for name, model in models.items() if model.ladder == ladder], key=params.__getitem__)
            if len(names) < 3:  # Page's L needs at least three ordered treatments
                continue
            full = np.array([[e.score_f1(runs[(m, d, "small/full")]) for m in names] for d in datasets])
            random = np.array([[e.score_f1(runs[(m, d, "random:0")]) for m in names] for d in datasets])
            trend_f1 = e.inferrer.test_trend(full)
            trend_gain = e.inferrer.test_trend(full - random)
            rows.append(
                ScalingRow(
                    ladder=ladder,
                    metric="f1",
                    p_increasing=trend_f1.p_increasing,
                    p_decreasing=trend_f1.p_decreasing,
                    p_two_sided=trend_f1.p_two_sided,
                )
            )
            rows.append(
                ScalingRow(
                    ladder=ladder,
                    metric="gain",
                    p_increasing=trend_gain.p_increasing,
                    p_decreasing=trend_gain.p_decreasing,
                    p_two_sided=trend_gain.p_two_sided,
                )
            )
            for i, dataset in enumerate(datasets):
                for j, name in enumerate(names):
                    points.append(ScalingPoint(ladder=ladder, params_b=params[name], dataset=dataset, f1=full[i, j]))
        self.reporter.write_table("S5", rows, "Scaling: Page's L trend test per ladder.")
        self.reporter.plot_lines("S5-scaling", points, x="params_b", y="f1", hue="dataset", col="ladder")

    def run_all(self) -> list[str]:
        """Write every table whose runs are complete (none before the format and k* are chosen)."""
        layout = self.runner.config.layout
        if not (Path(layout.frozen_format).exists() and Path(layout.k_star).exists()):
            return []
        done = []
        if self.is_complete("S1"):
            self.tabulate_s1()
            self.tabulate_s5()
            done += ["S1", "S5"]
            if self.is_complete("S2"):
                self.tabulate_friedman("S2", "condition")
                done.append("S2")
            if self.is_complete("S3"):
                self.tabulate_friedman("S3", "fmt")
                done.append("S3")
        if self.is_complete("S4"):
            self.tabulate_s4()
            done.append("S4")
        return done

    def score_baselines(self) -> list[BaselineRow]:
        """Score envs/baselines output with the shared aligner and scorer."""
        e = self.runner
        rows = []
        for tag, spec in e.config.baselines.items():
            systems = [tag]
            if spec.finetune:
                for seed in spec.seeds:
                    systems.append(f"{tag}-ft-s{seed}")
            for system in systems:
                for dataset in e.config.datasets:
                    path = Path(e.config.layout.baselines.format(system=system, dataset=dataset))
                    if not path.exists():
                        continue
                    labels = e.corpus.load_labels(dataset)
                    gold = e.corpus.load(dataset, "test")
                    predictions = [
                        ExternalPrediction.model_validate_json(line) for line in path.read_text().splitlines()
                    ]
                    tags = []
                    for prediction, query in zip(predictions, gold, strict=True):
                        entities = [JsonEntity(text=m.text, type=m.label) for m in prediction.entities]
                        answer = JsonAnswer(entities=entities).model_dump_json()
                        tags.append(e.formats["P2"].parse_output(labels, answer, query).tags)
                    counts = e.evaluator.count_matches(tags, [q.tags for q in gold])
                    rows.append(
                        BaselineRow(system=system, dataset=dataset, f1=float(e.evaluator.compute_f1(counts.sum(0))))
                    )
        self.reporter.write_table("baselines", rows, "Specialist baselines (micro-F1, test).")
        return rows
