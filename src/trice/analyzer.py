"""Pre-registered tables and figures (S1-S5) and the specialist baselines, from stored predictions."""

from collections import defaultdict

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
        try:
            return all(
                self.runner.config.layout.runs.fill(slug=r.slug).exists() for r in self.runner.expand_plan(study)
            )
        except RuntimeError, KeyError:
            return False

    def test_pairs(self, pairs: list[tuple[Run, Run, str]]) -> list[PairedRow]:
        """Paired tests of each (a, b, label of b) with Holm correction across the family."""
        e = self.runner
        comparisons = [
            e.inferrer.compare_pair(e.count_matches(a), e.count_matches(b), a.slug, b.slug) for a, b, _ in pairs
        ]
        holm = multipletests([c.p_value for c in comparisons], method="holm")[1] if pairs else []
        return [
            PairedRow(model=a.model, dataset=a.dataset, versus=versus, **c.model_dump(), p_holm=float(p))
            for (a, _, versus), c, p in zip(pairs, comparisons, holm, strict=True)
        ]

    def tabulate_s1(self) -> None:
        e, runs = self.runner, {(r.model, r.dataset, r.condition): r for r in self.runner.expand_plan("S1")}
        cells = []
        for (m, d, c), r in sorted(runs.items()):
            predictions = e.load_predictions(r)
            cells.append(
                CellRow(
                    model=m,
                    dataset=d,
                    condition=c,
                    k=r.k,
                    f1=e.score_f1(r),
                    invalid=sum(p.invalid for p in predictions) / max(1, sum(p.candidates for p in predictions)),
                )
            )
        self.reporter.write_table("S1", cells, "Micro-F1 per model, dataset and condition (test).")
        for baseline in ("random:0", "zero_shot"):
            name = f"S1-vs-{baseline.replace(':', '')}"
            rows = self.test_pairs(
                [(r, runs[(m, d, baseline)], baseline) for (m, d, c), r in sorted(runs.items()) if c == "small/full"]
            )
            self.reporter.write_table(name, rows, f"TRICE vs {baseline}: paired tests, Holm.")
            deltas: dict[str, list[float]] = defaultdict(list)
            for row in rows:
                deltas[row.model].append(row.delta)
            self.reporter.write_table(
                f"{name}-models",
                [
                    ModelDeltaRow(
                        model=m,
                        mean_delta=float(np.mean(v)),
                        wilcoxon_p=1.0 if not any(v) else float(stats.wilcoxon(v).pvalue),
                    )
                    for m, v in deltas.items()
                ],
                f"TRICE vs {baseline} across datasets (Wilcoxon).",
            )

    def tabulate_friedman(self, study: str, factor: str) -> None:
        """S2 (factor="condition") and S3 (factor="fmt"): Friedman/Nemenyi with random seeds averaged
        into one treatment, plus Holm-corrected paired tests of TRICE against each alternative."""
        e, alternatives = self.runner, self.runner.expand_plan(study)
        reference = {r.model for r in alternatives}
        s1 = [r for r in e.expand_plan("S1") if r.model in reference]
        trice = {(r.model, r.dataset): r for r in s1 if r.condition == "small/full"}
        alternatives += [r for r in s1 if r.condition == "random:0"] if factor == "condition" else []
        scores: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for r in [*trice.values(), *alternatives]:
            name = self.name_treatment(r, factor)
            scores[(r.model, r.dataset)]["random" if name.startswith("random") else name].append(e.score_f1(r))
        friedman = e.inferrer.test_friedman(
            pd.DataFrame({b: {t: np.mean(v) for t, v in s.items()} for b, s in scores.items()}).T
        )
        self.reporter.write_table(
            study,
            self.test_pairs([(trice[(r.model, r.dataset)], r, self.name_treatment(r, factor)) for r in alternatives]),
            f"{study}: TRICE vs each alternative, paired tests, Holm.",
        )
        self.reporter.write_table(
            f"{study}-friedman",
            [
                RankRow(
                    study=study,
                    treatment=str(treatment),
                    mean_rank=float(rank),
                    statistic=friedman.statistic,
                    p_value=friedman.p_value,
                )
                for treatment, rank in friedman.mean_ranks.items()
            ],
            f"{study}: Friedman test.",
        )
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
        loo = KTable.model_validate_json(e.config.layout.k_star.fill().read_text())
        curves: dict[tuple[str, str], dict[int, float]] = defaultdict(dict)
        for r in e.expand_plan("S4"):
            curves[(r.model, r.dataset)][r.k] = e.score_f1(r)
        rows, points = [], []
        for (m, d), curve in sorted(curves.items()):
            selection = loo.find(m, d).selection
            oracle = max(curve, key=lambda k: (curve[k], -k))
            rows.append(
                RegretRow(
                    model=m,
                    dataset=d,
                    k_star=selection.k_star,
                    k_oracle=oracle,
                    regret=curve[oracle] - curve[selection.k_star],
                )
            )
            points += [CurvePoint(dataset=d, k=k, f1=f, curve="test") for k, f in curve.items()]
            points += [
                CurvePoint(dataset=d, k=k, f1=f, curve="train LOO")
                for k, f in zip(selection.ks, selection.f1, strict=True)
            ]
        self.reporter.write_table("S4", rows, "Regret of k* (train LOO) against the oracle test k.")
        self.reporter.plot_lines("S4-curves", points, x="k", y="f1", hue="curve", col="dataset")

    def tabulate_s5(self) -> None:
        """Scaling: Page's L per ladder on F1 and on the gain over random demonstrations."""
        e, models = self.runner, self.runner.config.models
        runs = {(r.model, r.dataset, r.condition): r for r in e.expand_plan("S1")}
        datasets, rows, points = sorted({d for _, d, _ in runs}), [], []
        for ladder in sorted({m.ladder for m in models.values()}):
            names = sorted((n for n, m in models.items() if m.ladder == ladder), key=lambda n: models[n].params_b)
            if len(names) < 3:  # Page's L needs at least three ordered treatments
                continue
            full = np.array([[e.score_f1(runs[(m, d, "small/full")]) for m in names] for d in datasets])
            gain = full - np.array([[e.score_f1(runs[(m, d, "random:0")]) for m in names] for d in datasets])
            rows += [
                ScalingRow(ladder=ladder, metric=metric, **e.inferrer.test_trend(values).model_dump())
                for metric, values in (("f1", full), ("gain", gain))
            ]
            points += [
                ScalingPoint(ladder=ladder, params_b=models[m].params_b, dataset=d, f1=full[i, j])
                for i, d in enumerate(datasets)
                for j, m in enumerate(names)
            ]
        self.reporter.write_table("S5", rows, "Scaling: Page's L trend test per ladder.")
        self.reporter.plot_lines("S5-scaling", points, x="params_b", y="f1", hue="dataset", col="ladder")

    def run_all(self) -> list[str]:
        """Write every table whose runs are complete."""
        done = []
        if self.is_complete("S1"):
            self.tabulate_s1()
            self.tabulate_s5()
            done += ["S1", "S5"]
            for study, factor in (("S2", "condition"), ("S3", "fmt")):
                if self.is_complete(study):
                    self.tabulate_friedman(study, factor)
                    done.append(study)
        if self.is_complete("S4"):
            self.tabulate_s4()
            done.append("S4")
        return done

    def score_baselines(self) -> list[BaselineRow]:
        """Score envs/baselines output with the shared aligner and scorer."""
        e, rows = self.runner, []
        for tag, spec in e.config.baselines.items():
            for system in [tag, *(f"{tag}-ft-s{seed}" for seed in spec.seeds if spec.finetune)]:
                for dataset in e.config.datasets:
                    path = e.config.layout.baselines.fill(system=system, dataset=dataset)
                    if not path.exists():
                        continue
                    labels, gold = e.corpus.load_labels(dataset), e.corpus.load(dataset, "test")
                    tags = [
                        e.formats["P2"]
                        .parse_output(
                            labels,
                            JsonAnswer(
                                entities=[JsonEntity(text=m.text, type=m.label) for m in prediction.entities]
                            ).model_dump_json(),
                            query,
                        )
                        .tags
                        for prediction, query in zip(
                            map(ExternalPrediction.model_validate_json, path.read_text().splitlines()),
                            gold,
                            strict=True,
                        )
                    ]
                    counts = e.evaluator.count_matches(tags, [q.tags for q in gold])
                    rows.append(
                        BaselineRow(system=system, dataset=dataset, f1=float(e.evaluator.compute_f1(counts.sum(0))))
                    )
        self.reporter.write_table("baselines", rows, "Specialist baselines (micro-F1, test).")
        return rows
