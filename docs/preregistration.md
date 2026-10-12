# TRICE pre-registration (DRAFT)

> **Status: draft.** This file becomes binding when it is committed together with the frozen
> configs and tagged `prereg-v1` (milestone M7), **before any test-set call**. After the tag, any
> change to the items below must be reported as a deviation in the paper.

## 1. Frozen before any test-set call

| Item | Where |
|---|---|
| Datasets, sources, commits, file SHA-256, label verbalizations | `configs/datasets.yaml`, `data/manifest.json` |
| Models, OpenRouter providers and quantizations, HF tokenizer revisions | `configs/models.yaml`, `configs/endpoints.lock.json` |
| Retrieval stacks, model revisions, RRF k=60, rerank depth 200, reranker instruction | `configs/retrieval.yaml`, `data/rankings/manifest.json` |
| Study definitions, k grid, LOO query count, pool sizes | `configs/studies.yaml` |
| Prompt formats P0-P3, parsers, scorer | `src/trice/prompt/`, `src/trice/eval/metrics.py` |
| Seeds | derived only via `trice.seeds.derive_seed(purpose, ...)` |

Two decisions are made on training data **after** this tag and are themselves pre-registered
rules, not free choices:

1. **Prompt format (S0).** Leave-one-out over each training pool at k=8 with both reference
   models (Qwen3.5-9B, Qwen3.5-27B), `small/full` retrieval. The format with the highest mean
   micro-F1 over the 10 datasets x 2 models is frozen for every model (ties: P0 < P1 < P2 < P3).
   Written to `results/frozen_format.json`.
2. **Number of demonstrations k\* (S1-loo).** Per (model, dataset): LOO micro-F1 on the grid
   {0, 1, 2, 4, 8, 16, 32, 64} (capped at pool size - 1), isotonic regression, Kneedle knee
   (S = 1, concave, increasing) on a linear k axis; without a knee, the smallest k reaching the
   isotonic maximum. The 1-SE rule is reported for robustness only. Written to
   `results/k_star.json`.

No validation split is loaded anywhere.

## 2. Hypotheses

- **H1.** TRICE (`small/full`) > random demonstrations at the same k\*, per model (S1, S2).
- **H2.** TRICE > zero-shot (S1).
- **H3.** The tuning-free stack is not worse than v1's best retrievers (`v1_bm25`, `v1_dense`) (S2).
- **H4.** The train-chosen k\* has small regret against the oracle test k (S4, descriptive with CIs).
- **H5.** The retrieval gain over random changes monotonically with model size within a ladder
  (S5; direction not presumed, two-sided).

## 3. Primary metric

Micro-F1 over multisets of (whitespace-normalized span, label) per sentence, exact span match,
case-sensitive. Secondary: per-type F1, sentence exact match, invalid-output rate.

## 4. Tests (alpha = 0.05)

| Question | Test | Family for Holm-Bonferroni |
|---|---|---|
| A vs B on one dataset | Approximate randomization on micro-F1, sentence-level swaps, R = 10,000, seeded; 95% paired-bootstrap percentile CI, B = 10,000 | all (model, dataset) cells of one comparison type |
| A vs B, sentence exact match | Exact McNemar (binomial) | reported with the above |
| H1 across datasets, per model | Wilcoxon signed-rank on per-dataset differences | models |
| H1 against randomness | significance must hold against **all three** random seeds | - |
| Retriever / format comparison (S2, S3) | Friedman test + Nemenyi critical difference (Demsar 2006); blocks = datasets x reference models; random seeds averaged into one treatment | per study |
| H5 scaling | Page's L trend test per ladder, datasets as blocks, sizes ordered; both directions, Bonferroni x2; Kendall tau per dataset | ladders |

Reported with every difference: effect size, CI, minimum detectable effect from the bootstrap SE,
and the run-to-run SD from the determinism audit (S7).

## 5. Determinism

Greedy decoding (`temperature=0, top_p=1, seed=0`), pinned provider and quantization with
`allow_fallbacks=false, require_parameters=true`, thinking disabled for hybrid models. Every
response is stored in a SQLite record/replay cache keyed by the SHA-256 of the canonical request,
so analyses replay byte-identically. S7 resends 300 fixed prompts per model, three times on
different days, to quantify residual nondeterminism of hosted inference.
