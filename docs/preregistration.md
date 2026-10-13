# TRICE pre-registration (DRAFT)

> **Status: draft, not binding.** This file becomes binding when it is committed together with
> the frozen configuration and tagged `prereg-v1`, **before any test-set request**. After the tag,
> every change to an item below is reported in the paper as a deviation, with its reason. The
> method itself is described in `docs/methodology.md`. Items marked **[Dn]** are still open; they
> are listed in §9 and in `docs/methodology.md` §13 (with options and recommendations); every one
> must be resolved before the tag.

## 1. Frozen at the tag

| Item | Where |
|---|---|
| Datasets: sources, commits, file SHA-256, sentence counts, label verbalizations | `trice.toml` `[sources]`, `[datasets]` |
| Models: OpenRouter provider and quantization **[D8]** or local vLLM revision, tokenizer revisions | `trice.toml` `[models]`, `results/endpoints.json`, `envs/compose.yaml` |
| Retrieval: stacks and model revisions, RRF k = 60, rerank depth 200, store depth 200, reranker instruction, runtime device **[D7]**; the sparse encoder's remote code **[D18]** | `trice.toml` `[retrieval]` |
| Stored rankings and their SHA-256 | `data/rankings/` (published with the paper) **[D7]** |
| Study definitions, k grid **[D17]**, LOO query count, S6 pool sizes, format-selection k | `trice.toml` `[studies]`, `[selection]` |
| Decoding: temperature 0, top-p 1, seed 0, max 1,024 tokens, thinking off **[D20]** | `trice.toml` `[llm]`, `src/trice/llm.py` |
| Prompt formats P0–P3, parsers, aligner, scorer | `src/trice/templates/`, `src/trice/core/formats.py`, `src/trice/evaluation/scoring.py` |
| Statistical procedures and resample counts | this file §5, `trice.toml` `[stats]`, `src/trice/evaluation/inference.py`, `src/trice/analyzer.py` |
| Seeds | only through `trice.schema.derive_seed(purpose, ...)`; LLM seed 0 |

No validation split is loaded anywhere. No test-set output is looked at before the train-only
decisions in §2 are written to disk.

## 2. Decisions made after the tag, on training data only

These two settings are chosen after the tag, but the rules that choose them are fixed here.
**LOO queries** means each pool sentence in turn, with the remaining pool as its demonstration pool.
Pools of up to 200 sentences are used whole; larger pools use a seeded sample of 200 queries. The
query and any pool sentence with identical text are excluded from its candidates.

1. **Prompt format (S0).**
   - **Runs:** LOO at k = 8, with both reference models (Qwen3.5-9B and Qwen3.5-27B) and `small/full`
     retrieval.
   - **Rule:** the format with the highest mean micro-F1 over the 10 datasets × 2 models is frozen
     for every model and study. Ties go to the earlier format (P0 < P1 < P2 < P3).
   - **Output:** `results/frozen_format.json`.
2. **Number of demonstrations k\* (S1-loo; S6-loo for the S6 pools).**
   - **Curve:** per (model, dataset), LOO micro-F1 on the grid {0, 1, 2, 4, 8, 16, 32, 64}
     **[D17]**, keeping only k smaller than the pool size.
   - **Rule:** fit an isotonic regression to the curve, then take the Kneedle knee (S = 1,
     concave, increasing) on a linear k axis. If there is no knee, take the smallest k that reaches
     the isotonic maximum.
   - **Robustness only:** the 1-SE rule (the smallest k within one bootstrap SE of the best LOO F1)
     is reported, but never used.
   - **Output:** `results/k_star.json`.

## 3. Studies

Unless stated otherwise, a study uses the test split, k = k\* and the frozen format.

| Study | Models | Data | Conditions | k / format |
|---|---|---|---|---|
| S0 | reference | LOO | `small/full` | 8; P0–P3 |
| S1-loo | all 11 | LOO | `small/full` | grid |
| S1 | all 11 | test | zero-shot, `random:0`, `small/full` | k\* |
| S2 | reference | test | `random:1`, `random:2`, `small/{sparse,dense,rrf}`, `large/full`, `v1_bm25/full`, `v1_dense/full` | k\* |
| S3 | reference | test | `small/full` | k\*; the three formats not chosen |
| S4 | reference | test | `small/full` | grid |
| S5 | all, per ladder | S1 output | `small/full` vs `random:0` | – |
| S6-loo / S6 | Qwen3.5-9B | CoNLL-2003 pools of 50–1,600 (nested) | `small/full` / `random:0`, `small/full` | grid / k\* |
| S7 | **[D12]** | 300 seeded S1 `small/full` prompts, sent 3 times on different days with the cache bypassed **[D19]** | – | – |
| S8 | **[D12]** | the 40 lines after the first 40 of each raw test file | – | – |

The specialist baselines (GLiNER family, NuNER-Zero) are descriptive **[D14]**.

## 4. Hypotheses

- **H1.** TRICE (`small/full`) beats random demonstrations at the same k\*, per model (S1; S2
  for the extra seeds) **[D2]**.
- **H2.** TRICE beats zero-shot prompting, per model (S1).
- **H3.** The tuning-free `small` stack is not worse than v1's retrievers (`v1_bm25`, `v1_dense`)
  (S2) **[D3]**.
- **H4.** The train-chosen k\* has small regret against the oracle test k, where regret is the
  oracle test F1 minus the test F1 at k\* (S4). H4 is descriptive and is reported with CIs
  **[D4]**.
- **H5.** The retrieval gain over random demonstrations changes monotonically with model size within
  a ladder (S5). The direction is not presumed, so the test is two-sided. The trend of F1 itself is
  reported as secondary.

Everything else is exploratory, and the paper labels it as exploratory. That covers the S2 stage
ablation beyond H3, S3, S6, S7, S8 and the baselines.

## 5. Metric and tests

**Primary metric.**

- **Metric:** entity-level micro-F1 (seqeval, conlleval semantics) over IOB2 tags, computed per
  (model, dataset, condition) cell.
- **Aligning predictions:** a predicted mention is placed at its own position for inline tags (P3),
  and otherwise at the first exact token-sequence occurrence that is not already tagged.
- **Invalid and unreadable outputs:** a mention with an unknown type or no free occurrence is
  invalid. Unreadable output counts as no predictions.
- **Secondary metrics:** per-type F1, sentence exact match, and the invalid-mention rate.

**Tests.** All tests use α = 0.05, 10,000 resamples, and seeds from `derive_seed`.

| Question | Test | Holm family |
|---|---|---|
| A vs B in one cell | Two-sided paired permutation test on the micro-F1 difference: `scipy.stats.permutation_test`, swapping the two systems' outputs within each sentence; exact when 2^n ≤ 10,000. Reported with a 95% paired-bootstrap percentile CI (`scipy.stats.bootstrap`). | all cells of one comparison type **[D5]** |
| A vs B, sentence exact match | Exact McNemar test (`statsmodels`) | reported alongside the above |
| H1, H2 across datasets | Wilcoxon signed-rank test on the 10 per-dataset differences, per model | models **[D4]** |
| H1 robustness | the difference must be significant against every available random seed | – |
| S2, S3 | Friedman test + Nemenyi post-hoc test with a critical-difference diagram (`scikit-posthocs`). Blocks are dataset × reference model. Random seeds are averaged into one treatment. | per study |
| H5 | Page's L per ladder, datasets as blocks, sizes ordered by total parameters, on the per-dataset gain (`small/full` − `random:0`); both directions, Bonferroni × 2. Kendall τ per dataset as a descriptive statistic. | ladders **[D4]** |

Every reported difference comes with:

- its effect size (the F1 difference);
- its CI;
- the minimum detectable effect from the bootstrap SE;
- the S7 run-to-run SD **[D4]**.

## 6. Exclusions and failures

- **No data exclusions:** no query is excluded after the tag. Every test sentence is scored.
- **Failure policy:** truncated outputs, refusals, failed requests and provider changes are not yet
  covered **[D13]**.

## 7. Determinism

- **Hosted requests:**
  - greedy decoding (`temperature = 0`, `top_p = 1`, `seed = 0`);
  - one pinned provider and quantization, with `allow_fallbacks = false` and
    `require_parameters = true`;
  - thinking disabled for the hybrid-reasoning models.
- **Local models:** vLLM 0.31.0 in bf16 at a pinned revision, with batch-invariant kernels.
- **Rankings:** computed once in fp32 with deterministic torch, stored, and hashed.
- **Response cache:** every response is stored under the SHA-256 of its canonical request, so every
  analysis replays byte-identically from the cache.
- **Residual nondeterminism:** S7 quantifies the nondeterminism of hosted inference that remains
  **[D19]**.
- **Smoke test:** before any study, `trice smoke` checks that every format parses, that the pinned
  provider answered, and that no reasoning tokens were produced **[D20]**.

## 8. Known deviations from v1, declared in advance

- **Mention matching:** mentions are aligned to token positions instead of v1's surface-string
  multiset. The two differ only when a mention string repeats within a sentence.
- **CrossNER-Music test file:** it has 465 sentences; v1's Table 2 reported 456 **[D6]**.
- **Sparse encoder:** it runs remote code from `Alibaba-NLP/new-impl`, patched for transformers 5
  (`patch_remote_model`). Its output matches the transformers 4.57.6 reference exactly.

## 9. Open before the tag

These decisions are detailed in `docs/methodology.md` §13:

- **D1:** duplicate-text exclusion is asymmetric between random and retrieval for test queries.
- **D2:** three random seeds exist only for the reference models.
- **D3:** H3 needs a non-inferiority margin.
- **D4:** promised analyses are missing from `analyzer.py`.
- **D5:** the Holm family in S2 and S3.
- **D6:** CrossNER-Music size.
- **D7:** GPU device and CUDA determinism for the rankings.
- **D8:** mixed precision within ladders.
- **D12:** S7 and S8 models and days.
- **D13:** failure policy.
- **D14:** the baselines' role.
- **D17:** the range of the k grid.
- **D18:** pinning the sparse encoder's remote code.
- **D19:** how S7 is interpreted.
- **D20:** verifying that thinking is off.

When the decisions are resolved, delete this section, update the items above, commit, and tag
`prereg-v1`.
