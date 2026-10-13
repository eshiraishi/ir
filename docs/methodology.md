# TRICE: methodology (draft)

> Draft paper section. Every number below comes from `trice.toml`, `results/endpoints.json`,
> `docs/budget-partial.md` or the code in `src/trice/`. No experimental results exist yet. Open
> points are marked **[Dn]** and are listed in §10, "Decisions for the author". Citations still to
> be filled are marked `[ref]`.

## 1. The task, in plain words

**Named entity recognition (NER)** means finding the names of things in a sentence and saying what
kind of thing each one is. For example, in *"Geoffrey Hinton worked on backpropagation at the
University of Toronto"*, the entities are *Geoffrey Hinton* (researcher), *backpropagation*
(algorithm) and *University of Toronto* (university). This sentence was made up for illustration
and does not come from any dataset.

**In-context learning** solves the task without training the model. A large language model (LLM)
gets a prompt with a few solved examples, called *demonstrations*, followed by the new sentence,
called the *query*. The model then writes the answer for the query in the same style.

The demonstrations come from a labeled *pool* (a dataset's training split). Which demonstrations
are shown, how many, in what order and in what format all change the result. v1 of this work
(Shiraishi et al., BRACIS 2025) showed that *retrieving* demonstrations similar to the query beats
choosing them at random. TRICE (Tuning-free Retrieval for In-Context Entity recognition) keeps that
idea and adds one constraint: **no setting is tuned on held-out data**. Every choice is either
fixed in advance (§3) or made by a rule applied only to the training pool (§4). No validation split
is loaded anywhere (`src/trice/data.py`).

## 2. Data

There are ten English datasets. Each one has a training split, used only as the demonstration pool,
and a test split, used only for evaluation.

| Dataset | Domain | Types | Pool (train) | Test |
|---|---|---:|---:|---:|
| CrossNER-Politics | politics | 9 | 200 | 651 |
| CrossNER-Science | natural science | 17 | 200 | 543 |
| CrossNER-Music | music | 13 | 100 | 465 **[D6]** |
| CrossNER-Literature | literature | 12 | 100 | 416 |
| CrossNER-AI | artificial intelligence | 14 | 100 | 431 |
| CoNLL-2003 (copy shipped with CrossNER) | news | 4 | 14,041 | 3,453 |
| MIT-Movie ("eng" corpus) | movie queries | 12 | 9,775 | 2,443 |
| MIT-Restaurant | restaurant queries | 8 | 7,660 | 1,521 |
| WNUT-17 | social media | 6 | 3,394 | 1,287 |
| NCBI-Disease | biomedical | 1 | 5,432 | 940 |
| **Total** | | | | **12,150** |

- **Sources:**
  - Git sources (CrossNER, WNUT-17, NCBI-Disease) are pinned by commit.
  - The MIT corpora are plain downloads from `groups.csail.mit.edu`.
  - Every file is pinned by SHA-256 in `trice.toml`. `trice data` refuses a file whose hash or
    sentence count differs.
- **Preprocessing:**
  - Files are read as one token per line with its IOB2 tag, sentences separated by blank lines.
  - `-DOCSTART-` lines are dropped.
  - Each raw label is mapped one-to-one to a natural-language name shown to the model (for example
    `programlang` becomes "programming language"). These verbalizations are fixed in
    `[datasets.*].labels`.
- **Pool sizes:** only the five CrossNER pools are small (100 or 200 sentences). The other five
  datasets use their full training splits as pools **[D16]**.
- **Smaller pools for study S6:** S6 draws nested CoNLL-2003 pools of 50, 100, 200, 400, 800 and
  1,600 sentences. Each one is the first *n* sentences of a single seeded shuffle, so every smaller
  pool is contained in the larger ones. The test split is never subsampled.

## 3. The pipeline for one query

```
query ─┬─ sparse scores (whole pool) ─┐
       └─ dense scores  (whole pool) ─┴─ RRF fusion ─ top 200 ─ reranker ─ top k ─ shuffle ─ prompt ─ LLM ─ parse ─ score
```

### 3.1 Retrieval: finding similar training sentences

Three models score how similar each pool sentence is to the query. All three are under 1B
parameters, and all are pinned to a Hugging Face commit in `[retrieval.stacks.small]`.

1. **Learned sparse retrieval.** `opensearch-neural-sparse-encoding-doc-v3-gte` represents each
   sentence as weighted vocabulary terms. This is like keyword matching, but the weights are
   learned. Queries and documents go through the model's own query and document encoders
   (`encode_query`, `encode_document` in sentence-transformers).
2. **Dense retrieval.** `Qwen3-Embedding-0.6B` maps each sentence to a vector. The score is the
   cosine similarity. No instruction is prepended to the query, so query and pool sentences are
   embedded the same way **[D9]**.
3. **Fusion.** The two rankings are combined with reciprocal rank fusion (RRF; Cormack et al.,
   2009): score(d) = Σ 1/(60 + rank(d)), summed over the two rankings. The constant k = 60 is
   taken from the literature and never tuned.
4. **Reranking.** The 200 best fused candidates are rescored by `Qwen3-Reranker-0.6B`. The
   reranker reads the query and a candidate together, plus a fixed instruction: *"Given a sentence,
   retrieve sentences that mention similar kinds of named entities in a similar context."* The
   score is the probability of the token "yes" against "no", as on the model card. With depth 200,
   reranking covers the whole pool for every CrossNER dataset.

Some details apply to all four steps:

- **Ties:** ties in any ranking are broken by ascending sentence id.
- **Storage:** the top 200 ids of every stage (`sparse`, `dense`, `rrf`, `full`) are stored. The
  ablation in S2 can therefore use each stage alone without recomputing anything.
- **Leave-one-out:** when a training sentence is itself the query (§4), it is removed from its own
  pool, together with any pool sentence of identical text.
- **Computed once:** rankings are computed once, in fp32, with
  `torch.use_deterministic_algorithms(True)`. They are stored as Hugging Face datasets with their
  SHA-256 (`trice rank`). Everything downstream only reads them.

Study S2 compares this stack (`small`) with alternatives:

- each of its stages alone;
- a `large` stack in which the dense model and the reranker are the 8B Qwen3 models, called
  through OpenRouter;
- replicas of v1's two retrievers:
  - Okapi BM25 (k1 = 1.2, b = 0.75) over lowercased spaCy lookup lemmas;
  - `bge-large-en-v1.5` with v1's query instruction.

### 3.2 Choosing and ordering the demonstrations

- **Retrieval conditions:** the *k* top-ranked sentences are the demonstrations.
- **Random condition:** *k* pool sentences are drawn uniformly without replacement. The draw uses
  a seed derived from the random seed index, the dataset and the query id. Pool sentences with the
  same text as the query are excluded **[D1]**.
- **Zero-shot condition:** no demonstrations.
- **Order:** in every condition, the chosen demonstrations are sorted by id and then shuffled with
  a seed derived from (dataset, query id). This keeps the ranking order from leaking into the
  prompt, and gives the same set of demonstrations the same order in every condition and for every
  model.

### 3.3 Prompt formats

There are four candidate formats. One of them is chosen on training data (§4.1) and then used for
every model.

| Format | Prompt structure | Expected answer for the example in §1 |
|---|---|---|
| **P0** | v1's format: one user message with a definition line ("Defn: An entity is a …"), then `Example i:` / `Answer:` blocks | `1. Geoffrey Hinton (researcher)` / `2. backpropagation (algorithm)` / `3. University of Toronto (university)` |
| **P1** | chat format: a system message listing the entity types, then each demonstration as a user turn with its answer as an assistant turn | same numbered list as P0; `None` when there are no entities |
| **P2** | P1 plus a strict JSON Schema (structured generation) whose `type` field is an enum of the types | `{"entities": [{"text": "Geoffrey Hinton", "type": "researcher"}, …]}` |
| **P3** | P1, but the answer repeats the sentence with inline tags | `<researcher>Geoffrey Hinton</researcher> worked on <algorithm>backpropagation</algorithm> at the <university>University of Toronto</university>` |

All prompt text lives in Jinja templates (`src/trice/templates/p0.j2`, `system.j2`, `answer.j2`).
The answer of every demonstration is generated from its gold tags by the same template the model is
asked to imitate.

### 3.4 Calling the model

Each request uses the same settings:

- **Decoding:** greedy (temperature 0, top-p 1, seed 0), at most 1,024 output tokens.
- **Stop sequence:** P0 stops at `\nExample `.
- **Thinking:** off for the hybrid-reasoning Qwen3.5 models (OpenRouter `reasoning.enabled =
  false`; `enable_thinking = false` on vLLM).
- **Hosted models:** OpenRouter, pinned to one provider and one quantization, with
  `allow_fallbacks = false` and `require_parameters = true`. The pinning rule takes the providers
  that support every required parameter (including structured outputs, which P2 needs), then
  prefers the most faithful precision (bf16 > fp16 > fp32 > fp8 > …), then the lowest price, then
  the provider name.
- **Local models:** sizes that OpenRouter does not serve run on vLLM 0.31.0 in bf16, at a pinned
  revision, with `VLLM_BATCH_INVARIANT=1` (`envs/compose.yaml`).
- **Caching:** every response is cached under the SHA-256 of the full request. Rerunning a study
  in `replay` mode reproduces every prediction byte for byte without contacting any server.

### 3.5 Reading the answer back

The model's answer is turned back into one tag per query token:

1. **Read the candidate mentions:**
   - P0 and P1: lines matching `N. text (type)`.
   - P2: the JSON items.
   - P3: the `<tag>…</tag>` spans.
2. **Map each type to a dataset label.** The match ignores case, and the model may write the
   verbalization, the raw label or the tag form.
3. **Place each mention in the query:**
   - P3: at the position the model wrote it, if that position is valid.
   - Other formats: at the first occurrence of the exact token sequence that is not already tagged.
4. **Count invalid mentions:** a mention with an unknown type or no free occurrence is counted as
   invalid and dropped.
5. **Handle unreadable output:** it yields no entities, so its gold entities count as misses.

This alignment differs from v1's surface-string matching only when the same mention string occurs
more than once in a sentence.

### 3.6 Scoring

The primary metric is **entity-level micro-F1**, computed with seqeval (conlleval semantics) over
IOB2 tags. A predicted entity counts as correct only if both its boundaries and its type match the
gold entity. True positives, false positives and false negatives are kept per sentence, because
every statistical test resamples sentences (§7). The secondary metrics are per-type F1,
sentence-level exact match, and the invalid-mention rate.

## 4. Decisions made on training data only

Two settings are not fixed in advance. Each one is chosen by a fixed rule that sees only the
training pool, using **leave-one-out (LOO)** evaluation:

- **What LOO means:** each pool sentence in turn plays the query, and the rest of the pool serves
  as its demonstration pool.
- **Which sentences are used:**
  - Pools of up to 200 sentences (all CrossNER pools) are used whole.
  - Larger pools use a seeded sample of 200 query sentences.
  - This gives 1,700 LOO queries per model across the ten datasets.

### 4.1 Prompt format (study S0)

- **Run:** the two reference models (Qwen3.5-9B and Qwen3.5-27B) run LOO with every format at
  k = 8, using the full `small` stack.
- **Rule:** the format with the highest mean micro-F1 over the 10 datasets × 2 models is frozen
  for every model and study. Ties go to the earlier format (P0 < P1 < P2 < P3).

### 4.2 Number of demonstrations k\* (study S1-loo)

For every (model, dataset) pair:

1. **LOO curve:** compute LOO micro-F1 at k ∈ {0, 1, 2, 4, 8, 16, 32, 64}, keeping only k smaller
   than the pool size.
2. **Smoothing:** fit an isotonic (non-decreasing) regression to the curve. This removes noise
   without adding a parameter.
3. **Knee:** k\* is the knee of the smoothed curve found by Kneedle `[ref]`, with S = 1, concave,
   increasing, on a linear k axis (a linear axis because prompt cost grows linearly with k). The
   knee is where adding more demonstrations stops paying off.
4. **No knee:** if the curve has no knee, k\* is the smallest k that reaches the isotonic maximum.

For robustness only, the 1-SE rule is also reported: the smallest k whose LOO F1 is within one
bootstrap standard error of the best k.

## 5. Models

There are eleven instruction-tuned models in three size ladders.

| Model | Ladder | Params (B, total / active) | Endpoint | Precision |
|---|---|---|---|---|
| Qwen3.5-0.8B | Qwen3.5 dense | 0.8 / 0.8 | local vLLM | bf16 |
| Qwen3.5-2B | Qwen3.5 dense | 2 / 2 | local vLLM | bf16 |
| Qwen3.5-4B | Qwen3.5 dense | 4 / 4 | local vLLM | bf16 |
| Qwen3.5-9B (reference) | Qwen3.5 dense | 9 / 9 | DeepInfra | bf16 |
| Qwen3.5-27B (reference) | Qwen3.5 dense | 27 / 27 | DeepInfra | fp8 |
| Qwen3.5-35B-A3B | Qwen3.5 MoE | 35 / 3 | DeepInfra | fp8 |
| Qwen3.5-122B-A10B | Qwen3.5 MoE | 122 / 10 | Alibaba | not disclosed |
| Qwen3.5-397B-A17B | Qwen3.5 MoE | 397 / 17 | DeepInfra | fp8 |
| Gemma 3 4B | Gemma 3 | 4 / 4 | DeepInfra | bf16 |
| Gemma 3 12B | Gemma 3 | 12 / 12 | DeepInfra | bf16 |
| Gemma 3 27B | Gemma 3 | 27 / 27 | NextBit | fp8 |

Precision is mixed within ladders, as the table shows **[D8]**. Tokenizer revisions, providers and
prices are pinned in `results/endpoints.json`.

## 6. Studies

Studies S0 to S4 and S6 are defined in `[studies]` of `trice.toml`. S5 is an analysis of S1's
output. S7 and S8 are checks implemented in `src/trice/runner.py`. Unless stated otherwise, a study
uses the test split, k = k\*, and the frozen format.

| Study | Question | Models | Data | Conditions | k / format | Planned requests¹ |
|---|---|---|---|---|---|---:|
| S0 | Which prompt format? | 2 reference | LOO | `small/full` | k = 8; P0–P3 | 13,600 |
| S1-loo | Which k\*? | all 11 | LOO | `small/full` | k grid | 149,600 |
| S1 | Main result: does TRICE beat random and zero-shot? | all 11 | test | zero-shot, random (seed 0), `small/full` | k\* | 400,950 |
| S2 | Retriever ablation | 2 reference | test | random (seeds 1, 2), each `small` stage, `large/full`, v1 BM25, v1 dense | k\* | 194,400 |
| S3 | Format sensitivity | 2 reference | test | `small/full` | k\*; the three formats not chosen | 72,900 |
| S4 | How costly is choosing k on train only? | 2 reference | test | `small/full` | k grid | 194,400 |
| S5 | Does the retrieval gain change with model size? | all ladders | S1 output | – | – | 0 |
| S6-loo, S6 | How does pool size matter? | Qwen3.5-9B | CoNLL pools of 50 to 1,600 | random (seed 0), `small/full` | grid (LOO); k\* (test) | 7,550 + 41,436 |
| S7 | Run-to-run nondeterminism of hosted inference | **[D12]** | 300 fixed S1 prompts | `small/full` | – | 300 per repetition |
| S8 | Contamination probe | **[D12]** | the 40 lines after the first 40 of each raw test file | – | – | 10 per model |

¹ Runs × queries, computed from `trice.toml`. This is an upper bound: identical requests are sent
once and served from the cache afterwards. For example, the S4 runs at k = k\* repeat S1's runs.

Specialist baselines are also scored with the same aligner and scorer, as a descriptive reference
**[D14]**. They run in their own Docker image (`envs/baselines`):

- **Zero-shot:** GLiNER2-large, GLiNER2.5-base, GLiNER-v2.5-large, NuNER-Zero and GLiNER-BioMed.
- **Fine-tuned:** the last three are also fine-tuned on each training pool, with library-default
  hyperparameters, no validation set, and seeds 0, 1 and 2.

## 7. Statistical analysis

The tests, their families and α = 0.05 are fixed in `docs/preregistration.md` §5. In brief:

- **Per (model, dataset) cell:**
  - a paired permutation test on the micro-F1 difference (sentence-level swaps, 10,000
    resamples);
  - a 95% paired-bootstrap percentile CI (10,000 resamples);
  - an exact McNemar test on sentence exact match;
  - Holm correction within each family.
- **Across datasets, per model:** a Wilcoxon signed-rank test on the per-dataset differences.
- **Several treatments (S2, S3):** a Friedman test with the Nemenyi post-hoc test and a
  critical-difference diagram. The blocks are the dataset × reference-model pairs.
- **Size trends (S5):** Page's L test per ladder, with sizes ordered by total parameters (the
  order is the same by active parameters) and datasets as blocks. It is run in both directions with
  Bonferroni × 2.

## 8. Reproducibility

### 8.1 What is pinned

| Item | Pin |
|---|---|
| Data | git commit + SHA-256 + sentence count per file (`trice.toml`) |
| Retrieval models | Hugging Face commit per model (`[retrieval.stacks]`) |
| Rankings | stored once, with the SHA-256 that `trice rank` prints |
| LLMs | OpenRouter provider + quantization, or local vLLM image + revision; tokenizer revision (`results/endpoints.json`) |
| Prompts and parsers | templates in `src/trice/templates/`, code in `core/formats.py` |
| Every model response | SQLite cache keyed by the request hash (`cache/responses`) |
| Software | `uv.lock`; Python 3.14; vLLM 0.31.0 and GLiNER in their own containers |

### 8.2 Seeds

Every seed is the first 32 bits of SHA-256(`"|".join(purpose, …)`) (`trice.schema.derive_seed`),
so it depends only on what it is for and never on call order. The purposes are:

- `loo` (LOO query sample);
- `pool` (S6 pools);
- `random` (random demonstrations);
- `order` (demonstration order);
- `permutation` and `bootstrap` (tests);
- `se` (1-SE rule);
- `audit` (S7 sample).

The LLM seed is 0.

### 8.3 Order of commands

```bash
uv sync && uv run pytest
uv run trice data                                   # fetch, verify hashes, build datasets
uv run trice rank --stack small --pools             # needs a GPU in practice [D7]
uv run trice rank --stack v1_bm25 && uv run trice rank --stack v1_dense
uv run trice rank --stack large                     # paid OpenRouter embeddings/rerank [D10]
uv run trice pin                                    # free metadata call; writes endpoints.json
uv run trice smoke <model>                          # paid; every format parses, thinking off
git tag prereg-v1                                   # freeze (docs/preregistration.md)
uv run trice run S0 && uv run trice select-format   # results/frozen_format.json
uv run trice run S1-loo && uv run trice select-k    # results/k_star.json
uv run trice run S1   # then S2, S3, S4, S6-loo, select-k --study S6-loo, S6
uv run trice audit <model> <day-tag>                # S7, three days
uv run trice probe <model>                          # S8
uv run trice analyze                                # tables (.md/.tex/.json), figures (.pdf/TikZ)
uv run trice run <study> --mode replay              # reproduce from the cache, no network
```

To reproduce the analysis without paying for model calls, a reader needs three things: the
released response cache, the stored rankings and `results/*.json`.

### 8.4 Known caveats

- **Sparse encoder runs unpinned remote code.** The sparse model needs `trust_remote_code=True`,
  which runs code from `Alibaba-NLP/new-impl` that is not pinned by our config. That code was
  written for transformers 4. TRICE runs on transformers 5, because transformers 4 does not know
  the `qwen3_5` architecture. `patch_remote_model` (`src/trice/core/encoders.py`) makes three
  repairs:
  - restores the removed attention-mask helper;
  - ties the MLM decoder to the embeddings and reloads its bias from the checkpoint;
  - rebuilds the buffers that v5 leaves uninitialized.

  The patched encoder's output was identical (maximum absolute difference 0.0) to a
  transformers 4.57.6 reference. A change to the remote code could still alter the rankings, so
  the stored rankings, not the code, are the reproducible artifact.
- **Hosted inference is not bit-reproducible.** Greedy decoding on a pinned provider can still vary
  between calls. S7 measures how much. The cache makes every reported number replayable.
- **Hardware can change rankings.** fp32 results can differ between CPU and GPU, and between GPU
  models, which can reorder near-ties. The rankings must be released with their hashes **[D7]**.
- **Providers can change.** A provider can change or withdraw an endpoint during the study
  **[D13]**.

## 9. Compute and cost (status)

- **Retrieval:** on a 4-CPU machine the reranker took about 4.2 s per (query, candidate) pair, or
  about 14 minutes per query. The 10 datasets need 13,850 ranked queries (12,150 test + 1,700 LOO),
  so the reranker alone would need about 3,200 hours. One 8-pair measurement gave 0.58 s per pair,
  or about 450 hours. Both estimates leave out the S6 pools. The rankings will therefore be
  computed on a GPU. They have not been computed yet, and `trice rank` has not been tested end to
  end.
- **LLM calls:** a dry run counts every planned prompt with each model's own tokenizer and prices it
  at the pinned rates. It estimates output length as the gold answer's length × 1.25 + 8 tokens.
  The run has been measured only for the studies whose k is fixed in advance:

  | Study | Calls | Input tokens | Output tokens | Cost (USD) |
  |---|---:|---:|---:|---:|
  | S0 | 13,600 | 11,056,168 | 829,748 | 3.13 |
  | S1-loo | 149,578 | 256,647,296 | 11,527,310 | 42.14 |
  | Subtotal | 163,178 | 267,703,464 | 12,357,058 | 45.27 (67.91 with a 1.5× margin) |

  - **Not yet measured:** S1, S2, S3, S4, S6-loo and S6, because their cost depends on k\* **[D11]**.
  - **Retrieval conditions priced as random:** they were priced with random demonstrations of the
    same k, because no rankings exist yet.
  - **Local models cost $0 here:** the three local Qwen3.5 sizes run on the author's own
    hardware.
  - **Spent so far:** $0. A guard in the code (`llm.allow_spend = false`) blocks every OpenRouter
    request until the author enables it.

## 10. Decisions for the author

Each item below is undecided or unverified. None of them has been settled silently in the text
above.

- **D1. Random and retrieval treat duplicate sentences differently.**
  - For a *test* query, random sampling excludes pool sentences with the same text as the query.
    Retrieval does not exclude them, because the leave-one-out filter applies only to LOO queries.
  - In datasets with repeated sentences, retrieval could therefore show the model the answer
    itself.
  - Options: apply the same filter in both conditions, or measure how many test queries have an
    exact duplicate in the pool and report it.
- **D2. H1 asks for three random seeds that most models do not have.** H1 requires significance
  against all three random seeds. Only the two reference models get seeds 1 and 2 (in S2); the
  other nine models have seed 0 only. Either restrict that clause to the reference models or add
  seeds 1 and 2 to S1.
- **D3. H3 has no non-inferiority margin.** H3 says "not worse than" v1's retrievers, but no margin
  is defined, and the planned paired test is a superiority test. Choose a margin (in F1 points)
  and a one-sided test, or rephrase H3 as a two-sided comparison.
- **D4. The pre-registration promises analyses that `analyzer.py` does not compute yet.**
  - Holm correction of the per-model Wilcoxon tests across models.
  - Holm across ladders for Page's L.
  - Kendall τ per dataset.
  - The minimum detectable effect.
  - CIs for the S4 regret.
  - Attaching the S7 run-to-run SD to each difference.

  Implement them before the tag, or remove them from the pre-registration.
- **D5. The Holm family in S2 and S3 differs between the code and the pre-registration.** The code
  corrects all alternatives of a study (for example 9 alternatives × 20 cells in S2) as one Holm
  family. The pre-registration says one family per comparison type. Pick one.
- **D6. CrossNER-Music's test size differs from v1.** The pinned test file has 465 sentences, while
  v1's Table 2 reported 456. Check which file v1 used, and say in the paper whether v1's numbers
  for Music are comparable.
- **D7. Retrieval needs a GPU, and that touches the frozen config.**
  - `[retrieval.runtime] device = "cpu"` in `trice.toml`, but the rankings need a GPU, so the
    device must change before the tag.
  - On CUDA, `torch.use_deterministic_algorithms(True)` normally also needs
    `CUBLAS_WORKSPACE_CONFIG` set; otherwise cuBLAS operations raise an error. This is unverified
    here.
  - Decide the GPU model to record, and where to publish the rankings with their hashes.
- **D8. Precision varies within ladders.**
  - Within ladders, precision mixes bf16 (local and DeepInfra), fp8, and one undisclosed precision
    (Alibaba, Qwen3.5-122B-A10B). This confounds the scaling analysis (H5).
  - It is also unverified that OpenRouter accepts `quantizations = ["unknown"]` for the Alibaba
    pin, because no smoke test has been run.
  - Options: accept and report the mix, or re-pin to a uniform precision where one exists.
- **D9. The dense retriever uses no query instruction.** Qwen3-Embedding is used symmetrically,
  without one. Confirm that this is intended: the model card describes instruction-prefixed
  queries.
- **D10. The `large` stack is not budgeted.** Its embedding and rerank calls go through OpenRouter,
  but they are not in the budget. Whether OpenRouter's `/rerank` accepts the instruction is also
  unchecked (`ApiReranker`).
- **D11. The rest of the budget is unmeasured.** The cost of S1–S6 is not measured. Decide a
  spending cap, and whether to measure the rest of the budget (with `--k 16` and `--k 64`) before
  the tag.
- **D12. S7 and S8 have no model list or schedule.** Decide which models get the S7 audit and the S8
  probe, and on which days the three S7 repetitions run.
- **D13. There is no failure policy.** It should cover:
  - outputs cut off at 1,024 tokens (currently parsed as they are);
  - refusals;
  - requests that fail after 8 retries;
  - a provider withdrawing or changing an endpoint mid-study.
- **D14. The baselines' role is undefined.**
  - Fine-tuned baselines see the full training split, which is thousands of sentences for the
    non-CrossNER datasets.
  - Decide whether they are descriptive only, or part of a hypothesis.
- **D15. Citations are missing.** None are in the repo except Cormack et al. (2009). Kneedle,
  isotonic regression, seqeval/conlleval, the tests, every model and every dataset need
  references.
- **D16. "Low-resource" fits only the CrossNER pools.** Only those pools are small. The
  "low-resource" framing should either be limited to them and S6, or be justified for the full
  pools.
