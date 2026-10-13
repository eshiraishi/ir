# TRICE: methodology (draft)

> Draft paper section. Every number below comes from `trice.toml`, `results/endpoints.json`,
> `docs/budget-partial.md` or the code in `src/trice/`, or is simple arithmetic on those numbers
> (marked "estimate"). No experimental result exists yet. Open points are marked **[Dn]**. Each one
> is listed in §13, "Decisions for the author", with options and a recommendation. Citations still
> to be filled are marked `[ref]`.

## 1. The task, in plain words

**Named entity recognition (NER)** means finding the names of things in a sentence and saying what
kind of thing each one is. For example, in *"Geoffrey Hinton worked on backpropagation at the
University of Toronto"*, the entities are *Geoffrey Hinton* (researcher), *backpropagation*
(algorithm) and *University of Toronto* (university). This sentence was made up for illustration;
it is not from any dataset.

**In-context learning** solves the task without training the model. A large language model (LLM)
receives a prompt with a few solved examples, called *demonstrations*, followed by the new sentence,
called the *query*. The model then writes the answer for the query in the same style.

The demonstrations come from a labeled *pool*, which is a dataset's training split. Four things
change the result:

- which demonstrations are shown;
- how many;
- in what order;
- in what format.

v1 of this work (Shiraishi et al., BRACIS 2025) showed that *retrieving* demonstrations similar to
the query beats choosing them at random. TRICE (Tuning-free Retrieval for In-Context Entity
recognition) keeps that idea and adds one constraint: **no setting is tuned on held-out data**.
Every choice is either fixed in advance (§3, §5) or made by a fixed rule that sees only the training
pool (§4, §5.2). No validation split is loaded anywhere (`src/trice/data.py`).

```
query ─┬─ sparse scores (whole pool) ─┐
       └─ dense scores  (whole pool) ─┴─ RRF fusion ─ top 200 ─ reranker ─ top k* ─ shuffle
          ─ prompt (frozen format) ─ LLM (greedy) ─ parse and align ─ entity-level micro-F1
```

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
| CoNLL-2003 (the copy shipped with CrossNER) | news | 4 | 14,041 | 3,453 |
| MIT-Movie ("eng" corpus) | movie queries | 12 | 9,775 | 2,443 |
| MIT-Restaurant | restaurant queries | 8 | 7,660 | 1,521 |
| WNUT-17 | social media | 6 | 3,394 | 1,287 |
| NCBI-Disease | biomedical | 1 | 5,432 | 940 |
| **Total** | | | | **12,150** |

- **Sources:** git sources (CrossNER, WNUT-17, NCBI-Disease) are pinned by commit. The MIT corpora
  are plain downloads from `groups.csail.mit.edu`. Every file is pinned by SHA-256 and by sentence
  count in `trice.toml`. `trice data` refuses any file that does not match.
- **Preprocessing:**
  - Each line holds one token and its IOB2 tag. Blank lines separate sentences.
  - `-DOCSTART-` lines are dropped.
  - Each raw label maps one-to-one to a natural-language name shown to the model (for example
    `programlang` → "programming language"). These verbalizations are fixed in
    `[datasets.*].labels`.
- **Pool sizes:** only the five CrossNER pools are small (100 or 200 sentences). The other five
  datasets use their full training splits as pools **[D16]**.
- **S6 pools:** study S6 uses nested CoNLL-2003 pools of 50, 100, 200, 400, 800 and 1,600
  sentences. Each is the first *n* sentences of one seeded shuffle, so smaller pools are contained
  in larger ones. The test split is never subsampled.

## 3. Retrieval

### 3.1 The default stack (`small`)

Three models score how similar each pool sentence is to the query. All are under 1B parameters
and Apache-2.0, and each is pinned to a Hugging Face commit in `[retrieval.stacks.small]`.

1. **Learned sparse retrieval.** `opensearch-neural-sparse-encoding-doc-v3-gte` represents a
   sentence as weighted vocabulary terms. It works like keyword matching with learned weights. Each
   side goes through the model's own encoder (`encode_query` / `encode_document` in
   sentence-transformers). The model needs remote code **[D18]**.
2. **Dense retrieval.** `Qwen3-Embedding-0.6B` maps each sentence to a vector, and the score is
   cosine similarity. No instruction is prepended, so queries and pool sentences are embedded the
   same way **[D9]**.
3. **Fusion.** Reciprocal rank fusion (RRF; Cormack et al., 2009) combines the two rankings:
   score(d) = Σ 1/(60 + rank(d)), summed over the two rankings. The fusion uses each scorer's
   ranks over the whole pool. The constant 60 comes from the literature and is never tuned.
4. **Reranking.** `Qwen3-Reranker-0.6B` rescores the 200 best fused candidates. It reads the query
   and a candidate together with a fixed instruction: *"Given a sentence, retrieve sentences that
   mention similar kinds of named entities in a similar context."* The score is the probability of
   "yes" against "no", as on the model card. A depth of 200 covers the whole pool for every CrossNER
   dataset.

Some rules apply to every stage:

- **Ties:** ties are broken by ascending sentence id.
- **Storage:** the top 200 ids of every stage (`sparse`, `dense`, `rrf`, `full`) are stored, so
  the S2 ablation can use each stage alone.
- **Determinism:** rankings are computed once in fp32 with
  `torch.use_deterministic_algorithms(True)`. They are stored as Hugging Face datasets with their
  SHA-256 (`trice rank`), and everything downstream only reads them.

### 3.2 Comparison stacks (study S2)

- **Single stages:** each stage of `small` on its own.
- **`large`:** the 8B Qwen3 embedding and reranker, both called through OpenRouter **[D10]**.
- **v1's retrievers, replicated:**
  - Okapi BM25 (k1 = 1.2, b = 0.75) over lowercased spaCy lookup lemmas;
  - `bge-large-en-v1.5` with v1's query instruction.

### 3.3 Leave-one-out and duplicates

- **Leave-one-out queries:** when a training sentence plays the query (§4), the query and any pool
  sentence with identical text are removed from its candidates.
- **Random condition:** the same identical-text filter applies to test queries as well.
- **Retrieval for test queries:** the filter does not apply **[D1]**.

### 3.4 Selecting and ordering demonstrations

There are three conditions:

- **Retrieval:** the *k* top-ranked pool sentences.
- **Random:** *k* pool sentences drawn uniformly without replacement, with a seed derived from
  (random seed index, dataset, query id).
- **Zero-shot:** no demonstrations.

In every condition the chosen demonstrations are sorted by id, then shuffled with a seed derived
from (dataset, query id). The prompt order therefore never reveals the retrieval rank. The same set
of demonstrations also gets the same order in every condition and for every model.

## 4. Choosing the number of demonstrations k\*

k\* is chosen per (model, dataset) on the training pool only, by **leave-one-out (LOO)**: each
pool sentence in turn plays the query, and the rest of the pool is its demonstration pool.

- **LOO queries:** pools of up to 200 sentences (all CrossNER pools) are used whole. Larger pools
  use a seeded sample of 200 queries. That makes 1,700 LOO queries per model.
- **Grid:** LOO micro-F1 is computed with `small/full` retrieval at k ∈ {0, 1, 2, 4, 8, 16, 32, 64},
  keeping only k smaller than the pool size **[D17]**.
- **Smoothing:** the curve is smoothed with isotonic (non-decreasing) regression, which removes
  noise without adding a parameter.
- **Knee:** k\* is the Kneedle knee `[ref]` of the smoothed curve (S = 1, concave, increasing). The
  knee is computed on a linear k axis, because prompt cost grows linearly with k. It marks the
  point where more demonstrations stop paying off.
- **No knee:** k\* is the smallest k that reaches the isotonic maximum.
- **Robustness:** the 1-SE rule is reported but never used. It gives the smallest k whose LOO F1
  is within one bootstrap standard error of the best k.
- **S6 pools:** each S6 pool gets its own k\* by the same rule (study S6-loo).

### 4.1 The range of the k grid (open, to be decided before the tag)

Input tokens grow roughly linearly with k, at about 100 tokens per demonstration. For comparison,
S0 averages 813 input tokens per call at k = 8. The grid's eight points use 0 + 1 + 2 + … + 64 = 127
demonstrations per LOO query, and the k = 64 point accounts for 64 of them, so it carries about half
of S1-loo's demonstration tokens. The grid's top also bounds k\*, and with it the cost of every
test-set study that runs at k\* (S1, S2, S3, S6).

There are three options:

- **Cap the grid at 32:**
  - *Gains:* this removes one eighth of S1-loo's calls and about half of its demonstration tokens.
    It also caps every k\*, so the test-set studies at k\* can cost at most about half of the
    k = 64 worst case.
  - *Costs:* if a LOO curve is still rising at 32, the smoothed curve has no knee within the grid.
    k\* then lands on the boundary, and the paper cannot say whether more demonstrations would have
    helped.
- **Probe one model first:**
  - *What it is:* run S1-loo with one model, then fix the grid before the tag.
  - *Cost:* a hosted model needs `allow_spend`. At S1-loo's measured averages (about 1,716 input
    and 77 output tokens per call), Qwen3.5-9B's 13,600 LOO calls would cost about $2.5
    (estimate). A local Qwen3.5 size costs nothing in API fees.
  - *Gains:* this bases the decision on real curves.
  - *Costs:* the grid then depends on training-set results from one model, and the pre-registration
    must say so. The curve shape can also differ across model sizes.
- **Keep 64:**
  - *Gains:* the grid and its LOO cost are already measured ($42.14). A knee or plateau below 64 is
    then observed rather than assumed.
  - *Costs:* the S1–S6 costs depend on where the knees fall and are still unmeasured **[D11]**.
    Also, for 100-sentence CrossNER pools, k = 64 shows two thirds of the pool. Retrieved and random
    demonstrations then overlap heavily, so the comparison says little about retrieval at that k.

This is decision **D17**; it is framed here and left to the author.

## 5. Prompt formats

### 5.1 The four candidates

| Format | Prompt | Expected answer for the example in §1 |
|---|---|---|
| **P0** | v1's format: one user message with a definition line ("Defn: An entity is a …"), then `Example i:` / `Answer:` blocks; stops at `\nExample ` | `1. Geoffrey Hinton (researcher)` / `2. backpropagation (algorithm)` / `3. University of Toronto (university)` |
| **P1** | chat format: a system message lists the types; each demonstration is a user turn followed by an assistant turn | the same numbered list, or `None` when there are no entities |
| **P2** | P1 plus a strict JSON Schema (structured generation) whose `type` field is an enum of the types | `{"entities": [{"text": "Geoffrey Hinton", "type": "researcher"}, …]}` |
| **P3** | P1, but the answer repeats the sentence with inline tags | `<researcher>Geoffrey Hinton</researcher> worked on <algorithm>backpropagation</algorithm> at the <university>University of Toronto</university>` |

All prompt text is in Jinja templates (`src/trice/templates/p0.j2`, `system.j2`, `answer.j2`).
Demonstration answers are generated from gold tags by the same template the model is asked to
imitate.

### 5.2 Choosing the format (study S0)

- **Runs:** the two reference models (Qwen3.5-9B and Qwen3.5-27B) run LOO with every format at
  k = 8, using `small/full` retrieval.
- **Rule:** the format with the highest mean micro-F1 over the 10 datasets × 2 models is frozen
  for every model and study. Ties go to the earlier format (P0 < P1 < P2 < P3).
- **Output:** `results/frozen_format.json`.

## 6. Models and serving

There are eleven instruction-tuned models in three size ladders. The two reference models run
every ablation.

| Model | Ladder | Params (B, total / active) | Endpoint | Precision | Price (USD / M tokens, in / out) |
|---|---|---|---|---|---|
| Qwen3.5-0.8B | Qwen3.5 dense | 0.8 / 0.8 | local vLLM | bf16 | – |
| Qwen3.5-2B | Qwen3.5 dense | 2 / 2 | local vLLM | bf16 | – |
| Qwen3.5-4B | Qwen3.5 dense | 4 / 4 | local vLLM | bf16 | – |
| Qwen3.5-9B (reference) | Qwen3.5 dense | 9 / 9 | DeepInfra | bf16 | 0.10 / 0.15 |
| Qwen3.5-27B (reference) | Qwen3.5 dense | 27 / 27 | DeepInfra | fp8 | 0.26 / 2.60 |
| Qwen3.5-35B-A3B | Qwen3.5 MoE | 35 / 3 | DeepInfra | fp8 | 0.14 / 1.00 |
| Qwen3.5-122B-A10B | Qwen3.5 MoE | 122 / 10 | Alibaba | `unknown` **[D8]** | 0.26 / 2.08 |
| Qwen3.5-397B-A17B | Qwen3.5 MoE | 397 / 17 | DeepInfra | fp8 | 0.45 / 3.00 |
| Gemma 3 4B | Gemma 3 | 4 / 4 | DeepInfra | bf16 | 0.05 / 0.10 |
| Gemma 3 12B | Gemma 3 | 12 / 12 | DeepInfra | bf16 | 0.05 / 0.15 |
| Gemma 3 27B | Gemma 3 | 27 / 27 | NextBit | fp8 | 0.07 / 0.30 |

- **Decoding:** every request uses greedy decoding (temperature 0, top-p 1, seed 0), with at most
  1,024 output tokens.
- **Thinking:** all Qwen3.5 models are hybrid reasoners, and thinking is turned off. OpenRouter
  gets `reasoning.enabled = false`, and vLLM gets `chat_template_kwargs.enable_thinking = false`.
  The models are therefore evaluated as plain instruction followers **[D20]**.
- **Hosted models:** OpenRouter, pinned by `trice pin` to one provider and one quantization, with
  `allow_fallbacks = false` and `require_parameters = true`. The pin rule works in three steps:
  1. keep only the providers that support every required parameter, including structured outputs
     (needed by P2) and `reasoning` for hybrid models;
  2. among those, prefer the most faithful precision (bf16 > fp16 > fp32 > fp8 > …);
  3. break remaining ties by the lowest price, then by provider name.
- **Local models:** the three smallest Qwen3.5 sizes are not on OpenRouter. They run on vLLM 0.31.0
  in bf16, at a pinned revision, with `VLLM_BATCH_INVARIANT=1` and `--generation-config vllm`
  (`envs/compose.yaml`).
- **Spend guard:** while `llm.allow_spend = false`, the code raises an error before any request can
  reach OpenRouter. This covers chat, embedding and rerank requests, and tests check it.

## 7. Studies

Studies S0–S4 and S6 are defined in `[studies]` of `trice.toml`. S5 is an analysis of S1's output.
S7 and S8 are checks implemented in `src/trice/runner.py`. Unless stated otherwise, a study runs on
the test split at k = k\* with the frozen format.

| Study | Question | Models | Data | Conditions | k / format | Planned requests¹ |
|---|---|---|---|---|---|---:|
| S0 | Which prompt format? | 2 reference | LOO | `small/full` | k = 8; P0–P3 | 13,600 |
| S1-loo | Which k\*? | all 11 | LOO | `small/full` | grid | 149,600 |
| S1 | Main result: does TRICE beat random and zero-shot? | all 11 | test | zero-shot, `random:0`, `small/full` | k\* | 400,950 |
| S2 | Retriever ablation | 2 reference | test | `random:1`, `random:2`, `small/{sparse,dense,rrf}`, `large/full`, `v1_bm25/full`, `v1_dense/full` | k\* | 194,400 |
| S3 | Format sensitivity | 2 reference | test | `small/full` | k\*; the 3 other formats | 72,900 |
| S4 | Regret of choosing k on training data only | 2 reference | test | `small/full` | grid | 194,400 |
| S5 | Does the retrieval gain change with model size? | all ladders | S1 output | – | – | 0 |
| S6-loo, S6 | Effect of pool size | Qwen3.5-9B | CoNLL pools of 50–1,600 | `small/full`; S6 adds `random:0` | grid / k\* | 7,550 + 41,436 |
| S7 | Run-to-run nondeterminism | **[D12]** | 300 seeded S1 `small/full` prompts, sent 3 times with the cache bypassed | – | – | 900 per model |
| S8 | Contamination probe | **[D12]** | the 40 lines after the first 40 of each raw test file | – | – | 10 per model |

¹ Runs × queries, computed from `trice.toml`. This is an upper bound, because the cache serves
identical requests once. For example, the S4 runs at k = k\* repeat S1's runs.

**Specialist baselines** are scored with the same aligner and scorer, as a descriptive reference
**[D14]**:

- **Zero-shot:** GLiNER2-large, GLiNER2.5-base, GLiNER-v2.5-large, NuNER-Zero and GLiNER-BioMed.
- **Fine-tuned:** the last three are also fine-tuned on each training pool, with library-default
  hyperparameters, no validation set, and seeds 0, 1 and 2.
- **Environment:** they run in their own Docker image (`envs/baselines`).

## 8. Metrics

Model output is turned back into one tag per query token (`core/formats.py`):

1. **Read the candidate mentions:**
   - P0 and P1: lines matching `N. text (type)`;
   - P2: the JSON items;
   - P3: the `<tag>…</tag>` spans.
2. **Map each type to a dataset label.** The match ignores case. The type may be written as the
   verbalization, the raw label or the tag form.
3. **Place each mention in the query:**
   - P3: at the position the model gave, when that position is valid;
   - P0–P2: at the first occurrence of the exact token sequence that is not already tagged.
4. **Discard invalid mentions:** a mention with an unknown type or no free occurrence is invalid
   and dropped.
5. **Unreadable output:** it yields no predictions, so its gold entities count as misses.

This alignment differs from v1's surface-string matching only when a mention string repeats within
a sentence.

- **Primary metric:** entity-level micro-F1 with seqeval (conlleval semantics) over IOB2 tags,
  computed per (model, dataset, condition) cell. A prediction is correct only when both its
  boundaries and its type match. Per-sentence true positives, false positives and false negatives
  are kept, because every test resamples sentences.
- **Secondary metrics:** per-type F1, sentence exact match, and the invalid-mention rate.

## 9. Statistics

The tests are fixed in `docs/preregistration.md` §5 and implemented in
`src/trice/evaluation/inference.py` and `src/trice/analyzer.py`. All use α = 0.05 and 10,000
resamples.

- **One cell (A against B):**
  - two-sided paired permutation test on the micro-F1 difference: the two systems' outputs are
    swapped within each sentence, and the test is exact when 2^n ≤ 10,000;
  - 95% paired-bootstrap percentile CI;
  - exact McNemar test on sentence exact match;
  - Holm correction within each family **[D5]**.
- **Across datasets, per model (H1, H2):** Wilcoxon signed-rank test on the 10 per-dataset
  differences.
- **Several treatments (S2, S3):** Friedman test with Nemenyi post-hoc tests and a
  critical-difference diagram. The blocks are dataset × reference model, and the random seeds are
  averaged into one treatment.
- **Size trends (S5, H5):** Page's L test per ladder, on the per-dataset gain of TRICE over random:
  - sizes are ordered by total parameters, which gives the same order as active parameters;
  - both directions are tested, with Bonferroni × 2;
  - the trend of F1 itself is secondary.
- **Promised but not yet implemented:** the pre-registration promises analyses that the analyzer
  does not compute yet **[D4]**.

## 10. Determinism and reproducibility

### 10.1 What is pinned

| Item | Pin |
|---|---|
| Data | git commit + SHA-256 + sentence count per file (`trice.toml`) |
| Retrieval models | Hugging Face commit per model (`[retrieval.stacks]`); the sparse model's remote code is **not** pinned **[D18]** |
| Rankings | stored once, with the SHA-256 that `trice rank` prints **[D7]** |
| LLMs | OpenRouter provider + quantization, or local vLLM image + revision; tokenizer revision (`results/endpoints.json`) |
| Prompts, parsers, scorer | `src/trice/templates/`, `core/formats.py`, `evaluation/scoring.py` |
| Every model response | disk cache keyed by the SHA-256 of the request (`cache/responses`) |
| Software | `uv.lock`; Python 3.14; vLLM 0.31.0 and GLiNER in their own containers |

### 10.2 Seeds

Every seed is the first 32 bits of SHA-256(`"|".join(purpose, …)`) (`trice.schema.derive_seed`),
so it depends only on its purpose and never on call order. The purposes are:

- `loo`: the LOO sample;
- `pool`: the S6 pools;
- `random`: random demonstrations;
- `order`: the demonstration order;
- `permutation` and `bootstrap`: the tests;
- `se`: the 1-SE rule;
- `audit`: the S7 sample.

The LLM seed is 0.

### 10.3 Replay

A response is computed once and cached. `trice run <study> --mode replay` then reproduces every
prediction byte for byte without any network access, and an end-to-end test checks this. To
reproduce every table without paying for any model call, a reader needs three released artifacts:

- the response cache;
- the stored rankings;
- `results/*.json`.

### 10.4 Order of commands

```bash
uv sync && uv run pytest
uv run trice data                                   # fetch, verify hashes, build datasets
uv run trice rank --stack small --pools             # GPU in practice [D7]
uv run trice rank --stack v1_bm25 && uv run trice rank --stack v1_dense
uv run trice rank --stack large                     # paid: OpenRouter embeddings and rerank [D10]
uv run trice pin                                    # free metadata call; writes endpoints.json
uv run trice smoke <model>                          # paid: every format parses, thinking off
uv run trice budget --k 16 --k 64                   # dry run, no calls [D11, D21]
git tag prereg-v1                                   # freeze (docs/preregistration.md)
uv run trice run S0 && uv run trice select-format
uv run trice run S1-loo && uv run trice select-k
uv run trice run S1   # then S2, S3, S4; S6-loo, `select-k --study S6-loo`, S6
uv run trice audit <model> <day-tag>                # S7, on three days
uv run trice probe <model>                          # S8
uv run trice analyze                                # tables (.md/.tex/.json), figures (.pdf/TikZ)
```

## 11. Threats to validity

- **Unpinned remote code in the sparse encoder [D18].**
  - *What happens:* `trust_remote_code=True` loads modeling code from `Alibaba-NLP/new-impl`, which
    is not pinned by our config. That code targets transformers 4. TRICE needs transformers 5,
    because transformers 4 does not know `qwen3_5`.
  - *The patch:* `patch_remote_model` (`src/trice/core/encoders.py`) restores a removed
    attention-mask helper, ties the MLM decoder to the embeddings, reloads its bias from the
    checkpoint, and rebuilds the buffers that v5 leaves uninitialized.
  - *Verification so far:* the patched output matched a transformers 4.57.6 reference exactly
    (maximum absolute difference 0.0).
  - *Remaining risk:* a later change to the remote code could still change the rankings.
- **Hosted nondeterminism [D19].** Greedy decoding on a pinned provider can still differ between
  calls, for example through batching or kernel choice. S7 measures this. The cache makes every
  reported number replayable, but not re-derivable bit for bit from the live API.
- **Undisclosed quantization and mixed precision [D8].**
  - *The mix:* the pins mix bf16, fp8, and one quantization OpenRouter lists as `unknown`
    (Alibaba, Qwen3.5-122B-A10B). The local models run in bf16. Precision therefore varies within
    each ladder, which confounds the size trend (H5).
  - *The `unknown` pin:* it is sent to OpenRouter as `quantizations = ["unknown"]`. Whether
    OpenRouter routes that filter as intended has not been checked, because no smoke test has run.
- **Thinking mode [D20].** Thinking is turned off by request parameters, but only the smoke test
  verifies that, and it has not run. The check (`reasoning_tokens` is 0 or not reported) also
  passes when a provider does not report reasoning tokens at all. Results describe the models with
  thinking off, not their default behavior.
- **Hardware-dependent rankings [D7].** fp32 results can differ between CPU and GPU, or between
  GPU types, and such differences can reorder near-ties. The stored rankings are the reproducible
  artifact.
- **Possible leakage through duplicates [D1].** For test queries, retrieval can return a pool
  sentence identical to the query, while random sampling cannot.
- **Contamination.** Public test sets may be in the models' training data. S8 checks for verbatim
  memorization of the raw files only. It cannot rule out contamination that does not show up as
  memorization.
- **Comparability with v1.** v1's test sizes and scoring differ: for Music, 465 sentences against
  456 in v1 **[D6]**, and the mention alignment differs (§8). v1's numbers are therefore not
  directly comparable.
- **Generality.** The data covers English only, and five of the ten pools are full training sets
  **[D16]**.
- **Tooling.** `ty` reports 13 diagnostics that predate this work **[D22]**. None is in
  `src/trice/`:
  - 6 are in `envs/baselines/baselines.py`, including the unresolved `gliner` and `gliner2`
    imports, which are installed only in the Docker image;
  - 7 are in `tests/`.

## 12. Compute and cost

### 12.1 Retrieval

- **Measured speed:** on a 4-CPU machine the reranker took about 4.2 s per (query, candidate)
  pair, or about 14 minutes per query.
- **Workload:** the ten datasets need 13,850 ranked queries (12,150 test + 1,700 LOO).
- **Total time:** that is about 3,200 hours for the reranker alone. One 8-pair measurement gave
  0.58 s per pair, which would make it about 450 hours. Neither figure includes the S6 pools.
- **Status:** the rankings will be computed on a GPU. They have not been computed yet, and
  `trice rank` has not been tested end to end.

### 12.2 LLM calls

`trice budget` is a dry run with no model calls:

- it renders every planned prompt;
- it counts the prompt's tokens with each model's own tokenizer and chat template;
- it estimates output length as the gold answer's tokens × 1.25 + 8;
- it prices both at the pinned rates, counting identical requests once.

The full run did not finish on the 4-CPU machine, so it was split by study. Only the studies whose
k is fixed in advance are measured:

| Study | Calls | Input tokens | Output tokens | Cost (USD) |
|---|---:|---:|---:|---:|
| S0 | 13,600 | 11,056,168 | 829,748 | 3.13 |
| S1-loo | 149,578 | 256,647,296 | 11,527,310 | 42.14 |
| Subtotal | 163,178 | 267,703,464 | 12,357,058 | 45.27 (67.91 with a 1.5× margin) |

- **Not measured:** S1, S2, S3, S4, S6-loo and S6, because their cost depends on k\* **[D11]**.
- **Proxy pricing:** retrieval conditions were priced with random demonstrations of the same k and
  format, because no rankings exist yet **[D21]**.
- **`large` stack:** its embedding and rerank calls are not priced **[D10]**.
- **Local models:** they cost $0 here, because they run on the author's hardware.
- **Spent so far:** $0 on OpenRouter.

## 13. Decisions for the author

Each entry gives the options and a recommendation, except D17, which is left neutral as requested.
"Before the tag" means before `prereg-v1`.

**D1. Duplicate sentences in test-query retrieval.**

- *The issue:* for test queries, retrieval can return a pool sentence identical to the query, while
  random sampling excludes such sentences.
- *Options:*
  - (a) apply the identical-text filter to retrieval for test queries, which is a small change in
    `Retriever.rank`;
  - (b) keep the current behavior and report how many test queries have an exact duplicate in the
    pool;
  - (c) remove duplicates from the test sets.
- *Recommendation:* (a), plus the duplicate count from (b) as a descriptive table. (a) makes the
  conditions symmetric and matches the LOO rule. (c) would change the standard test sets.

**D2. H1 requires three random seeds, but only the reference models have them.**

- *Options:*
  - (a) apply the "all three seeds" clause to the reference models only; the other nine models are
    tested against `random:0`;
  - (b) add `random:1` and `random:2` to S1, which adds up to 2 × 11 × 12,150 = 267,300 requests.
- *Recommendation:* (a), stated explicitly in H1.

**D3. H3 ("not worse than v1") has no margin.**

- *Options:*
  - (a) a non-inferiority test: the lower bound of the one-sided 95% paired-bootstrap CI must be
    above −δ, with δ in F1 points fixed before the tag;
  - (b) rephrase H3 as a two-sided comparison, reported with CIs.
- *Recommendation:* (a) if a principled δ can be justified before the tag. Otherwise use (b)
  rather than choose δ after the fact.

**D4. Promised analyses missing from `analyzer.py`.** The missing analyses are:

- Holm across models for the Wilcoxon tests;
- Holm across ladders for Page's L;
- Kendall τ per dataset;
- the minimum detectable effect (MDE);
- CIs for the S4 regret;
- the S7 SD attached to each difference.

- *Options:*
  - (a) implement them all before the tag;
  - (b) remove them from the pre-registration;
  - (c) implement all but the per-difference S7 SD, and report S7 once as its own table.
- *Recommendation:* (c). Each of the others is a few lines over scipy and statsmodels (for
  example, the MDE at 80% power is about 2.8 × bootstrap SE). The S7 SD exists only for the audited
  models, so attaching it to every difference is not possible.

**D5. Holm family for S2 and S3.**

- *The issue:* the code corrects all alternatives in a study as one family, while the
  pre-registration says one family per comparison type.
- *Options:*
  - (a) change the analyzer to use one family per alternative across cells, which matches S1;
  - (b) keep the code and update the pre-registration; this is more conservative.
- *Recommendation:* (a), for consistency with S1.

**D6. CrossNER-Music has 465 test sentences against 456 in v1.**

- *Options:*
  - (a) use the pinned file and footnote the difference;
  - (b) find and reproduce v1's 456-sentence version.
- *Recommendation:* (a). The pinned file is the published CrossNER file at a fixed commit. Check
  v1's code or logs for the cause and mention it in the footnote.

**D7. Ranking hardware and determinism.**

- *The issue:* `device = "cpu"` in `trice.toml`, but rankings need a GPU.
- *Options:*
  - (a) set `device = "cuda"` before the tag, and set `CUBLAS_WORKSPACE_CONFIG=:4096:8`, which
    deterministic cuBLAS normally needs (unverified here);
  - (b) rank on CPU with more cores.
- *Recommendation:* (a), and also:
  - run `trice rank` end to end on one CrossNER dataset first, as the missing end-to-end test;
  - compare the top-k overlap between GPU and CPU on a small sample;
  - record the GPU model, driver and torch version;
  - publish the rankings with their SHA-256 values (for example on the Hugging Face Hub or Zenodo).

**D8. Mixed precision and the `unknown` pin.**

- *Options:*
  - (a) accept the mix and report it as a covariate in S5;
  - (b) re-pin towards bf16. Structured outputs are required only because of P2, so if S0 freezes
    another format, dropping `response_format` and `structured_outputs` from the required
    parameters may make bf16 providers eligible. That would require re-pinning between S0 and
    S1-loo;
  - (c) for the `unknown` pin, send only the provider and no quantization filter. This is a small
    change in `LLM.render_request`.
- *Recommendation:* (a), plus (c). (b) changes endpoints in the middle of the protocol. Before the
  tag, run `trice pin` with the reduced parameter list as a free check of whether bf16 providers
  exist. Then smoke-test the 122B model first, since it is the riskiest pin.

**D9. Dense retriever without a query instruction.**

- *Options:*
  - (a) keep it symmetric;
  - (b) reuse the reranker's instruction for queries.
- *Recommendation:* (a). Query and pool are the same kind of text (sentences), and an instruction
  would be one more untuned choice. Say this in the paper.

**D10. The `large` stack is unbudgeted, and its instruction support is unchecked.**

- *Recommendation:* count its embedding and rerank tokens offline with the pool and query texts,
  which is free, and price them before the tag. Then check in OpenRouter's documentation whether
  `/rerank` accepts an instruction. If it does not, report that `large/full` reranks without one,
  which confounds the small-versus-large comparison.

**D11. S1–S6 costs are unmeasured.**

- *Options:*
  - (a) run the per-study dry runs at `--k 16` and `--k 64` on a larger machine before the tag;
  - (b) set a spending cap now and measure after S1-loo.
- *Recommendation:* both. Bound the cost with (a) before the tag, then re-price with the real k\*
  and real rankings before spending on S1.

**D12. S7 and S8 coverage.**

- *Recommendation:*
  - run S7 on all eight hosted models: 300 prompts × 3 days × 8 models = 7,200 requests;
  - run S7 on one local model too, to show that vLLM is deterministic;
  - schedule the three S7 days at least 24 hours apart, within the S1 window;
  - run S8 on all eleven models (110 requests).

**D13. Failure policy (not defined yet).**

- *Recommendation:* pre-register these rules:
  - **Truncated outputs** (`finish_reason = length`): parse them as they are, and report the
    truncation rate per cell.
  - **Refusals and empty outputs:** treat them as no predictions.
  - **Requests that fail after 8 retries:** retry them later. A cell is reported only when it is
    complete.
  - **A withdrawn or changed endpoint:** re-pin by the same rule and rerun every cell of that
    model, including S1-loo and its k\*. Report this as a deviation.

**D14. Role of the specialist baselines.**

- *Options:*
  - (a) descriptive only;
  - (b) add a hypothesis.
- *Recommendation:* (a). Fine-tuned baselines and TRICE see the same labeled pool, but the
  fine-tuned baselines use library-default hyperparameters with no tuning. A hypothesis would need
  a fairness argument that is not settled yet.

**D15. Citations.**

- *The issue:* only Cormack et al. (2009) is in the repo.
- *Recommendation:* reuse v1's bibliography, and add:
  - Kneedle and isotonic regression;
  - seqeval/conlleval;
  - the permutation, bootstrap, Holm, Friedman/Nemenyi, Wilcoxon and Page tests;
  - every model and dataset;
  - OpenRouter and vLLM.

**D16. Scope of the "low-resource" claim.**

- *Recommendation:* lead with "tuning-free", limit low-resource claims to CrossNER and S6, and
  describe the other five datasets as full-pool settings.

**D17. k grid up to 64, up to 32, or set after a one-model probe.** This is framed in §4.1. The
recommendation is left to the author, as requested. Whatever the choice, it must be in
`trice.toml` before the tag.

**D18. The sparse encoder's remote code is unpinned.**

- *Options:*
  - (a) pin the code revision: transformers accepts `code_revision` for code in a separate
    repository (unverified for this model and sentence-transformers' `SparseEncoder`);
  - (b) vendor the remote modules at a recorded commit, and record their SHA-256 values;
  - (c) record the resolved `Alibaba-NLP/new-impl` commit at ranking time and rely on the stored
    rankings.
- *Recommendation:* (b). It pins exactly what runs, and the patch already depends on that code's
  internals. Use (c) as a minimum.

**D19. Interpreting hosted nondeterminism.**

- *Recommendation:* pre-register how S7 is read:
  - report exact-output agreement and the F1 SD across the three repetitions;
  - treat any difference smaller than that SD as not meaningful;
  - run S7's first repetition at the start of S1, so that drift can be seen during the main runs.

**D20. Verifying that thinking is off.**

- *The issue:* the smoke check passes when a provider does not report reasoning tokens.
- *Options:*
  - (a) also fail the smoke test when the output contains `<think>`, or when the completion tokens
    greatly exceed the visible text;
  - (b) keep the current check.
- *Recommendation:* (a), and also state in the paper that the models were evaluated with thinking
  off.

**D21. Proxy pricing in the budget.**

- *The issue:* retrieved demonstrations resemble the query, so their length can differ
  systematically from random ones, in a direction that is not known.
- *Recommendation:* treat the measured S0 and S1-loo costs as estimates until the rankings exist,
  then rerun `trice budget`. That run uses the stored rankings automatically.

**D22. Pre-existing `ty` diagnostics.**

- *Options:*
  - (a) fix them;
  - (b) suppress each one with a reason;
  - (c) leave them and record them.
- *Recommendation:*
  - for the two `gliner` imports, (b), or exclude `envs/baselines` from `ty`, because those packages
    exist only in the container;
  - for the tests, (a) where the fix is local, otherwise (b).

  Do this before the tag, so that the frozen code is type-clean and a future diagnostic stands
  out.
