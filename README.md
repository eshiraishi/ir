# TRICE: Tuning-free Retrieval for In-Context Entity recognition

TRICE is the second version of *Information Retrieval and In-Context Learning for Low-Resource
Named Entity Recognition* (Shiraishi, de Camargo, Silva, Prati; BRACIS 2025, LNAI 16180,
pp. 135-149). An LLM recognizes entities from demonstrations retrieved from a small labeled
pool. v2 removes every hyperparameter that v1 tuned on held-out data:

- **Retrieval:** learned sparse plus dense retrieval, fused with RRF (k = 60), then reranked.
  The default stack uses only models under 1B parameters.
- **Choosing k:** the number of demonstrations k\* is chosen from the training pool alone.
  Leave-one-out curves are smoothed with isotonic regression and k\* is the Kneedle knee.
- **Prompt format:** chosen on training data among four candidates (PromptNER list, chat list,
  JSON with schema-constrained decoding, inline tags).
- **Demonstration order:** shuffled with a per-query seed, never rank order.

The evaluation covers 10 English datasets and 11 open-weight models in two size ladders
(Qwen3.5, Gemma 3), with pre-registered statistics (`docs/preregistration.md`).

## Layout

```
configs/            datasets, models, retrieval stacks, studies (single source of truth)
src/trice/
  data/             pinned fetch (git commit / SHA-256) -> canonical JSONL + manifest
  retrieval/        scorers, exact RRF, reranking, stored rankings, demonstration selection
  prompt/           formats P0-P3, label verbalizations, parsers
  llm/              OpenAI-compatible client, OpenRouter pinning, record/replay cache
  eval/             scorer, statistical tests
  select_k.py       isotonic + Kneedle k selection
  budget.py         dry-run token budget with each model's own tokenizer
  studies.py        study expansion shared by the runner and the budget
  runner.py         demos -> prompt -> cached call -> parse -> predictions
  analysis.py       S0/k* decisions and pre-registered tables
  audit.py          S7 determinism audit, S8 contamination probe
envs/serve/         separate uv project: vLLM server for sizes OpenRouter does not serve
envs/baselines/     separate uv project: GLiNER / GLiNER2 / NuNER-Zero baselines
docs/               pre-registration
tests/              unit tests and an end-to-end run against a mock OpenRouter
```

There are three uv projects because vLLM 0.31 pins torch 2.13 / transformers < 5.18 and gliner
0.2.29 pins transformers < 5.17. The core uses the newest releases (Python 3.14.6, torch 2.14.1,
transformers 5.19, sentence-transformers 6.1).

## Setup

```bash
uv sync                       # core: data, prompts, client, stats, budget (no torch)
uv sync --extra retrieval     # + torch / sentence-transformers for the local retrieval stack
uv sync --extra v1            # + bm25s / spaCy for the v1 BM25 replica
uv run pytest
```

What the environment needs:

- **Network allowlist:** `huggingface.co` and `*.hf.co` (models and tokenizers), `openrouter.ai`,
  `github.com` (pinned data repos) and `groups.csail.mit.edu` (MIT corpora).
- **Secrets:**
  - `OPENROUTER_API_KEY`
  - `HF_TOKEN`, with the Gemma license accepted (the Gemma 3 tokenizers are gated).
- **GPU (optional):** set `TRICE_DEVICE=cuda` for the local retrieval stack, and use
  `envs/serve/serve.sh` for any model size OpenRouter does not serve.

## Pipeline (milestones)

```bash
uv run trice data build                     # M1: fetch, verify hashes, canonical splits
uv run trice rank --stack small --pools     # M2: rankings for test + LOO (+ S6 pools)
uv run trice rank --stack v1_bm25           #     v1 replicas and the API-backed large stack
uv run trice endpoints                      # M4: pin providers, quantizations, tokenizer revisions
uv run trice smoke -m qwen3.5-9b            #     every format parses, thinking off, pin honoured
uv run trice budget --k 16 --k 64           # M5: exact token budget, no model calls
uv run trice run -s S0 && uv run trice select-format   # M6: format chosen on training data
git tag prereg-v1                           # M7: freeze (see docs/preregistration.md)
uv run trice run -s S1-loo && uv run trice select-k    # M8: k* per (model, dataset)
uv run trice run -s S1   # ... S2, S3, S4, S6-loo, S6  # M9
uv run trice audit -m qwen3.5-9b --tag day1 # S7 (repeat on different days)
uv run trice contamination -m qwen3.5-9b    # S8
uv run trice analyze                        # M10: results/tables/*.json
```

Baselines (zero API tokens):

```bash
cd envs/baselines
uv run python predict.py --library gliner2 --model fastino/gliner2-large-v1 --revision <sha> --tag gliner2-large
uv run python finetune.py --model urchade/gliner_large-v2.5 --revision <sha> --tag gliner-v2.5-large --seed 0
cd ../.. && uv run trice score-baselines
```

## Determinism

- **Data:** pinned by git commit and SHA-256.
- **Rankings:** computed once in fp32 with deterministic torch algorithms, then stored and hashed.
- **Seeds:** every seed is a hash of its purpose, so it never depends on call order.
- **Model calls:** greedy decoding on a pinned provider and quantization. Every response goes to
  a SQLite record/replay cache, so `trice run --mode replay` and `trice analyze` reproduce
  results byte for byte. The end-to-end test checks this.

## Status

**Implemented and tested here** (Python 3.14.6; ruff and ty clean; 51 tests):
- data layer
- prompts and parsers
- scorer and statistics
- k selection
- client and cache
- ranking pipeline
- runner
- budget
- analysis
- CLI

**Built here from the pinned git sources, with the expected split sizes:**
- CrossNER ×5
- CoNLL-2003
- WNUT-17
- NCBI-disease

**Corrections to v1's Table 2** found on the way:
- CrossNER Music has 465 test sentences (not 456).
- CrossNER AI has 14 labels and Literature 12 (not 12 and 11).

**Pending, needs the allowlist and keys above:**
- MIT hashes, sentence counts and label names (fail-loud checks are in place).
- Model downloads for the retrieval stacks.
- `trice endpoints`, which also confirms the OpenRouter slugs and rerank API shape.
- The budget with real tokenizers.
