# TRICE: Tuning-free Retrieval for In-Context Entity recognition

TRICE is v2 of *Information Retrieval and In-Context Learning for Low-Resource Named Entity
Recognition* (Shiraishi, de Camargo, Silva, Prati; BRACIS 2025, LNAI 16180, pp. 135-149). An LLM
recognizes entities from demonstrations retrieved from a small labeled pool, and nothing is tuned
on held-out data:

- **Retrieval:** learned sparse plus dense retrieval, fused with RRF (k = 60), then reranked. The
  default stack uses only models under 1B parameters.
- **Choosing k:** the number of demonstrations k\* comes from leave-one-out curves on the
  training pool, smoothed with isotonic regression; k\* is the Kneedle knee.
- **Prompt format:** chosen on training data among four candidates.
- **Demonstration order:** shuffled with a per-query seed.

The method is described in `docs/methodology.md`; the pre-registered design is in
`docs/preregistration.md`.

## Design

- **One config:** `trice.toml` holds every setting (datasets with pinned commits and SHA-256,
  models, retrieval stacks, studies, sampling, statistics, and the `[layout]`: a path string with
  `{placeholders}` for every file the pipeline reads or writes, filled with `str.format`). Secrets go in `.env` (`OPENROUTER_API_KEY`, `HF_TOKEN`). Both are
  parsed by `trice.config.Config` (pydantic-settings); nothing derives settings with functions.
- **Composition root:** `trice.compose.compose` is one explicit function that builds and wires
  every component and returns them as an `App`; it is a context manager that closes the cache and
  clients. The CLI and the tests use it, and a test replaces the network by passing a `transport`.
- **Protocols, not base classes:** `Scorer`, `Reranker`, `Format`, `Completer` and `Transport` are
  `typing.Protocol`s; implementations are plain pydantic models that satisfy them.
- **Modular:**
  - `core/` is the ML (retrieval, the four prompt formats, train-only selection), with no I/O,
    CLI or network.
  - `evaluation/` is scoring, statistics (`Inferrer`) and reports (`Reporter`).
  - `runner.py` runs studies.
  - `analyzer.py` writes the pre-registered tables and figures.
- **Libraries over own code:**

  | Job | Library |
  |---|---|
  | Data | `datasets` |
  | Scoring | `evaluate` / `seqeval` |
  | Statistics | scipy, statsmodels, scikit-posthocs |
  | Model calls | official `openai` SDK (OpenRouter and local vLLM), `diskcache` record/replay |
  | Retrieval | sentence-transformers |
  | k\* selection | `kneed` |
  | Figures | seaborn, exported to TikZ with `matplot2tikz` |

- **Text lives in templates:** every prompt and report string is a Jinja template in
  `src/trice/templates/`.

```
trice.toml                 the configuration
src/trice/
  config.py  schema.py  data.py  llm.py  runner.py  analyzer.py  budget.py  compose.py  cli.py
  core/        retrieval.py  encoders.py  formats.py  selection.py
  evaluation/  scoring.py  inference.py  report.py
  templates/   prompts (p0, system, answer, probe, reranker) and reports (tables, budget)
envs/compose.yaml   Docker services: `serve` (vLLM, for model sizes OpenRouter does not serve) and `baselines`
envs/baselines/     Dockerfile + baselines.py for the GLiNER / GLiNER2 / NuNER-Zero baselines
```

vLLM 0.31 and gliner 0.2.29 pin older torch/transformers, which is why those two run in their own
Docker containers. The core runs on Python 3.14.6 with the newest releases.

```bash
# local model sizes (OpenAI-compatible API at llm.local_url)
MODEL=Qwen/Qwen3.5-0.8B REVISION=<hf-sha> docker compose -f envs/compose.yaml up serve
# specialist baselines: tags are the [baselines.*] entries of trice.toml; results/baselines/ is the output
docker compose -f envs/compose.yaml run --rm baselines predict gliner2-large
docker compose -f envs/compose.yaml run --rm baselines finetune nuner-zero
uv run trice baselines                 # score them with the shared aligner and scorer
```

## Use

```bash
uv sync && uv run pytest
uv run trice data                      # fetch pinned files, verify hashes, build HF datasets
uv run trice rank --stack small --pools
uv run trice pin                       # one OpenRouter provider + quantization per model
uv run trice smoke qwen3.5-9b
uv run trice budget --k 16 --k 64      # exact token budget, no model calls
uv run trice run S0 && uv run trice select-format
git tag prereg-v1                      # freeze (docs/preregistration.md)
uv run trice run S1-loo && uv run trice select-k
uv run trice run S1                    # ... S2, S3, S4, S6-loo, S6
uv run trice audit qwen3.5-9b day1     # S7, repeat on other days
uv run trice probe qwen3.5-9b          # S8
uv run trice analyze                   # results/tables: .md, .tex, .json, figures as .pdf + TikZ
```

What the environment needs:

- **Network allowlist:** `huggingface.co` and `*.hf.co`, `openrouter.ai`, `github.com`,
  `groups.csail.mit.edu`.
- **Gated tokenizers:** the Gemma 3 tokenizers need an `HF_TOKEN` with the Gemma license
  accepted.

## Determinism

- **Data:** pinned by git commit and SHA-256.
- **Rankings:** computed once in fp32 with deterministic torch, then stored.
- **Seeds:** every seed is a hash of what it is for.
- **Model calls:** greedy decoding on a pinned provider and quantization, with thinking off.
  Every response is cached under the hash of its request, so `trice run --mode replay`
  reproduces predictions byte for byte. The end-to-end test checks this.
