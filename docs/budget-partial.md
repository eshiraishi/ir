# Partial token budget (dry run, no model calls)

`trice budget` for the whole plan did not finish on the 4-CPU machine (one process needs more than
2 hours), so it was split by study. Only the studies whose k is fixed in `trice.toml` completed;
the rest depend on the assumed k* and are **not measured**. Prices are the pinned OpenRouter
endpoints in `results/endpoints.json`; `qwen3.5-0.8b/2b/4b` run on local vLLM and cost nothing here.

| Study | k | Calls | Input tokens | Output tokens | Cost (USD) |
|---|---|---:|---:|---:|---:|
| S0 | 8 (fixed) | 13,600 | 11,056,168 | 829,748 | 3.13 |
| S1-loo | grid (fixed) | 149,578 | 256,647,296 | 11,527,310 | 42.14 |
| **Measured subtotal** | | 163,178 | 267,703,464 | 12,357,058 | **45.27** (x1.5 ask: 67.91) |

Not measured: S1, S2, S3, S4, S6-loo, S6 (k = k*, so they scale with the k* ask of 16 to 64).
Retrieval conditions were priced with random demonstrations of the same length, because no rankings
exist yet. Money actually spent on OpenRouter so far: $0.
