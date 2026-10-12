#!/usr/bin/env bash
# Serve one model locally with deterministic settings, exposing the same OpenAI-compatible API
# the core `trice` client uses for OpenRouter.
#
#   envs/serve/serve.sh Qwen/Qwen3.5-0.8B <hf-revision> [port]
#
# - bf16 weights, greedy decoding is requested per call by the client (temperature=0, seed=0);
# - VLLM_BATCH_INVARIANT=1 makes outputs independent of batch composition;
# - --generation-config vllm ignores the model's generation_config.json defaults (e.g. Qwen's
#   recommended presence penalty), so only the parameters the client sends apply.
set -euo pipefail
model="${1:?model repo}"; revision="${2:?pinned HF revision}"; port="${3:-8000}"
cd "$(dirname "$0")"
export VLLM_BATCH_INVARIANT=1
exec uv run vllm serve "$model" \
  --revision "$revision" \
  --dtype bfloat16 \
  --seed 0 \
  --generation-config vllm \
  --port "$port"
