"""Endpoint discovery and pinning (M4): choose one OpenRouter provider per model and record it,
with prices and the HF tokenizer revision, in configs/endpoints.lock.json."""

import json
from pathlib import Path

import httpx

from trice.config import LLMSpec, repo_root
from trice.llm.client import OPENROUTER_URL, EndpointPin

# Prefer the most faithful weights; within a precision, the cheapest provider; then by name.
QUANT_PREFERENCE = ("bf16", "fp16", "fp32", "fp8", "int8", "fp6", "fp4", "int4", "unknown")
# Every format must work on the pinned provider (P2 needs structured outputs).
REQUIRED = frozenset(
    {"temperature", "top_p", "seed", "max_tokens", "stop", "response_format", "structured_outputs"}
)
REQUIRED_THINKING = REQUIRED | {"reasoning"}


def lock_path() -> Path:
    return repo_root() / "configs" / "endpoints.lock.json"


def fetch_endpoints(slug: str, client: httpx.Client | None = None) -> list[dict]:
    client = client or httpx.Client(base_url=OPENROUTER_URL, timeout=60)
    r = client.get(f"/models/{slug}/endpoints")
    r.raise_for_status()
    return r.json()["data"]["endpoints"]


def _quant_rank(e: dict) -> int:
    q = (e.get("quantization") or "unknown").lower()
    return QUANT_PREFERENCE.index(q) if q in QUANT_PREFERENCE else len(QUANT_PREFERENCE)


def choose_endpoint(endpoints: list[dict], thinking: bool) -> dict | None:
    required = REQUIRED_THINKING if thinking else REQUIRED
    ok = [e for e in endpoints if required <= set(e.get("supported_parameters") or [])]
    if not ok:
        return None
    return min(
        ok,
        key=lambda e: (
            _quant_rank(e),
            float(e["pricing"]["prompt"]),
            float(e["pricing"]["completion"]),
            e.get("provider_name") or e.get("name") or "",
        ),
    )


def pin_for(spec: LLMSpec, endpoints: list[dict]) -> dict:
    """Lock entry for one model: an OpenRouter pin, or a local (vLLM) pin if no provider fits."""
    chosen = choose_endpoint(endpoints, spec.thinking)
    if chosen is None:
        return {"backend": "local", "model": spec.hf_repo, "thinking": spec.thinking}
    return {
        "backend": "openrouter",
        "model": spec.openrouter,
        "provider": chosen.get("provider_name") or chosen["name"],
        "quantization": chosen.get("quantization"),
        "thinking": spec.thinking,
        "prompt_price": float(chosen["pricing"]["prompt"]),
        "completion_price": float(chosen["pricing"]["completion"]),
        "context_length": chosen.get("context_length"),
    }


def load_pins() -> dict[str, EndpointPin]:
    path = lock_path()
    if not path.exists():
        raise FileNotFoundError("configs/endpoints.lock.json missing; run `trice endpoints`")
    raw = json.loads(path.read_text())
    fields = set(EndpointPin.__dataclass_fields__)
    return {
        name: EndpointPin(**{k: v for k, v in entry.items() if k in fields})
        for name, entry in raw.items()
    }
