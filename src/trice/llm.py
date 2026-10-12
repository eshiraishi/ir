"""Model calls through the OpenAI client (OpenRouter or the local vLLM server), with pinned
routing, greedy decoding, thinking disabled, and a record/replay cache: every response is stored
under the hash of its canonical request, so analyses replay byte-identically."""

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from typing import Literal, cast

from diskcache import Cache
from openai import OpenAI
from pydantic import BaseModel, ConfigDict

from trice.schema import Completion, Prompt
from trice.settings import Backend, LLMSettings, ModelSpec

Mode = Literal["readwrite", "replay", "bypass"]


class Endpoint(BaseModel):
    """How one model is called; chosen by :class:`Pinner` and stored in results/endpoints.json."""

    backend: Backend
    model: str
    provider: str | None = None
    quantization: str | None = None
    thinking: bool = False
    prompt_price: float = 0.0  # USD per token
    completion_price: float = 0.0
    hf_revision: str | None = None


class LLM(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    endpoint: Endpoint
    client: OpenAI
    cache: Cache
    settings: LLMSettings
    mode: Mode = "readwrite"

    def request(self, prompt: Prompt) -> dict:
        extra: dict = {}
        if self.endpoint.backend == "openrouter":
            extra["provider"] = {
                "order": [self.endpoint.provider],
                "allow_fallbacks": False,
                "require_parameters": True,
            }
            if self.endpoint.quantization:
                extra["provider"]["quantizations"] = [self.endpoint.quantization]
            if self.endpoint.thinking:
                extra["reasoning"] = {"enabled": False}
        elif self.endpoint.thinking:
            extra["chat_template_kwargs"] = {"enable_thinking": False}
        optional = {"stop": prompt.stop, "response_format": prompt.response_format}
        return {
            "model": self.endpoint.model,
            "messages": prompt.messages,
            "temperature": self.settings.temperature,
            "top_p": self.settings.top_p,
            "seed": self.settings.seed,
            "max_tokens": self.settings.max_tokens,
            "extra_body": extra,
            **{k: v for k, v in optional.items() if v is not None},
        }

    def key(self, request: dict) -> str:
        return hashlib.sha256(json.dumps([self.endpoint.backend, request], sort_keys=True).encode()).hexdigest()

    def send(self, request: dict) -> dict:
        return self.client.chat.completions.create(**request).model_dump()

    def complete(self, prompt: Prompt, audit_tag: str = "") -> Completion:
        request = self.request(prompt)
        key = self.key(request)
        if self.mode != "bypass" and key in self.cache:
            return self.completion(self.cache[key])
        if self.mode == "replay":
            raise LookupError(f"replay miss for request {key}")
        data = self.send(request)
        self.cache[f"audit/{audit_tag}/{key}" if self.mode == "bypass" else key] = data
        return self.completion(data)

    def complete_many(self, prompts: list[Prompt], audit_tag: str = "") -> list[Completion]:
        with ThreadPoolExecutor(self.settings.workers) as pool:
            return list(pool.map(lambda p: self.complete(p, audit_tag), prompts))

    @staticmethod
    def completion(data: dict) -> Completion:
        choice, usage = data["choices"][0], data.get("usage") or {}
        return Completion(
            text=choice["message"].get("content") or "",
            finish_reason=choice.get("finish_reason"),
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            reasoning_tokens=(usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
            provider=data.get("provider"),
        )


class Pinner(BaseModel):
    """Pick one OpenRouter provider per model: every required parameter supported, then the most
    faithful quantization, then the cheapest, then by name. No provider -> local vLLM."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    client: OpenAI
    settings: LLMSettings

    def choose(self, endpoints: list[dict], thinking: bool) -> dict | None:
        required = set(self.settings.required_parameters) | ({"reasoning"} if thinking else set())
        prefs = self.settings.quantization_preference
        ok = [e for e in endpoints if required <= set(e.get("supported_parameters") or [])]

        def preference(e: dict) -> tuple:
            quantization = e.get("quantization") or "unknown"
            precision = prefs.index(quantization) if quantization in prefs else len(prefs)
            price = (float(e["pricing"]["prompt"]), float(e["pricing"]["completion"]))
            return precision, price, e.get("provider_name") or ""

        return min(ok, key=preference) if ok else None

    def pin(self, spec: ModelSpec, revision: str | None) -> Endpoint:
        data = cast(dict, self.client.get(f"/models/{spec.openrouter}/endpoints", cast_to=object))
        chosen = self.choose(data["data"]["endpoints"], spec.thinking)
        if chosen is None:
            return Endpoint(backend="local", model=spec.hf_repo, thinking=spec.thinking, hf_revision=revision)
        return Endpoint(
            backend="openrouter",
            model=spec.openrouter,
            provider=chosen.get("provider_name"),
            quantization=chosen.get("quantization"),
            thinking=spec.thinking,
            prompt_price=float(chosen["pricing"]["prompt"]),
            completion_price=float(chosen["pricing"]["completion"]),
            hf_revision=revision,
        )
