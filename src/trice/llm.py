"""Model calls through the OpenAI client (OpenRouter or the local vLLM server), with pinned
routing, greedy decoding, thinking disabled, and a record/replay cache: every response is stored
under the hash of its canonical request, so analyses replay byte-identically."""

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from typing import Protocol, runtime_checkable

from diskcache import Cache
from openai import NotFoundError, OpenAI
from pydantic import BaseModel, ConfigDict

from trice.config import LLMConfig, ModelSpec
from trice.schema import Backend, Completion, Mode, Prompt


@runtime_checkable
class Transport(Protocol):
    def send(self, request: dict) -> dict:
        """One chat-completions request in, the raw response out."""
        ...


@runtime_checkable
class Completer(Protocol):
    def complete(self, prompt: Prompt, audit_tag: str = "") -> Completion: ...

    def complete_many(self, prompts: list[Prompt], audit_tag: str = "") -> list[Completion]: ...


class OpenAITransport(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    client: OpenAI
    backend: Backend
    allow_spend: bool = False

    def send(self, request: dict) -> dict:
        if self.backend == "openrouter" and not self.allow_spend:
            raise PermissionError("OpenRouter requests are blocked: llm.allow_spend is false in trice.toml")
        return self.client.chat.completions.create(**request).model_dump()


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
    transport: Transport
    cache: Cache
    config: LLMConfig
    mode: Mode = "readwrite"

    def render_request(self, prompt: Prompt) -> dict:
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
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "seed": self.config.seed,
            "max_tokens": self.config.max_tokens,
            "extra_body": extra,
            **{k: v for k, v in optional.items() if v is not None},
        }

    def complete(self, prompt: Prompt, audit_tag: str = "") -> Completion:
        request = self.render_request(prompt)
        key = hashlib.sha256(json.dumps([self.endpoint.backend, request], sort_keys=True).encode()).hexdigest()
        if self.mode != "bypass" and key in self.cache:
            return self.parse_completion(self.cache[key])
        if self.mode == "replay":
            raise LookupError(f"replay miss for request {key}")
        data = self.transport.send(request)
        self.cache[f"audit/{audit_tag}/{key}" if self.mode == "bypass" else key] = data
        return self.parse_completion(data)

    def complete_many(self, prompts: list[Prompt], audit_tag: str = "") -> list[Completion]:
        with ThreadPoolExecutor(self.config.workers) as pool:
            return list(pool.map(lambda p: self.complete(p, audit_tag), prompts))

    @staticmethod
    def parse_completion(data: dict) -> Completion:
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
    config: LLMConfig

    def pin(self, spec: ModelSpec, revision: str | None) -> Endpoint:
        required = set(self.config.required_parameters) | ({"reasoning"} if spec.thinking else set())
        precision = {q: i for i, q in enumerate(self.config.quantization_preference)}
        try:
            listing = self.client.get(f"/models/{spec.openrouter}/endpoints", cast_to=dict)["data"]["endpoints"]
        except NotFoundError:  # the model is not listed on OpenRouter at all
            listing = []
        supported = [e for e in listing if required <= set(e.get("supported_parameters") or [])]
        chosen = min(
            supported,
            key=lambda e: (
                precision.get(e.get("quantization") or "unknown", len(precision)),
                (float(e["pricing"]["prompt"]), float(e["pricing"]["completion"])),
                e.get("provider_name") or "",
            ),
            default=None,
        )
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
