"""OpenAI-compatible chat client for OpenRouter and the local vLLM server, with pinned routing,
thinking disabled, greedy decoding and the record/replay cache."""

import os
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Literal

import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from trice.llm.cache import ResponseCache, request_key
from trice.prompt.formats import Prompt

OPENROUTER_URL = "https://openrouter.ai/api/v1"
# Greedy decoding. Penalties are left at provider defaults (0) rather than sent, because
# `require_parameters` would otherwise exclude providers that do not list them.
SAMPLING = {"temperature": 0, "top_p": 1, "seed": 0}
MAX_TOKENS = 1024

Mode = Literal["readwrite", "replay", "bypass"]


@dataclass(frozen=True)
class EndpointPin:
    """How one model is called; produced by `trice endpoints` (configs/endpoints.lock.json)."""

    backend: Literal["openrouter", "local"]
    model: str  # OpenRouter slug, or the served HF repo for local
    provider: str | None = None
    quantization: str | None = None
    thinking: bool = False
    prompt_price: float = 0.0  # USD per token
    completion_price: float = 0.0


def build_body(pin: EndpointPin, prompt: Prompt, max_tokens: int = MAX_TOKENS) -> dict:
    body: dict = {
        "model": pin.model,
        "messages": prompt.messages,
        **SAMPLING,
        "max_tokens": max_tokens,
    }
    if prompt.stop:
        body["stop"] = prompt.stop
    if prompt.response_format:
        body["response_format"] = prompt.response_format
    if pin.backend == "openrouter":
        routing: dict = {
            "order": [pin.provider],
            "allow_fallbacks": False,
            "require_parameters": True,
        }
        if pin.quantization:
            routing["quantizations"] = [pin.quantization]
        body["provider"] = routing
        if pin.thinking:
            body["reasoning"] = {"enabled": False}
    elif pin.thinking:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    return body


@dataclass(frozen=True)
class Completion:
    text: str
    finish_reason: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    reasoning_tokens: int | None
    provider: str | None
    model: str | None
    id: str | None
    cached: bool

    @classmethod
    def from_response(cls, data: dict, cached: bool) -> Completion:
        choice = data["choices"][0]
        usage = data.get("usage") or {}
        details = usage.get("completion_tokens_details") or {}
        return cls(
            text=choice["message"].get("content") or "",
            finish_reason=choice.get("finish_reason"),
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            reasoning_tokens=details.get("reasoning_tokens"),
            provider=data.get("provider"),
            model=data.get("model"),
            id=data.get("id"),
            cached=cached,
        )


class CacheMissError(RuntimeError):
    pass


class RetryableError(RuntimeError):
    pass


def _retryable(exc: BaseException) -> bool:
    return isinstance(exc, RetryableError | httpx.TransportError)


@dataclass
class Backend:
    name: str  # logical name, part of every cache key: "openrouter" or "local"
    base_url: str
    api_key: str | None = None

    @classmethod
    def openrouter(cls) -> Backend:
        key = os.environ.get("OPENROUTER_API_KEY")
        return cls("openrouter", os.environ.get("OPENROUTER_BASE_URL", OPENROUTER_URL), key)

    @classmethod
    def local(cls, base_url: str = "http://127.0.0.1:8000/v1") -> Backend:
        return cls("local", base_url, None)


class ChatClient:
    def __init__(
        self,
        backend: Backend,
        cache: ResponseCache,
        mode: Mode = "readwrite",
        timeout: float = 300.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.backend, self.cache, self.mode = backend, cache, mode
        headers = {"Authorization": f"Bearer {backend.api_key}"} if backend.api_key else {}
        self._http = httpx.Client(
            base_url=backend.base_url, headers=headers, timeout=timeout, transport=transport
        )

    def close(self) -> None:
        self._http.close()

    @retry(
        retry=retry_if_exception(_retryable),
        wait=wait_exponential(multiplier=2, min=2, max=120),
        stop=stop_after_attempt(8),
        reraise=True,
    )
    def _post(self, body: dict) -> dict:
        r = self._http.post("/chat/completions", json=body)
        if r.status_code == 429 or r.status_code >= 500:
            raise RetryableError(f"HTTP {r.status_code}: {r.text[:300]}")
        r.raise_for_status()
        data = r.json()
        if "error" in data or not data.get("choices"):
            # OpenRouter reports upstream failures inside a 200 response.
            raise RetryableError(f"provider error: {str(data.get('error', data))[:300]}")
        return data

    def complete(self, body: dict, audit_tag: str | None = None) -> Completion:
        key = request_key(self.backend.name, body)
        if self.mode != "bypass":
            hit = self.cache.get(key)
            if hit is not None:
                return Completion.from_response(hit, cached=True)
            if self.mode == "replay":
                raise CacheMissError(key)
        data = self._post(body)
        if self.mode == "bypass":
            self.cache.put_audit(key, audit_tag or "live", data)
        else:
            self.cache.put(key, body, data)
        return Completion.from_response(data, cached=False)

    def complete_many(
        self,
        bodies: Iterable[dict],
        workers: int = 16,
        on_done: Callable[[int, Completion], None] | None = None,
        audit_tag: str | None = None,
    ) -> list[Completion]:
        bodies = list(bodies)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(self.complete, b, audit_tag) for b in bodies]
            out = []
            for i, f in enumerate(futures):
                c = f.result()
                if on_done:
                    on_done(i, c)
                out.append(c)
        return out
