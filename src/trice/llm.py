"""Model calls through the OpenAI client (OpenRouter or the local vLLM server), with pinned
routing, greedy decoding, thinking disabled, and a record/replay cache: every response is stored
under the hash of its canonical request, so analyses replay byte-identically."""

import hashlib
from concurrent.futures import ThreadPoolExecutor
from typing import Protocol, runtime_checkable

from diskcache import Cache
from openai import NotFoundError, OpenAI
from pydantic import BaseModel, ConfigDict, RootModel

from trice.config import LLMConfig, ModelSpec
from trice.schema import Backend, Completion, Message, Mode, Prompt


class Routing(BaseModel):
    """OpenRouter provider routing: one pinned provider, no fallbacks."""

    order: list[str | None]
    allow_fallbacks: bool = False
    require_parameters: bool = True
    quantizations: list[str] | None = None


class Reasoning(BaseModel):
    enabled: bool


class TemplateKwargs(BaseModel):
    enable_thinking: bool


class ExtraBody(BaseModel):
    provider: Routing | None = None  # OpenRouter
    reasoning: Reasoning | None = None  # OpenRouter
    chat_template_kwargs: TemplateKwargs | None = None  # local vLLM


class ChatRequest(BaseModel):
    """What is sent: it is also what the response cache is keyed on."""

    model: str
    messages: list[Message]
    temperature: float
    top_p: float
    seed: int
    max_tokens: int
    extra_body: ExtraBody
    stop: list[str] | None = None
    response_format: dict | None = None


@runtime_checkable
class Transport(Protocol):
    def send(self, request: ChatRequest) -> Completion: ...


@runtime_checkable
class Completer(Protocol):
    def complete(self, prompt: Prompt, audit_tag: str = "") -> Completion: ...

    def complete_many(self, prompts: list[Prompt], audit_tag: str = "") -> list[Completion]: ...


class OpenAITransport(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    client: OpenAI
    backend: Backend
    allow_spend: bool = False

    def send(self, request: ChatRequest) -> Completion:
        if self.backend == "openrouter" and not self.allow_spend:
            raise PermissionError("OpenRouter requests are blocked: llm.allow_spend is false in trice.toml")
        response = self.client.chat.completions.create(**request.model_dump(exclude_none=True))
        usage = response.usage
        return Completion(
            text=response.choices[0].message.content or "",
            finish_reason=response.choices[0].finish_reason,
            prompt_tokens=usage.prompt_tokens if usage else None,
            completion_tokens=usage.completion_tokens if usage else None,
            reasoning_tokens=usage.completion_tokens_details.reasoning_tokens
            if usage and usage.completion_tokens_details
            else None,
            provider=(response.model_extra or {}).get("provider"),
        )


class Endpoint(BaseModel):
    """How one model is called; chosen by :class:`Pinner` and stored in results/endpoints.json."""

    name: str  # the key of [models] in trice.toml
    backend: Backend
    model: str
    provider: str | None = None
    quantization: str | None = None
    thinking: bool = False
    prompt_price: float = 0.0  # USD per token
    completion_price: float = 0.0
    hf_revision: str | None = None


class EndpointTable(RootModel[list[Endpoint]]):
    """results/endpoints.json: one pinned endpoint per model."""


class LLM(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    endpoint: Endpoint
    transport: Transport
    cache: Cache
    config: LLMConfig
    mode: Mode = "readwrite"

    def render_request(self, prompt: Prompt) -> ChatRequest:
        endpoint = self.endpoint
        match endpoint.backend, endpoint.thinking:
            case "openrouter", thinking:
                extra = ExtraBody(
                    provider=Routing(
                        order=[endpoint.provider],
                        quantizations=[endpoint.quantization] if endpoint.quantization else None,
                    ),
                    reasoning=Reasoning(enabled=False) if thinking else None,
                )
            case "local", True:
                extra = ExtraBody(chat_template_kwargs=TemplateKwargs(enable_thinking=False))
            case "local", False:
                extra = ExtraBody()
        return ChatRequest(
            model=endpoint.model,
            messages=prompt.messages,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            seed=self.config.seed,
            max_tokens=self.config.max_tokens,
            extra_body=extra,
            stop=prompt.stop,
            response_format=prompt.response_format,
        )

    def complete(self, prompt: Prompt, audit_tag: str = "") -> Completion:
        request = self.render_request(prompt)
        key = hashlib.sha256(f"{self.endpoint.backend}|{request.model_dump_json()}".encode()).hexdigest()
        if self.mode != "bypass" and key in self.cache:
            return self.cache[key]
        if self.mode == "replay":
            raise LookupError(f"replay miss for request {key}")
        completion = self.transport.send(request)
        self.cache[f"audit/{audit_tag}/{key}" if self.mode == "bypass" else key] = completion
        return completion

    def complete_many(self, prompts: list[Prompt], audit_tag: str = "") -> list[Completion]:
        with ThreadPoolExecutor(self.config.workers) as pool:
            return list(pool.map(lambda p: self.complete(p, audit_tag), prompts))


class Pricing(BaseModel):
    prompt: float  # USD per token
    completion: float


class ProviderEndpoint(BaseModel):
    provider_name: str | None = None
    quantization: str | None = None
    supported_parameters: list[str] | None = None
    pricing: Pricing


class Listing(BaseModel):
    """OpenRouter's GET /models/<slug>/endpoints."""

    class Data(BaseModel):
        endpoints: list[ProviderEndpoint]

    data: Data


class Pinner(BaseModel):
    """Pick one OpenRouter provider per model: every required parameter supported, then the most
    faithful quantization, then the cheapest, then by name. No provider -> local vLLM."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    client: OpenAI
    config: LLMConfig

    def pin(self, name: str, spec: ModelSpec, revision: str | None) -> Endpoint:
        required = set(self.config.required_parameters) | ({"reasoning"} if spec.thinking else set())
        precision = {q: i for i, q in enumerate(self.config.quantization_preference)}
        try:
            listing = Listing.model_validate(self.client.get(f"/models/{spec.openrouter}/endpoints", cast_to=dict))
        except NotFoundError:  # the model is not listed on OpenRouter at all
            listing = Listing(data=Listing.Data(endpoints=[]))
        chosen = min(
            (e for e in listing.data.endpoints if required <= set(e.supported_parameters or [])),
            key=lambda e: (
                precision.get(e.quantization or "unknown", len(precision)),
                (e.pricing.prompt, e.pricing.completion),
                e.provider_name or "",
            ),
            default=None,
        )
        if chosen is None:
            return Endpoint(
                name=name, backend="local", model=spec.hf_repo, thinking=spec.thinking, hf_revision=revision
            )
        return Endpoint(
            name=name,
            backend="openrouter",
            model=spec.openrouter,
            provider=chosen.provider_name,
            quantization=chosen.quantization,
            thinking=spec.thinking,
            prompt_price=chosen.pricing.prompt,
            completion_price=chosen.pricing.completion,
            hf_revision=revision,
        )
