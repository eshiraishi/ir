import httpx2 as httpx
import pytest
from diskcache import Cache
from openai import OpenAI
from pydantic import BaseModel, Field

from trice.core.encoders import ApiClient
from trice.llm import LLM, Endpoint, OpenAITransport, Pinner
from trice.schema import Prompt

PROMPT = Prompt(messages=[{"role": "user", "content": "hi"}], response_format={"type": "json_schema"}, stop=["\nX"])
PINNED = Endpoint(
    backend="openrouter", model="qwen/qwen3.5-9b", provider="DeepInfra", quantization="bf16", thinking=True
)


class Recording(BaseModel):
    """Answers locally and records what would have been sent."""

    sent: list[dict] = Field(default_factory=list)

    def send(self, request: dict) -> dict:
        self.sent.append(request)
        return {
            "choices": [{"message": {"content": f"answer {len(self.sent)}"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2},
            "provider": "DeepInfra",
        }


@pytest.fixture
def transport() -> Recording:
    return Recording()


@pytest.fixture
def llm(app, tmp_path, transport):
    def make(mode: str = "readwrite", endpoint: Endpoint = PINNED) -> LLM:
        return LLM(
            endpoint=endpoint, transport=transport, cache=Cache(tmp_path / "cache"), config=app.config.llm, mode=mode
        )

    return make


def test_openrouter_request_pins_routing_and_disables_thinking(llm):
    request = llm().render_request(PROMPT)
    assert request["extra_body"] == {
        "provider": {
            "order": ["DeepInfra"],
            "allow_fallbacks": False,
            "require_parameters": True,
            "quantizations": ["bf16"],
        },
        "reasoning": {"enabled": False},
    }
    assert (request["temperature"], request["seed"], request["stop"]) == (0.0, 0, ["\nX"])
    local = llm(endpoint=Endpoint(backend="local", model="Qwen/Qwen3.5-0.8B", thinking=True)).render_request(PROMPT)
    assert local["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}


def test_record_replay_and_audit(llm, transport):
    writer = llm()
    first = writer.complete(PROMPT)
    assert writer.complete(PROMPT) == first and len(transport.sent) == 1
    assert llm("replay").complete(PROMPT) == first
    with pytest.raises(LookupError):
        llm("replay").complete(PROMPT.model_copy(update={"stop": None}))
    audit = llm("bypass")
    audit.complete(PROMPT, audit_tag="day1")
    assert len(transport.sent) == 2  # always a live call, stored under the audit tag only
    assert writer.complete(PROMPT) == first and len(transport.sent) == 2


def test_pinner_prefers_precision_then_price(app):
    pinner = Pinner(client=OpenAI(base_url="http://unused", api_key="x"), config=app.config.llm)
    params = [*app.config.llm.required_parameters, "reasoning"]

    def endpoint(name: str, quant: str, price: float, supported: list[str] = params) -> dict:
        return {
            "provider_name": name,
            "quantization": quant,
            "pricing": {"prompt": price, "completion": 1e-6},
            "supported_parameters": supported,
        }

    choice = pinner.choose_endpoint(
        [endpoint("Cheap", "fp8", 1e-8), endpoint("A", "bf16", 5e-8), endpoint("B", "bf16", 3e-8)], thinking=True
    )
    assert choice["provider_name"] == "B"
    assert pinner.choose_endpoint([endpoint("NoSchema", "bf16", 1e-9, ["temperature"])], thinking=False) is None


REQ = {"model": "m", "messages": []}
OK = {"id": "x", "object": "chat.completion", "created": 0, "model": "m", "choices": []}


class Network:
    """A real OpenAI client over a mock HTTP transport; counts requests that would leave the machine."""

    def __init__(self, body: dict):
        self.calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            self.calls += 1
            return httpx.Response(200, json=body)

        self.client = OpenAI(
            base_url="http://mock/v1",
            api_key="x",
            max_retries=0,
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        )


def test_allow_spend_defaults_to_false(app):
    assert app.config.llm.allow_spend is False


def test_openrouter_transport_blocked_without_allow_spend():
    net = Network({})
    with pytest.raises(PermissionError):
        OpenAITransport(client=net.client, backend="openrouter").send(REQ)
    assert net.calls == 0


def test_local_transport_allowed_without_allow_spend():
    net = Network(OK)
    assert OpenAITransport(client=net.client, backend="local").send(REQ)["id"] == OK["id"]


def test_openrouter_transport_allowed_when_enabled():
    net = Network(OK)
    assert OpenAITransport(client=net.client, backend="openrouter", allow_spend=True).send(REQ)["id"] == OK["id"]


def test_api_client_blocks_misses_but_replays_cache(tmp_path):
    net, cache = Network({"data": 1}), Cache(tmp_path / "api")
    api = ApiClient(client=net.client, cache=cache)
    with pytest.raises(PermissionError):
        api.post("/embeddings", {"input": ["a"]})
    assert net.calls == 0
    ApiClient(client=net.client, cache=cache, allow_spend=True).post("/embeddings", {"input": ["a"]})
    assert net.calls == 1
    assert api.post("/embeddings", {"input": ["a"]}) == {"data": 1}  # replay needs no permission
    assert net.calls == 1


def test_llm_replays_openrouter_cache_without_allow_spend(app, tmp_path):
    cache, net = Cache(tmp_path / "llm"), Network(OK)
    transport = OpenAITransport(client=net.client, backend="openrouter")
    live = LLM(endpoint=PINNED, transport=transport, cache=cache, config=app.config.llm)
    with pytest.raises(PermissionError):
        live.complete(PROMPT)
    cache[live.hash_request(live.render_request(PROMPT))] = {
        "choices": [{"message": {"content": "cached"}, "finish_reason": "stop"}]
    }
    assert live.complete(PROMPT).text == "cached"
    assert net.calls == 0


def test_pinner_falls_back_to_local_when_model_not_on_openrouter(app):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": {"message": "Not Found", "code": 404}})

    client = OpenAI(
        base_url="http://mock/v1",
        api_key="x",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    spec = app.config.models["qwen3.5-2b"]
    pin = Pinner(client=client, config=app.config.llm).pin(spec, "abc")
    assert (pin.backend, pin.model, pin.hf_revision) == ("local", spec.hf_repo, "abc")
