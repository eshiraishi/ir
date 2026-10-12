import json

import httpx
import pytest
from tenacity import wait_none

from trice.config import LLMSpec
from trice.llm.cache import ResponseCache, request_key
from trice.llm.client import Backend, CacheMissError, ChatClient, EndpointPin, build_body
from trice.llm.endpoints import choose_endpoint, pin_for
from trice.prompt.formats import Prompt

PIN = EndpointPin(
    backend="openrouter",
    model="qwen/qwen3.5-9b",
    provider="DeepInfra",
    quantization="bf16",
    thinking=True,
)
PROMPT = Prompt(
    messages=[{"role": "user", "content": "hi"}],
    response_format={"type": "json_schema"},
    stop=["\nX"],
)


def _response(text="ok", provider="DeepInfra"):
    return {
        "id": "gen-1",
        "provider": provider,
        "model": "qwen/qwen3.5-9b",
        "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": 5,
            "completion_tokens": 1,
            "completion_tokens_details": {"reasoning_tokens": 0},
        },
    }


class Recorder:
    def __init__(self, replies):
        self.replies, self.bodies = list(replies), []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.bodies.append(json.loads(request.content))
        status, payload = self.replies.pop(0)
        return httpx.Response(status, json=payload)


@pytest.fixture
def cache(tmp_path):
    c = ResponseCache(tmp_path / "cache.sqlite")
    yield c
    c.close()


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch):
    monkeypatch.setattr(ChatClient._post.retry, "wait", wait_none())


def test_build_body_openrouter_pins_routing_and_disables_thinking():
    body = build_body(PIN, PROMPT)
    assert body["temperature"] == 0 and body["top_p"] == 1 and body["seed"] == 0
    assert body["provider"] == {
        "order": ["DeepInfra"],
        "allow_fallbacks": False,
        "require_parameters": True,
        "quantizations": ["bf16"],
    }
    assert body["reasoning"] == {"enabled": False}
    assert body["stop"] == ["\nX"] and body["response_format"] == {"type": "json_schema"}
    assert "chat_template_kwargs" not in body


def test_build_body_local_uses_chat_template_kwargs():
    body = build_body(
        EndpointPin(backend="local", model="Qwen/Qwen3.5-0.8B", thinking=True), PROMPT
    )
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert "provider" not in body and "reasoning" not in body


def test_cache_hit_replay_and_bypass(cache):
    rec = Recorder([(200, _response("first")), (200, _response("audit"))])
    client = ChatClient(
        Backend("openrouter", "http://x"), cache, transport=httpx.MockTransport(rec)
    )
    body = build_body(PIN, PROMPT)
    first = client.complete(body)
    again = client.complete(body)
    assert (first.text, first.cached, again.text, again.cached) == ("first", False, "first", True)
    assert len(rec.bodies) == 1 and rec.bodies[0] == body

    replay = ChatClient(
        Backend("openrouter", "http://x"), cache, mode="replay", transport=httpx.MockTransport(rec)
    )
    assert replay.complete(body).text == "first"
    with pytest.raises(CacheMissError):
        replay.complete({**body, "seed": 1})

    audit = ChatClient(
        Backend("openrouter", "http://x"), cache, mode="bypass", transport=httpx.MockTransport(rec)
    )
    assert audit.complete(body, audit_tag="day1").text == "audit"
    assert (
        cache.audit_responses(request_key("openrouter", body))["day1"]["choices"][0]["message"][
            "content"
        ]
        == "audit"
    )
    assert client.complete(body).text == "first"  # audit never overwrites the replay record


def test_retries_on_429_and_in_band_errors_but_not_400(cache):
    rec = Recorder([(429, {}), (200, {"error": {"message": "upstream"}}), (200, _response("ok"))])
    client = ChatClient(
        Backend("openrouter", "http://x"), cache, transport=httpx.MockTransport(rec)
    )
    assert client.complete(build_body(PIN, PROMPT)).text == "ok"
    assert len(rec.bodies) == 3

    rec = Recorder([(400, {"error": "bad"})])
    client = ChatClient(
        Backend("openrouter", "http://x"), cache, transport=httpx.MockTransport(rec)
    )
    with pytest.raises(httpx.HTTPStatusError):
        client.complete({**build_body(PIN, PROMPT), "seed": 7})
    assert len(cache) == 1  # failures are never cached


def _ep(
    name,
    quant,
    prompt,
    params=(
        "temperature",
        "top_p",
        "seed",
        "max_tokens",
        "stop",
        "response_format",
        "structured_outputs",
        "reasoning",
    ),
):
    return {
        "provider_name": name,
        "quantization": quant,
        "pricing": {"prompt": str(prompt), "completion": "0.000001"},
        "supported_parameters": list(params),
    }


def test_choose_endpoint_prefers_precision_then_price():
    eps = [
        _ep("Cheap", "fp8", 1e-8),
        _ep("Exact", "bf16", 5e-8),
        _ep("Exact2", "bf16", 3e-8),
        _ep("NoSchema", "bf16", 1e-9, ("temperature",)),
    ]
    assert choose_endpoint(eps, thinking=True)["provider_name"] == "Exact2"
    assert (
        choose_endpoint([_ep("NoSchema", "bf16", 1e-9, ("temperature",))], thinking=False) is None
    )


def test_pin_for_falls_back_to_local():
    spec = LLMSpec(
        name="m",
        family="qwen3.5",
        ladder="qwen3.5-dense",
        params_b=0.8,
        active_params_b=0.8,
        openrouter="qwen/qwen3.5-0.8b",
        hf_repo="Qwen/Qwen3.5-0.8B",
        thinking=True,
    )
    assert pin_for(spec, [])["backend"] == "local"
    pinned = pin_for(spec, [_ep("A", "bf16", 2e-8)])
    assert (pinned["backend"], pinned["provider"], pinned["prompt_price"]) == (
        "openrouter",
        "A",
        2e-8,
    )
