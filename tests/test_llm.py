import pytest
from diskcache import Cache
from openai import OpenAI
from pydantic import BaseModel, Field

from trice.llm import LLM, Endpoint, Pinner
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
