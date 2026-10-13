"""Scorer implementations: local models (sentence-transformers, transformers, bm25s) and the
hosted large stack (OpenRouter through the OpenAI client). All run deterministically in fp32."""

import hashlib
import operator
from functools import cached_property

import bm25s
import numpy as np
import spacy
import torch
from diskcache import Cache
from jinja2 import Environment
from openai import OpenAI
from huggingface_hub import hf_hub_download
from pydantic import BaseModel, ConfigDict
from safetensors import safe_open
from sentence_transformers import SentenceTransformer, SparseEncoder
from transformers import AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizerFast

from trice.schema import LocalRef, Message, Runtime


def fix_determinism(runtime: Runtime) -> None:
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(runtime.threads)


def extended_attention_mask(attention_mask: torch.Tensor, input_shape: object, dtype: torch.dtype) -> torch.Tensor:
    """transformers 4 ``get_extended_attention_mask``: [batch, seq] -> additive [batch, 1, 1, seq]."""
    return (1.0 - attention_mask[:, None, None, :].to(dtype)) * torch.finfo(dtype).min


def patch_remote_model(model: torch.nn.Module, checkpoint: str) -> None:
    """Make a transformers-4 remote-code model (opensearch sparse, Alibaba-NLP/new-impl) work on
    transformers 5: restore the removed ``get_extended_attention_mask``, tie the MLM decoder to the
    word embeddings (the checkpoint omits the weight, v5 leaves it random and zeroes the bias,
    which is restored from ``checkpoint``), and rebuild the
    non-persistent buffers v5 leaves uninitialized (garbage ``position_ids``, rotary tables) the
    way the model's own __init__ does. Idempotent: a correctly initialized model gets the same values."""
    for module in model.modules():
        if isinstance(getattr(module, "lm_head", None), torch.nn.Module) and hasattr(module.lm_head, "decoder"):
            module.lm_head.decoder.weight = module.get_input_embeddings().weight
            with safe_open(checkpoint, "pt") as f:  # v5 re-initialized the whole decoder, bias included
                module.lm_head.decoder.bias.data.copy_(f.get_tensor("lm_head.decoder.bias"))
        if type(module).__name__ == "NewModel" and not hasattr(module, "get_extended_attention_mask"):
            module.get_extended_attention_mask = lambda mask, shape, device=None, dtype=None: extended_attention_mask(
                mask, shape, dtype or torch.float32
            )
        if "position_ids" in module._buffers:
            module.position_ids = torch.arange(module.position_ids.size(0), device=module.position_ids.device)
        if "inv_freq" in module._buffers and hasattr(module, "_set_cos_sin_cache"):
            device = module.inv_freq.device
            module.inv_freq = 1.0 / (module.base ** (torch.arange(0, module.dim, 2).float().to(device) / module.dim))
            module._set_cos_sin_cache(module.max_seq_len_cached, device, torch.get_default_dtype())


class SparseScorer(BaseModel):
    """Learned sparse retrieval (OpenSearch v3 / SPLADE family)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    ref: LocalRef
    runtime: Runtime

    @cached_property
    def model(self) -> SparseEncoder:
        fix_determinism(self.runtime)
        model = SparseEncoder(
            self.ref.id, revision=self.ref.revision, device=self.runtime.device, trust_remote_code=True
        )
        patch_remote_model(model, hf_hub_download(self.ref.id, "model.safetensors", revision=self.ref.revision))
        return model

    def score(self, queries: list[str], pool: list[str]) -> np.ndarray:
        size = self.runtime.batch_size
        q = self.model.encode_query(queries, batch_size=size, save_to_cpu=True)
        d = self.model.encode_document(pool, batch_size=size, save_to_cpu=True)
        return self.model.similarity(q, d).to_dense().float().numpy()


class DenseScorer(BaseModel):
    """Bi-encoder, symmetric (no instruction) unless ``query_prompt`` is set (v1 replica)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    ref: LocalRef
    runtime: Runtime
    query_prompt: str | None = None

    @cached_property
    def model(self) -> SentenceTransformer:
        fix_determinism(self.runtime)
        return SentenceTransformer(self.ref.id, revision=self.ref.revision, device=self.runtime.device)

    def score(self, queries: list[str], pool: list[str]) -> np.ndarray:
        def embed(texts: list[str], prompt: str | None) -> np.ndarray:
            return self.model.encode(
                texts, prompt=prompt, batch_size=self.runtime.batch_size, normalize_embeddings=True
            )

        return embed(queries, self.query_prompt) @ embed(pool, None).T


class BM25Scorer(BaseModel):
    """v1 replica: Okapi BM25 over spaCy lookup-lemmatized lowercase tokens (bm25s' Robertson
    variant drops the constant (k1 + 1) factor; rankings are identical)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    k1: float
    b: float

    @cached_property
    def nlp(self) -> spacy.language.Language:
        nlp = spacy.blank("en")
        nlp.add_pipe("lemmatizer", config={"mode": "lookup"})
        nlp.initialize()
        return nlp

    def score(self, queries: list[str], pool: list[str]) -> np.ndarray:
        def tokens(text: str) -> list[str]:
            return [t.lemma_.lower() for t in self.nlp(text) if not t.is_space]

        index = bm25s.BM25(method="robertson", k1=self.k1, b=self.b)
        index.index([tokens(t) for t in pool], show_progress=False)
        return np.stack([index.get_scores(tokens(q)) for q in queries])


class QwenReranker(BaseModel):
    """Qwen3-Reranker scored as on its model card: P("yes") from the next-token logits of an
    instruction-formatted (query, document) pair."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    ref: LocalRef
    runtime: Runtime
    instruction: str
    env: Environment
    batch_size: int = 16

    @cached_property
    def tokenizer(self) -> PreTrainedTokenizerFast:
        return PreTrainedTokenizerFast.from_pretrained(self.ref.id, revision=self.ref.revision, padding_side="left")

    @cached_property
    def model(self) -> PreTrainedModel:
        fix_determinism(self.runtime)
        return AutoModelForCausalLM.from_pretrained(
            self.ref.id, revision=self.ref.revision, dtype=torch.float32, device_map=self.runtime.device
        ).eval()

    def score_pairs(self, query: str, docs: list[str]) -> np.ndarray:
        system = self.env.get_template("reranker_system.j2").render()
        user = self.env.get_template("reranker.j2")
        texts = [
            self.tokenizer.apply_chat_template(
                [
                    Message(role="system", content=system).model_dump(),
                    Message(
                        role="user", content=user.render(instruction=self.instruction, query=query, document=doc)
                    ).model_dump(),
                ],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for doc in docs
        ]
        yes = self.tokenizer.convert_tokens_to_ids("yes")
        no = self.tokenizer.convert_tokens_to_ids("no")
        scores = []
        for i in range(0, len(texts), self.batch_size):
            batch = self.tokenizer(texts[i : i + self.batch_size], padding=True, return_tensors="pt").to(
                self.runtime.device
            )
            with torch.no_grad():
                logits = self.model(**batch).logits[:, -1, [no, yes]]
            scores.append(torch.log_softmax(logits, dim=1)[:, 1].exp().numpy())
        return np.concatenate(scores) if scores else np.zeros(0)


class EmbeddingRequest(BaseModel):
    model: str
    input: list[str]


class EmbeddingRow(BaseModel):
    index: int
    embedding: list[float]


class EmbeddingResponse(BaseModel):
    data: list[EmbeddingRow]


class RerankRequest(BaseModel):
    model: str
    query: str
    documents: list[str]


class RerankRow(BaseModel):
    index: int
    relevance_score: float


class RerankResponse(BaseModel):
    results: list[RerankRow]


class ApiClient(BaseModel):
    """The OpenAI-compatible client for hosted models, with every raw response cached."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    client: OpenAI
    cache: Cache
    allow_spend: bool = False

    def post[T: BaseModel](self, path: str, body: BaseModel, response: type[T]) -> T:
        key = hashlib.sha256(f"{path}|{body.model_dump_json()}".encode()).hexdigest()
        if key not in self.cache:
            if not self.allow_spend:
                raise PermissionError("OpenRouter requests are blocked: llm.allow_spend is false in trice.toml")
            self.cache[key] = self.client.post(path, body=body.model_dump(), cast_to=dict)
        return response.model_validate(self.cache[key])


class ApiScorer(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    api: ApiClient
    model_id: str
    batch_size: int = 128

    def score(self, queries: list[str], pool: list[str]) -> np.ndarray:
        def embed(texts: list[str]) -> np.ndarray:
            rows: list[list[float]] = []
            for i in range(0, len(texts), self.batch_size):
                body = EmbeddingRequest(model=self.model_id, input=texts[i : i + self.batch_size])
                response = self.api.post("/embeddings", body, EmbeddingResponse)
                rows += [r.embedding for r in sorted(response.data, key=operator.attrgetter("index"))]
            m = np.asarray(rows)
            return m / np.linalg.norm(m, axis=1, keepdims=True)

        return embed(queries) @ embed(pool).T


class ApiReranker(BaseModel):
    """Cohere-style /rerank (OpenRouter); whether the instruction can be set is checked at M4."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    api: ApiClient
    model_id: str

    def score_pairs(self, query: str, docs: list[str]) -> np.ndarray:
        response = self.api.post(
            "/rerank", RerankRequest(model=self.model_id, query=query, documents=docs), RerankResponse
        )
        scores = np.zeros(len(docs))
        for r in response.results:
            scores[r.index] = r.relevance_score
        return scores
