"""Scorer implementations: local models (sentence-transformers, transformers, bm25s) and the
hosted large stack (OpenRouter through the OpenAI client). All run deterministically in fp32."""

import hashlib
import json
from functools import cached_property
from typing import cast

import bm25s
import numpy as np
import spacy
import torch
from diskcache import Cache
from jinja2 import Environment
from openai import OpenAI
from pydantic import BaseModel, ConfigDict
from sentence_transformers import SentenceTransformer, SparseEncoder
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from trice.schema import ModelRef, Runtime


def fix_determinism(runtime: Runtime) -> None:
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(runtime.threads)


class SparseScorer(BaseModel):
    """Learned sparse retrieval (OpenSearch v3 / SPLADE family)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    ref: ModelRef
    runtime: Runtime

    @cached_property
    def model(self) -> SparseEncoder:
        fix_determinism(self.runtime)
        return SparseEncoder(self.ref.id, revision=self.ref.revision, device=self.runtime.device)

    def score(self, queries: list[str], pool: list[str]) -> np.ndarray:
        size = self.runtime.batch_size
        q = self.model.encode_query(queries, batch_size=size, save_to_cpu=True)
        d = self.model.encode_document(pool, batch_size=size, save_to_cpu=True)
        return self.model.similarity(q, d).to_dense().float().numpy()


class DenseScorer(BaseModel):
    """Bi-encoder, symmetric (no instruction) unless ``query_prompt`` is set (v1 replica)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    ref: ModelRef
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

    ref: ModelRef
    runtime: Runtime
    instruction: str
    env: Environment
    batch_size: int = 16

    @cached_property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        tokenizer = AutoTokenizer.from_pretrained(self.ref.id, revision=self.ref.revision, padding_side="left")
        return cast(PreTrainedTokenizerBase, tokenizer)

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
                    {"role": "system", "content": system},
                    {"role": "user", "content": user.render(instruction=self.instruction, query=query, document=doc)},
                ],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for doc in docs
        ]
        yes, no = self.tokenizer.convert_tokens_to_ids("yes"), self.tokenizer.convert_tokens_to_ids("no")
        scores = []
        for i in range(0, len(texts), self.batch_size):
            batch = self.tokenizer(texts[i : i + self.batch_size], padding=True, return_tensors="pt").to(
                self.runtime.device
            )
            with torch.no_grad():
                logits = self.model(**batch).logits[:, -1, [no, yes]]
            scores.append(torch.log_softmax(logits, dim=1)[:, 1].exp().numpy())
        return np.concatenate(scores) if scores else np.zeros(0)


class ApiClient(BaseModel):
    """The OpenAI-compatible client for hosted models, with every raw response cached."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    client: OpenAI
    cache: Cache
    allow_spend: bool = False

    def post(self, path: str, body: dict) -> dict:
        key = hashlib.sha256(json.dumps([path, body], sort_keys=True).encode()).hexdigest()
        if key not in self.cache:
            if not self.allow_spend:
                raise PermissionError("OpenRouter requests are blocked: llm.allow_spend is false in trice.toml")
            self.cache[key] = self.client.post(path, body=body, cast_to=object)
        return cast(dict, self.cache[key])


class ApiScorer(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    api: ApiClient
    model_id: str
    batch_size: int = 128

    def score(self, queries: list[str], pool: list[str]) -> np.ndarray:
        def embed(texts: list[str]) -> np.ndarray:
            rows = []
            for i in range(0, len(texts), self.batch_size):
                body = {"model": self.model_id, "input": texts[i : i + self.batch_size]}
                rows += [
                    r["embedding"] for r in sorted(self.api.post("/embeddings", body)["data"], key=lambda r: r["index"])
                ]
            m = np.asarray(rows)
            return m / np.linalg.norm(m, axis=1, keepdims=True)

        return embed(queries) @ embed(pool).T


class ApiReranker(BaseModel):
    """Cohere-style /rerank (OpenRouter); whether the instruction can be set is checked at M4."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    api: ApiClient
    model_id: str

    def score_pairs(self, query: str, docs: list[str]) -> np.ndarray:
        body = {"model": self.model_id, "query": query, "documents": docs}
        scores = np.zeros(len(docs))
        for r in self.api.post("/rerank", body)["results"]:
            scores[r["index"]] = r["relevance_score"]
        return scores
