"""Scorers behind each retrieval stage. Heavy dependencies (torch, sentence-transformers,
transformers models, spaCy) are imported lazily so the rest of TRICE runs without them.

Every scorer returns a dense ``(n_queries, n_pool)`` float array; the pipeline owns ordering,
exclusions and fusion.
"""

import os
from collections.abc import Sequence
from functools import cached_property
from typing import Any

import httpx
import numpy as np

from trice.config import ModelRef
from trice.llm.cache import ResponseCache, request_key
from trice.llm.client import OPENROUTER_URL


def _torch_determinism():
    import torch

    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(int(os.environ.get("TRICE_THREADS", "8")))
    return torch


def _device() -> str:
    return os.environ.get("TRICE_DEVICE", "cpu")


class SparseScorer:
    """Learned sparse model (OpenSearch v3 / SPLADE family) via sentence-transformers."""

    def __init__(self, ref: ModelRef, batch_size: int = 32) -> None:
        self.ref, self.batch_size = ref, batch_size

    @cached_property
    def model(self):
        _torch_determinism()
        from sentence_transformers import SparseEncoder

        return SparseEncoder(self.ref.id, revision=self.ref.revision, device=_device())

    def score(self, queries: Sequence[str], pool: Sequence[str]) -> np.ndarray:
        q = self.model.encode_query(list(queries), batch_size=self.batch_size, save_to_cpu=True)
        d = self.model.encode_document(list(pool), batch_size=self.batch_size, save_to_cpu=True)
        return self.model.similarity(q, d).to_dense().float().numpy()


class DenseScorer:
    """Dense bi-encoder; symmetric (no instruction) unless ``query_prompt`` is set (v1 replica)."""

    def __init__(
        self, ref: ModelRef, query_prompt: str | None = None, batch_size: int = 32
    ) -> None:
        self.ref, self.query_prompt, self.batch_size = ref, query_prompt, batch_size

    @cached_property
    def model(self):
        _torch_determinism()
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(self.ref.id, revision=self.ref.revision, device=_device())

    def embed(self, texts: Sequence[str], prompt: str | None = None) -> np.ndarray:
        return self.model.encode(
            list(texts),
            prompt=prompt,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)

    def score(self, queries: Sequence[str], pool: Sequence[str]) -> np.ndarray:
        return self.embed(queries, self.query_prompt) @ self.embed(pool).T


class QwenReranker:
    """Qwen3-Reranker scored exactly as on its model card: P("yes") for an instruction-formatted
    (query, document) pair from the causal LM's next-token logits."""

    PREFIX = (
        "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query "
        'and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n'
        "<|im_start|>user\n"
    )
    SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    def __init__(
        self, ref: ModelRef, instruction: str, batch_size: int = 16, max_length: int = 8192
    ) -> None:
        self.ref, self.instruction = ref, instruction
        self.batch_size, self.max_length = batch_size, max_length

    @cached_property
    def _model(self):
        torch = _torch_determinism()
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(
            self.ref.id, revision=self.ref.revision, padding_side="left"
        )
        model: Any = AutoModelForCausalLM.from_pretrained(
            self.ref.id, revision=self.ref.revision, dtype=torch.float32
        )
        return tok, model.to(_device()).eval()

    def _pairs(self, query: str, docs: Sequence[str]) -> list[str]:
        return [f"<Instruct>: {self.instruction}\n<Query>: {query}\n<Document>: {d}" for d in docs]

    def score_pairs(self, query: str, docs: Sequence[str]) -> np.ndarray:
        import torch

        tok, model = self._model
        prefix = tok.encode(self.PREFIX, add_special_tokens=False)
        suffix = tok.encode(self.SUFFIX, add_special_tokens=False)
        yes, no = tok.convert_tokens_to_ids("yes"), tok.convert_tokens_to_ids("no")
        out = []
        pairs = self._pairs(query, docs)
        for i in range(0, len(pairs), self.batch_size):
            enc = tok(
                pairs[i : i + self.batch_size],
                padding=False,
                truncation="longest_first",
                return_attention_mask=False,
                max_length=self.max_length - len(prefix) - len(suffix),
            )
            enc["input_ids"] = [prefix + ids + suffix for ids in enc["input_ids"]]
            batch = tok.pad(enc, padding=True, return_tensors="pt").to(model.device)
            with torch.no_grad():
                logits = model(**batch).logits[:, -1, :]
            two = torch.stack([logits[:, no], logits[:, yes]], dim=1)
            out.append(torch.log_softmax(two, dim=1)[:, 1].exp().float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0, np.float32)


class _OpenRouterAPI:
    """Shared plumbing for the hosted large-stack models; raw responses go to the replay cache."""

    def __init__(
        self, ref: ModelRef, cache: ResponseCache, transport: httpx.BaseTransport | None = None
    ) -> None:
        self.ref, self.cache = ref, cache
        key = os.environ.get("OPENROUTER_API_KEY")
        self._http = httpx.Client(
            base_url=os.environ.get("OPENROUTER_BASE_URL", OPENROUTER_URL),
            headers={"Authorization": f"Bearer {key}"} if key else {},
            timeout=300,
            transport=transport,
        )

    def _call(self, path: str, body: dict) -> dict:
        key = request_key(f"openrouter{path}", body)
        if (hit := self.cache.get(key)) is not None:
            return hit
        r = self._http.post(path, json=body)
        r.raise_for_status()
        data = r.json()
        self.cache.put(key, body, data)
        return data


class OpenRouterEmbeddings(_OpenRouterAPI):
    def embed(self, texts: Sequence[str], batch_size: int = 128) -> np.ndarray:
        vecs = []
        for i in range(0, len(texts), batch_size):
            data = self._call(
                "/embeddings", {"model": self.ref.id, "input": list(texts[i : i + batch_size])}
            )
            rows = sorted(data["data"], key=lambda r: r["index"])
            vecs += [r["embedding"] for r in rows]
        m = np.asarray(vecs, dtype=np.float32)
        return m / np.linalg.norm(m, axis=1, keepdims=True)

    def score(self, queries: Sequence[str], pool: Sequence[str]) -> np.ndarray:
        return self.embed(queries) @ self.embed(pool).T


class OpenRouterReranker(_OpenRouterAPI):
    """Cohere-style /rerank call. Whether the instruction can be set is checked at M4; if not,
    configs/retrieval.yaml switches both stacks to the provider's default instruction."""

    def score_pairs(self, query: str, docs: Sequence[str]) -> np.ndarray:
        data = self._call(
            "/rerank", {"model": self.ref.id, "query": query, "documents": list(docs)}
        )
        scores = np.zeros(len(docs), np.float32)
        for r in data["results"]:
            scores[r["index"]] = r["relevance_score"]
        return scores


class BM25V1:
    """v1 replica: Okapi BM25 (k1, b from config) over spaCy lookup-lemmatized lowercase tokens.

    bm25s' Robertson variant omits the constant (k1 + 1) numerator factor; rankings are identical.
    """

    def __init__(self, k1: float, b: float, lemmatize: bool) -> None:
        self.k1, self.b, self.lemmatize = k1, b, lemmatize

    @cached_property
    def _nlp(self):
        import spacy

        nlp = spacy.blank("en")
        if self.lemmatize:
            nlp.add_pipe("lemmatizer", config={"mode": "lookup"})
            nlp.initialize()
        return nlp

    def tokens(self, text: str) -> list[str]:
        doc = self._nlp(text)
        return [(t.lemma_ if self.lemmatize else t.text).lower() for t in doc if not t.is_space]

    def score(self, queries: Sequence[str], pool: Sequence[str]) -> np.ndarray:
        import bm25s

        retriever = bm25s.BM25(method="robertson", k1=self.k1, b=self.b)
        retriever.index([self.tokens(t) for t in pool], show_progress=False)
        return np.stack([retriever.get_scores(self.tokens(q)) for q in queries]).astype(np.float32)
