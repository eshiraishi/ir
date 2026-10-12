"""Composition root: the only place that reads Settings and builds/wires components."""

import json
from functools import cached_property

from diskcache import Cache
from jinja2 import Environment, PackageLoader, StrictUndefined
from openai import OpenAI
from pydantic import BaseModel, Field

from trice.analysis import Analysis
from trice.budget import Budget
from trice.core.encoders import ApiClient, ApiReranker, ApiScorer, BM25Scorer, DenseScorer, QwenReranker, SparseScorer
from trice.core.prompting import Prompter
from trice.core.retrieval import Reranker, Retriever, Scorer
from trice.data import Corpus
from trice.evaluation.report import Reporter
from trice.evaluation.scoring import Evaluator
from trice.evaluation.stats import Stats
from trice.experiment import Experiment
from trice.llm import LLM, Endpoint, Mode, Pinner
from trice.settings import ModelRef, Settings


def cell(value: object) -> str:
    return f"{value:.4f}" if isinstance(value, float) else str(value)


def latex(value: object) -> str:
    return "".join("\\" + c if c in "&%$#_{}" else c for c in str(value))


class Container(BaseModel):
    settings: Settings = Field(default_factory=Settings)

    @cached_property
    def env(self) -> Environment:
        env = Environment(loader=PackageLoader("trice"), undefined=StrictUndefined, autoescape=False)
        env.filters.update(json=lambda v: json.dumps(v, ensure_ascii=False), cell=cell, latex=latex)
        return env

    @cached_property
    def corpus(self) -> Corpus:
        return Corpus(root=self.settings.paths.data, sources=self.settings.sources, specs=self.settings.datasets)

    @cached_property
    def experiment(self) -> Experiment:
        return Experiment(
            settings=self.settings,
            corpus=self.corpus,
            prompter=Prompter(env=self.env),
            evaluator=Evaluator(),
            stats=Stats(resamples=self.settings.stats.resamples, alpha=self.settings.stats.alpha),
        )

    @cached_property
    def analysis(self) -> Analysis:
        return Analysis(
            experiment=self.experiment, reporter=Reporter(env=self.env, out=self.settings.paths.results / "tables")
        )

    @cached_property
    def cache(self) -> Cache:
        return Cache(self.settings.paths.cache / "responses")

    @cached_property
    def openrouter(self) -> OpenAI:
        key = self.settings.openrouter_api_key
        return OpenAI(
            base_url=self.settings.llm.openrouter_url,
            api_key=key.get_secret_value() if key else "missing",
            max_retries=self.settings.llm.max_retries,
            timeout=self.settings.llm.timeout,
        )

    @cached_property
    def local(self) -> OpenAI:
        return OpenAI(base_url=self.settings.llm.local_url, api_key="local", timeout=self.settings.llm.timeout)

    @cached_property
    def endpoints(self) -> dict[str, Endpoint]:
        path = self.settings.paths.results / "endpoints.json"
        raw = json.loads(path.read_text()) if path.exists() else {}
        return {name: Endpoint.model_validate(e) for name, e in raw.items()}

    @cached_property
    def pinner(self) -> Pinner:
        return Pinner(client=self.openrouter, settings=self.settings.llm)

    @cached_property
    def budget(self) -> Budget:
        return Budget(experiment=self.experiment, endpoints=self.endpoints)

    def llm(self, model: str, mode: Mode = "readwrite") -> LLM:
        endpoint = self.endpoints[model]
        client = self.openrouter if endpoint.backend == "openrouter" else self.local
        return LLM(endpoint=endpoint, client=client, cache=self.cache, settings=self.settings.llm, mode=mode)

    def scorer(self, stage: str, ref: ModelRef, query_prompt: str | None = None) -> Scorer:
        if ref.backend == "openrouter":
            return ApiScorer(api=ApiClient(client=self.openrouter, cache=self.cache), model_id=ref.id)
        r = self.settings.retrieval
        local = {
            "model_id": ref.id,
            "revision": ref.revision,
            "device": r.device,
            "threads": r.threads,
            "batch_size": r.batch_size,
        }
        return SparseScorer(**local) if stage == "sparse" else DenseScorer(**local, query_prompt=query_prompt)

    def reranker(self, ref: ModelRef) -> Reranker:
        if ref.backend == "openrouter":
            return ApiReranker(api=ApiClient(client=self.openrouter, cache=self.cache), model_id=ref.id)
        r = self.settings.retrieval
        return QwenReranker(
            model_id=ref.id, revision=ref.revision, instruction=r.reranker_instruction, env=self.env, device=r.device
        )

    def retriever(self, stack: str) -> Retriever:
        r, spec = self.settings.retrieval, self.settings.retrieval.stacks[stack]
        if spec.kind == "bm25":
            stages, reranker = {"bm25": BM25Scorer(k1=spec.bm25_k1, b=spec.bm25_b)}, None
        elif spec.kind == "dense" and spec.dense:
            stages, reranker = {"dense": self.scorer("dense", spec.dense, spec.query_instruction)}, None
        elif spec.sparse and spec.dense and spec.reranker:
            stages = {"sparse": self.scorer("sparse", spec.sparse), "dense": self.scorer("dense", spec.dense)}
            reranker = self.reranker(spec.reranker)
        else:
            raise ValueError(f"stack {stack} needs sparse, dense and reranker models")
        return Retriever(
            stages=stages, reranker=reranker, rrf_k=r.rrf_k, rerank_depth=r.rerank_depth, store_depth=r.store_depth
        )
