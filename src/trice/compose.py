"""The composition root: one explicit function that builds and wires every component.

Read top to bottom, it is the whole architecture. To customize, edit it, or pass a ``transport`` to
replace the network (tests do). It is the only place that turns a Config into components."""

import json
from collections.abc import Generator
from contextlib import ExitStack, contextmanager

from diskcache import Cache
from jinja2 import Environment, PackageLoader, StrictUndefined
from openai import OpenAI
from pydantic import BaseModel, ConfigDict

from trice.analyzer import Analyzer
from trice.budget import Budget
from trice.config import BM25Stack, Config, DenseStack, TriceStack
from trice.core.encoders import ApiClient, ApiReranker, ApiScorer, BM25Scorer, DenseScorer, QwenReranker, SparseScorer
from trice.core.formats import Format, InlineFormat, JsonFormat, ListFormat, ListingFormat
from trice.core.retrieval import Reranker, Retriever, Scorer, Stage
from trice.data import Corpus
from trice.evaluation.inference import Inferrer
from trice.evaluation.report import Reporter
from trice.evaluation.scoring import Evaluator
from trice.llm import LLM, Endpoint, EndpointTable, OpenAITransport, Pinner, Transport
from trice.runner import Runner
from trice.schema import FormatName, ModelRef


class App(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    config: Config
    corpus: Corpus
    formats: dict[FormatName, Format]
    retrievers: dict[str, Retriever]
    endpoints: list[Endpoint]
    llms: dict[str, LLM]
    pinner: Pinner
    runner: Runner
    budget: Budget
    analyzer: Analyzer


@contextmanager
def compose(config: Config, transport: Transport | None = None) -> Generator[App]:
    with ExitStack() as stack:
        layout, llm, retrieval = config.layout, config.llm, config.retrieval

        # Templates: every prompt and report string is rendered from src/trice/templates.
        env = Environment(loader=PackageLoader("trice"), undefined=StrictUndefined, autoescape=False)
        env.filters.update(
            json=lambda v: json.dumps(v, ensure_ascii=False),
            cell=lambda v: f"{v:.4f}" if isinstance(v, float) else str(v),
            latex=lambda v: "".join("\\" + c if c in "&%$#_{}" else c for c in str(v)),
        )

        # Network and storage: one client per backend, one record/replay cache.
        cache = stack.enter_context(Cache(layout.cache.fill()))
        openrouter = stack.enter_context(
            OpenAI(
                base_url=llm.openrouter_url,
                api_key=config.openrouter_api_key.get_secret_value() if config.openrouter_api_key else "missing",
                max_retries=llm.max_retries,
                timeout=llm.timeout,
            )
        )
        hosted = transport or OpenAITransport(client=openrouter, backend="openrouter", allow_spend=llm.allow_spend)
        local = transport or OpenAITransport(
            client=stack.enter_context(OpenAI(base_url=llm.local_url, api_key="local", timeout=llm.timeout)),
            backend="local",
        )

        # Data and prompt formats (the order is the order of the S0 candidates).
        corpus = Corpus(raw=layout.raw, canonical=layout.canonical, sources=config.sources, specs=config.datasets)
        formats: dict[FormatName, Format] = {
            "P0": ListingFormat(env=env),
            "P1": ListFormat(env=env),
            "P2": JsonFormat(env=env),
            "P3": InlineFormat(env=env),
        }

        # Retrieval stacks: local models by default, OpenRouter when a model says so.
        api = ApiClient(client=openrouter, cache=cache, allow_spend=llm.allow_spend)

        def scorer(ref: ModelRef, stage: str, prompt: str | None = None) -> Scorer:
            if ref.backend == "openrouter":
                return ApiScorer(api=api, model_id=ref.id)
            if stage == "sparse":
                return SparseScorer(ref=ref, runtime=retrieval.runtime)
            return DenseScorer(ref=ref, runtime=retrieval.runtime, query_prompt=prompt)

        def reranker(ref: ModelRef) -> Reranker:
            if ref.backend == "openrouter":
                return ApiReranker(api=api, model_id=ref.id)
            return QwenReranker(ref=ref, runtime=retrieval.runtime, instruction=retrieval.reranker_instruction, env=env)

        retrievers = {}
        for name, spec in retrieval.stacks.items():
            match spec:
                case TriceStack():
                    stages = [
                        Stage(name="sparse", scorer=scorer(spec.sparse, "sparse")),
                        Stage(name="dense", scorer=scorer(spec.dense, "dense")),
                    ]
                    ranker = reranker(spec.reranker)
                case BM25Stack():
                    stages, ranker = [Stage(name="bm25", scorer=BM25Scorer(k1=spec.k1, b=spec.b))], None
                case DenseStack():
                    stages, ranker = (
                        [Stage(name="dense", scorer=scorer(spec.dense, "dense", spec.query_instruction))],
                        None,
                    )
            retrievers[name] = Retriever(
                stages=stages,
                reranker=ranker,
                rrf_k=retrieval.rrf_k,
                rerank_depth=retrieval.rerank_depth,
                store_depth=retrieval.store_depth,
            )

        # Model calls: the pinned endpoint of every model (written by `trice pin`).
        endpoints = (
            EndpointTable.model_validate_json(layout.endpoints.fill().read_text()).root
            if layout.endpoints.fill().exists()
            else []
        )

        # Studies, budget and analysis.
        runner = Runner(
            config=config,
            corpus=corpus,
            formats=formats,
            env=env,
            evaluator=Evaluator(),
            inferrer=Inferrer(resamples=config.stats.resamples, alpha=config.stats.alpha),
        )
        yield App(
            config=config,
            corpus=corpus,
            formats=formats,
            retrievers=retrievers,
            endpoints=endpoints,
            llms={
                e.name: LLM(
                    endpoint=e, transport=hosted if e.backend == "openrouter" else local, cache=cache, config=llm
                )
                for e in endpoints
            },
            pinner=Pinner(client=openrouter, config=llm),
            runner=runner,
            budget=Budget(runner=runner, env=env, endpoints=endpoints),
            analyzer=Analyzer(runner=runner, reporter=Reporter(env=env, tables=layout.tables)),
        )
