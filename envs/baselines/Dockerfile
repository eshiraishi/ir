# Baselines image: gliner pins transformers <5.17, so it cannot share an environment with the core.
# Built by envs/compose.yaml; the repository is mounted at /work at run time, where trice.toml and
# data/ are read and results/ is written.
FROM ghcr.io/astral-sh/uv:python3.14-bookworm-slim

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-install-project
COPY baselines.py ./

ENV PATH="/app/.venv/bin:$PATH" HF_HOME=/work/.cache/huggingface
WORKDIR /work
ENTRYPOINT ["python", "/app/baselines.py"]
