# syntax=docker/dockerfile:1.4

FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim AS builder
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
WORKDIR /app

# Install dependencies first (without project)
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project --no-dev

# Copy project and install it
COPY . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

# Runtime stage (lighter, no uv)
FROM python:3.11-slim-bookworm AS runtime

# Install curl for healthcheck
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Copy only app files and its venv
COPY --from=builder /app /app
ENV PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Exposed ports (API on 8000, Streamlit on 8501)
EXPOSE 8000 8501

# Use ENTRYPOINT so CMD can be overridden in docker-compose
ENTRYPOINT []
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]


