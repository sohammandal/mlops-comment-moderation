# syntax=docker/dockerfile:1.4

########## BUILDER ##########
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim AS builder
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
WORKDIR /app

# Install dependencies (without project code) - cache layer
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project --no-dev

# Copy only the source code (big stuff excluded via .dockerignore)
COPY . /app

# Install project itself
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

# Remove build cache
RUN rm -rf /root/.cache/uv

########## RUNTIME ##########
FROM python:3.11-slim-bookworm AS runtime

# Install minimal OS deps (curl only if needed for healthchecks)
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy venv from builder (all Python deps)
COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Copy only runtime code & metadata
COPY --from=builder /app/src /app/src
COPY --from=builder /app/pyproject.toml /app/

# Expose ports
EXPOSE 8000 8501

# Default command
ENTRYPOINT []
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
