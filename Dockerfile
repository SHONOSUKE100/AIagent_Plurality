# syntax=docker/dockerfile:1.4

FROM python:3.12-slim AS base

ENV UV_PROJECT_ENVIRONMENT=/app/.venv \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install build essentials for any deps that need compilation
RUN apt-get update && apt-get install -y --no-install-recommends build-essential curl ca-certificates && rm -rf /var/lib/apt/lists/*

# Install uv and tmux
RUN pip install --no-cache-dir uv && apt-get update && apt-get install -y --no-install-recommends tmux && rm -rf /var/lib/apt/lists/*

# Install mise (https://mise.jdx.dev)
RUN curl -fsSL https://mise.run | sh
ENV PATH="/root/.local/bin:${PATH}"

# Copy dependency manifests and install
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Copy project files
COPY . .

# Default shell; override in docker compose or `docker run ... <cmd>`
CMD ["bash"]
