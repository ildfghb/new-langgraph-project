# syntax=docker/dockerfile:1.7

FROM python:3.12-slim AS base

ARG PIP_INDEX_URL=https://g0gb7qq6rcc7mh.xuanyuan.run/simple
ARG PIP_EXTRA_INDEX_URL=https://pypi.org/simple
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_INDEX_URL=${PIP_INDEX_URL} \
    PIP_EXTRA_INDEX_URL=${PIP_EXTRA_INDEX_URL}

WORKDIR /app

# System packages required to compile some optional deps (e.g. orjson, xxhash)
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml langgraph.json README.md ./
COPY src ./src
COPY tests ./tests
COPY static ./static
COPY .env.example ./.env.example

RUN pip install --upgrade pip \
    && pip install "langgraph-cli[inmem]" \
    && pip install -e .

EXPOSE 2024

CMD ["langgraph", "dev", "--host", "0.0.0.0"]
