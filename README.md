# New LangGraph Project

[![CI](https://github.com/langchain-ai/new-langgraph-project/actions/workflows/unit-tests.yml/badge.svg)](https://github.com/langchain-ai/new-langgraph-project/actions/workflows/unit-tests.yml)
[![Integration Tests](https://github.com/langchain-ai/new-langgraph-project/actions/workflows/integration-tests.yml/badge.svg)](https://github.com/langchain-ai/new-langgraph-project/actions/workflows/integration-tests.yml)

This template demonstrates a simple application implemented using [LangGraph](https://github.com/langchain-ai/langgraph), designed for showing how to get started with [LangGraph Server](https://langchain-ai.github.io/langgraph/concepts/langgraph_server/#langgraph-server) and using [LangGraph Studio](https://langchain-ai.github.io/langgraph/concepts/langgraph_studio/), a visual debugging IDE.

<div align="center">
  <img src="./static/studio_ui.png" alt="Graph view in LangGraph studio UI" width="75%" />
</div>

The core logic defined in `src/agent/graph.py`, showcases an single-step application that responds with a fixed string and the configuration provided.

You can extend this graph to orchestrate more complex agentic workflows that can be visualized and debugged in LangGraph Studio.

## Getting Started

1. Install dependencies, along with the [LangGraph CLI](https://langchain-ai.github.io/langgraph/concepts/langgraph_cli/), which will be used to run the server.

```bash
cd path/to/your/app
pip install -e . "langgraph-cli[inmem]"
```

2. (Optional) Customize the code and project as needed. Create a `.env` file if you need to use secrets.

```bash
cp .env.example .env
```

If you want to enable LangSmith tracing, add your LangSmith API key to the `.env` file.

```text
# .env
LANGSMITH_API_KEY=lsv2...
```

To call DeepSeek via its OpenAI-compatible API, set the following variables (already present in `.env.example`):

```text
DEEPSEEK_API_KEY=sk-...
OPENAI_BASE_URL=https://api.deepseek.com
REDIS_URL=redis://app:Str0ng!Passw0rd-ChangeMe@120.46.5.144:6379/1
PAYLOAD_REDIS_URL=redis://app:Str0ng!Passw0rd-ChangeMe@120.46.5.144:6379/4
PAYLOAD_TTL_SECONDS=3600
```

`fetch_fixture` 会把 API-Football 的完整 JSON 存入 `PAYLOAD_REDIS_URL` 指向的 Redis（未配置时会退回到 `REDIS_URL` 的第 4 库），LangGraph 状态只携带摘要与引用 key，避免上下文膨胀。`PAYLOAD_TTL_SECONDS` 控制引用写入后的生存时间。

3. Start the LangGraph Server.

```shell
langgraph dev
```

For more information on getting started with LangGraph Server, [see here](https://langchain-ai.github.io/langgraph/tutorials/langgraph-platform/local-server/).

## How to customize

1. **Define runtime context**: Modify the `Context` class in the `graph.py` file to expose the arguments you want to configure per assistant. For example, in a chatbot application you may want to define a dynamic system prompt or LLM to use. For more information on runtime context in LangGraph, [see here](https://langchain-ai.github.io/langgraph/agents/context/?h=context#static-runtime-context).

2. **Extend the graph**: The core logic of the application is defined in [graph.py](./src/agent/graph.py). You can modify this file to add new nodes, edges, or change the flow of information.

The current implementation adds a `resolve_fixture` node that looks up `mapping:matchid-fixtureid:{match_id}` in Redis and forwards the resulting `fixture_id` to the DeepSeek chat node. The downstream `fetch_fixture` node now offloads large payloads into Redis DB 4 and keeps only a lightweight `_kind=external_ref` summary in the LangGraph state.

## Development

While iterating on your graph in LangGraph Studio, you can edit past state and rerun your app from previous states to debug specific nodes. Local changes will be automatically applied via hot reload.

Follow-up requests extend the same thread. You can create an entirely new thread, clearing previous history, using the `+` button in the top right.

For more advanced features and examples, refer to the [LangGraph documentation](https://langchain-ai.github.io/langgraph/). These resources can help you adapt this template for your specific use case and build more sophisticated conversational agents.

LangGraph Studio also integrates with [LangSmith](https://smith.langchain.com/) for more in-depth tracing and collaboration with teammates, allowing you to analyze and optimize your chatbot's performance.

## Docker

Build a development image that bundles dependencies and the CLI (Dockerfile defaults to the `https://g0gb7qq6rcc7mh.xuanyuan.run/simple` mirror and falls back to PyPI):

```bash
docker build -t langgraph-agent .
```

If you use a private/accelerated index, pass it via build args (example mirrors shown below). Keep
the official index in `PIP_EXTRA_INDEX_URL` so pip can fall back when a package is missing:

```bash
docker build \
  --build-arg PIP_INDEX_URL=https://g0gb7qq6rcc7mh.xuanyuan.run/simple \
  --build-arg PIP_EXTRA_INDEX_URL=https://pypi.org/simple \
  -t langgraph-agent .
```

Run the container, mounting a local `.env` with your secrets and exposing the dev server:

```bash
docker run --rm -p 2024:2024 --env-file .env langgraph-agent
```

The service listens on `0.0.0.0:2024`, so you can connect via `http://localhost:2024` (or your host/forwarded port). Provide the real Redis/DeepSeek credentials via environment variables or an `.env` file; the image only includes `.env.example`.
