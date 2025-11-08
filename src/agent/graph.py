"""LangGraph graph that resolves 比赛信息并调用 DeepSeek 模型。"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin

import httpx
import redis.asyncio as redis
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from langgraph.runtime import Runtime
from typing_extensions import TypedDict

from agent.payload_store import (
    DEFAULT_PRIMARY_REDIS_URL,
    PayloadStoreError,
    store_payload,
)

DEFAULT_API_SPORTS_KEY = "ff503be5af62c3de324b71e439883fbf"
DEFAULT_DEEPSEEK_MODEL = "deepseek-reasoner"


class Context(TypedDict, total=False):
    """Context parameters for the agent.

    Set these when creating assistants OR when invoking the graph.
    See: https://langchain-ai.github.io/langgraph/cloud/how-tos/configuration_cloud/
    """

    my_configurable_param: str
    model: str
    temperature: float


@dataclass
class State:
    """Input state for the agent.

    Defines the initial structure of incoming data.
    See: https://langchain-ai.github.io/langgraph/concepts/low_level/#state
    """

    changeme: str = "example"
    user_message: Optional[str] = None
    match_id: Optional[str] = None
    fixture_id: Optional[str] = None
    error: Optional[str] = None
    fixture_payload: Optional[Dict[str, Any]] = None
    fixture_payload_ref: Optional[str] = None
    intent: Optional[str] = None
    query_plan: Optional[List[Dict[str, Any]]] = None


SUPPORTED_DATA_SUMMARY = (
    "当前助手可提供：比赛详情(fixtures)、交锋记录(fixtures/headtohead)、技术统计(fixtures/statistics)、"
    "阵容(fixtures/lineups)、伤病(injuries)、伤停(sidelined)、赔率(odds)、赔率商(odds/bookmakers)、积分榜(standings)。"
)


async def analyze_intent(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Validate that the request is football-related and derive a query plan."""

    message = _resolve_user_message(state).strip()
    if not message:
        guidance = (
            "请提供具体的足球比赛问题，例如给出比赛编号或想查询的内容。"
        )
        return {
            "intent": None,
            "error": guidance,
            "changeme": f"{guidance}\n{SUPPORTED_DATA_SUMMARY}",
        }

    normalized = message.lower()
    non_football_keywords = {
        "篮球",
        "nba",
        "cba",
        "网球",
        "tennis",
        "棒球",
        "mlb",
        "baseball",
        "橄榄球",
        "nfl",
        "冰球",
        "nhl",
        "排球",
        "羽毛球",
        "电竞",
        "dota",
        "lol",
        "csgo",
    }
    if any(
        keyword in message or keyword.lower() in normalized
        for keyword in non_football_keywords
    ):
        guidance = (
            "抱歉，当前助手仅支持足球比赛相关的数据查询。"
        )
        return {
            "intent": "rejected",
            "error": guidance,
            "changeme": f"{guidance}\n{SUPPORTED_DATA_SUMMARY}",
        }

    query_plan = _build_query_plan(message, normalized)
    return {
        "intent": "football",
        "query_plan": query_plan,
        "changeme": state.changeme,
        "error": None,
    }


async def resolve_fixture(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Resolve fixture_id for a match_id via Redis mapping."""

    if state.error:
        return {}

    message = _resolve_user_message(state)
    plan_requires_fixture = any(
        entry.get("requires_fixture") is True
        for entry in (state.query_plan or [])
        if isinstance(entry, dict)
    )
    match_id = state.match_id or _extract_match_id(message)
    if not plan_requires_fixture:
        return {
            "match_id": match_id,
            "fixture_id": state.fixture_id,
            "user_message": message,
            "error": None,
        }

    if not match_id:
        guidance = (
            "为了查到准确的比赛数据，我需要先知道比赛编号（9-10 位 match_id 或 fixture_id）。"
            "请从页面列表里点选目标比赛，或直接把 match_id/fixture_id 告诉我，我才能继续查询。"
        )
        return {
            "fixture_id": None,
            "match_id": None,
            "changeme": (
                f"{guidance}\n{SUPPORTED_DATA_SUMMARY}"
            ),
            "error": guidance,
            "user_message": message,
        }

    redis_url = os.getenv("REDIS_URL") or DEFAULT_PRIMARY_REDIS_URL
    if not redis_url:
        return {
            "fixture_id": None,
            "match_id": match_id,
            "changeme": (
                f"Redis 未配置（缺少 REDIS_URL）。本次 match_id={match_id} 无法查询 fixture_id。"
            ),
            "error": (
                f"Redis 未配置（缺少 REDIS_URL）。本次 match_id={match_id} 无法查询 fixture_id。"
            ),
            "user_message": message,
        }

    client = redis.from_url(redis_url, encoding="utf-8", decode_responses=True)
    try:
        key = f"mapping:matchid-fixtureid:{match_id}"
        fixture_id = await client.get(key)
    except Exception as exc:  # noqa: BLE001 - surface redis errors
        return {
            "fixture_id": None,
            "match_id": match_id,
            "changeme": f"查询 Redis 时出错：{exc}",
            "error": f"查询 Redis 时出错：{exc}",
            "user_message": message,
        }
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            await close()

    if not fixture_id:
        return {
            "fixture_id": None,
            "match_id": match_id,
            "changeme": (
                f"Redis 中不存在匹配项：match_id={match_id}，键 {key}。"
            ),
            "error": f"Redis 中不存在匹配项：match_id={match_id}，键 {key}。",
            "user_message": message,
        }

    return {
        "fixture_id": fixture_id,
        "match_id": match_id,
        "user_message": message,
        "error": None,
    }


async def fetch_fixture(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Fetch fixture details from the API-Football endpoint."""

    if state.error:
        return {}

    runtime_context = runtime.context or {}
    api_key = (
        runtime_context.get("fixture_api_key")
        or os.getenv("API_SPORTS_KEY")
        or os.getenv("APISPORTS_KEY")
        or DEFAULT_API_SPORTS_KEY
    )
    base_url = runtime_context.get(
        "fixture_api_url", "https://v3.football.api-sports.io/fixtures"
    )
    path = runtime_context.get("fixture_api_path")
    if path:
        base_url = urljoin(base_url.rstrip("/") + "/", path.lstrip("/"))

    if isinstance(path, str) and path:
        endpoint_hint = path.rstrip("/").split("/")[-1]
    else:
        endpoint_hint = base_url.rstrip("/").split("/")[-1]

    raw_params = runtime_context.get("fixture_api_params")
    params: Dict[str, Any]
    if isinstance(raw_params, dict):
        params = dict(raw_params)
    elif isinstance(raw_params, (list, tuple)):
        params = dict(raw_params)
    elif isinstance(raw_params, str):
        params = dict(httpx.QueryParams(raw_params))
    else:
        params = {}

    if state.fixture_id:
        if not params:
            if endpoint_hint == "fixtures":
                params["id"] = state.fixture_id
            elif endpoint_hint in {"statistics", "lineups", "injuries", "odds"}:
                params["fixture"] = state.fixture_id
        elif "fixture" not in params and "id" not in params:
            if endpoint_hint in {"statistics", "lineups", "injuries", "odds"}:
                params["fixture"] = state.fixture_id
            elif endpoint_hint == "fixtures":
                params["id"] = state.fixture_id

    if not params:
        return {
            "fixture_payload": None,
            "changeme": "未提供任何查询参数，无法请求比赛数据。",
            "error": "未提供任何查询参数，无法请求比赛数据。",
        }

    headers_context = runtime_context.get("fixture_api_headers")
    headers: Dict[str, str] = (
        dict(headers_context) if isinstance(headers_context, dict) else {}
    )

    if api_key:
        headers.setdefault("x-apisports-key", api_key)
    else:
        return {
            "fixture_payload": None,
            "changeme": "API-Football 未配置密钥（缺少 fixture_api_key 或 API_SPORTS_KEY）。",
            "error": "API-Football 未配置密钥（缺少 fixture_api_key 或 API_SPORTS_KEY）。",
        }

    timeout_seconds = runtime_context.get("fixture_api_timeout", 10.0)

    async with httpx.AsyncClient(timeout=timeout_seconds) as client:
        try:
            response = await client.get(base_url, params=params, headers=headers)
        except httpx.HTTPError as exc:
            return {
                "fixture_payload": None,
                "changeme": f"请求 API-Football 失败：{exc}",
                "error": f"请求 API-Football 失败：{exc}",
            }

    if response.status_code >= 400:
        return {
            "fixture_payload": None,
            "changeme": (
                f"API-Football 返回错误状态码 {response.status_code}：{response.text}"
            ),
            "error": (
                f"API-Football 返回错误状态码 {response.status_code}：{response.text}"
            ),
        }

    try:
        payload = response.json()
    except json.JSONDecodeError:
        return {
            "fixture_payload": None,
            "changeme": "API-Football 响应不是有效的 JSON。",
            "error": "API-Football 响应不是有效的 JSON。",
        }

    payload_summary = _summarize_fixture(payload)
    payload_ref: Optional[str] = None
    compact_payload: Optional[Dict[str, Any]] = None
    metadata = {
        "match_id": state.match_id,
        "fixture_id": state.fixture_id,
        "endpoint": endpoint_hint,
    }

    try:
        payload_ref = await store_payload(
            payload,
            namespace="fixture",
            metadata=metadata,
        )
    except PayloadStoreError:
        # 回退到原始行为，直接把 JSON 写入 state，避免阻断查询流程。
        compact_payload = payload
    else:
        compact_payload = {
            "_kind": "external_ref",
            "ref": payload_ref,
            "summary": payload_summary,
            "endpoint": endpoint_hint,
        }

    return {
        "fixture_payload": compact_payload,
        "fixture_payload_ref": payload_ref,
        "changeme": state.changeme,
        "error": None,
    }


async def call_model(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Call DeepSeek (OpenAI compatible) and handle tool calling."""

    api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
    runtime_context = runtime.context or {}
    configurable_value = runtime_context.get("my_configurable_param", "unset")
    message = _resolve_user_message(state)
    plan_summary = _summarize_plan(state.query_plan)

    if state.error:
        return {"changeme": state.error}

    if not api_key:
        return {
            "changeme": (
                "DeepSeek API key not configured. "
                f"State received: {message!r}. "
                f"Config parameter: {configurable_value!r}."
            )
        }

    base_url = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com")
    model = runtime_context.get("model", DEFAULT_DEEPSEEK_MODEL)
    temperature = runtime_context.get("temperature", 0.2)

    llm = ChatOpenAI(
        api_key=api_key,
        base_url=base_url,
        model=model,
        temperature=temperature,
    )

    messages = [
        SystemMessage(
            content=(
                "你是一名足球数据助手。上游节点已经获取了 API-Football JSON，"
                "必须依据提供的摘要或上下文回答，除非明确缺少字段，否则不得说无法查询。"
                "如数据不足，请精确说明缺失项并提示用户补充。"
            )
        ),
        HumanMessage(
            content=(
                f"当前用户消息: {message!r}. "
                f"match_id={state.match_id!r}, fixture_id={state.fixture_id!r}. "
                f"比赛数据摘要: {_summarize_fixture(state.fixture_payload)}. "
                f"意图: {state.intent!r}. "
                f"查询计划: {plan_summary}. "
                f"可配置参数: {configurable_value!r}. "
                "请基于上述比赛数据输出关键信息（如赛事、时间、球场、状态等），"
                "不得再次声称无法访问数据。若确有缺项，请说明缺少哪些字段。"
            )
        ),
    ]

    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup_config",
                "description": "Return configuration values from the runtime context.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "key": {
                            "type": "string",
                            "description": "Name of the configuration parameter to fetch.",
                        }
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_api_sop",
                "description": "Return the SOP (standard operating procedure) for selecting API-Football endpoints and parameters.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
    ]

    response: AIMessage = await llm.ainvoke(messages, tools=tools)
    messages.append(response)

    while response.tool_calls:
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            arguments = tool_call.get("arguments") or "{}"
            parsed_args = _safe_json_loads(arguments)

            if tool_name == "lookup_config":
                result = _lookup_config(runtime_context, parsed_args)
            elif tool_name == "get_api_sop":
                result = _get_api_sop()
            else:
                result = {"error": f"Unsupported tool: {tool_name}"}

            messages.append(
                ToolMessage(
                    content=json.dumps(result, ensure_ascii=False),
                    tool_call_id=tool_call["id"],
                )
            )

        response = await llm.ainvoke(messages, tools=tools)
        messages.append(response)

    return {"changeme": _render_content(response.content)}


def _lookup_config(
    runtime_context: Dict[str, Any], arguments: Dict[str, Any]
) -> Dict[str, Any]:
    """Return runtime configuration details for the tool call."""
    key = arguments.get("key")
    if key:
        return {"key": key, "value": runtime_context.get(key)}

    return {
        "available_keys": sorted(runtime_context.keys()),
        "context": runtime_context,
    }


def _get_api_sop() -> Dict[str, Any]:
    """Return SOP describing how to choose and call API-Football endpoints."""

    document = (
        "## API-Football 调用 SOP\n"
        "\n"
        "### 1. 理解需求\n"
        "- 确定用户是想要比赛详情、头对头、技术统计、阵容、伤病、伤停、赔率、赔率商还是积分榜。\n"
        "- 若关键信息缺失（例如球队、日期、赛事编号等），先追问补齐。\n"
        "\n"
        "### 2. 选择端点并设置 `fixture_api_path`\n"
        "- `fixtures`：单场/多场比赛、live、按联赛或日期过滤。\n"
        "- `fixtures/headtohead`：`h2h=主队ID-客队ID`，可叠加 status/date/from/to/league/season/last/next/timezone。\n"
        "- `fixtures/statistics`：`fixture=比赛ID`，可加 half=true、type=Total Shots、team=球队ID。\n"
        "- `fixtures/lineups`：`fixture=比赛ID`，可加 team、player、type=startXI。\n"
        "- `injuries`：按 league+season、fixture、ids、team、player、date 等查询伤病。\n"
        "- `sidelined`：`player=` 或 `players=` 查询球员，`coach=` 或 `coachs=` 查询教练。\n"
        "- `odds`：`fixture`、`league&season`、`date`、`bookmaker`、`bet` 等组合。\n"
        "- `odds/bookmakers`：列出全部赔率商、按 id 查询或 `search=` 模糊搜索。\n"
        "- `standings`：`league&season`、`league&team&season` 或 `team&season` 获取积分榜。\n"
        "\n"
        "### 3. 配置 `fixture_api_params`\n"
        "- 使用 dict、列表或查询字符串（如 `\"league=39&season=2019\"`）。\n"
        "- `fetch_fixture` 会在必要时自动把 `state.fixture_id` 映射为 `id` 或 `fixture`。\n"
        "- 若端点仍缺必需参数，节点会返回中文错误，需要先补齐后再次调用。\n"
        "\n"
        "### 4. 可选上下文键\n"
        "- `fixture_api_headers`：附加自定义 header，默认自动添加 `x-apisports-key`。\n"
        "- `fixture_api_timeout`：浮点数，单位秒，默认 10。\n"
        "- `fixture_api_key`：若不使用环境变量，可在 context 中直接提供。\n"
        "\n"
        "### 5. 错误处理\n"
        "- 缺少密钥、HTTP >= 400、JSON 解析失败时，`state.error` 会记录中文描述，后续节点应优先反馈错误。\n"
        "- 同一次对话中需要多条数据，可分多次设置 context 并重新执行 `fetch_fixture`。\n"
        "\n"
        "### 6. 输出交接\n"
        "- `fetch_fixture` 的结果写入 `state.fixture_payload`。\n"
        "- `call_model` 会在系统信息中包含压缩后的 JSON 片段，模型据此总结或继续追问。\n"
    )

    return {"document": document}


def _summarize_fixture(payload: Optional[Dict[str, Any]]) -> str:
    """Return a short string representation of the fixture payload."""
    if not payload:
        return "无"

    if isinstance(payload, dict) and payload.get("_kind") == "external_ref":
        summary = payload.get("summary") or "无摘要"
        ref = payload.get("ref")
        if ref:
            return f"{summary} (ref={ref})"
        return summary

    try:
        serialized = json.dumps(payload, ensure_ascii=False)
    except TypeError:
        serialized = str(payload)

    if len(serialized) > 500:
        return f"{serialized[:500]}…"

    return serialized


def _build_query_plan(message: str, normalized: str) -> List[Dict[str, Any]]:
    """Derive a high-level query plan based on the user question."""

    plan: List[Dict[str, Any]] = [
        {
            "step": "resolve_fixture",
            "description": "解析 match_id 并从 Redis 匹配 fixture_id。",
            "uses": "resolve_fixture",
            "requires": "提供 9-10 位 match_id 或上下文中的 fixture_id。",
        }
    ]

    keyword_map = [
        (
            {
                "交锋",
                "对战",
                "对阵记录",
                "h2h",
                "head to head",
            },
            {
                "step": "fetch_head_to_head",
                "endpoint": "fixtures/headtohead",
                "description": "查询两队历史交锋，将先通过 fixture 获取主客队 ID 后再调用。",
                "uses": "fetch_fixture",
                "requires_fixture": True,
            },
        ),
        (
            {
                "统计",
                "技术统计",
                "数据统计",
                "stats",
                "possession",
                "射门",
            },
            {
                "step": "fetch_statistics",
                "endpoint": "fixtures/statistics",
                "description": "获取比赛技术统计，需要 fixture 参数。",
                "uses": "fetch_fixture",
                "requires_fixture": True,
            },
        ),
        (
            {
                "阵容",
                "首发",
                "首发阵容",
                "lineup",
                "line-up",
            },
            {
                "step": "fetch_lineups",
                "endpoint": "fixtures/lineups",
                "description": "获取首发与替补阵容，需要 fixture 参数，可选 team。",
                "uses": "fetch_fixture",
                "requires_fixture": True,
            },
        ),
        (
            {
                "伤病",
                "伤停",
                "injury",
                "伤势",
            },
            {
                "step": "fetch_injuries",
                "endpoint": "injuries",
                "description": "查询球员或球队伤病情况，需要联赛+赛季或 fixture/队伍信息。",
                "uses": "fetch_fixture",
                "requires_fixture": False,
            },
        ),
        (
            {
                "禁赛",
                "停赛",
                "sidelined",
            },
            {
                "step": "fetch_sidelined",
                "endpoint": "sidelined",
                "description": "查询球员/教练停赛情况，需要提供球员或教练 ID。",
                "uses": "fetch_fixture",
                "requires_fixture": False,
            },
        ),
        (
            {
                "赔率",
                "盘口",
                "博彩",
                "odds",
            },
            {
                "step": "fetch_odds",
                "endpoint": "odds",
                "description": "查询比赛赔率，需要 fixture 或联赛+赛季等参数。",
                "uses": "fetch_fixture",
                "requires_fixture": True,
            },
        ),
        (
            {
                "赔率商",
                "bookmaker",
            },
            {
                "step": "fetch_bookmakers",
                "endpoint": "odds/bookmakers",
                "description": "列出或搜索赔率商，可选 bookmaker id 或 search。",
                "uses": "fetch_fixture",
                "requires_fixture": False,
            },
        ),
        (
            {
                "积分榜",
                "排名",
                "standings",
                "table",
            },
            {
                "step": "fetch_standings",
                "endpoint": "standings",
                "description": "查询联赛积分榜，需要 league 与 season 或 team。",
                "uses": "fetch_fixture",
                "requires_fixture": False,
            },
        ),
    ]

    matched_steps: List[Dict[str, Any]] = []
    for keywords, entry in keyword_map:
        if any(keyword.lower() in normalized for keyword in keywords):
            matched_steps.append(entry)

    if not matched_steps:
        matched_steps.append(
            {
                "step": "fetch_fixture",
                "endpoint": "fixtures",
                "description": "获取比赛基础数据（比分、时间、场地等）。",
                "uses": "fetch_fixture",
                "requires_fixture": True,
            }
        )

    plan.extend(matched_steps)
    plan.append(
        {
            "step": "call_model",
            "description": "调用模型整合数据并生成回答或追问缺失信息。",
            "uses": "call_model",
        }
    )

    return plan


def _summarize_plan(plan: Optional[List[Dict[str, Any]]]) -> str:
    """Return a concise string representation of the query plan."""

    if not plan:
        return "无计划"

    parts: List[str] = []
    for item in plan:
        step = item.get("step", "unknown")
        description = item.get("description", "")
        endpoint = item.get("endpoint")
        if endpoint:
            parts.append(f"{step}({endpoint}): {description}")
        else:
            parts.append(f"{step}: {description}")

    return " | ".join(parts)


def _render_content(content: Any) -> str:
    """Normalize LangChain content payloads to a single string."""
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "".join(parts)

    return str(content)


def _safe_json_loads(payload: str) -> Dict[str, Any]:
    """Parse JSON, swallowing errors so tool calls never crash graph execution."""
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return {}
    if isinstance(data, dict):
        return data
    return {}


def _extract_match_id(text: str) -> Optional[str]:
    """Extract a 9-10 digit match_id from free text."""
    if not text:
        return None

    match = re.search(r"\b(\d{9,10})\b", text)
    return match.group(1) if match else None


def _resolve_user_message(state: State) -> str:
    """Return the primary user message from state inputs."""
    return state.user_message or state.changeme or ""


# Define the graph
graph = (
    StateGraph(State, context_schema=Context)
    .add_node(analyze_intent)
    .add_node(resolve_fixture)
    .add_node(fetch_fixture)
    .add_node(call_model)
    .add_edge("__start__", "analyze_intent")
    .add_edge("analyze_intent", "resolve_fixture")
    .add_edge("resolve_fixture", "fetch_fixture")
    .add_edge("fetch_fixture", "call_model")
    .compile(name="New Graph")
)
