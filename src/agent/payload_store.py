"""Helpers for persisting large payloads in Redis DB 4."""

from __future__ import annotations

import json
import os
import time
import uuid
from typing import Any, Dict, Optional, Tuple

import redis.asyncio as redis

DEFAULT_PRIMARY_REDIS_URL = "redis://app:Str0ng!Passw0rd-ChangeMe@120.46.5.144:6379/1"
DEFAULT_PAYLOAD_REDIS_URL = "redis://app:Str0ng!Passw0rd-ChangeMe@120.46.5.144:6379/4"


class PayloadStoreError(RuntimeError):
    """Raised when payload persistence is not available or fails."""


def _resolve_connection() -> Tuple[str, Dict[str, Any]]:
    """Return the Redis connection URL and kwargs for the payload store."""

    payload_url = os.getenv("PAYLOAD_REDIS_URL")
    if payload_url:
        return payload_url, {}

    fallback_url = os.getenv("REDIS_URL")
    if fallback_url:
        # Force DB 4 when reusing the primary Redis connection string.
        return fallback_url, {"db": 4}

    if DEFAULT_PAYLOAD_REDIS_URL:
        return DEFAULT_PAYLOAD_REDIS_URL, {}

    if DEFAULT_PRIMARY_REDIS_URL:
        return DEFAULT_PRIMARY_REDIS_URL, {"db": 4}

    raise PayloadStoreError(
        "Payload Redis 未配置（请设置 PAYLOAD_REDIS_URL 或 REDIS_URL）。"
    )


def _default_ttl() -> int:
    raw = os.getenv("PAYLOAD_TTL_SECONDS")
    if not raw:
        return 3600
    try:
        return max(int(raw), 60)
    except ValueError:
        return 3600


def _build_key(namespace: str, metadata: Optional[Dict[str, Any]]) -> str:
    pieces = ["payload", namespace]
    if metadata:
        for field in ("match_id", "fixture_id", "endpoint"):
            value = metadata.get(field)
            if value:
                pieces.append(str(value))
    pieces.append(uuid.uuid4().hex)
    return ":".join(pieces)


async def store_payload(
    payload: Any,
    *,
    namespace: str = "fixture",
    metadata: Optional[Dict[str, Any]] = None,
    ttl_seconds: Optional[int] = None,
) -> str:
    """Persist the payload and return its Redis reference key."""

    url, extra_kwargs = _resolve_connection()
    client = redis.from_url(
        url,
        encoding="utf-8",
        decode_responses=True,
        **extra_kwargs,
    )

    key = _build_key(namespace, metadata)
    envelope = {
        "payload": payload,
        "metadata": metadata or {},
        "stored_at": time.time(),
    }

    try:
        await client.set(
            key,
            json.dumps(envelope, ensure_ascii=False),
            ex=ttl_seconds or _default_ttl(),
        )
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            await close()

    return key


async def load_payload(reference: str) -> Optional[Dict[str, Any]]:
    """Load the payload envelope for a previously stored reference key."""

    url, extra_kwargs = _resolve_connection()
    client = redis.from_url(
        url,
        encoding="utf-8",
        decode_responses=True,
        **extra_kwargs,
    )

    try:
        raw = await client.get(reference)
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            await close()

    if not raw:
        return None

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None

    if isinstance(data, dict):
        return data
    return None


async def delete_payload(reference: str) -> None:
    """Delete a stored payload reference, ignoring missing keys."""

    url, extra_kwargs = _resolve_connection()
    client = redis.from_url(
        url,
        encoding="utf-8",
        decode_responses=True,
        **extra_kwargs,
    )

    try:
        await client.delete(reference)
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            await close()
