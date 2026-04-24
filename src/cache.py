"""Tiny disk-backed JSON cache with TTL.

Used to make player-profile, schedule and weather lookups survive across
requests (and within an instance lifetime on Render), so the app stops
hammering Baseball Savant on every page load.
"""

from __future__ import annotations

import json
import os
import re
import time
import threading
from typing import Any, Callable, Optional

CACHE_DIR = os.getenv("MLB_CACHE_DIR", "data_cache")
_LOCK = threading.Lock()


def _safe_key(key: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", key)[:180]


def _path(namespace: str, key: str) -> str:
    return os.path.join(CACHE_DIR, namespace, _safe_key(key) + ".json")


def _ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def get(namespace: str, key: str, ttl_seconds: int) -> Optional[Any]:
    path = _path(namespace, key)
    try:
        with open(path, "r") as f:
            payload = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None

    ts = payload.get("ts", 0)
    if time.time() - ts > ttl_seconds:
        return None
    return payload.get("data")


def put(namespace: str, key: str, data: Any) -> None:
    path = _path(namespace, key)
    with _LOCK:
        try:
            _ensure_dir(path)
            tmp = path + ".tmp"
            with open(tmp, "w") as f:
                json.dump({"ts": time.time(), "data": data}, f)
            os.replace(tmp, path)
        except OSError:
            # disk write failed -- silently skip caching, never break the app
            pass


def memoize(
    namespace: str,
    ttl_seconds: int,
    key_fn: Callable[..., str],
):
    """Decorator: cache the function's return value to disk by key_fn(*args)."""

    def decorator(fn):
        def wrapper(*args, **kwargs):
            key = key_fn(*args, **kwargs)
            cached = get(namespace, key, ttl_seconds)
            if cached is not None:
                return cached
            result = fn(*args, **kwargs)
            if result is not None:
                put(namespace, key, result)
            return result

        wrapper.__wrapped__ = fn
        return wrapper

    return decorator
