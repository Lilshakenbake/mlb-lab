"""OpenAI second-opinion layer for top picks.

Wraps the official `openai` SDK with:
  - lazy client init (no key -> module is a no-op)
  - in-memory cache keyed by pick set (avoid burning tokens on identical asks)
  - structured JSON output via response_format
  - hard timeout + try/except so a flaky API call NEVER blocks the slate refresh

The pick objects passed in are dicts produced by predict.* builders; we read
only a handful of safe fields so the prompt stays tiny.
"""
from __future__ import annotations

import os
import json
import time
import hashlib
from typing import Any, Iterable

try:
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover - openai is optional at runtime
    OpenAI = None  # type: ignore

_client = None
_CACHE: dict[str, dict] = {}
_CACHE_TTL = int(os.getenv("AI_REVIEW_TTL", "21600"))  # 6h default — keeps cost minimal
_MODEL = os.getenv("AI_REVIEW_MODEL", "gpt-4o-mini")
_TIMEOUT = float(os.getenv("AI_REVIEW_TIMEOUT", "20"))
_MAX_PICKS = int(os.getenv("AI_REVIEW_MAX_PICKS", "12"))
_DISABLED = os.getenv("AI_REVIEW_DISABLED", "0") in ("1", "true", "True")
# Hard daily call cap (defensive — prevents any runaway cost no matter
# what triggers a refresh). Resets every 24h. Set AI_REVIEW_DAILY_CAP=0
# to disable the cap.
_DAILY_CAP = int(os.getenv("AI_REVIEW_DAILY_CAP", "20"))
_call_log: list[float] = []


def _get_client():
    global _client
    if _DISABLED:
        return None
    if _client is not None:
        return _client
    key = os.getenv("OPENAI_API_KEY")
    if not key or OpenAI is None:
        return None
    try:
        _client = OpenAI(api_key=key)
    except Exception as e:
        print(f"[ai-review] client init failed: {e}")
        return None
    return _client


def _under_daily_cap() -> bool:
    """Drop entries older than 24h, then check whether we're still under
    the per-day call cap. Cheap defensive guard against runaway cost."""
    if _DAILY_CAP <= 0:
        return True
    cutoff = time.time() - 86400
    while _call_log and _call_log[0] < cutoff:
        _call_log.pop(0)
    return len(_call_log) < _DAILY_CAP


def _pick_key(p: dict) -> str:
    """Stable identifier for one pick. Falls back to `player`/`pitcher` since
    game-detail picks store the name there instead of `headline`."""
    name = p.get("headline") or p.get("player") or p.get("pitcher") or "?"
    stat = p.get("stat_label") or "?"
    matchup = p.get("matchup") or "?"
    return f"{name}|{stat}|{matchup}|{p.get('game_pk','?')}"


def _cache_key(picks: list[dict]) -> str:
    raw = "||".join(_pick_key(p) for p in picks)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def _summarize_pick(i: int, p: dict) -> str:
    pick_side = p.get("pick") or "OVER"
    line = p.get("line")
    line_str = f" {pick_side} {line}" if line is not None else f" {pick_side}"
    name = p.get("headline") or p.get("player") or p.get("pitcher") or "?"
    parts = [
        f"{i}. {name}",
        f"{p.get('stat_label','?')}{line_str}",
        f"prob {p.get('probability','?')}%",
    ]
    vs = p.get("vs")
    if vs:
        parts.append(f"vs {vs}")
    park = p.get("park")
    if park:
        parts.append(f"@ {park}")
    note = p.get("matchup_note") or p.get("rationale")
    if note:
        parts.append(str(note)[:140])
    return " | ".join(parts)


def review_picks(picks: list[dict], kind: str = "picks") -> dict[str, dict]:
    """Ask GPT to sanity-check each pick. Returns a {_pick_key: review} map.

    review fields:
      - verdict:    "AGREE" | "LEAN" | "FADE"
      - confidence: "HIGH"  | "MED"  | "LOW"
      - note:       short reason (1 sentence)

    Returns empty dict on any failure (no key, network error, parse error).
    """
    if not picks:
        return {}
    picks = list(picks)[:_MAX_PICKS]
    client = _get_client()
    if client is None:
        return {}

    ck = _cache_key(picks)
    cached = _CACHE.get(ck)
    if cached and time.time() - cached["ts"] < _CACHE_TTL:
        return cached["data"]
    # Defensive: enforce per-day call cap.
    if not _under_daily_cap():
        print(f"[ai-review] daily cap of {_DAILY_CAP} hit — skipping call")
        return {}

    summary = "\n".join(_summarize_pick(i + 1, p) for i, p in enumerate(picks))
    prompt = (
        "You are a sharp, NEUTRAL MLB betting analyst giving a quick "
        "second-opinion on a model's top picks tonight. The model uses Poisson "
        "rate projections with park/weather/recent-form adjustments. For each "
        "pick give an honest, balanced verdict — neither rubber-stamp nor "
        "reflexively contrarian:\n"
        "  - verdict: AGREE (you support the pick), LEAN (mostly support, with "
        "    a caveat), or FADE (you actively disagree because of a concrete "
        "    angle the model may be missing)\n"
        "  - confidence: HIGH, MED, or LOW\n"
        "  - note: ONE short sentence (<=18 words) citing a specific angle "
        "    (park, weather, lineup spot, recent form, pitcher matchup, "
        "    bullpen, weather, regression risk). Do NOT default to FADE just "
        "    because the probability is high.\n\n"
        f"Category: {kind}\n"
        f"Picks:\n{summary}\n\n"
        'Respond as JSON: {"reviews":[{"i":1,"verdict":"AGREE","confidence":"HIGH","note":"..."}]}'
    )

    try:
        _call_log.append(time.time())
        resp = client.chat.completions.create(
            model=_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.3,
            max_tokens=500,
            timeout=_TIMEOUT,
        )
        raw = resp.choices[0].message.content or "{}"
        parsed = json.loads(raw)
        try:
            usage = getattr(resp, "usage", None)
            if usage:
                print(
                    f"[ai-review] call ok — model={_MODEL} "
                    f"in={getattr(usage,'prompt_tokens','?')} "
                    f"out={getattr(usage,'completion_tokens','?')} "
                    f"daily={len(_call_log)}/{_DAILY_CAP}"
                )
        except Exception:
            pass
    except Exception as e:
        print(f"[ai-review] {kind} failed: {e}")
        return {}

    out: dict[str, dict] = {}
    for r in parsed.get("reviews", []) or []:
        idx = r.get("i")
        if not isinstance(idx, int) or idx < 1 or idx > len(picks):
            continue
        verdict = str(r.get("verdict", "")).strip().upper()
        if verdict not in ("AGREE", "LEAN", "FADE"):
            continue
        confidence = str(r.get("confidence", "")).strip().upper()
        if confidence not in ("HIGH", "MED", "LOW"):
            confidence = "MED"
        note = str(r.get("note", "")).strip()
        out[_pick_key(picks[idx - 1])] = {
            "verdict": verdict,
            "confidence": confidence,
            "note": note[:240],
        }

    _CACHE[ck] = {"ts": time.time(), "data": out}
    return out


def attach_reviews(picks: Iterable[dict], reviews: dict[str, dict]) -> list[dict]:
    """Mutates and returns the pick list with `ai_review` attached where known."""
    out = []
    for p in picks:
        r = reviews.get(_pick_key(p))
        if r:
            p["ai_review"] = r
        out.append(p)
    return out


def is_enabled() -> bool:
    return _get_client() is not None
