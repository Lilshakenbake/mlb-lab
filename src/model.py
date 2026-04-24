"""
Loads trained projection models from ./models/ and exposes simple inference
helpers used by src/predict.py. If no models are present (or sklearn/joblib
are unavailable), every helper returns None so callers can fall back to the
existing heuristic projections.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

try:
    import joblib  # type: ignore
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    joblib = None
    np = None

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

_BUNDLES: dict = {}


def _load(name: str):
    if joblib is None:
        return None
    if name in _BUNDLES:
        return _BUNDLES[name]
    path = MODELS_DIR / f"{name}.joblib"
    if not path.exists():
        _BUNDLES[name] = None
        return None
    try:
        _BUNDLES[name] = joblib.load(path)
    except Exception:
        _BUNDLES[name] = None
    return _BUNDLES[name]


def _row(bundle, values: dict):
    feats = bundle["features"]
    arr = [[float(values.get(f, 0.0)) for f in feats]]
    return arr


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def models_available() -> bool:
    return any(
        (MODELS_DIR / f"{n}.joblib").exists()
        for n in ("hitter_hits", "hitter_total_bases", "pitcher_strikeouts")
    )


def _predict(name: str, values: dict) -> Optional[tuple[float, float]]:
    bundle = _load(name)
    if bundle is None:
        return None
    try:
        proj = float(bundle["model"].predict(_row(bundle, values))[0])
        return proj, float(bundle.get("residual_std", 1.0))
    except Exception:
        return None


def hitter_hits(hits_avg_5, tb_avg_5, hits_std_5=0.0, tb_std_5=0.0, indoor=0,
                hr_avg_5=0.0, rbi_avg_5=0.0):
    return _predict("hitter_hits", {
        "hits_avg_5": hits_avg_5,
        "tb_avg_5": tb_avg_5,
        "hr_avg_5": hr_avg_5,
        "rbi_avg_5": rbi_avg_5,
        "hits_std_5": hits_std_5,
        "tb_std_5": tb_std_5,
        "indoor": indoor,
    })


def hitter_total_bases(hits_avg_5, tb_avg_5, hits_std_5=0.0, tb_std_5=0.0, indoor=0,
                       hr_avg_5=0.0, rbi_avg_5=0.0):
    return _predict("hitter_total_bases", {
        "hits_avg_5": hits_avg_5,
        "tb_avg_5": tb_avg_5,
        "hr_avg_5": hr_avg_5,
        "rbi_avg_5": rbi_avg_5,
        "hits_std_5": hits_std_5,
        "tb_std_5": tb_std_5,
        "indoor": indoor,
    })


def hitter_home_runs(hits_avg_5, tb_avg_5, hr_avg_5, rbi_avg_5=0.0,
                     hits_std_5=0.0, tb_std_5=0.0, indoor=0):
    return _predict("hitter_home_runs", {
        "hits_avg_5": hits_avg_5,
        "tb_avg_5": tb_avg_5,
        "hr_avg_5": hr_avg_5,
        "rbi_avg_5": rbi_avg_5,
        "hits_std_5": hits_std_5,
        "tb_std_5": tb_std_5,
        "indoor": indoor,
    })


def hitter_rbis(hits_avg_5, tb_avg_5, hr_avg_5, rbi_avg_5,
                hits_std_5=0.0, tb_std_5=0.0, indoor=0):
    return _predict("hitter_rbis", {
        "hits_avg_5": hits_avg_5,
        "tb_avg_5": tb_avg_5,
        "hr_avg_5": hr_avg_5,
        "rbi_avg_5": rbi_avg_5,
        "hits_std_5": hits_std_5,
        "tb_std_5": tb_std_5,
        "indoor": indoor,
    })


def pitcher_strikeouts(k_avg_5, hits_allowed_avg_5, k_std_5=0.0, indoor=0):
    return _predict("pitcher_strikeouts", {
        "k_avg_5": k_avg_5,
        "k_std_5": k_std_5,
        "hits_allowed_avg_5": hits_allowed_avg_5,
        "indoor": indoor,
    })


def over_probability(projection: float, line: float, residual_std: float) -> int:
    """Probability the actual outcome exceeds the line, clipped to a sane range."""
    if residual_std <= 0:
        return 50
    z = (projection - line) / residual_std
    p = _norm_cdf(z)
    pct = int(round(max(p, 1 - p) * 100))
    return max(50, min(pct, 85))
