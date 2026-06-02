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

import os
MODELS_DIR = Path(os.getenv("MODELS_DIR") or Path(__file__).resolve().parent.parent / "models")

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


def _poisson_tail(lam: float, k: int) -> float:
    """P(X >= k) for a Poisson(lam) count."""
    lam = max(1e-9, float(lam))
    cum = 0.0
    term = math.exp(-lam)  # P(X = 0)
    for i in range(0, k):
        cum += term
        term *= lam / (i + 1)
    return max(0.0, 1.0 - cum)


def _nb_tail(mu: float, k: int, dispersion: float = 0.6) -> float:
    """P(X >= k) for an over-dispersed count modeled as negative-binomial
    with mean `mu` and variance `mu + dispersion*mu^2`.

    Used for total bases: a single home run adds 4 bases, so the per-game
    distribution has a fatter upper tail than a plain Poisson of the same
    mean. Dispersion 0 collapses back to Poisson.
    """
    mu = max(1e-9, float(mu))
    if dispersion <= 0:
        return _poisson_tail(mu, k)
    r = 1.0 / dispersion              # NB "size" parameter
    ratio = mu / (r + mu)
    p_i = (r / (r + mu)) ** r         # P(X = 0)
    cum = 0.0
    for i in range(0, k):
        cum += p_i
        # Recurrence: P(i+1) = P(i) * (r + i)/(i + 1) * mu/(r + mu)
        p_i *= (r + i) / (i + 1) * ratio
    return max(0.0, 1.0 - cum)


def count_prob_over(projection: float, line: float, stat_type: str) -> float:
    """Probability the OVER hits for a count stat at an X.5 line.

    OVER X.5 means the integer count must be >= floor(line)+1 (Hits/HR/RBI
    at 0.5 need >=1; Total Bases at 1.5 needs >=2). A Poisson / negative-
    binomial tail on the projected mean is far better calibrated near a .5
    line than a normal CDF or an edge-bucket table, both of which treated a
    low-count integer outcome as continuous and ran ~7pp overconfident
    (model said 74%, reality 67%).

    - hits / home_runs / rbis: Poisson tail (per-game counts ~Poisson near 0)
    - total_bases: lightly over-dispersed negative-binomial (HR adds 4 bases)
    """
    k = int(math.floor(float(line))) + 1
    mu = max(1e-9, float(projection))
    if stat_type == "total_bases":
        p = _nb_tail(mu, k, dispersion=0.6)
    else:
        p = _poisson_tail(mu, k)
    return max(0.02, min(0.97, p))
