"""W11 — split-conformal uncertainty intervals around the forecast probability.

A point probability ("0.62 up") carries no statement about how much it can be
trusted. This module attaches an honest, distribution-free interval:

- **Method (split conformal):** nonconformity score ``s = |q - y|`` on a
  calibration slice the model never trained on (here: the pipeline's older
  80% calibration-fit slice); the interval is
  ``q ± Quantile_{ceil((n+1)*(1-alpha))/n}(s)`` — the *higher* conservative
  empirical quantile, so finite-sample coverage holds for EXCHANGEABLE rows.
- **Honesty rules (mirroring W1/W9/W10):**
  - Too few calibration rows (``MIN_N``) => ``status="insufficient_sample"``
    — a wide-but-fake interval from 3 rows is never shown.
  - NaN probs are dropped; if the query prob itself is NaN the interval is
    ``None`` with ``status="unavailable"`` — never fabricated.
  - The target is 20-day-horizon OOF rows, which OVERLAP (the horizon spans
    ~20 trading days while rows are daily), so the exchangeability needed
    for the finite-sample guarantee only holds for the *effective*
    independent windows. The report therefore includes
    ``effective_independent_windows`` and the coverage claim is explicitly
    conditional on it — we surface the caveat instead of hiding it.
  - ``warn_wide`` marks intervals wider than ``WIDE_INTERVAL`` (a purely
    uninformative band) — reported, never silently dropped.
- Deterministic: sorting is stable, quantile index is a pure function of n
  and alpha. Nothing here calls the network or changes any prediction.
"""

from __future__ import annotations

import numpy as np

SCHEMA_VERSION = "conformal-v1"

MIN_N = 30                 # below this an interval is noise, not knowledge
MIN_EVAL_ROWS = 10         # below this a coverage fraction is 0/1 noise
WIDE_INTERVAL = 0.5        # half-width >= 0.5 => band spans the whole [0,1]
DEFAULT_ALPHA = 0.10       # target ~90% coverage


def _clean(probs: np.ndarray) -> np.ndarray:
    p = np.asarray(probs, dtype=float)
    return p[~np.isnan(p)]


def _higher_quantile_index(n: int, alpha: float) -> int:
    """Rank (1-based) of the conservative empirical quantile: ceil((n+1)(1-a))/n."""
    k = int(np.ceil((n + 1) * (1.0 - float(alpha))))
    return min(max(k, 1), n)


def calibration_scores(cal_probs: np.ndarray, cal_y: np.ndarray) -> np.ndarray:
    """Absolute nonconformity scores |q - y| on the calibration slice."""
    q = np.asarray(cal_probs, dtype=float)
    y = np.asarray(cal_y, dtype=float)
    if len(q) != len(y):
        raise ValueError("cal_probs and cal_y length mismatch")
    keep = ~np.isnan(q) & ~np.isnan(y)
    return np.abs(q[keep] - y[keep])


def conformal_interval(
    prob: float | None,
    scores: np.ndarray,
    *,
    alpha: float = DEFAULT_ALPHA,
    min_n: int = MIN_N,
) -> dict:
    """Interval for ONE probability from precomputed calibration scores.

    Returns ``{"lower","upper","half_width"}`` (None when unavailable) plus
    the status/honesty metadata; pure and deterministic.
    """
    s = _clean(scores)
    out: dict = {
        "schema_version": SCHEMA_VERSION,
        "alpha": round(float(alpha), 4),
        "target_coverage": round(1.0 - float(alpha), 4),
        "n_calibration": int(len(s)),
        "min_n": int(min_n),
        "effective_independent_windows": None,
        "lower": None, "upper": None, "half_width": None,
        "is_wide": False,
        "status": "unavailable",
        "note": (
            "split-conformal band around the calibrated P(up); coverage is "
            "finite-sample valid for EXCHANGEABLE rows — the 20d-horizon OOF "
            "labels overlap, so treat the guarantee as approximate and check "
            "effective_independent_windows"
        ),
        "warning": None,
    }
    if prob is None or not np.isfinite(prob):
        out["warning"] = "probability unavailable — no interval fabricated"
        return out
    if len(s) < min_n:
        out["status"] = "insufficient_sample"
        out["warning"] = (
            f"only {len(s)} usable calibration score(s); below the {min_n} "
            "floor a conformal band would be noise — interval withheld"
        )
        return out
    q = float(np.clip(float(prob), 0.0, 1.0))
    idx = _higher_quantile_index(len(s), alpha)
    width = float(np.sort(s, kind="mergesort")[idx - 1])
    lower, upper = float(np.clip(q - width, 0.0, 1.0)), float(np.clip(q + width, 0.0, 1.0))
    out["lower"] = round(lower, 4)
    out["upper"] = round(upper, 4)
    out["half_width"] = round(width, 4)
    out["is_wide"] = bool(width >= WIDE_INTERVAL)
    out["status"] = "ok"
    if out["is_wide"]:
        out["warning"] = (
            f"interval half-width {width:.4f} >= {WIDE_INTERVAL:.2f} — the "
            "band spans (almost) the whole [0,1] range and carries no "
            "practical information"
        )
    return out


def interval_for_holdout_prob(
    prob: float | None,
    cal_probs: np.ndarray,
    cal_y: np.ndarray,
    *,
    alpha: float = DEFAULT_ALPHA,
    horizon: int = 20,
    min_n: int = MIN_N,
) -> dict:
    """End-to-end helper: scores from the calibration slice + one interval."""
    out = conformal_interval(prob, calibration_scores(cal_probs, cal_y),
                             alpha=alpha, min_n=min_n)
    out["effective_independent_windows"] = effective_independent_windows(
        out["n_calibration"], horizon
    )
    return out


def effective_independent_windows(n: int, horizon: int) -> int:
    """ceil(n / horizon): overlapping 20d labels shrink the effective sample."""
    try:
        h = int(horizon)
        n_v = int(n)
    except (TypeError, ValueError):
        return 0
    if h <= 0 or n_v <= 0:
        return 0
    return int(-(-n_v // h))


def empirical_coverage(
    probs: np.ndarray,
    y: np.ndarray,
    scores: np.ndarray,
    *,
    alpha: float = DEFAULT_ALPHA,
    min_n: int = MIN_N,
) -> float | None:
    """Fraction of holdout rows whose true label falls inside their band.

    A DIAGNOSTIC on rows the scores never saw (caller must pass a distinct
    slice, e.g. the newest 20% holdout). Two gates keep it honest: the score
    floor (``min_n``) and ``MIN_EVAL_ROWS`` on the evaluation rows themselves —
    a coverage fraction from 2 rows is 0% or 100% pure noise, so it is
    withheld (None) instead of reported. Overlap shrinks the effective n, so
    the number stays a sanity signal, never a guarantee.
    """
    p = np.asarray(probs, dtype=float)
    yy = np.asarray(y, dtype=float)
    s = _clean(scores)
    keep = ~np.isnan(p) & ~np.isnan(yy)
    p, yy = p[keep], yy[keep]
    if len(s) < min_n or len(p) < MIN_EVAL_ROWS:
        return None
    idx = _higher_quantile_index(len(s), alpha)
    width = float(np.sort(s, kind="mergesort")[idx - 1])
    inside = np.abs(p - yy) <= width
    return round(float(inside.mean()), 4)


def summarize(
    cal_probs: np.ndarray,
    cal_y: np.ndarray,
    hold_probs: np.ndarray,
    hold_y: np.ndarray,
    *,
    prob: float | None = None,
    alpha: float = DEFAULT_ALPHA,
    horizon: int = 20,
    min_n: int = MIN_N,
) -> dict:
    """One-call report for the pipeline: latest-prob interval + holdout
    empirical coverage. Never raises; missing pieces stay honestly None."""
    try:
        out = interval_for_holdout_prob(
            prob, cal_probs, cal_y, alpha=alpha, horizon=horizon, min_n=min_n
        )
        cov = empirical_coverage(
            hold_probs, hold_y, calibration_scores(cal_probs, cal_y),
            alpha=alpha, min_n=min_n,
        )
    except Exception as exc:  # noqa: BLE001 — degrade, never fabricate
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "unavailable",
            "warning": f"{type(exc).__name__}: conformal report failed",
        }
    out["holdout_empirical_coverage"] = cov
    if cov is None and out["status"] == "ok":
        out["warning"] = (
            "holdout empirical coverage unavailable (thin holdout) — band "
            "shown without a holdout sanity check"
        )
    return out
