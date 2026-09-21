"""Probability calibration (Task 9).

Raw model probabilities (logistic sigmoid, RF/XGB mean leaves) are usually
miscalibrated in magnitude. This module provides two sklearn-free,
deterministic calibrators trained on a hold-out set of P(up) vs actual 0/1:

- Platt scaling  : sigmoid(a * logit(p) + b) fit by iterative reweighted
                   least squares (IRLS) — monotone, defined everywhere.
- Isotonic       : weighted pool-adjacent-violators (PAVA) — canonical
                   monotone regression onto observed outcome frequencies.
- `ece()`        : expected calibration error (binned |predicted - actual|),
                   used by tests and by monitor.
- `fit_calibrator()` returns an object with `.predict()` + `.to_dict()`.

Both transforms are monotone non-decreasing, keep outputs in [0, 1], and are
fit on a *separate* set than the model was trained on (caller's responsibility).
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _logit(p: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    positive = x >= 0
    out = np.empty_like(x)
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    ex = np.exp(x[~positive])
    out[~positive] = ex / (1.0 + ex)
    return out


def platt_fit(probs: np.ndarray, y: np.ndarray,
              iters: int = 200, tol: float = 1e-9) -> dict[str, float]:
    """IRLS logistic fit of sigmoid(a*logit(p)+b) against 0/1 labels."""
    p = np.asarray(probs, dtype=float)
    yv = np.asarray(y, dtype=float)
    design = np.column_stack([_logit(p), np.ones_like(p)])
    b = np.zeros(2)
    for _ in range(iters):
        lin = design @ b
        mu = _sigmoid(lin)
        mu = np.clip(mu, 1e-9, 1.0 - 1e-9)
        W = mu * (1.0 - mu)
        z = lin + (yv - mu) / np.clip(W, 1e-12, None)
        root_w = np.sqrt(W)
        b_new = np.linalg.lstsq(design * root_w[:, None],
                                z * root_w, rcond=None)[0]
        if np.allclose(b_new, b, atol=tol):
            b = b_new
            break
        b = b_new
    return {"method": "platt", "a": float(b[0]), "b": float(b[1])}


def platt_predict(probs: np.ndarray, params: dict[str, float]) -> np.ndarray:
    lin = params["a"] * _logit(probs) + params["b"]
    return np.clip(_sigmoid(lin), 0.0, 1.0)


def _isotonic_blocks(probs: np.ndarray, y: np.ndarray) -> tuple[list, list, list]:
    """PAVA isotonic blocks. Returns (steps_x, steps_w, steps_s) where
    ``steps_w`` are the per-block row supports and ``steps_s`` the weighted
    sums of y per block (so the output level of a block = s/w). Exposing the
    block support lets callers detect degeneracy/collapse (a block with tiny
    support has a high-variance empirical mean, and very few blocks means the
    map carries almost no information).
    """
    p = np.asarray(probs, dtype=float)
    yv = np.asarray(y, dtype=float)
    order = np.argsort(p, kind="mergesort")
    px, py = p[order], yv[order]
    steps_x: list[float] = [float(px[0])]
    steps_w: list[float] = [1.0]
    steps_s: list[float] = [float(py[0])]  # weighted sum of y in block
    for xi, yi in zip(px[1:], py[1:]):
        steps_x.append(float(xi))
        steps_w.append(1.0)
        steps_s.append(float(yi))
        while (len(steps_x) >= 2 and
               steps_s[-1] / steps_w[-1] < steps_s[-2] / steps_w[-2]):
            w = steps_w[-2] + steps_w[-1]
            steps_x[-2] = (steps_x[-2] * steps_w[-2] +
                           steps_x[-1] * steps_w[-1]) / w
            steps_s[-2] += steps_s[-1]
            steps_w[-2] = w
            del steps_x[-1], steps_s[-1], steps_w[-1]
    return steps_x, steps_w, steps_s


def isotonic_fit(probs: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """PAVA isotonic regression. Returns the monotone step function params.

    The returned dict also carries ``block_sizes`` (per-block row support) and
    ``n_blocks`` / ``n_distinct_levels`` so downstream code can apply the
    degeneracy diagnostics defined in this module. Existing callers (the
    ``Calibrator`` wrapper, tests) keep working: ``predict`` uses only
    ``steps_x``/``values`` and ``to_dict`` copies the params.
    """
    steps_x, steps_w, steps_s = _isotonic_blocks(probs, y)
    values = np.clip(np.asarray(steps_s) / np.asarray(steps_w), 0.0, 1.0)
    return {
        "method": "isotonic",
        "steps_x": steps_x,
        "values": values,
        "block_sizes": [int(w) for w in steps_w],
        "n_blocks": int(len(steps_x)),
        "n_distinct_levels": int(len(np.unique(np.round(values, 12)))),
    }


# ---------------------------------------------------------------------------
# Calibration-stability diagnostics (change A-D; Section 1-3 of the spec)
# ---------------------------------------------------------------------------

def isotonic_n_distinct_levels(params: dict[str, Any]) -> int:
    """Number of distinct output levels of a fitted isotonic step function.

    PAVA collapses input probs into constant blocks, each outputting its
    empirical label frequency. When the map has very few distinct levels the
    calibrated output is near-constant (carries little information) — the
    degeneracy that made the 88-long run snap all 88 holdout probs to ~0.5.
    """
    return int(len(np.unique(np.round(np.asarray(params.get("values", [])), 12))))


def isotonic_block_sizes(params: dict[str, Any]) -> list[int]:
    """Per-block row support of a fitted isotonic step function."""
    return list(params.get("block_sizes", []))


def brier_standard_error(probs: np.ndarray, y: np.ndarray) -> float:
    """Empirical standard error of the mean per-row Brier loss.

    Brier = mean((p_i - y_i)^2). The per-row loss (p_i - y_i)^2 is bounded in
    [0, 1], so SE = sample std of the losses / sqrt(n). This is the *empirical*
    non-constant value used by the 1SE model-selection tie-break (change A); it
    is computed here from actual per-row losses rather than a fixed constant.
    """
    p = np.asarray(probs, dtype=float)
    yv = np.asarray(y, dtype=float)
    keep = ~np.isnan(p) & ~np.isnan(yv)
    if keep.sum() == 0:
        return float("nan")
    loss = (p[keep] - yv[keep]) ** 2
    n = int(keep.sum())
    return float(np.std(loss, ddof=1) / np.sqrt(n))


def raw_prob_support(p: np.ndarray, near: float = 0.05) -> dict[str, float]:
    """Diagnostic of raw-probability support / collapse (change B).

    Near-chance raw probs cluster around 0.50; if most of the OOF raw probs
    are within ``near`` of 0.50 the calibration fit has very little spread to
    learn a non-trivial isotonic/Platt map — it is "collapsed". Reported (not
    hidden) so we never claim smoothing resolved it.

    ``near`` is a swept sensitivity parameter (default 0.05), never an
    arbitrary production constant on its own — the report returns the pct at
    several bands so the collapse is visible regardless of a single choice.
    """
    p = np.asarray(p, dtype=float)
    p = p[~np.isnan(p)]
    if len(p) == 0:
        return {"n": 0, "pct_near_0p02": float("nan"),
                "pct_near_0p05": float("nan"), "pct_near_0p10": float("nan"),
                "min": float("nan"), "max": float("nan"), "std": float("nan")}
    return {
        "n": int(len(p)),
        "pct_near_0p02": float(np.mean(np.abs(p - 0.5) <= 0.02)),
        "pct_near_0p05": float(np.mean(np.abs(p - 0.5) <= 0.05)),
        "pct_near_0p10": float(np.mean(np.abs(p - 0.5) <= 0.10)),
        "min": float(np.min(p)),
        "max": float(np.max(p)),
        "std": float(np.std(p)),
    }


def isotonic_predict(probs: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    steps_x = np.asarray(params["steps_x"], dtype=float)
    values = np.asarray(params["values"], dtype=float)
    idx = np.searchsorted(steps_x, np.asarray(probs, dtype=float),
                          side="right") - 1
    idx = np.clip(idx, 0, len(values) - 1)
    return np.asarray(values[idx], dtype=float)


class Calibrator:
    """Small wrapper exposing a uniform `.predict()` + `.to_dict()` API."""

    def __init__(self, params: dict[str, Any]):
        self.params = params

    def predict(self, probs: np.ndarray) -> np.ndarray:
        p = np.asarray(probs, dtype=float)
        out = np.full(len(p), np.nan)
        ok = ~np.isnan(p)
        if self.params["method"] == "platt":
            out[ok] = platt_predict(p[ok], self.params)
        else:
            out[ok] = isotonic_predict(p[ok], self.params)
        return out

    def to_dict(self) -> dict[str, Any]:
        return dict(self.params)


def fit_calibrator(probs: np.ndarray, y: np.ndarray,
                   method: str = "platt") -> Calibrator:
    """Fit `method` ("platt" | "isotonic") on (P(up), actual 0/1) pairs."""
    p = np.asarray(probs, dtype=float)
    yv = np.asarray(y, dtype=float)
    keep = ~np.isnan(p) & ~np.isnan(yv) & (yv.astype(int).astype(float) == yv)
    p, yv = p[keep], yv[keep].astype(int)
    if method == "platt":
        return Calibrator(platt_fit(p, yv))
    if method == "isotonic":
        return Calibrator(isotonic_fit(p, yv))
    raise ValueError(f"unknown calibration method: {method}")


def ece(probs: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error (binned, count-weighted)."""
    p = np.asarray(probs, dtype=float)
    yv = np.asarray(y, dtype=float)
    valid = ~np.isnan(p)
    p, yv = p[valid], yv[valid]
    if len(p) == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p <= hi) if hi == 1.0 else (p >= lo) & (p < hi)
        if m.sum() == 0:
            continue
        total += (m.sum() / len(p)) * abs(float(p[m].mean()) - float(yv[m].mean()))
    return float(total)