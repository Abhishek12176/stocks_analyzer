"""Deterministic SHA-256 fingerprinting of forecast input snapshots.

Reproducibility layer: every successful forecast is tagged with a SHA-256
fingerprint of the EXACT upstream inputs the forecast path actually consumed
(OHLCV records, market-index series, fundamentals snapshot). The fingerprint is
a pure function of the data content only (dict key order is irrelevant, NaN and
-0.0 are normalised), so:

- the same input snapshot always yields the same fingerprint;
- any change to any consumed value yields a different fingerprint;
- the persistent forecast cache can therefore refuse to serve a previously
  cached result when the current upstream snapshot differs from the one the
  cached result was built from (no more silently reusing a different data
  snapshot as if it were the same result).

Scope note: the fingerprint IDENTIFIES the exact input snapshot; it does NOT
persist the inputs, so offline reconstruction of the full run from the
fingerprint alone is not supported. This is identification, not replay.
"""

from __future__ import annotations

import hashlib
import math
from datetime import date, datetime  # noqa: F401  (re-exported for callers)
from typing import Any

import numpy as np
import pandas as pd

_NAG = type(pd.NaT)


def sha256_hex(data: bytes) -> str:
    """SHA-256 hex digest of raw bytes."""
    return hashlib.sha256(data).hexdigest()


def _canon_scalar(value: Any) -> str:
    """Canonical string for one scalar value (deterministic and lossless)."""
    if value is None:
        return "null"
    if isinstance(value, _NAG):
        return "NaT"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, np.bool_):
        return "true" if bool(value) else "false"
    if isinstance(value, np.integer):
        return str(int(value))
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "+inf" if value > 0 else "-inf"
        if value == 0.0:  # normalise -0.0
            value = 0.0
        return repr(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, str):
        return value
    return repr(value)


def _frame_bytes(frame: pd.DataFrame) -> bytes:
    """Deterministic bytes for a DataFrame (index labels + column dtype +
    full column values). Float columns are serialised as raw IEEE-754 bytes so
    even 1-ULP differences change the digest."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"expected pd.DataFrame, got {type(frame).__name__}")
    parts: list[bytes] = []
    parts.append(("idx=" + "|".join(_canon_scalar(i) for i in frame.index)).encode("utf-8"))
    for col in frame.columns:
        series = frame[col]
        values = series.to_numpy()
        prefix = (
            "col=" + _canon_scalar(col)
            + ";dtype=" + str(series.dtype)
            + ";"
        ).encode("utf-8")
        if values.dtype.kind in "bifu":
            try:
                raw = np.ascontiguousarray(values, dtype=values.dtype).tobytes()
            except (TypeError, ValueError):
                raw = ("text:" + "|".join(_canon_scalar(x) for x in values)).encode("utf-8")
            parts.append(prefix)
            parts.append(raw)
        else:
            parts.append(prefix + (">" + "|".join(_canon_scalar(x) for x in values)).encode("utf-8"))
    return b"\x1f".join(parts)


def canonical_bytes(obj: Any) -> bytes:
    """Deterministic bytes for any JSON-ish / pandas object."""
    if isinstance(obj, pd.DataFrame):
        return b"D" + _frame_bytes(obj)
    if isinstance(obj, dict):
        chunks: list[bytes] = []
        for key in sorted(obj, key=_canon_scalar):
            chunks.append(b"K" + _canon_scalar(key).encode("utf-8")
                          + b"V" + canonical_bytes(obj[key]))
        return b"{" + b"\n".join(chunks) + b"}"
    if isinstance(obj, (list, tuple)):
        return b"[" + b",".join(canonical_bytes(x) for x in obj) + b"]"
    return b"S" + _canon_scalar(obj).encode("utf-8")


def fingerprint_object(name: str, obj: Any) -> str:
    """SHA-256 hex of one named object (name is inlined, so two objects of
    different kinds never collide on the same digest)."""
    return sha256_hex(
        ("obj=" + str(name)).encode("utf-8") + b"\x00" + canonical_bytes(obj)
    )


def fingerprint_dataframe(name: str, frame: pd.DataFrame | None) -> str:
    """Fingerprint a DataFrame; a missing frame is a distinct, stable state."""
    if frame is None:
        return sha256_hex(("obj=" + str(name) + b"").decode("utf-8").encode() + b"@absent")
    return fingerprint_object(name, frame)


def compose_fingerprint(parts: dict[str, str], name: str = "input-snapshot") -> str:
    """Combine component digests into one snapshot fingerprint. Order of the
    `parts` dict is irrelevant (sorted); names are inlined."""
    lines = [f"{name}:{k}={v}" for k, v in sorted(parts.items())]
    return sha256_hex(("\n".join(lines)).encode("utf-8"))


def feature_frame_fingerprint(frame: pd.DataFrame) -> str:
    """Fingerprint of a built feature frame (used when only the frame is
    available, i.e. the pure `forecast_frame` path)."""
    return fingerprint_object("feature_frame", frame)