"""Research-only point-in-time pooled panel builder (Task 13).

This module deliberately stops at a clean, versioned panel.  It does not
normalise features, fit a model, calibrate probabilities, or alter the live
forecast path.  The default input provider reuses the existing causal feature
assembly and target/fingerprint helpers; tests can inject a fully offline
provider.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np
import pandas as pd

from app.config import settings
from app.ml import dataset as ds
from app.ml import fingerprint as fp
from app.utils.validators import clean_symbol

logger = logging.getLogger("equitylens.pooled_dataset")

MANIFEST_VERSION = "universe-manifest-v1"
DATASET_VERSION = "pooled-dataset-v1"
SCHEMA_VERSION = "pooled-panel-v1"

DEFAULT_RESEARCH_UNIVERSE = (
    "HDFCBANK",
    "BAJFINANCE",
    "TCS",
    "INFY",
    "RELIANCE",
    "ICICIBANK",
    "SBIN",
    "LT",
    "ITC",
    "SUNPHARMA",
    "AXISBANK",
    "MARUTI",
    "TATASTEEL",
    "BHARTIARTL",
    "ASIANPAINT",
    "WIPRO",
    "KOTAKBANK",
    "HINDUNILVR",
    "TITAN",
    "ADANIENT",
)
SMOKE_UNIVERSE = ("TCS", "RELIANCE", "HDFCBANK", "INFY")

Provider = Callable[[str, str, bool], Mapping[str, Any]]


class PooledDatasetError(ValueError):
    """Invalid research input (duplicate keys, dates, or malformed schema)."""


def _date(value: Any, *, name: str) -> pd.Timestamp | None:
    if value is None or value == "":
        return None
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        raise PooledDatasetError(f"{name} must be a valid date")
    if stamp.tzinfo is not None:
        stamp = stamp.tz_localize(None)
    return stamp.normalize()


def _iso(value: Any) -> str | None:
    stamp = _date(value, name="date")
    return stamp.date().isoformat() if stamp is not None else None


def _canonical_member(raw: Any) -> dict[str, Any]:
    if isinstance(raw, str):
        symbol = clean_symbol(raw)
        item: dict[str, Any] = {"symbol": symbol}
    elif isinstance(raw, Mapping):
        if not raw.get("symbol"):
            raise PooledDatasetError("universe member is missing symbol")
        symbol = clean_symbol(str(raw["symbol"]))
        item = {"symbol": symbol}
        for key in ("sector", "industry", "exchange", "source"):
            if raw.get(key) is not None:
                item[key] = str(raw[key])
        # Accept common source names, but persist one unambiguous PIT contract.
        item["valid_from"] = _iso(raw.get(
            "valid_from",
            raw.get("first_available_date",
                    raw.get("listing_date", raw.get("inception_date"))),
        ))
        item["valid_to"] = _iso(raw.get(
            "valid_to",
            raw.get("last_available_date",
                    raw.get("delisted_date", raw.get("delisting_date"))),
        ))
        item["inclusion_status"] = str(raw.get("inclusion_status", "included"))
        item["exclusion_reason"] = (
            str(raw["exclusion_reason"])
            if raw.get("exclusion_reason") is not None
            else None
        )
    else:
        raise PooledDatasetError("universe members must be strings or mappings")

    item.setdefault("valid_from", None)
    item.setdefault("valid_to", None)
    item.setdefault("inclusion_status", "included")
    item.setdefault("exclusion_reason", None)
    item["first_available_date"] = item["valid_from"]
    item["last_available_date"] = item["valid_to"]
    start = _date(item["valid_from"], name="valid_from")
    end = _date(item["valid_to"], name="valid_to")
    if start is not None and end is not None and start > end:
        raise PooledDatasetError(f"{symbol}: valid_from is after valid_to")
    return item


def build_universe_manifest(
    symbols: Iterable[str | Mapping[str, Any]] | None = None,
    *,
    members: Iterable[str | Mapping[str, Any]] | None = None,
    as_of: Any = None,
    manifest_version: str = MANIFEST_VERSION,
    source: str = "research-input",
) -> dict[str, Any]:
    """Build a canonical, versioned universe manifest.

    Members are sorted by symbol and duplicate symbols are rejected rather than
    silently changing panel weights. ``valid_from``/``valid_to`` are applied to
    every historical row, not just to the manifest creation date.
    """
    if symbols is not None and members is not None:
        raise PooledDatasetError("pass either symbols or members, not both")
    raw_members = members if members is not None else symbols
    if raw_members is None:
        raw_members = DEFAULT_RESEARCH_UNIVERSE
    canonical = [_canonical_member(item) for item in raw_members]
    seen: set[str] = set()
    for item in canonical:
        if item["symbol"] in seen:
            raise PooledDatasetError(f"duplicate universe symbol: {item['symbol']}")
        seen.add(item["symbol"])
    canonical.sort(key=lambda item: item["symbol"])
    manifest_as_of = _iso(as_of)
    if manifest_as_of is not None:
        cutoff = _date(manifest_as_of, name="as_of")
        for item in canonical:
            if not _eligible(item, cutoff):
                item["inclusion_status"] = "excluded"
                item["exclusion_reason"] = "outside validity window at as_of"
    body: dict[str, Any] = {
        "manifest_version": str(manifest_version),
        "schema_version": MANIFEST_VERSION,
        "source": str(source),
        "as_of": manifest_as_of,
        "members": canonical,
        "symbols": [item["symbol"] for item in canonical],
        "n_symbols": len(canonical),
    }
    body["manifest_fingerprint"] = fp.fingerprint_object(
        "universe-manifest", body
    )
    return body


def manifest_fingerprint(manifest: Mapping[str, Any]) -> str:
    """Fingerprint manifest content while ignoring its stored digest field."""
    body = {key: value for key, value in manifest.items()
            if key != "manifest_fingerprint"}
    return fp.fingerprint_object("universe-manifest", body)


def _member_by_symbol(manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    members = manifest.get("members")
    if not isinstance(members, list):
        raise PooledDatasetError("manifest.members must be a list")
    out: dict[str, Mapping[str, Any]] = {}
    for raw in members:
        member = _canonical_member(raw)
        symbol = member["symbol"]
        if symbol in out:
            raise PooledDatasetError(f"duplicate universe symbol: {symbol}")
        out[symbol] = member
    expected = sorted(out)
    if list(manifest.get("symbols", expected)) != expected:
        raise PooledDatasetError("manifest symbols are not canonical/sorted")
    return out


def _normalise_frame(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise PooledDatasetError(f"{symbol}: empty feature frame")
    out = frame.copy()
    if "date" in out.columns and not isinstance(out.index, pd.DatetimeIndex):
        out = out.set_index("date")
    if "date" in out.columns:
        out = out.drop(columns=["date"])
    if "symbol" in out.columns:
        out = out.drop(columns=["symbol"])
    try:
        index = pd.DatetimeIndex(out.index)
    except Exception as exc:
        raise PooledDatasetError(f"{symbol}: index is not date-like") from exc
    if index.tz is not None:
        index = index.tz_localize(None)
    index = index.normalize()
    if index.has_duplicates:
        raise PooledDatasetError(f"{symbol}: duplicate dates in feature frame")
    out.index = index
    out = out.sort_index()
    if "Close" not in out.columns:
        raise PooledDatasetError(f"{symbol}: feature frame has no Close column")
    if not pd.api.types.is_numeric_dtype(out["Close"]):
        raise PooledDatasetError(f"{symbol}: Close column is not numeric")
    return out


def _eligible(member: Mapping[str, Any], stamp: pd.Timestamp) -> bool:
    start = _date(member.get("valid_from"), name="valid_from")
    end = _date(member.get("valid_to"), name="valid_to")
    return (start is None or stamp >= start) and (end is None or stamp <= end)


def _feature_columns(frames: Mapping[str, pd.DataFrame]) -> list[str]:
    preferred = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    names = {
        column for frame in frames.values() for column in frame.columns
        if column not in {"symbol", "date"} and not str(column).startswith("target_")
    }
    return [column for column in preferred if column in names] + sorted(
        names.difference(preferred), key=str
    )


def _missingness(frame: pd.DataFrame, feature_columns: list[str]) -> dict[str, Any]:
    total = len(frame)
    by_feature = {}
    for column in feature_columns:
        missing = int(frame[column].isna().sum()) if column in frame else total
        by_feature[column] = {
            "missing": missing,
            "total": total,
            "rate": round(missing / total, 6) if total else None,
        }
    return by_feature


def build_pooled_dataset(
    frames: Mapping[str, pd.DataFrame],
    manifest: Mapping[str, Any] | None = None,
    *,
    horizon: int = ds.DEFAULT_HORIZON,
    as_of: Any = None,
    drop_incomplete_target: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Pool pre-built causal feature frames into a deterministic panel.

    The returned report is deterministic and contains missingness, coverage,
    and target availability.  Missing source columns remain NaN; no imputation
    or normalization occurs.  ``as_of`` is an inclusive information cutoff.
    """
    if not isinstance(frames, Mapping):
        raise PooledDatasetError("frames must be a symbol -> DataFrame mapping")
    manifest_obj = dict(manifest or build_universe_manifest(frames.keys()))
    members = _member_by_symbol(manifest_obj)
    canonical_inputs = [clean_symbol(symbol) for symbol in frames]
    if len(canonical_inputs) != len(set(canonical_inputs)):
        raise PooledDatasetError("duplicate input symbol after canonicalization")
    cutoff = _date(as_of if as_of is not None else manifest_obj.get("as_of"),
                   name="as_of")
    prepared: dict[str, pd.DataFrame] = {}
    skipped: list[dict[str, str]] = []
    for raw_symbol, frame in sorted(frames.items(), key=lambda pair: clean_symbol(pair[0])):
        symbol = clean_symbol(raw_symbol)
        if symbol not in members:
            skipped.append({"symbol": symbol, "reason": "not in universe manifest"})
            continue
        if members[symbol].get("inclusion_status") == "excluded":
            skipped.append({
                "symbol": symbol,
                "reason": str(members[symbol].get("exclusion_reason") or "excluded"),
            })
            continue
        try:
            work = _normalise_frame(frame, symbol)
            if cutoff is not None:
                work = work.loc[work.index <= cutoff]
            work = work.loc[[_eligible(members[symbol], stamp) for stamp in work.index]]
            if work.empty:
                raise PooledDatasetError("no rows inside point-in-time universe window")
            work = ds.add_target(work, horizon=horizon)
            work.insert(0, "symbol", symbol)
            prepared[symbol] = work
        except Exception as exc:  # deterministic per-symbol quality report
            skipped.append({"symbol": symbol, "reason": str(exc)})
    for symbol in sorted(members):
        if symbol not in prepared and not any(
            item["symbol"] == symbol for item in skipped
        ):
            skipped.append({"symbol": symbol, "reason": "no feature frame supplied"})

    feature_columns = _feature_columns(prepared)
    target_columns = [f"target_ret_{horizon}d", f"target_up_{horizon}d"]
    columns = ["symbol", "date"] + feature_columns + target_columns
    rows: list[pd.DataFrame] = []
    for symbol in sorted(prepared):
        work = prepared[symbol].copy()
        work.insert(1, "date", work.index)
        for column in feature_columns + target_columns:
            if column not in work:
                work[column] = np.nan
        rows.append(work[columns])
    panel = (pd.concat(rows, ignore_index=True)
             if rows else pd.DataFrame(columns=columns))
    if not panel.empty:
        panel = panel.sort_values(["date", "symbol"], kind="mergesort").reset_index(drop=True)
    if drop_incomplete_target and target_columns:
        panel = panel.loc[panel[target_columns[0]].notna()].reset_index(drop=True)

    coverage: dict[str, Any] = {}
    for symbol in sorted(prepared):
        part = panel.loc[panel["symbol"] == symbol]
        coverage[symbol] = {
            "rows": int(len(part)),
            "date_start": part["date"].min().date().isoformat() if len(part) else None,
            "date_end": part["date"].max().date().isoformat() if len(part) else None,
            "target_rows": int(part[target_columns[0]].notna().sum()) if len(part) else 0,
        }
    report = {
        "schema_version": SCHEMA_VERSION,
        "dataset_version": DATASET_VERSION,
        "horizon": int(horizon),
        "as_of": cutoff.date().isoformat() if cutoff is not None else None,
        "symbols_requested": int(len(members)),
        "symbols_ok": int(len(prepared)),
        "symbols_skipped": int(len(skipped)),
        "rows": int(len(panel)),
        "columns": columns,
        "feature_columns": feature_columns,
        "target_columns": target_columns,
        "coverage": coverage,
        "missingness": _missingness(panel, feature_columns + target_columns),
        "missingness_by_symbol": {
            symbol: _missingness(
                panel.loc[panel["symbol"] == symbol],
                feature_columns + target_columns,
            )
            for symbol in sorted(prepared)
        },
        "skipped": sorted(skipped, key=lambda item: item["symbol"]),
        "normalization": "none",
        "model_training": "not performed",
        "point_in_time": {
            "cutoff_inclusive": True,
            "universe_validity_windows_applied": True,
            "future_rows_excluded_before_target": True,
        },
    }
    panel.attrs["pooled_dataset_version"] = DATASET_VERSION
    panel.attrs["schema_version"] = SCHEMA_VERSION
    return panel, report


def _default_provider(symbol: str, period: str, enrich: bool) -> Mapping[str, Any]:
    from app.ml import pipeline
    return pipeline._build_symbol_input(symbol, period=period, enrich=enrich)


def _provider_frame(payload: Mapping[str, Any], symbol: str) -> pd.DataFrame:
    frame = payload.get("features", payload.get("frame"))
    if isinstance(frame, pd.DataFrame):
        return frame
    ohlcv = payload.get("ohlcv")
    if isinstance(ohlcv, pd.DataFrame):
        from app.ml import pipeline
        return pipeline.build_feature_frame(ohlcv, market=payload.get("market"))
    raise PooledDatasetError(f"{symbol}: provider returned no feature frame")


def run_pooled_dataset(
    symbols: Iterable[str | Mapping[str, Any]] | None = None,
    *,
    manifest: Mapping[str, Any] | None = None,
    provider: Provider | None = None,
    period: str = "5y",
    enrich: bool = False,
    horizon: int = settings.ml_horizon,
    as_of: Any = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Fetch/build each member independently and return panel + report metadata."""
    manifest_obj = dict(manifest or build_universe_manifest(symbols, as_of=as_of))
    members = _member_by_symbol(manifest_obj)
    fetch = provider or _default_provider
    frames: dict[str, pd.DataFrame] = {}
    failures: list[dict[str, str]] = []
    source_fingerprints: dict[str, str] = {}
    for symbol in sorted(members):
        try:
            payload = fetch(symbol, period=period, enrich=enrich)
            if isinstance(payload, pd.DataFrame):
                payload = {"ok": True, "features": payload}
            if not payload.get("ok", True):
                raise PooledDatasetError(str(payload.get("error") or "provider unavailable"))
            frames[symbol] = _provider_frame(payload, symbol)
        except Exception as exc:  # one bad symbol never aborts the panel
            failures.append({"symbol": symbol, "reason": f"{type(exc).__name__}: {exc}"})
    panel, report = build_pooled_dataset(
        frames, manifest_obj, horizon=horizon, as_of=as_of
    )
    skipped_by_symbol = {item["symbol"]: item for item in report["skipped"]}
    skipped_by_symbol.update({item["symbol"]: item for item in failures})
    report["skipped"] = sorted(skipped_by_symbol.values(), key=lambda item: item["symbol"])
    report["symbols_skipped"] = len(report["skipped"])
    # Fingerprint the rows actually admitted to the PIT panel.  In particular,
    # a provider's full-history fingerprint must not make an as-of run depend
    # on data dated after its cutoff.
    for symbol in sorted(frames):
        admitted = panel.loc[panel["symbol"] == symbol].reset_index(drop=True)
        if not admitted.empty:
            source_fingerprints[symbol] = fp.fingerprint_object(
                f"admitted-source:{symbol}", admitted
            )
    components = {
        "manifest": manifest_fingerprint(manifest_obj),
        "panel": fp.fingerprint_object("pooled-panel", panel),
        "sources": source_fingerprints,
        "config": {
            "period": period, "enrich": bool(enrich), "horizon": int(horizon),
            "as_of": report["as_of"], "seed": int(seed),
        },
    }
    dataset_fingerprint = fp.fingerprint_object("pooled-dataset-components", components)
    metadata = {
        "dataset_version": DATASET_VERSION,
        "schema_version": SCHEMA_VERSION,
        "manifest_version": manifest_obj["manifest_version"],
        "manifest_fingerprint": manifest_fingerprint(manifest_obj),
        "dataset_fingerprint": dataset_fingerprint,
        "fingerprint": {
            "algorithm": "sha256",
            "scope": "manifest + source inputs + canonical pooled panel + config",
            "reconstruction": "identification only; raw inputs are not persisted",
        },
        "versions": {
            "feature": settings.ml_feature_version,
            "model": settings.ml_model_version,
            "dataset": DATASET_VERSION,
        },
        "period": period,
        "enrich": bool(enrich),
        "horizon": int(horizon),
        "seed": int(seed),
        "as_of": report["as_of"],
        "no_production_integration": True,
    }
    return {
        "metadata": metadata,
        "manifest": manifest_obj,
        "report": report,
        "dataset": panel,
    }


def save_pooled_dataset(
    result: Mapping[str, Any], output_dir: str = "ml_pooled_dataset", label: str = "panel"
) -> dict[str, str]:
    """Persist the research panel and JSON report; paths are intentionally local."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"pooled_{label}"
    csv_path = directory / f"{stem}.csv"
    json_path = directory / f"{stem}.json"
    panel = result["dataset"]
    if not isinstance(panel, pd.DataFrame):
        raise PooledDatasetError("result.dataset must be a DataFrame")
    panel.to_csv(csv_path, index=False)
    serializable = {key: value for key, value in result.items() if key != "dataset"}
    serializable["dataset"] = {"rows": int(len(panel)), "columns": list(panel.columns)}
    # Keep JSON deterministic; no wall-clock timestamp is part of the report.
    json_path.write_text(
        json.dumps(serializable, indent=2, sort_keys=True), encoding="utf-8"
    )
    return {"csv": str(csv_path), "json": str(json_path)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the research-only PIT pooled panel")
    parser.add_argument("--symbols", help="comma-separated symbols")
    parser.add_argument("--smoke", action="store_true", help="use the four-symbol smoke universe")
    parser.add_argument("--period", default="5y")
    parser.add_argument("--horizon", type=int, default=settings.ml_horizon)
    parser.add_argument("--as-of", dest="as_of")
    parser.add_argument("--output-dir", default="ml_pooled_dataset")
    args = parser.parse_args(argv)
    if args.symbols:
        symbols = [clean_symbol(item) for item in args.symbols.split(",") if item.strip()]
    elif args.smoke:
        symbols = list(SMOKE_UNIVERSE)
    else:
        symbols = list(DEFAULT_RESEARCH_UNIVERSE)
    result = run_pooled_dataset(
        symbols, period=args.period, horizon=args.horizon, as_of=args.as_of
    )
    paths = save_pooled_dataset(result, output_dir=args.output_dir)
    print(json.dumps({
        "dataset_fingerprint": result["metadata"]["dataset_fingerprint"],
        "rows": result["report"]["rows"],
        "symbols_ok": result["report"]["symbols_ok"],
        "symbols_skipped": result["report"]["symbols_skipped"],
        "csv": paths["csv"],
        "json": paths["json"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
