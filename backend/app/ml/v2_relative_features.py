"""Research-only relative and cross-sectional features for V2 experiments.

This module is intentionally not imported by the production V1 pipeline.  It
operates on a long panel with one row per ``(symbol, date)`` and returns the
same panel, sorted deterministically by date and symbol.  All features use
only closes and labels available on the row's date; no model is fitted here.

The public entry point is :func:`build_relative_features`.  ``add_relative_features``
is retained as a readable alias for experiment notebooks.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from app.ml.fingerprint import fingerprint_dataframe
from app.ml.market_features import excess_return, relative_strength_ratio

FEATURE_VERSION = "v2-relative-research-1"
RELATIVE_FEATURE_VERSION = FEATURE_VERSION
SCHEMA_VERSION = "v2-relative-panel-1"
MANIFEST_VERSION = "v2-relative-manifest-1"
DEFAULT_HORIZONS: tuple[int, ...] = (1, 5, 20)
DEFAULT_MIN_GROUP_SIZE = 3


def _feature_names(horizons: Iterable[int]) -> list[str]:
    names: list[str] = []
    for h in horizons:
        names.extend(
            [
                f"market_excess_return_{h}d",
                f"market_relative_strength_{h}d",
                f"sector_return_{h}d",
                f"sector_excess_return_{h}d",
                f"peer_median_divergence_{h}d",
                f"cross_sectional_return_rank_{h}d",
                f"sector_return_rank_{h}d",
                f"peer_return_rank_{h}d",
            ]
        )
    return sorted(names)


def feature_manifest(horizons: Iterable[int] = DEFAULT_HORIZONS) -> dict[str, Any]:
    """Return a copy of the explicit research feature contract."""
    hs = tuple(sorted({int(h) for h in horizons if int(h) > 0}))
    if not hs:
        raise ValueError("horizons must contain a positive integer")
    features: dict[str, dict[str, Any]] = {}
    for h in hs:
        features.update(
            {
                f"market_excess_return_{h}d": {
                    "family": "market-relative",
                    "formula": "R_stock(T,T-h) - R_market(T,T-h)",
                    "source": "Close + market_close",
                    "min_group_size": None,
                },
                f"market_relative_strength_{h}d": {
                    "family": "market-relative",
                    "formula": "(1+R_stock)/(1+R_market)-1",
                    "source": "Close + market_close",
                    "min_group_size": None,
                },
                f"sector_return_{h}d": {
                    "family": "sector-relative",
                    "formula": "mean of eligible sector trailing returns",
                    "source": "Close + point-in-time sector",
                    "min_group_size": DEFAULT_MIN_GROUP_SIZE,
                },
                f"sector_excess_return_{h}d": {
                    "family": "sector-relative",
                    "formula": "R_stock - mean(sector R)",
                    "source": "Close + point-in-time sector",
                    "min_group_size": DEFAULT_MIN_GROUP_SIZE,
                },
                f"peer_median_divergence_{h}d": {
                    "family": "peer-relative",
                    "formula": "R_stock - median(R_peers), excluding self",
                    "source": "Close + point-in-time peer group",
                    "min_group_size": DEFAULT_MIN_GROUP_SIZE,
                },
                f"cross_sectional_return_rank_{h}d": {
                    "family": "cross-sectional",
                    "formula": "(1 + #less + 0.5*#ties) / n",
                    "source": "same-date trailing returns",
                    "min_group_size": DEFAULT_MIN_GROUP_SIZE,
                },
                f"sector_return_rank_{h}d": {
                    "family": "cross-sectional",
                    "formula": "(1 + #less + 0.5*#ties) / n_sector",
                    "source": "same-date trailing returns + sector",
                    "min_group_size": DEFAULT_MIN_GROUP_SIZE,
                },
                f"peer_return_rank_{h}d": {
                    "family": "cross-sectional",
                    "formula": "(1 + #less + 0.5*#ties) / (n_peers + 1)",
                    "source": "same-date trailing returns, excluding self",
                    "min_group_size": DEFAULT_MIN_GROUP_SIZE,
                },
            }
        )
    return {
        "feature_version": FEATURE_VERSION,
        "schema_version": SCHEMA_VERSION,
        "manifest_version": MANIFEST_VERSION,
        "horizons": list(hs),
        "features": {key: features[key] for key in sorted(features)},
        "tie_method": "average-midrank",
        "missing_policy": "NaN; no fill, backfill, or fabricated benchmark",
        "holiday_policy": "absent dates are not reindexed; windows count observations",
        "pit_guarantee": "a row at T reads closes and mappings available no later than T",
        "mapping_semantics": (
            "sector/peer labels are point-in-time; valid_from/valid_to are "
            "inclusive; out-of-window manifest rows remain but are ineligible"
        ),
    }


def get_feature_manifest(horizons: Iterable[int] = DEFAULT_HORIZONS) -> dict[str, Any]:
    """Compatibility/readability alias for :func:`feature_manifest`."""
    return feature_manifest(horizons)


def list_relative_features(horizons: Iterable[int] = DEFAULT_HORIZONS) -> list[str]:
    """Return the deterministic sorted output feature names."""
    return _feature_names(_normalise_horizons(horizons))


def _normalise_horizons(horizons: Iterable[int]) -> tuple[int, ...]:
    values = tuple(sorted({int(h) for h in horizons}))
    if not values or any(h <= 0 for h in values):
        raise ValueError("horizons must contain positive integers")
    return values


def _normalise_panel(
    panel: pd.DataFrame,
    *,
    symbol_col: str,
    date_col: str,
    close_col: str,
) -> pd.DataFrame:
    if not isinstance(panel, pd.DataFrame):
        raise TypeError("panel must be a pandas DataFrame")
    out = panel.copy()
    if isinstance(out.index, pd.MultiIndex) and symbol_col not in out.columns:
        names = list(out.index.names)
        if symbol_col in names and date_col in names:
            out = out.reset_index()
    if date_col not in out.columns and isinstance(out.index, pd.DatetimeIndex):
        out[date_col] = out.index
    required = {symbol_col, date_col, close_col}
    missing = required.difference(out.columns)
    if missing:
        raise ValueError(f"panel missing required columns: {sorted(missing)}")
    dates = pd.to_datetime(out[date_col], errors="coerce")
    if dates.isna().any():
        raise ValueError("date contains invalid values")
    if getattr(dates.dt, "tz", None) is not None:
        dates = dates.dt.tz_localize(None)
    out[date_col] = dates.dt.normalize()
    if out[[symbol_col, date_col]].duplicated().any():
        raise ValueError("panel must contain one row per (symbol, date)")
    out[symbol_col] = out[symbol_col].astype(str)
    out[close_col] = pd.to_numeric(out[close_col], errors="coerce")
    return out.sort_values([date_col, symbol_col], kind="mergesort").reset_index(drop=True)


def _mapping_frame(
    mapping: Any,
    *,
    symbol_col: str,
    date_col: str,
    sector_col: str,
) -> pd.DataFrame | None:
    """Canonicalise static or PIT mapping input.

    A mapping DataFrame may use ``valid_from``/``valid_to`` (inclusive).  A
    simple dictionary is deliberately static and is documented as such.
    """
    if mapping is None:
        return None
    if isinstance(mapping, Mapping):
        rows = [{symbol_col: str(k), sector_col: v} for k, v in mapping.items()]
        return pd.DataFrame(rows)
    if not isinstance(mapping, pd.DataFrame):
        raise TypeError("sector_mapping must be a dict or DataFrame")
    m = mapping.copy()
    if symbol_col not in m.columns:
        raise ValueError(f"sector_mapping missing {symbol_col!r}")
    if sector_col not in m.columns and "sector" in m.columns:
        m = m.rename(columns={"sector": sector_col})
    if sector_col not in m.columns:
        raise ValueError(f"sector_mapping missing {sector_col!r}")
    for col in ("valid_from", "valid_to"):
        if col in m.columns:
            m[col] = pd.to_datetime(m[col], errors="coerce").dt.normalize()
    if date_col in m.columns:
        m[date_col] = pd.to_datetime(m[date_col], errors="coerce").dt.normalize()
    return m


def _apply_sector_mapping(
    out: pd.DataFrame,
    mapping: Any,
    *,
    symbol_col: str,
    date_col: str,
    sector_col: str,
) -> pd.DataFrame:
    m = _mapping_frame(mapping, symbol_col=symbol_col, date_col=date_col, sector_col=sector_col)
    if m is None:
        return out
    result = out.copy()
    if date_col in m.columns:
        # Exact dated mapping is useful for manifests that already contain a
        # PIT row per date; it is not silently treated as a future mapping.
        result = result.drop(columns=[sector_col], errors="ignore").merge(
            m[[symbol_col, date_col, sector_col]].drop_duplicates(
                [symbol_col, date_col], keep="last"
            ),
            on=[symbol_col, date_col],
            how="left",
            sort=False,
        )
        return result.sort_values([date_col, symbol_col], kind="mergesort").reset_index(drop=True)
    if "valid_from" not in m.columns and "valid_to" not in m.columns:
        values = m.drop_duplicates(symbol_col, keep="last").set_index(symbol_col)[sector_col]
        result[sector_col] = result[symbol_col].map(values)
        return result
    result[sector_col] = pd.NA
    for _, row in m.sort_values([symbol_col, "valid_from"], na_position="first").iterrows():
        mask = result[symbol_col].eq(row[symbol_col])
        if pd.notna(row.get("valid_from")):
            mask &= result[date_col] >= row["valid_from"]
        if pd.notna(row.get("valid_to")):
            mask &= result[date_col] <= row["valid_to"]
        result.loc[mask, sector_col] = row[sector_col]
    return result


def _trailing_returns(out: pd.DataFrame, symbol_col: str, close_col: str, h: int) -> pd.Series:
    return out.groupby(symbol_col, sort=False, observed=True)[close_col].pct_change(
        h, fill_method=None
    )


def _rank_by_group(
    values: pd.Series,
    groups: pd.Series,
    *,
    min_group_size: int,
    exclude_self: bool = False,
    symbols: pd.Series | None = None,
) -> pd.Series:
    """Midrank percentile, with NaN and thin groups remaining NaN."""
    result = pd.Series(np.nan, index=values.index, dtype=float)
    frame = pd.DataFrame({"value": values, "group": groups}, index=values.index)
    if symbols is not None:
        frame["symbol"] = symbols
    for _, positions in frame.groupby("group", sort=True, dropna=True, observed=True).groups.items():
        valid = frame.loc[positions].dropna(subset=["value"])
        if exclude_self and "symbol" in valid:
            for idx, row in valid.iterrows():
                peers = valid.loc[valid["symbol"] != row["symbol"], "value"]
                if len(peers) < max(1, min_group_size - 1):
                    continue
                less = float((peers < row["value"]).sum())
                ties = float((peers == row["value"]).sum())
                # Keep the percentile on [0, 1] while excluding the target
                # from the peer value set: the denominator is the target plus
                # its eligible peers.
                result.loc[idx] = (1.0 + less + 0.5 * ties) / (len(peers) + 1)
        elif len(valid) >= min_group_size:
            ranks = valid["value"].rank(method="average", ascending=True)
            result.loc[valid.index] = ranks / len(valid)
    return result


def percentile_rank(
    values: pd.Series,
    *,
    group: pd.Series | None = None,
    min_group_size: int = DEFAULT_MIN_GROUP_SIZE,
    exclude_self: bool = False,
    symbols: pd.Series | None = None,
) -> pd.Series:
    """Public deterministic same-date/group midrank helper."""
    if min_group_size < 1:
        raise ValueError("min_group_size must be >= 1")
    groups = group if group is not None else pd.Series("all", index=values.index)
    return _rank_by_group(
        values, groups, min_group_size=min_group_size,
        exclude_self=exclude_self, symbols=symbols,
    )


def cross_sectional_rank(
    values: pd.Series,
    *,
    group: pd.Series | None = None,
    min_group_size: int = DEFAULT_MIN_GROUP_SIZE,
) -> pd.Series:
    """Named wrapper for the same-date rank primitive."""
    return percentile_rank(values, group=group, min_group_size=min_group_size)


def peer_median_divergence(
    values: pd.Series,
    *,
    group: pd.Series,
    symbols: pd.Series,
    min_group_size: int = DEFAULT_MIN_GROUP_SIZE,
) -> pd.Series:
    """Leave-one-out peer median divergence for already-aligned values."""
    result = pd.Series(np.nan, index=values.index, dtype=float)
    for _, positions in pd.DataFrame(
        {"value": values, "group": group, "symbol": symbols}, index=values.index
    ).groupby("group", sort=True, dropna=True, observed=True).groups.items():
        valid = pd.DataFrame(
            {"value": values.loc[positions], "symbol": symbols.loc[positions]},
            index=positions,
        ).dropna(subset=["value"])
        if len(valid) < min_group_size:
            continue
        for idx, row in valid.iterrows():
            peers = valid.loc[valid["symbol"] != row["symbol"], "value"]
            if len(peers) >= max(1, min_group_size - 1):
                result.loc[idx] = row["value"] - float(peers.median())
    return result


def _group_mean(values: pd.Series, groups: pd.Series, minimum: int) -> pd.Series:
    frame = pd.DataFrame({"value": values, "group": groups}, index=values.index)
    counts = frame.groupby("group", dropna=True, observed=True)["value"].transform("count")
    means = frame.groupby("group", dropna=True, observed=True)["value"].transform("mean")
    return means.where(counts >= minimum)


def build_relative_features(
    panel: pd.DataFrame,
    *,
    symbol_col: str = "symbol",
    date_col: str = "date",
    close_col: str = "Close",
    market_close_col: str = "market_close",
    market_col: str | None = None,
    sector_col: str = "sector",
    peer_group_col: str | None = None,
    sector_mapping: Mapping[str, Any] | pd.DataFrame | None = None,
    manifest: Mapping[str, Any] | None = None,
    horizons: Iterable[int] = DEFAULT_HORIZONS,
    min_group_size: int = DEFAULT_MIN_GROUP_SIZE,
) -> pd.DataFrame:
    """Build causal market/sector/peer/cross-sectional research features.

    ``manifest`` may contain ``members`` with ``symbol``, optional ``sector``,
    ``valid_from`` and ``valid_to``.  Ineligible rows remain in the returned
    panel but receive NaN relative features, making coverage loss explicit.
    """
    if min_group_size < 1:
        raise ValueError("min_group_size must be >= 1")
    hs = _normalise_horizons(horizons)
    out = _normalise_panel(
        panel, symbol_col=symbol_col, date_col=date_col, close_col=close_col
    )
    if market_col is not None:
        market_close_col = market_col
    if manifest is not None:
        members = manifest.get("members", [])
        if not isinstance(members, list):
            raise ValueError("manifest.members must be a list")
        member_rows = pd.DataFrame(members)
        if not member_rows.empty:
            if sector_mapping is not None:
                raise ValueError("pass either manifest or sector_mapping, not both")
            sector_mapping = member_rows.rename(columns={"symbol": symbol_col})
            # Manifest windows are applied below as an eligibility mask.
    out = _apply_sector_mapping(
        out, sector_mapping, symbol_col=symbol_col, date_col=date_col, sector_col=sector_col
    )
    eligible = pd.Series(True, index=out.index)
    if manifest is not None and not out.empty:
        members = pd.DataFrame(manifest.get("members", []))
        if not members.empty:
            members = members.rename(columns={"symbol": symbol_col})
            eligible = out[symbol_col].isin(members[symbol_col].astype(str))
            for _, member in members.iterrows():
                mask = out[symbol_col].eq(str(member[symbol_col]))
                if pd.notna(member.get("valid_from")):
                    eligible &= ~(mask & (out[date_col] < pd.Timestamp(member["valid_from"])))
                if pd.notna(member.get("valid_to")):
                    eligible &= ~(mask & (out[date_col] > pd.Timestamp(member["valid_to"])))
    feature_columns = _feature_names(hs)
    for name in feature_columns:
        out[name] = np.nan

    # A stock's trailing return uses its own observed trading rows.  No date
    # reindex/fill means weekends and exchange holidays cannot create returns.
    trailing: dict[int, pd.Series] = {
        h: _trailing_returns(out, symbol_col, close_col, h).where(eligible)
        for h in hs
    }
    # By default peers are same-sector, point-in-time peers.  A separate
    # peer_group_col permits industry/basket peers; absent sectors fall back
    # to the whole same-date cross-section.
    peer_source = peer_group_col or (sector_col if sector_col in out.columns else None)
    if peer_source is not None and peer_source not in out.columns:
        raise ValueError(f"peer_group_col {peer_source!r} is not a panel column")
    if peer_source is None:
        peer_group = pd.Series("all", index=out.index, dtype="string")
    else:
        peer_group = out[peer_source].astype("string")
    peer_group_key = out[date_col].astype("string") + "\x1f" + peer_group
    market = None
    if market_close_col in out.columns:
        market_rows = out[[date_col, market_close_col]].copy()
        market_rows[market_close_col] = pd.to_numeric(
            market_rows[market_close_col], errors="coerce"
        )
        market = (
            market_rows.dropna(subset=[market_close_col])
            .drop_duplicates(date_col, keep="last")
            .set_index(date_col)[market_close_col]
            .sort_index()
        )

    for h in hs:
        ret = trailing[h]
        if market is not None:
            # Reuse the established pairwise helpers: common non-NaN dates,
            # no fill, then reindex to the panel rows.
            market_excess = pd.Series(np.nan, index=out.index, dtype=float)
            market_rs = pd.Series(np.nan, index=out.index, dtype=float)
            for _, positions in out.groupby(symbol_col, sort=False, observed=True).groups.items():
                stock = out.loc[positions].set_index(date_col)[close_col]
                market_excess.loc[positions] = excess_return(stock, market, h).reindex(
                    out.loc[positions, date_col]
                ).to_numpy()
                market_rs.loc[positions] = relative_strength_ratio(stock, market, h).reindex(
                    out.loc[positions, date_col]
                ).to_numpy()
            out[f"market_excess_return_{h}d"] = market_excess.where(eligible)
            out[f"market_relative_strength_{h}d"] = market_rs.where(eligible)

        # Sector statistics are cross-sectional *on the same date*.  Grouping
        # by sector alone would let a future observation alter an earlier row.
        sector_group = (
            out[date_col].astype("string")
            + "\x1f"
            + out[sector_col].astype("string")
        )
        sector_mean = _group_mean(ret, sector_group, min_group_size)
        out[f"sector_return_{h}d"] = sector_mean.where(eligible)
        out[f"sector_excess_return_{h}d"] = (ret - sector_mean).where(eligible)
        out[f"cross_sectional_return_rank_{h}d"] = percentile_rank(
            ret.where(eligible), group=out[date_col],
            min_group_size=min_group_size,
        )
        out[f"sector_return_rank_{h}d"] = percentile_rank(
            ret.where(eligible), group=sector_group,
            min_group_size=min_group_size,
        )
        out[f"peer_return_rank_{h}d"] = percentile_rank(
            ret.where(eligible), group=peer_group_key,
            min_group_size=min_group_size, exclude_self=True,
            symbols=out[symbol_col],
        )

        peer_div = pd.Series(np.nan, index=out.index, dtype=float)
        frame = pd.DataFrame(
            {
                "value": ret,
                "date": out[date_col],
                "peer_group": peer_group,
                "symbol": out[symbol_col],
            },
            index=out.index,
        )
        for _, positions in frame.groupby(
            ["date", "peer_group"], sort=True, observed=True
        ).groups.items():
            valid = frame.loc[positions].dropna(subset=["value"])
            if len(valid) < min_group_size:
                continue
            for idx, row in valid.iterrows():
                peers = valid.loc[valid["symbol"] != row["symbol"], "value"]
                if len(peers) >= max(1, min_group_size - 1):
                    peer_div.loc[idx] = row["value"] - float(peers.median())
        out[f"peer_median_divergence_{h}d"] = peer_div.where(eligible)

    out.attrs["feature_version"] = FEATURE_VERSION
    out.attrs["schema_version"] = SCHEMA_VERSION
    out.attrs["feature_manifest"] = feature_manifest(hs)
    out.attrs["feature_metadata"] = deepcopy(out.attrs["feature_manifest"])
    out.attrs["min_group_size"] = int(min_group_size)
    out.attrs["coverage"] = coverage_report(
        out,
        feature_columns=feature_columns,
        symbol_col=symbol_col,
        date_col=date_col,
    )
    out.attrs["fingerprint"] = fingerprint_dataframe(
        "v2-relative-features", out[[symbol_col, date_col] + feature_columns]
    )
    return out


def add_relative_features(*args: Any, **kwargs: Any) -> pd.DataFrame:
    """Alias for :func:`build_relative_features`."""
    return build_relative_features(*args, **kwargs)


def coverage_report(
    frame: pd.DataFrame,
    *,
    feature_columns: Iterable[str] | None = None,
    symbol_col: str = "symbol",
    date_col: str = "date",
) -> dict[str, Any]:
    """Return deterministic row/feature coverage and thin-group diagnostics."""
    features = list(feature_columns or [
        c for c in frame.columns
        if c.startswith(("market_", "sector_", "peer_", "cross_sectional_"))
    ])
    total = len(frame)
    by_feature = {}
    for name in sorted(features):
        present = int(frame[name].notna().sum()) if name in frame else 0
        by_feature[name] = {
            "present": present,
            "missing": total - present,
            "coverage": round(present / total, 6) if total else None,
        }
    return {
        "rows": total,
        "symbols": int(frame[symbol_col].nunique()) if symbol_col in frame else None,
        "dates": int(frame[date_col].nunique()) if date_col in frame else None,
        "features": by_feature,
    }


def diagnostics(frame: pd.DataFrame, *, feature_columns: Iterable[str] | None = None) -> dict[str, Any]:
    """Compatibility alias for the coverage/diagnostics report."""
    return coverage_report(frame, feature_columns=feature_columns)


__all__ = [
    "FEATURE_VERSION",
    "RELATIVE_FEATURE_VERSION",
    "SCHEMA_VERSION",
    "MANIFEST_VERSION",
    "DEFAULT_HORIZONS",
    "DEFAULT_MIN_GROUP_SIZE",
    "feature_manifest",
    "get_feature_manifest",
    "list_relative_features",
    "percentile_rank",
    "cross_sectional_rank",
    "peer_median_divergence",
    "build_relative_features",
    "add_relative_features",
    "coverage_report",
    "diagnostics",
]
