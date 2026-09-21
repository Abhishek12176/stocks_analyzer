"""Network-free proof tests for the V2 relative research block."""

import numpy as np
import pandas as pd
import pytest

from app.ml.v2_relative_features import (
    FEATURE_VERSION,
    build_relative_features,
    feature_manifest,
    percentile_rank,
)


def _panel(days=5):
    rows = []
    for i in range(days):
        date = pd.Timestamp("2024-01-01") + pd.Timedelta(days=i)
        # A/B/C are one sector; D is intentionally a thin sector.
        for symbol, close, sector in (
            ("A", 100 + i * 10, "tech"),
            ("B", 100 + i * 5, "tech"),
            ("C", 100 + i * 2, "tech"),
            ("D", 100 + i, "thin"),
        ):
            rows.append(
                {
                    "symbol": symbol,
                    "date": date,
                    "Close": float(close),
                    "market_close": float(100 + i * 3),
                    "sector": sector,
                }
            )
    return pd.DataFrame(rows)


def test_market_and_sector_formulas_are_explicit():
    out = build_relative_features(_panel(), horizons=(1,), min_group_size=3)
    row = out[(out.symbol == "A") & (out.date == "2024-01-03")].iloc[0]
    stock_ret = 120 / 110 - 1
    market_ret = 106 / 103 - 1
    expected_sector = np.mean([stock_ret, 110 / 105 - 1, 104 / 102 - 1])
    assert row.market_excess_return_1d == pytest.approx(stock_ret - market_ret)
    assert row.sector_return_1d == pytest.approx(expected_sector)
    assert row.sector_excess_return_1d == pytest.approx(stock_ret - expected_sector)


def test_peer_median_divergence_excludes_target():
    out = build_relative_features(_panel(), horizons=(1,), min_group_size=3)
    row = out[(out.symbol == "A") & (out.date == "2024-01-03")].iloc[0]
    # peers are B (+4.76%) and C (+1.96%); D is in a different sector.
    peer_median = np.median([110 / 105 - 1, 104 / 102 - 1])
    assert row.peer_median_divergence_1d == pytest.approx(120 / 110 - 1 - peer_median)


def test_ranks_are_midranked_and_ties_are_deterministic():
    values = pd.Series([1.0, 2.0, 2.0, 4.0])
    out = percentile_rank(values, min_group_size=3)
    assert out.tolist() == pytest.approx([0.25, 0.625, 0.625, 1.0])
    panel = _panel()
    panel.loc[(panel.symbol == "A") & (panel.date == "2024-01-02"), "Close"] = 105
    panel.loc[(panel.symbol == "B") & (panel.date == "2024-01-02"), "Close"] = 105
    a = build_relative_features(panel, horizons=(1,), min_group_size=3)
    b = build_relative_features(panel.sample(frac=1, random_state=9), horizons=(1,), min_group_size=3)
    pd.testing.assert_frame_equal(a, b)


def test_missing_values_and_thin_groups_remain_nan():
    panel = _panel()
    panel.loc[(panel.symbol == "C") & (panel.date == "2024-01-03"), "Close"] = np.nan
    out = build_relative_features(panel, horizons=(1,), min_group_size=3)
    row_d = out[(out.symbol == "D") & (out.date == "2024-01-03")].iloc[0]
    assert pd.isna(row_d.sector_return_1d)
    assert pd.isna(row_d.sector_return_rank_1d)
    assert pd.isna(out.loc[(out.symbol == "C") & (out.date == "2024-01-03"), "cross_sectional_return_rank_1d"].iloc[0])


def test_holidays_are_absent_and_truncation_is_equivalent():
    panel = _panel(8)
    panel = panel[panel.date != pd.Timestamp("2024-01-04")].reset_index(drop=True)
    full = build_relative_features(panel, horizons=(1, 5), min_group_size=3)
    cutoff = pd.Timestamp("2024-01-05")
    short = build_relative_features(panel[panel.date <= cutoff], horizons=(1, 5), min_group_size=3)
    cols = [c for c in full.columns if c.startswith(("market_", "sector_", "peer_", "cross_"))]
    pd.testing.assert_frame_equal(full[full.date <= cutoff][cols].reset_index(drop=True), short[cols].reset_index(drop=True))


def test_future_changes_do_not_change_earlier_rank_but_same_day_changes_do():
    panel = _panel(5)
    base = build_relative_features(panel, horizons=(1,), min_group_size=3)
    future = panel.copy()
    future.loc[future.date == pd.Timestamp("2024-01-05"), "Close"] *= 100
    changed = build_relative_features(future, horizons=(1,), min_group_size=3)
    earlier = base.date < pd.Timestamp("2024-01-05")
    cols = ["cross_sectional_return_rank_1d", "peer_median_divergence_1d"]
    pd.testing.assert_frame_equal(base.loc[earlier, cols].reset_index(drop=True), changed.loc[earlier, cols].reset_index(drop=True))
    same_day = panel.copy()
    same_day.loc[(same_day.symbol == "D") & (same_day.date == "2024-01-04"), "Close"] *= 10
    same = build_relative_features(same_day, horizons=(1,), min_group_size=3)
    old = base.loc[(base.date == "2024-01-04") & (base.symbol == "A"), "cross_sectional_return_rank_1d"].iloc[0]
    new = same.loc[(same.date == "2024-01-04") & (same.symbol == "A"), "cross_sectional_return_rank_1d"].iloc[0]
    assert old != new


def test_manifest_pit_mapping_and_metadata_are_deterministic():
    panel = _panel()
    manifest = {
        "members": [
            {"symbol": "A", "sector": "tech", "valid_from": "2024-01-01", "valid_to": "2024-01-03"},
            {"symbol": "B", "sector": "tech"},
            {"symbol": "C", "sector": "tech"},
            {"symbol": "D", "sector": "thin"},
        ]
    }
    out = build_relative_features(panel, manifest=manifest, horizons=(1,))
    assert out.attrs["feature_version"] == FEATURE_VERSION
    assert out.attrs["feature_manifest"]["manifest_version"] == "v2-relative-manifest-1"
    assert out.attrs["coverage"]["rows"] == len(panel)
    assert out.attrs["fingerprint"] == build_relative_features(panel, manifest=manifest, horizons=(1,)).attrs["fingerprint"]
    assert out.loc[(out.symbol == "A") & (out.date == "2024-01-04"), "sector_excess_return_1d"].isna().all()
    assert out.loc[(out.symbol == "A") & (out.date == "2024-01-04"), "sector_return_1d"].isna().all()


def test_duplicate_keys_and_invalid_horizons_fail_loudly():
    panel = _panel().iloc[:4].copy()
    panel = pd.concat([panel, panel.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="one row"):
        build_relative_features(panel)
    with pytest.raises(ValueError, match="positive"):
        build_relative_features(_panel(), horizons=(0,))


def test_manifest_feature_list_is_sorted_and_versioned():
    manifest = feature_manifest((20, 1, 5))
    assert manifest["horizons"] == [1, 5, 20]
    assert sorted(manifest["features"]) == list(manifest["features"])
    assert manifest["feature_version"] == FEATURE_VERSION
