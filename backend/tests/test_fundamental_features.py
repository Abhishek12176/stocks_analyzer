"""Unit tests for PIT fundamental features (Task 6).

Core invariant: a fundamental value with `available_at = D` must never appear
on rows before D (no lookahead); a later snapshot supersedes an earlier one.
"""

import pandas as pd
import pytest

from app.config import settings
from app.ml import alpha as al
from app.ml import fundamental_features as ff


def _frame() -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=7, freq="D")
    return pd.DataFrame({"Close": 100 + pd.Series(range(7), index=index)}, index=index)


class TestPitSeries:
    def test_latest_known_step_function(self):
        snapshots = [
            {"available_at": "2024-01-03", "pe_ratio": 10.0},
            {"available_at": "2024-01-05", "pe_ratio": 12.0},
        ]
        df = _frame()
        pe = ff.pit_series(snapshots, "pe_ratio", pd.DatetimeIndex(df.index))
        assert pd.isna(pe.loc["2024-01-01"]) and pd.isna(pe.loc["2024-01-02"])
        assert pe.loc["2024-01-03"] == 10.0 and pe.loc["2024-01-04"] == 10.0
        assert pe.loc["2024-01-05"] == 12.0 and pe.loc["2024-01-06"] == 12.0

    def test_elementwise_missing_key(self):
        snapshots = [
            {"available_at": "2024-01-03", "pe_ratio": 10.0},
            {"available_at": "2024-01-05"},  # no pe -> NaN after 5th
        ]
        df = _frame()
        pe = ff.pit_series(snapshots, "pe_ratio", pd.DatetimeIndex(df.index))
        assert pe.loc["2024-01-04"] == 10.0
        assert pd.isna(pe.loc["2024-01-05"])

    def test_single_snapshot_defaults_to_last_row(self):
        df = _frame()
        out = ff.add_fundamental_features(df, snapshot={"pe_ratio": 15.0})
        assert out["fund_pe"].iloc[:-1].isna().all()
        assert out["fund_pe"].iloc[-1] == 15.0

    def test_single_snapshot_with_available_at(self):
        df = _frame()
        out = ff.add_fundamental_features(
            df, snapshot={"pe_ratio": 15.0, "roe": 0.2, "available_at": "2024-01-03"}
        )
        assert out.loc["2024-01-02", "fund_pe"] != out.loc["2024-01-03", "fund_pe"]
        assert out.loc["2024-01-03", "fund_pe"] == 15.0
        assert out.loc["2024-01-03", "fund_roe"] == 0.2
        assert pd.isna(out.loc["2024-01-02", "fund_roe"])

    def test_snapshots_required_to_carry_available_at(self):
        with pytest.raises(ValueError):
            ff.add_fundamental_features(
                _frame(), snapshots=[{"pe_ratio": 10.0}, {"pe_ratio": 12.0}]
            )

    def test_missing_keys_not_created(self):
        out = ff.add_fundamental_features(_frame(), snapshot={"pe_ratio": 15.0})
        assert "fund_pe" in out.columns
        assert "fund_score" not in out.columns  # not provided -> graceful

    def test_all_service_keys_mapped(self):
        snapshot = {
            "pe_ratio": 15.0, "eps": 10.0, "roe": 0.2, "roce": 0.25,
            "debt_to_equity": 0.4, "operating_margin": 0.3,
            "revenue_growth": 0.1, "profit_growth": 0.12,
            "fundamental_score": 80.0, "available_at": "2024-01-03",
        }
        out = ff.add_fundamental_features(_frame(), snapshot=snapshot)
        expected = {
            "fund_pe", "fund_eps", "fund_roe", "fund_roce", "fund_de",
            "fund_opm", "fund_rev_growth", "fund_profit_growth", "fund_score",
        }
        assert expected <= set(out.columns)
        assert out.attrs["feature_version"] == settings.ml_feature_version

    def test_as_of_alias(self):
        df = _frame()
        out = ff.add_fundamental_features(
            df, snapshot={"pe_ratio": 9.0, "as_of": "2024-01-04"}
        )
        assert pd.isna(out.loc["2024-01-03", "fund_pe"])
        assert out.loc["2024-01-04", "fund_pe"] == 9.0

    def test_no_lookahead_via_truncation(self):
        snapshots = [
            {"available_at": "2024-01-03", "pe_ratio": 10.0},
            {"available_at": "2024-01-05", "pe_ratio": 12.0},
            {"available_at": "2024-01-06", "roe": 0.3},
        ]
        df = _frame()
        full = ff.add_fundamental_features(df, snapshots=snapshots)
        k = 4
        truncated = ff.add_fundamental_features(df.iloc[:k], snapshots=snapshots)
        pd.testing.assert_frame_equal(full.iloc[:k], truncated)

    def test_feature_list_deterministic(self):
        cols = ff.list_fundamental_features()
        assert cols == sorted(set(cols))
        assert "fund_pe" in cols and "fund_score" in cols


class TestAlphaIntegration:
    def test_pit_pe_feeds_value_factor(self):
        df = _frame()
        out = ff.add_fundamental_features(
            df, snapshot={"pe_ratio": 15.0, "available_at": "2024-01-03"}
        )
        pe = al.value_factor(pe=out["fund_pe"], window=3)
        assert pe is not None
        assert pe.notna().all()
        # rows before the PIT value are the composite NEUTRAL (0.5), never a
        # fabricated value score; later rows reflect the real PIT data
        assert pe.iloc[0] == pytest.approx(al.NEUTRAL)
        assert pe.iloc[-1] != pytest.approx(al.NEUTRAL)