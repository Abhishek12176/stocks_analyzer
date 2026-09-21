"""Unit tests for the market-regime engine (Task 7).

Invariants: every regime signal is causal (trailing windows / SMA only),
unknown warm-up rows carry NaN (never a fabricated bucket), and missing
sources simply omit the corresponding columns.
"""

import numpy as np
import pandas as pd
import pytest

from app.config import settings
from app.ml import regime as rg


def _index(n: int = 120) -> pd.DatetimeIndex:
    return pd.bdate_range("2023-06-01", periods=n)


def test_trend_regime_bull():
    idx = _index(120)
    close = pd.Series(200 + np.arange(120) * 0.5, index=idx)  # steady uptrend
    out = rg.trend_regime(close, fast=5, slow=20)
    assert out.iloc[-1] == rg.BULL
    assert out.iloc[-10:].eq(rg.BULL).all()


def test_trend_regime_bear():
    idx = _index(120)
    close = pd.Series(300 - np.arange(120) * 0.5, index=idx)  # steady downtrend
    out = rg.trend_regime(close, fast=5, slow=20)
    assert out.iloc[-1] == rg.BEAR
    assert out.iloc[-10:].eq(rg.BEAR).all()


def test_trend_regime_sideways_or_nan():
    idx = _index(120)
    flat = pd.Series(np.full(120, 100.0), index=idx)
    out = rg.trend_regime(flat, fast=5, slow=20)
    # flat close == its own SMA -> neither strictly above nor below -> 0
    assert out.iloc[-1] == rg.SIDEWAYS


def test_trend_warmup_is_nan():
    idx = _index(30)
    close = pd.Series(np.full(30, 100.0), index=idx)
    out = rg.trend_regime(close, fast=10, slow=5)  # fast > slow is nonsense order
    # with fast=10 slow=5 the SMA-10 is NaN for first 9 rows -> NaN there
    assert out.iloc[:9].isna().all()


def test_vol_regime_high_bucket():
    n = 140
    idx = _index(n)
    amp = np.linspace(0.002, 0.06, n)  # volatility grows linearly
    rng = np.random.default_rng(0)
    ret = rng.normal(0, amp)
    close = pd.Series(100 * np.cumprod(1 + ret), index=idx)
    bucket, pctile = rg.vol_regime(close, vol_window=5, rank_window=60)
    assert bucket.iloc[-1] == 1                  # newest vol is the max
    assert pctile.iloc[-1] >= 0.99
    assert bucket.iloc[:20].isna().any()          # warm-up stays unknown
    assert (bucket == 1).sum() >= 2


def test_vol_regime_low_bucket():
    n = 140
    idx = _index(n)
    amp = np.linspace(0.06, 0.002, n)  # volatility shrinks linearly
    rng = np.random.default_rng(7)
    ret = rng.normal(0, amp)
    close = pd.Series(100 * np.cumprod(1 + ret), index=idx)
    bucket, _ = rg.vol_regime(close, vol_window=5, rank_window=60)
    assert bucket.iloc[-1] == -1                 # newest vol is the min
    assert (bucket == -1).sum() >= 2


def test_risk_regime_both_directions():
    n = 160
    half = n // 2
    idx = _index(n)
    nifty = pd.Series(
        np.concatenate([np.linspace(200, 150, half), np.linspace(150, 260, n - half)]),
        index=idx,
    )
    vix = pd.Series(
        np.concatenate([np.linspace(14, 28, half), np.linspace(28, 12, n - half)]),
        index=idx,
    )
    bucket, _ = rg.risk_regime(nifty, vix, fast=10, slow=30, rank_window=60)
    assert (bucket == rg.RISK_ON).any()          # bull trend + falling VIX tail
    assert (bucket == rg.RISK_OFF).any()         # bear trend + climbing VIX
    assert bucket.iloc[-1] == rg.RISK_ON


def test_risk_warmup_nan_when_vix_unknown():
    n = 30
    idx = _index(n)
    nifty = pd.Series(np.linspace(100, 200, n), index=idx)
    vix = pd.Series(np.linspace(30, 10, n), index=idx)
    bucket, _ = rg.risk_regime(nifty, vix, fast=5, slow=10, rank_window=15)
    assert bucket.isna().any()                   # NaN while VIX pctile is warm
    assert bucket.iloc[-1] == rg.RISK_ON


def test_breadth_fraction_above_ma():
    idx = _index(60)
    sym1 = pd.Series(100 + np.arange(60) * 1.0, index=idx)  # always above own SMA
    sym2 = pd.Series(np.full(60, 100.0), index=idx)          # == own SMA, not above
    panel = pd.DataFrame({"a": sym1, "b": sym2})
    br = rg.breadth(panel, ma_len=20)
    assert br.iloc[2] == pytest.approx(0.5)      # only sym1 counts as above
    assert br.iloc[-1] == pytest.approx(0.5)


def test_add_regime_features_columns_and_version():
    idx = _index(140)
    nifty = pd.Series(np.linspace(100, 300, len(idx)), index=idx)
    vix = pd.Series(np.linspace(30, 10, len(idx)), index=idx)
    panel = pd.DataFrame({"a": nifty, "b": 100.0})
    df = pd.DataFrame({"Close": nifty}, index=idx)
    out = rg.add_regime_features(df, nifty_close=nifty, vix_close=vix,
                                 breadth_series=rg.breadth(panel))
    for col in ("regime_trend", "regime_vol", "regime_risk",
                "regime_vix_pctile", "regime_nifty_vol_pctile", "regime_breadth"):
        assert col in out.columns
    assert out.attrs["feature_version"] == settings.ml_feature_version


def test_add_regime_features_graceful_disable():
    idx = _index(30)
    df = pd.DataFrame({"Close": 100.0}, index=idx)
    out = rg.add_regime_features(df)             # no market inputs at all
    assert not any(c.startswith("regime_") for c in out.columns)


def test_add_regime_no_lookahead_via_truncation():
    idx = _index(180)
    nifty = pd.Series(np.linspace(100, 400, len(idx)), index=idx)
    vix = pd.Series(np.linspace(30, 10, len(idx)), index=idx)
    panel = pd.DataFrame({"a": nifty, "b": nifty[::-1].values}, index=idx)
    df = pd.DataFrame({"Close": nifty}, index=idx)
    k = 90
    full = rg.add_regime_features(df, nifty_close=nifty, vix_close=vix,
                                  breadth_series=rg.breadth(panel))
    trunc = rg.add_regime_features(
        df.iloc[:k], nifty_close=nifty.iloc[:k], vix_close=vix.iloc[:k],
        breadth_series=rg.breadth(panel.iloc[:k]),
    )
    pd.testing.assert_frame_equal(full.iloc[:k], trunc)


def test_feature_list_deterministic():
    feats = rg.list_regime_features()
    assert feats == sorted(set(feats))
    assert len(feats) == 6


def test_risk_regime_ragged_calendars():
    """nifty & vix on DIFFERENT calendars must not crash (regression).

    When the two calendars diverge (real 10y history: nifty ~2466 rows, vix
    ~2451) the aligned boolean masks used to form the risk bucket on the
    UNION calendar, then `pd.Series(bucket, index=nifty_close.index)` blew up
    with a length mismatch. VIX must be classified on its own calendar first,
    then the percentile reindexed to the nifty calendar (still causal).
    """

    def _mkt_index(start, periods):
        return pd.bdate_range(start, periods=periods)

    nifty = pd.Series(
        np.linspace(100, 300, 2466),
        index=_mkt_index("2016-09-15", 2466),
    )
    vix = pd.Series(
        np.linspace(30, 12, 2451),
        index=_mkt_index("2016-10-01", 2451),
    )
    bucket, pctile = rg.risk_regime(nifty, vix, fast=50, slow=200, rank_window=252)
    assert len(bucket) == len(nifty)          # never exceeds the nifty calendar
    assert len(pctile) == len(nifty)
    assert bucket.index.equals(nifty.index)
    # VIX-unknown days (outside the shorter vix calendar) stay NaN, not 0
    vix_present = vix.index.isin(nifty.index)
    vix_present_dates = vix.index[vix_present]
    assert bucket.loc[vix_present_dates].notna().any()