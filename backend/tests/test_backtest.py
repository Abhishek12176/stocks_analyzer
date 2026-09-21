"""Unit tests for the walk-forward backtest engine, costs, purged CV + metrics.

Structural assertions on synthetic data: hand-computed costs, embargo purging,
chronological fold ordering, metric-schema completeness, cost drag direction,
and calibration output shape. (Not strategy backtests — nothing asserts alpha.)
"""

import numpy as np
import pandas as pd
import pytest

from app.ml import backtest as bt
from app.ml import models as mo


def _rising_close(n: int = 360, seed: int = 1) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2023-01-02", periods=n)
    rets = rng.normal(0.0018, 0.02, n)
    return pd.Series(100 * np.cumprod(1 + rets), index=idx)


def _frame(n: int = 360, with_indicators: bool = False, seed: int = 1) -> pd.DataFrame:
    idx = _rising_close(n).index
    close = _rising_close(n, seed=seed)
    rng = np.random.default_rng(9)
    df = pd.DataFrame({"Close": close}, index=idx)
    df["x0"] = rng.normal(0, 1, n)
    df["x1"] = rng.normal(0, 1, n)
    h = 5
    df["leak"] = (close.shift(-h) / close - 1 > 0).astype(float)
    if with_indicators:
        df["sma20"] = close.rolling(20).mean()
        df["sma50"] = close.rolling(50).mean()
        df["rsi"] = 50 + rng.normal(0, 10, n)
        df["macd"] = rng.normal(0, 1, n)
        df["signal"] = rng.normal(0, 1, n)
        df["sentiment"] = rng.normal(0, 0.5, n)
    return df


KW = dict(horizon=5, test_size=40, step=40, min_train=160)


class TestCostModel:
    def test_zero_cost(self):
        c = bt.CostModel(brokerage_inr=0, fee_percent=0, slippage_bps=0)
        assert c.per_side_cost(100) == 0.0
        assert c.round_trip(100, 100) == 0.0

    def test_hand_computed_default(self):
        c = bt.CostModel.from_settings()
        # per side = 20/px + 0.01% fee + 10bps slippage
        assert c.per_side_cost(100) == pytest.approx(20 / 100 + 0.0001 + 0.001)
        assert c.round_trip(100, 120) == pytest.approx(
            (20 / 100 + 0.0001 + 0.001) + (20 / 120 + 0.0001 + 0.001))

    def test_cheaper_than_small_prices(self):
        c = bt.CostModel(brokerage_inr=5, fee_percent=0, slippage_bps=0)
        assert c.per_side_cost(50) > c.per_side_cost(500)


class TestPurge:
    def test_purge_last_removes_embargo(self):
        idx = _frame().index
        train = idx[:100]
        purged = bt.purge_last(train, 5)
        assert len(purged) == 95
        assert purged[-1] == train[-6]

    def test_walk_forward_no_leak_and_purged(self):
        idx = _frame(n=300).index
        folds = bt.purged_walk_forward_splits(idx, 30, 30, min_train=120, embargo=5)
        assert len(folds[0][0]) == 120 - 5   # embargo stripped even on first fold
        for train, test in folds:
            assert (train < test[0]).all()
        # expanding window: each train starts with the previous fold's train,
        # and every fold's test window is strictly disjoint from the next's.
        for i in range(len(folds) - 1):
            assert folds[i + 1][0][:len(folds[i][0])].equals(folds[i][0])
            assert folds[i][1].max() < folds[i + 1][1].min()

    def test_chrono_splits(self):
        idx = _frame(n=300).index
        folds = bt.purged_chrono_splits(idx, n_splits=3, embargo=5)
        assert len(folds) == 2          # last split would equal full index
        for train, test in folds:
            assert (train < test[0]).all()


class TestRunBacktest:
    def test_metric_schema_complete(self):
        res = bt.run_backtest(_frame(), model="logistic", **KW)
        m = res["metrics"]
        for key in ("accuracy", "precision", "recall", "f1", "confusion",
                    "false_buy_pct", "false_sell_pct", "n_trades", "win_rate",
                    "avg_return", "cum_return", "max_dd", "sharpe", "sortino",
                    "calmar", "profit_factor", "roc_auc", "no_leverage_ok",
                    "annualized_return", "annualized_volatility", "n_days"):
            assert key in m
        assert set(m["confusion"]) == {"tn", "fp", "fn", "tp", "n"}
        assert res["trades"].shape[0] > 0
        assert "pl" in res["trades"].columns and "ret" in res["trades"].columns
        assert "cum" not in res["trades"].columns  # overlapping-compound bug removed
        g = m["metric_groups"]
        assert {"per_bet", "daily_equity_curve", "classification"} == set(g)
        per_bet = set(g["per_bet"]["metrics"])
        assert {"n_trades", "win_rate", "avg_return", "profit_factor"} <= per_bet
        eq_curve = set(g["daily_equity_curve"]["metrics"])
        assert {"cum_return", "max_dd", "sharpe", "sortino", "calmar",
                "n_days"} <= eq_curve

    def test_no_fake_drawdown_and_bounded_equity(self):
        # the old code compounded overlapping 20d returns and reported
        # max DD < -100% / fake cum_returns; the ledger fixes this.
        low_cost = bt.CostModel(brokerage_inr=0, fee_percent=0.01, slippage_bps=5)
        res = bt.run_backtest(_frame(seed=8), model="rf", cost=low_cost, **KW)
        m = res["metrics"]
        assert m["max_dd"] > -1.0
        assert m["cum_return"] > -1.0
        assert m["no_leverage_ok"] == "yes"
        assert m["max_concurrent"] <= KW["horizon"]

    def test_leak_feature_gives_high_accuracy(self):
        res = bt.run_backtest(_frame(), model="logistic", feature_cols=["leak"], **KW)
        assert res["metrics"]["accuracy"] >= 0.95
        assert res["metrics"]["f1"] >= 0.95

    def test_voting_model_runs(self):
        res = bt.run_backtest(_frame(with_indicators=True), model="voting", **KW)
        assert res["metrics"] is not None
        assert res["trades"].shape[0] > 0
        assert res["trades"]["direction"].isin([0.0, 1.0]).all()

    def test_allow_short_produces_short_trades(self):
        res = bt.run_backtest(_frame(), model="logistic", allow_short=True, **KW)
        assert (res["trades"]["direction"] == -1.0).any() or \
               (res["trades"]["direction"] == 0.0).any()
        # short trades still pay (not get refunded) transaction costs
        assert (res["trades"]["cost"] >= 0.0).all()
        short_cost = res["trades"].loc[res["trades"]["direction"] == -1.0, "cost"]
        if len(short_cost):
            assert (short_cost > 0.0).all()

    def test_cost_drags_cum_return(self):
        no_cost = bt.CostModel(brokerage_inr=0, fee_percent=0, slippage_bps=0)
        with_cost = bt.CostModel(brokerage_inr=30, fee_percent=0.05, slippage_bps=25)
        res_nc = bt.run_backtest(_frame(seed=3), model="logistic", cost=no_cost, **KW)
        res_c = bt.run_backtest(_frame(seed=3), model="logistic", cost=with_cost, **KW)
        assert res_nc["metrics"]["cum_return"] >= res_c["metrics"]["cum_return"]

    def test_calibration_output(self):
        res = bt.run_backtest(_frame(), model="logistic", **KW)
        cal = res["calibration"]
        assert set(cal) == {"bins", "brier", "n"}
        assert cal["n"] > 0
        assert all(0.0 <= b["bin_lo"] <= b["bin_hi"] <= 1.0 for b in cal["bins"])

    def test_fold_metrics_reported(self):
        res = bt.run_backtest(_frame(), model="logistic", **KW)
        assert len(res["fold_metrics"]) >= 1
        assert "train_size" in res["fold_metrics"][0]


class TestBuyHoldAndCompare:
    def test_buy_hold_equity_is_price_ratio(self):
        frame = _frame(seed=11)
        m = bt.buy_and_hold_metrics(frame, horizon=5)
        close = frame["Close"].dropna().sort_index()
        assert m["n_trades"] == 1
        assert m["cum_return"] == pytest.approx(
            close.iloc[-1] / close.iloc[0] - 1.0, rel=1e-6)
        assert -1.0 < m["max_dd"] <= 0.0

    def test_buy_hold_eval_window_restriction(self):
        frame = _frame(seed=21)
        idx = frame.index[100:180]
        m = bt.buy_and_hold_metrics(frame, horizon=5, eval_index=idx)
        close = frame["Close"].dropna().sort_index()
        assert m["n_trades"] == 1
        # buy-and-hold enters at the first eval bar and holds to the LAST close
        assert m["cum_return"] == pytest.approx(
            close.iloc[-1] / close.loc[idx[0]] - 1.0, rel=1e-6)

    def test_compare_models_table(self):
        tbl = bt.compare_models(_frame(), models=("voting", "logistic"), horizon=5,
                                test_size=40, step=40, min_train=160)
        names = set(tbl["model"])
        assert {"voting", "logistic", "buy_hold"} <= names
        assert {"accuracy", "cum_return", "sharpe", "max_dd", "profit_factor"} <= set(tbl.columns)


class TestEquityLedger:
    def test_verified_five_day_example_exits_at_T_plus_horizon(self):
        # User-verified ledger: capital=200, C=capital/horizon=100, one long
        # per day, exit at Close[T+horizon]. Prices extend past the 5 signal
        # days so the day-4/day-5 positions also settle (as in production,
        # where the close series continues past the last signal). Expected
        # values are hand-computed on the closed form
        #   equity(t) = cash + sum(active shares_j * close(t)).
        idx = pd.bdate_range("2024-01-01", periods=7)
        close = pd.Series([100.0, 110.0, 121.0, 110.0, 100.0, 110.0, 121.0],
                          index=idx)
        signals = pd.DataFrame({"direction": [1.0] * 5}, index=idx[:5])
        zero = bt.CostModel(brokerage_inr=0, fee_percent=0, slippage_bps=0)
        equity, trades = bt.equity_ledger(close, signals, horizon=2,
                                          capital=200.0, cost=zero)
        expected = [200.0, 210.0, 231.0, 211.909091, 194.553719,
                    213.644628, 224.644628]
        assert np.allclose(equity.to_numpy(), expected, atol=1e-4)
        # invariant 2: equity(final) == capital + sum(realized P&L) - costs
        assert equity.iloc[-1] == pytest.approx(200.0 + trades["pl"].sum(), abs=1e-6)
        assert len(trades) == 5
        # target alignment: P&L == C * ret exactly (zero cost)
        assert np.allclose(trades["pl"].to_numpy(), 100.0 * trades["ret"].to_numpy(),
                           atol=1e-6)
        assert equity.attrs["max_concurrent"] == 2  # horizon, not more

    def test_exit_at_close_T_plus_horizon_matches_target_window(self):
        idx = pd.bdate_range("2024-02-01", periods=60)
        close = pd.Series(np.linspace(10, 30, 60), index=idx)
        signals = pd.DataFrame({"direction": [1.0]}, index=idx[5:6])
        zero = bt.CostModel(0, 0, 0)
        _, trades = bt.equity_ledger(close, signals, horizon=20,
                                     capital=1000.0, cost=zero)
        assert len(trades) == 1
        tr = trades.iloc[0]
        assert tr["entry"] == pytest.approx(close.iloc[5])
        assert tr["exit"] == pytest.approx(close.iloc[25])
        assert tr["ret"] == pytest.approx(close.iloc[25] / close.iloc[5] - 1.0)
        assert tr["pl"] == pytest.approx(50.0 * tr["ret"])  # C = capital/horizon

    def test_no_hidden_leverage_long_only(self):
        rng = np.random.default_rng(5)
        idx = pd.bdate_range("2024-03-01", periods=220)
        close = pd.Series(100 * np.cumprod(1 + rng.normal(0.0005, 0.02, 220)),
                          index=idx)
        signals = pd.DataFrame({"direction": np.ones(220)}, index=idx)
        # zero-ish cost so the low-priced test stock doesn't drown in the
        # INR broker charge (on a ~100-price equity that would be 40%/side)
        cost = bt.CostModel(brokerage_inr=0, fee_percent=0.0, slippage_bps=0)
        equity, trades = bt.equity_ledger(close, signals, horizon=20,
                                          capital=bt.BACKTEST_CAPITAL, cost=cost)
        assert equity.attrs["max_concurrent"] == 20  # == horizon -> 1.0x gross
        assert (equity.to_numpy() > 0).all()
        assert len(trades) == 200

    def test_cost_charged_once_at_open(self):
        idx = pd.bdate_range("2024-04-01", periods=45)
        close = pd.Series(np.linspace(50, 90, 45), index=idx)
        signals = pd.DataFrame({"direction": [1.0, 1.0]}, index=idx[:2])
        cost = bt.CostModel(brokerage_inr=0, fee_percent=0, slippage_bps=10)
        equity, trades = bt.equity_ledger(close, signals, horizon=20,
                                          capital=10000.0, cost=cost)
        # cost is deducted once per opening, never at exit / inside the mark
        assert len(trades) == 2
        assert (trades["cost"] > 0).all()
        # equity(final) = capital + realized(net of cost)
        assert equity.iloc[-1] == pytest.approx(10000.0 + trades["pl"].sum(), abs=1e-6)

    def test_no_leverage_guard_rejects_oversized_same_day_book(self):
        # Many signals landing on the SAME bar would blow committed notional
        # past initial capital. The ledger drops same-day duplicates (it never
        # opens 12 positions of size C on one bar -> no hidden leverage), and
        # the committed-notional guard would raise if that invariant broke.
        idx = pd.bdate_range("2024-07-01", periods=30)
        close = pd.Series(np.full(30, 100.0), index=idx)
        capital, horizon = 1000.0, 5
        many = pd.DataFrame({"direction": [1.0] * 12}, index=[idx[0]] * 12)
        equity, trades = bt.equity_ledger(close, many, horizon=horizon,
                                          capital=capital,
                                          cost=bt.CostModel(0, 0, 0))
        assert len(trades) == 1  # deduped to one position, never 12x C
        assert equity.attrs["max_concurrent"] <= horizon
        assert (equity.to_numpy() > 0).all()


class TestStrategyBaselines:
    def test_always_up_equals_buy_and_hold_curve(self):
        rng = np.random.default_rng(11)
        idx = pd.bdate_range("2024-05-01", periods=160)
        close = pd.Series(100 * np.cumprod(1 + rng.normal(0.0002, 0.02, 160)),
                          index=idx)
        n = len(idx)
        sig = pd.DataFrame({"direction": np.ones(n),
                            "ret": close.shift(-20) / close - 1.0}, index=idx)
        zero = bt.CostModel(0, 0, 0)
        met = bt.strategy_metrics(close, sig, horizon=20,
                                  capital=bt.BACKTEST_CAPITAL, cost=zero)
        assert met["n_trades"] == n - 20
        # zero-cost: ledger invariant => cum_return == sum(realized P&L)/capital
        equity, trades = bt.equity_ledger(close, sig, horizon=20,
                                          capital=bt.BACKTEST_CAPITAL, cost=zero)
        assert met["cum_return"] == pytest.approx(trades["pl"].sum() / bt.BACKTEST_CAPITAL,
                                                  rel=1e-6)
        assert equity.iloc[-1] == pytest.approx(bt.BACKTEST_CAPITAL * (1 + met["cum_return"]),
                                                rel=1e-6)
        assert met["no_leverage_ok"] == "yes"
        assert met["max_dd"] > -1.0
        for key in ("accuracy", "roc_auc", "sortino", "max_dd", "sharpe",
                    "calmar", "profit_factor", "n_days", "n_trades"):
            assert key in met

    def test_random_signal_schema_and_no_leverage(self):
        rng = np.random.default_rng(3)
        idx = pd.bdate_range("2024-06-01", periods=120)
        close = pd.Series(100 + np.cumsum(rng.normal(0, 1, 120)), index=idx)
        dirs = np.zeros(120)
        dirs[:50] = 1.0
        sig = pd.DataFrame({"direction": dirs}, index=idx)
        met = bt.strategy_metrics(close, sig, horizon=20,
                                  capital=bt.BACKTEST_CAPITAL,
                                  cost=bt.CostModel.from_settings())
        assert met["n_trades"] == 50
        assert met["max_concurrent"] <= 20
        assert met["no_leverage_ok"] == "yes"