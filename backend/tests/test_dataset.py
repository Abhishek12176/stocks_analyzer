"""Unit tests for prediction-target builder + chronological splits (Task 7).

No-leakage contract: a row at day T carries only data <= T in its features;
the forward-return target is the *label* stored at row T and is evaluated
after the horizon. Chronological splits are strictly forward-only.
"""

import numpy as np
import pandas as pd
import pytest

from app.ml import dataset as ds


def _frame(n: int = 100) -> pd.DataFrame:
    idx = pd.bdate_range("2023-01-02", periods=n)
    rng = np.random.default_rng(0)
    close = pd.Series(np.cumprod(1 + rng.normal(0.0005, 0.01, n)), index=idx)
    return pd.DataFrame(
        {
            "Close": close,
            "rsi": 50 + rng.normal(0, 5, n),
            "trend": rng.normal(0, 1, n),
        },
        index=idx,
    )


class TestTarget:
    def test_forward_return_uses_future_close_only(self):
        df = _frame()
        out = ds.add_target(df, horizon=20)
        ret = out["target_ret_20d"]
        i = 10
        expected = df["Close"].iloc[i + 20] / df["Close"].iloc[i] - 1
        assert ret.iloc[i] == pytest.approx(expected)

    def test_tail_rows_are_nan(self):
        df = _frame()
        out = ds.add_target(df, horizon=20)
        assert out["target_ret_20d"].iloc[-20:].isna().all()
        assert out["target_up_20d"].iloc[-20:].isna().all()

    def test_label_sign_matches_forward_return(self):
        df = _frame()
        out = ds.add_target(df, horizon=20)
        ret = out["target_ret_20d"]
        lab = out["target_up_20d"]
        assert lab.max() == ds.UP
        assert lab.min() == ds.DOWN
        assert ((lab == ds.UP) == (ret > 0)).all()

    def test_custom_horizon(self):
        df = _frame()
        out = ds.add_target(df, horizon=5)
        assert "target_ret_5d" in out.columns
        assert "target_up_5d" in out.columns

    def test_no_lookahead_row_values_unchanged(self):
        # build_dataset must NOT shift/rewrite feature rows (only the label).
        df = _frame()
        X, y = ds.build_dataset(df, horizon=20)
        assert X.index.equals(y.index)
        assert "target_ret_20d" not in X.columns
        assert "target_up_20d" not in X.columns
        pd.testing.assert_frame_equal(X, df.reindex(X.index)[X.columns])


class TestBuildDataset:
    def test_drops_incomplete_tail(self):
        df = _frame(n=100)
        X, y = ds.build_dataset(df, horizon=20)
        assert len(X) == len(df) - 20
        assert y.notna().all()

    def test_selected_feature_cols(self):
        df = _frame(n=100)
        X, _ = ds.build_dataset(df, horizon=20, feature_cols=["rsi", "trend"])
        assert list(X.columns) == ["rsi", "trend"]

    def test_default_feature_cols_exclude_targets(self):
        df = _frame(n=100)
        X, _ = ds.build_dataset(df, horizon=10)
        assert not any(c.startswith("target_") for c in X.columns)


class TestSplits:
    def test_split_dates_contiguous(self):
        idx = _frame(n=120).index
        train, test = ds.split_dates(idx, train_end=idx[70])
        assert len(train) == 71                     # inclusive of train_end
        assert len(train) + len(test) == len(idx)
        assert train.max() < test.min()
        assert (train <= idx[70]).all()
        assert (test > idx[70]).all()

    def test_split_dates_with_test_end(self):
        idx = _frame(n=120).index
        train, test = ds.split_dates(idx, train_end=idx[60], test_end=idx[100])
        assert test.min() == idx[61]
        assert test.max() == idx[100]

    def test_walk_forward_strictly_chronological(self):
        idx = _frame(n=200).index
        folds = ds.walk_forward_splits(idx, test_size=20, step=20, min_train=60)
        for train, test in folds:
            assert len(test) == 20
            assert train.max() < test.min()          # no leakage across folds
            assert len(train) + 20 <= len(idx)
        assert folds[0][0].min() == idx[0]
        assert len(folds) >= 4                      # expanding window continues
        assert folds[-1][0].max() < folds[-1][1].min()

    def test_walk_forward_train_grows(self):
        idx = _frame(n=200).index
        folds = ds.walk_forward_splits(idx, test_size=20, step=20, min_train=40)
        train_lens = [len(t) for t, _ in folds]
        assert train_lens == sorted(train_lens)
        assert train_lens[-1] > train_lens[0]

    def test_walk_forward_min_train_respected(self):
        idx = _frame(n=200).index
        folds = ds.walk_forward_splits(idx, test_size=20, min_train=100)
        assert all(len(t) >= 100 for t, _ in folds)