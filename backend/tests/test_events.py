"""Unit tests for the PIT event layer (Task 6).

The critical invariant: an event that happens *after* the close of day T must
NOT be visible to the day-T prediction row — it only surfaces from day T+1.
"""

import numpy as np
import pandas as pd
import pytest

from app.config import settings
from app.ml import events as ev


def _frame() -> pd.DataFrame:
    index = pd.bdate_range("2024-05-06", periods=9)  # Mon 06 -> Thu 16 May 2024
    return pd.DataFrame({"Close": 100.0 + 0.1 * np.arange(len(index))}, index=index)


def _event(event_timestamp: str, **kw) -> dict:
    event: dict = {"event_timestamp": event_timestamp}
    event.update(kw)
    return event


class TestEffectiveDate:
    def test_before_close_same_day(self):
        e = _event("2024-05-10 12:00")
        assert ev.effective_date(e) == pd.Timestamp("2024-05-10")

    def test_at_close_same_day(self):
        e = _event("2024-05-10 15:30")
        assert ev.effective_date(e) == pd.Timestamp("2024-05-10")

    def test_after_close_next_day(self):
        e = _event("2024-05-10 19:00")
        assert ev.effective_date(e) == pd.Timestamp("2024-05-11")

    def test_utc_aware_after_close(self):
        # 13:30 UTC == 19:00 IST -> next day; 10:00 UTC == 15:30 IST -> same day
        assert ev.effective_date(_event("2024-05-10 13:30:00+00:00")) == pd.Timestamp("2024-05-11")
        assert ev.effective_date(_event("2024-05-10 10:00:00+00:00")) == pd.Timestamp("2024-05-10")

    def test_explicit_effective_date_wins(self):
        e = _event("2024-05-10 12:00", effective_date="2024-05-15")
        assert ev.effective_date(e) == pd.Timestamp("2024-05-15")

    def test_available_timestamp_overrides_event_timestamp(self):
        e = _event("2024-05-10 12:00", available_timestamp="2024-05-10 19:30")
        assert ev.effective_date(e) == pd.Timestamp("2024-05-11")

    def test_missing_timestamps_raises(self):
        with pytest.raises(ValueError):
            ev.effective_date({})


class TestVisibility:
    def test_no_leak_event_after_close_not_visible_on_day_T(self):
        e = _event("2024-05-10 19:00")  # Friday after close -> effective Sat 11th
        assert ev.is_visible(e, "2024-05-10") is False            # day T: NOT visible
        assert ev.is_visible(e, "2024-05-13") is True             # next trading day
        assert ev.is_visible(e, "2024-05-11") is True             # effective date itself

    def test_event_before_close_visible_on_day_T(self):
        e = _event("2024-05-10 09:30")
        assert ev.is_visible(e, "2024-05-10") is True
        assert ev.is_visible(e, "2024-05-09") is False

    def test_filter_visible(self):
        before = _event("2024-05-09 10:00")
        after_close = _event("2024-05-10 19:00")
        out = ev.filter_visible([before, after_close], "2024-05-10")
        assert out == [before]

    def test_filter_visible_next_day_includes_after_close(self):
        before = _event("2024-05-09 10:00")
        after_close = _event("2024-05-10 19:00")
        out = ev.filter_visible([before, after_close], "2024-05-13")
        assert after_close in out and before in out


class TestEventFeatures:
    def _events(self):
        return [
            _event("2024-05-10 09:30"),  # Fri before close  -> effective 10th
            _event("2024-05-10 19:00"),  # Fri after  close  -> effective 11th (Sat)
            _event("2024-05-13 09:30"),  # Mon before close  -> effective 13th
        ]

    def test_event_today_no_leak(self):
        df = _frame()
        out = ev.add_event_features(df, self._events(), window=5)
        # day T (2024-05-10): only the morning event is visible -> 1
        assert out.loc["2024-05-10", "event_today"] == 1
        # weekend event (Sat 11th) surfaces on Monday 13th together with 13th's own
        assert out.loc["2024-05-13", "event_today"] == 1

    def test_count_window_hand_computed(self):
        df = _frame()
        out = ev.add_event_features(df, self._events(), window=5)
        assert out.loc["2024-05-09", "event_count_5d"] == 0
        assert out.loc["2024-05-10", "event_count_5d"] == 1
        # by Wed 15th all three events are inside the 5-day window
        assert out.loc["2024-05-15", "event_count_5d"] == 3

    def test_future_event_never_counts(self):
        df = _frame()
        out = ev.add_event_features(df, [_event("2024-05-31 19:00")], window=5)
        assert (out["event_count_5d"] == 0).all()
        assert (out["event_today"] == 0).all()

    def test_no_lookahead_via_truncation(self):
        df = _frame()
        events = self._events()
        full = ev.add_event_features(df, events, window=5)
        k = 5
        truncated = ev.add_event_features(df.iloc[:k], events, window=5)
        pd.testing.assert_frame_equal(full.iloc[:k], truncated)

    def test_feature_version_attached(self):
        out = ev.add_event_features(_frame(), self._events())
        assert out.attrs["feature_version"] == settings.ml_feature_version

    def test_feature_list_deterministic(self):
        assert ev.list_event_features(window=7) == ["event_count_7d", "event_today"]