"""Unit tests for PIT sentiment features (Task 6).

Articles follow the same no-lookahead rule as events: an article published
after the NSE close of day T must not influence the day-T row.
"""

import numpy as np
import pandas as pd
import pytest

from app.config import settings
from app.ml import sentiment_features as se


def _frame() -> pd.DataFrame:
    index = pd.bdate_range("2024-05-06", periods=9)  # Mon 06 -> Thu 16 May 2024
    return pd.DataFrame({"Close": 100.0 + 0.1 * np.arange(len(index))}, index=index)


def _article(published_dt: str, score: float | None = None,
             label: str = "neutral") -> dict:
    article = {"title": "test", "published_dt": published_dt}
    if score is not None:
        article["sentiment"] = {"label": label, "score": score}
    return article


class TestArticlesToEvents:
    def test_missing_published_dt_skipped(self):
        assert se.articles_to_events([{"title": "x"}]) == []

    def test_score_from_sentiment_and_plain(self):
        arts = [
            _article("2024-05-10 12:00", score=0.4),
            {"title": "b", "published_dt": "2024-05-10 12:00", "score": -0.2},
            {"title": "c", "published_dt": "2024-05-10 12:00"},
        ]
        events = se.articles_to_events(arts)
        assert events[0]["_score"] == 0.4
        assert events[1]["_score"] == -0.2
        assert events[2]["_score"] is None


class TestSentimentFeatures:
    def test_no_leak_after_close_article(self):
        arts = [
            _article("2024-05-10 12:00", score=0.3),  # before close -> visible Fri
            _article("2024-05-10 19:00", score=0.5),  # after close  -> NOT on Fri row
        ]
        out = se.add_sentiment_features(_frame(), arts, window=5)
        # day T (Fri): only the morning article is knowable
        assert out.loc["2024-05-10", "sent_recent"] == pytest.approx(0.3)
        assert out.loc["2024-05-10", "sent_score_5d"] == pytest.approx(0.3)
        assert out.loc["2024-05-10", "sent_count_5d"] == 1
        # by Monday both articles are inside the 5-day window (effective Sat + Fri)
        assert out.loc["2024-05-13", "sent_count_5d"] == 2
        assert out.loc["2024-05-13", "sent_score_5d"] == pytest.approx(0.4)

    def test_window_aggregates_hand_computed(self):
        arts = [
            _article("2024-05-10 12:00", score=0.3),
            _article("2024-05-10 19:00", score=0.5),   # effective Sat 11th
            _article("2024-05-13 09:30", score=-0.2),  # effective Mon 13th
        ]
        out = se.add_sentiment_features(_frame(), arts, window=5)
        assert out.loc["2024-05-09", "sent_count_5d"] == 0
        assert out.loc["2024-05-10", "sent_count_5d"] == 1
        assert out.loc["2024-05-10", "sent_score_5d"] == pytest.approx(0.3)
        assert out.loc["2024-05-10", "sent_pos_ratio_5d"] == pytest.approx(1.0)
        assert out.loc["2024-05-13", "sent_count_5d"] == 3
        assert out.loc["2024-05-13", "sent_score_5d"] == pytest.approx(0.2)
        assert out.loc["2024-05-13", "sent_pos_ratio_5d"] == pytest.approx(2 / 3)

    def test_unsaved_article_counts_but_not_scored(self):
        arts = [_article("2024-05-10 12:00")]  # no score
        out = se.add_sentiment_features(_frame(), arts, window=5)
        assert out.loc["2024-05-10", "sent_count_5d"] == 1
        assert np.isnan(out.loc["2024-05-10", "sent_recent"])
        assert np.isnan(out.loc["2024-05-10", "sent_score_5d"])

    def test_no_articles_all_nan_zero(self):
        out = se.add_sentiment_features(_frame(), [], window=5)
        assert (out["sent_count_5d"] == 0).all()
        assert out["sent_recent"].isna().all()

    def test_no_lookahead_via_truncation(self):
        arts = [
            _article("2024-05-09 10:00", score=0.2),
            _article("2024-05-10 12:00", score=0.3),
            _article("2024-05-10 19:00", score=0.5),
            _article("2024-05-13 09:30", score=-0.2),
        ]
        df = _frame()
        full = se.add_sentiment_features(df, arts, window=5)
        k = 5
        truncated = se.add_sentiment_features(df.iloc[:k], arts, window=5)
        pd.testing.assert_frame_equal(full.iloc[:k], truncated)

    def test_feature_version_and_list(self):
        arts = [_article("2024-05-10 12:00", score=0.1)]
        out = se.add_sentiment_features(_frame(), arts)
        assert out.attrs["feature_version"] == settings.ml_feature_version
        cols = se.list_sentiment_features(5)
        assert cols == sorted(set(cols))
        assert {"sent_score_5d", "sent_count_5d", "sent_pos_ratio_5d", "sent_recent"} == set(cols)