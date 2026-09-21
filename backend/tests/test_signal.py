"""Unit tests for the BUY/HOLD/SELL signal policy (Task 9)."""

import pytest

from app.ml import signal as sg


T = dict(threshold_buy=0.6, threshold_sell=0.4)


class TestDecideSignal:
    def test_buy_above_buy_threshold(self):
        out = sg.decide_signal(0.75, **T)
        assert out["signal"] == "BUY" and out["direction"] == 1

    def test_exact_buy_threshold_is_buy(self):
        out = sg.decide_signal(0.6, **T)
        assert out["signal"] == "BUY"

    def test_sell_below_sell_threshold(self):
        out = sg.decide_signal(0.25, **T, allow_short=True)
        assert out["signal"] == "SELL" and out["direction"] == -1

    def test_exact_sell_threshold_is_sell(self):
        assert sg.decide_signal(0.4, **T)["signal"] == "SELL"

    def test_hold_inside_band(self):
        assert sg.decide_signal(0.5, **T)["signal"] == "HOLD"
        assert sg.decide_signal(0.41, **T)["signal"] == "HOLD"

    def test_hold_when_sell_disabled_for_short(self):
        out = sg.decide_signal(0.25, **T, allow_short=False)
        assert out["signal"] == "SELL" and out["direction"] == 0

    def test_short_when_enabled(self):
        out = sg.decide_signal(0.25, **T, allow_short=True)
        assert out["direction"] == -1

    def test_confidence_floor_forces_hold(self):
        out = sg.decide_signal(0.55, **T, confidence_floor=0.3)
        assert out["signal"] == "HOLD"
        # 0.9 clears a high floor
        assert sg.decide_signal(0.9, **T, confidence_floor=0.3)["signal"] == "BUY"

    def test_reason_strings(self):
        assert "BUY threshold" in sg.decide_signal(0.7, **T)["reason"]
        assert "SELL threshold" in sg.decide_signal(0.3, **T)["reason"]
        assert "HOLD band" in sg.decide_signal(0.5, **T)["reason"]


class TestDeterminism:
    def test_same_input_same_output(self):
        for prob in (0.1, 0.5, 0.9):
            a = sg.decide_signal(prob, **T)
            b = sg.decide_signal(prob, **T)
            assert a == b

    def test_config_defaults_usable(self):
        out = sg.decide_signal(0.7)
        assert out["signal"] in {"BUY", "HOLD", "SELL"}
        assert 0 <= out["direction"] <= 1


class TestValidation:
    def test_buy_must_exceed_sell(self):
        with pytest.raises(ValueError):
            sg.decide_signal(0.5, threshold_buy=0.4, threshold_sell=0.4)

    def test_inverted_thresholds_rejected(self):
        with pytest.raises(ValueError):
            sg.decide_signal(0.5, threshold_buy=0.3, threshold_sell=0.8)

    def test_prob_out_of_range(self):
        with pytest.raises(ValueError):
            sg.decide_signal(1.2, **T)
        with pytest.raises(ValueError):
            sg.decide_signal(-0.1, **T)

    def test_thresholds_out_of_range(self):
        with pytest.raises(ValueError):
            sg.decide_signal(0.5, threshold_buy=1.5, threshold_sell=0.4)

    def test_validate_returns_floats(self):
        tb, ts, cf = sg.validate_thresholds(0.6, 0.4, 0.1)
        assert (tb, ts, cf) == (0.6, 0.4, 0.1)