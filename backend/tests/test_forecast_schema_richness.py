"""Schema richness test — ForecastResponse must NOT drop pipeline fields.

Regression for the MEDIUM-3 gap: the pipeline computes `discrimination`,
`validation`, `data_fingerprint`/`input_fingerprint` and rich calibration
keys, but the response schema silently stripped them (`extra=ignore`).
These tests prove a pipeline-shaped payload round-trips end-to-end through
the pydantic model (the same path FastAPI's response_model uses).
"""

from app.schemas.forecast import ForecastResponse


def _pipeline_shaped_payload() -> dict:
    return {
        "symbol": "TEST",
        "is_available": True,
        "horizon": 20,
        "generated_at": "2026-09-14T00:00:00Z",
        "versions": {"feature": "v1", "model": "baseline-v1"},
        "latest": {
            "as_of": "2026-09-14",
            "close": 1000.0,
            "raw_probability": 0.5123,
            "probability": 0.5100,
            "signal": "BUY",
            "direction": 1,
            "confidence": 0.5,
            "reason": "test",
        },
        "calibration": {
            "method": "isotonic",
            "n": 320,
            "n_holdout": 88,
            "brier": 0.2412,
            "brier_calibrated": 0.2300,
            "ece": 0.0312,
            "ece_calibrated": 0.0284,
            "brier_fit": 0.2440,
            "brier_calibrated_fit": 0.2300,
            "ece_fit": 0.0302,
            "ece_calibrated_fit": 0.0290,
            "report": {
                "selection_reason": "stable",
                "n_distinct_calibrated_levels": 12,
                "pct_within_0p05_0p50": 0.42,
            },
        },
        "discrimination": {
            "metric": "full-OOF AUC is the primary stable RANKING metric",
            "full_oof_auc": 0.5703,
            "holdout_auc": 0.5100,
            "full_oof_accuracy_raw": 0.55,
            "holdout_accuracy_calibrated": 0.53,
            "holdout_baseline_accuracy": {"always_up": 0.52, "always_down": 0.48},
            "holdout_best_baseline_accuracy": 0.52,
            "holdout_edge_vs_best_baseline_accuracy": 0.01,
            "effective_independent_windows": {"oof": 6, "holdout": 1},
        },
        "validation": {
            "target_formula": "target_ret_20d = Close[T+20]/Close[T] - 1",
            "oof_rows": 408,
            "calibration_fit_rows": 320,
            "holdout_rows": 88,
            "oof_rows_effective_independent": 6,
            "holdout_rows_effective_independent": 1,
            "final_holdout_never_used_for_fitting": True,
        },
        "data_fingerprint": "abcdef1234567890" * 8,
        "input_fingerprint": {
            "data_fingerprint": "abcdef1234567890" * 8,
            "algorithm": "sha256",
            "scope": "upstream_input",
            "as_of": "2026-09-14",
            "sources": {"stock": {"rows": 1240}},
            "reconstruction": "not supported",
        },
        "backtest": [{"model": "ensemble", "n_trades": 88, "cum_return": -0.27}],
        "explanation": None,
        "uncertainty": None,
        "monitor": {"mode": "fresh", "alarms": []},
        "snapshot": {},
    }


def test_forecast_response_preserves_discrimination_and_validation():
    model = ForecastResponse(**_pipeline_shaped_payload())
    dumped = model.model_dump(by_alias=True)

    assert dumped["discrimination"]["full_oof_auc"] == 0.5703
    assert dumped["discrimination"]["effective_independent_windows"]["holdout"] == 1
    assert dumped["validation"]["oof_rows"] == 408
    assert dumped["validation"]["final_holdout_never_used_for_fitting"] is True


def test_forecast_response_preserves_fingerprints():
    model = ForecastResponse(**_pipeline_shaped_payload())
    dumped = model.model_dump(by_alias=True)

    fp = "abcdef1234567890" * 8
    assert dumped["dataFingerprint"] == fp
    assert dumped["inputFingerprint"]["algorithm"] == "sha256"
    assert dumped["inputFingerprint"]["scope"] == "upstream_input"
    assert dumped["inputFingerprint"]["sources"]["stock"]["rows"] == 1240


def test_forecast_response_preserves_rich_calibration():
    model = ForecastResponse(**_pipeline_shaped_payload())
    cal = model.model_dump(by_alias=True)["calibration"]

    assert cal["method"] == "isotonic"
    assert cal["nHoldout"] == 88
    assert cal["brierFit"] == 0.2440
    assert cal["brierCalibratedFit"] == 0.2300
    assert cal["eceFit"] == 0.0302
    assert cal["eceCalibratedFit"] == 0.0290
    assert cal["report"]["n_distinct_calibrated_levels"] == 12


def test_forecast_response_graceful_empty_blocks():
    # Pipeline never emits empty blocks, but a graceful-degraded payload must
    # still serialize without error.
    model = ForecastResponse(
        symbol="TEST",
        is_available=False,
        error="boom",
        horizon=20,
        generated_at="2026-09-14T00:00:00Z",
        discrimination={},
        validation={},
    )
    dumped = model.model_dump(by_alias=True)
    assert dumped["discrimination"] == {}
    assert dumped["validation"] == {}
    assert dumped["calibration"] is None
    assert dumped["dataFingerprint"] is None
    assert dumped["inputFingerprint"] is None