"""Unit tests for ML settings in app.config (Task 1)."""

from app.config import settings


class TestMLSettings:
    def test_horizon_is_20(self):
        assert settings.ml_horizon == 20

    def test_feature_and_model_versions(self):
        assert settings.ml_feature_version
        assert settings.ml_model_version

    def test_cost_model_values(self):
        assert settings.ml_cost_brokerage_inr >= 0
        assert settings.ml_cost_fee_percent >= 0
        assert settings.ml_cost_slippage_bps >= 0