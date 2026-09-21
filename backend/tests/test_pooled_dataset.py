"""Network-free Task 13 tests for the research-only pooled panel."""

import numpy as np
import pandas as pd
import pytest

from app.ml import pooled_dataset as pdset


def _frame(start="2024-01-01", n=8, close_offset=0.0, extra=False):
    index = pd.bdate_range(start, periods=n)
    close = pd.Series(np.arange(n, dtype=float) + 100 + close_offset, index=index)
    out = pd.DataFrame(
        {
            "Close": close,
            "Open": close - 1,
            "Volume": np.arange(n, dtype=float) + 1000,
            "shared": np.arange(n, dtype=float),
        },
        index=index,
    )
    if extra:
        out["only_a"] = 1.0
    return out


def test_manifest_is_sorted_versioned_and_rejects_duplicates():
    manifest = pdset.build_universe_manifest(
        [{"symbol": "zeta", "listing_date": "2024-01-03"}, "AAA"]
    )
    assert manifest["manifest_version"] == pdset.MANIFEST_VERSION
    assert manifest["schema_version"] == pdset.MANIFEST_VERSION
    assert manifest["symbols"] == ["AAA", "ZETA"]
    zeta = manifest["members"][1]
    assert zeta["first_available_date"] == "2024-01-03"
    assert zeta["last_available_date"] is None
    assert zeta["inclusion_status"] == "included"
    assert zeta["exclusion_reason"] is None
    assert manifest["manifest_fingerprint"] == pdset.manifest_fingerprint(manifest)
    with pytest.raises(pdset.PooledDatasetError, match="duplicate"):
        pdset.build_universe_manifest(["AAA", "aaa"])


def test_pool_has_canonical_order_no_duplicate_symbol_dates_and_missingness():
    frames = {"ZZZ": _frame(close_offset=10), "AAA": _frame(extra=True)}
    panel, report = pdset.build_pooled_dataset(frames, horizon=2)
    assert list(panel.columns[:2]) == ["symbol", "date"]
    assert report["columns"] == list(panel.columns)
    assert list(panel[["date", "symbol"]].itertuples(index=False, name=None)) == sorted(
        panel[["date", "symbol"]].itertuples(index=False, name=None)
    )
    assert not panel.duplicated(["symbol", "date"]).any()
    assert report["missingness"]["only_a"]["missing"] == len(frames["ZZZ"])
    assert report["normalization"] == "none"


def test_point_in_time_cutoff_and_member_window_are_enforced():
    frames = {"AAA": _frame(n=8)}
    manifest = pdset.build_universe_manifest(
        [{"symbol": "AAA", "valid_from": "2024-01-04", "valid_to": "2024-01-09"}]
    )
    panel, _ = pdset.build_pooled_dataset(
        frames, manifest, as_of="2024-01-10", horizon=2
    )
    assert panel["date"].min() == pd.Timestamp("2024-01-04")
    assert panel["date"].max() == pd.Timestamp("2024-01-09")
    manifest_at_cutoff = pdset.build_universe_manifest(
        [{"symbol": "AAA", "valid_from": "2024-01-04", "valid_to": "2024-01-09"}],
        as_of="2024-01-10",
    )
    assert manifest_at_cutoff["members"][0]["inclusion_status"] == "excluded"
    assert manifest_at_cutoff["members"][0]["exclusion_reason"] == "outside validity window at as_of"


def test_target_is_existing_forward_target_and_tail_is_unlabelled():
    frame = _frame(n=8)
    panel, report = pdset.build_pooled_dataset({"AAA": frame}, horizon=2)
    target = "target_ret_2d"
    first = panel.iloc[0]
    assert first[target] == pytest.approx(frame["Close"].iloc[2] / frame["Close"].iloc[0] - 1)
    assert panel[target].iloc[-2:].isna().all()
    assert panel["target_up_2d"].iloc[-2:].isna().all()
    assert report["coverage"]["AAA"]["target_rows"] == 6


def test_graceful_per_symbol_failures_and_reproducibility():
    def provider(symbol, period="5y", enrich=False):
        if symbol == "BAD":
            raise RuntimeError("offline source failed")
        return {"ok": True, "features": _frame(close_offset=1 if symbol == "BBB" else 0)}

    manifest = pdset.build_universe_manifest(["BBB", "BAD"])
    first = pdset.run_pooled_dataset(
        manifest=manifest, provider=provider, horizon=2, as_of="2024-01-20", seed=7
    )
    second = pdset.run_pooled_dataset(
        manifest=manifest, provider=provider, horizon=2, as_of="2024-01-20", seed=7
    )
    assert first["report"]["symbols_ok"] == second["report"]["symbols_ok"] == 1
    assert first["report"]["symbols_skipped"] == 1
    assert first["metadata"]["dataset_fingerprint"] == second["metadata"]["dataset_fingerprint"]
    assert first["report"] == second["report"]
    assert first["metadata"]["no_production_integration"] is True


def test_as_of_fingerprint_ignores_rows_after_cutoff():
    first = _frame(n=10)
    second = first.copy()
    second.iloc[-1, second.columns.get_loc("Close")] += 999
    current = [first]

    def provider(symbol, period="5y", enrich=False):
        return {"ok": True, "features": current[0]}

    one = pdset.run_pooled_dataset(
        ["AAA"], provider=provider, horizon=2, as_of="2024-01-10"
    )
    current[0] = second
    second_result = pdset.run_pooled_dataset(
        ["AAA"], provider=provider, horizon=2, as_of="2024-01-10"
    )
    assert one["dataset"].equals(second_result["dataset"])
    assert one["metadata"]["dataset_fingerprint"] == second_result["metadata"]["dataset_fingerprint"]


def test_duplicate_dates_are_reported_per_symbol_without_aborting_pool():
    bad = _frame(n=4)
    bad.index = [bad.index[0], bad.index[1], bad.index[1], bad.index[3]]
    panel, report = pdset.build_pooled_dataset({"BAD": bad, "GOOD": _frame(n=4)})
    assert report["symbols_ok"] == 1
    assert report["skipped"][0]["symbol"] == "BAD"
    assert set(panel["symbol"]) == {"GOOD"}
