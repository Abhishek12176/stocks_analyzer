# Model V2 research: point-in-time pooled dataset (Task 13)

Task 13 creates research infrastructure only. `backend/app/ml/pooled_dataset.py`
builds a dated panel whose row key is `(symbol, date)` and leaves production V1
unchanged. It reuses the existing causal feature assembly, `dataset.add_target`
and SHA-256 fingerprint helpers.

## Contract

- `universe-manifest-v1` is canonical and sorted by symbol. Duplicate symbols
  are rejected. Members carry explicit symbol/exchange/sector/industry,
  first/last-available dates, inclusion status, and exclusion reason fields
  (nullable when the source does not provide them). Members may carry
  `valid_from`/`valid_to` (listing/delisting dates); those windows are applied
  to every historical row.
- The `as_of` cutoff is inclusive. Rows after it, and rows outside a member's
  validity window, are excluded. A feature frame with duplicate dates is
  rejected for that symbol.
- Feature columns are the deterministic union across symbols. Missing source
  columns remain `NaN`; there is no normalization, imputation, feature
  selection, or model training.
- `target_ret_{h}d` and `target_up_{h}d` are attached through the existing
  forward target helper. The final `h` rows remain unlabeled unless a later
  research step explicitly drops them.
- A failing/unavailable symbol is recorded in `report.skipped` and does not
  abort the other symbols.

## Reproducibility and outputs

`metadata.dataset_fingerprint` is a SHA-256 digest of the canonical manifest,
per-symbol source fingerprints, canonical pooled panel, and run configuration.
The JSON report contains schema/version, coverage, missingness, target coverage,
and skipped-symbol details. Wall-clock timestamps are intentionally excluded
from deterministic metadata. `save_pooled_dataset` writes a CSV panel and a
JSON report under a caller-selected research output directory.

Network-free usage injects a provider returning
`{"ok": True, "features": dataframe}` (or a graceful `{"ok": False,
"error": ...}` response):

```bash
cd backend
python -m app.ml.pooled_dataset --smoke --output-dir ml_pooled_dataset
```

The CLI's default provider calls the existing input builder and may require
network access. Tests use synthetic frames only. This module is not imported
by API routes and does not change V1 forecasts, normalization, training,
calibration, thresholds, or backtest behavior.
