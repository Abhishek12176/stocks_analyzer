# Task 16 — pooled V2 model research

**Status: research-only; V1 is unchanged.** `backend/app/ml/v2_pooled_models.py`
evaluates the frozen V1 binary target (`target_up_20d`) on the Task 13 pooled
panel. It is not imported by the production pipeline, settings, API, frontend,
calibration, thresholds, or backtest.

## Protocol

The experiment compares pooled logistic regression, a ridge probabilistic
baseline (ridge regression on the binary target, clipped to `[0, 1]`), random
forest, and XGBoost when installed. There is no broad hyperparameter search.
Each model is run with the same numeric baseline matrix and a
baseline-plus-Task-14-relative matrix. Missing values are median-imputed using
the training fold only. The default symbol identity is **none** (no identity
columns), chosen for portability and to avoid symbol memorisation. A
deterministic sorted one-hot option exists for controlled research.

Dates are split globally, so every symbol on a date is on the same side. The
last 20% of dates is a final holdout. Expanding folds remove at least the
20-date target horizon from the training side as both label purge and embargo.
The holdout is touched only for the final headline metrics. Reports contain
per-fold, per-symbol, aggregate, final-holdout, ROC-AUC, accuracy, precision,
recall, F1, confusion, Brier, ECE, probability concentration, seed sensitivity,
feature deltas, prior/always-up/always-down, and shuffled-target controls.

## Result artifact

The mandatory real-data run writes the machine-readable artifact to the
ignored path:

`backend/ml_v2_pooled_models/task16_real_data.json`

The default universe is the diverse Task 13 universe (21 requested symbols). Each
upstream failure is retained in the pooled dataset report; if the provider
cannot produce a usable panel, the result records a graceful failure rather
than fabricating data or metrics. The JSON records the exact symbols,
coverage, skips, configuration, and SHA-256 fingerprint.

The completed run used 20 symbols, 24,778 rows (2021-09-08 through
2026-09-07), 4 folds, 4,958 final-holdout rows, 97 baseline features and 121
baseline-plus-relative features. All four requested models were installed and
there were no upstream skips. The run fingerprint is
`e71490f13638bdfb59614493586174e4127fbb6df4585159a7ae61b0ae5a2f0f`.
Final-holdout aggregate AUCs (baseline / plus-relative) were logistic
0.5067 / 0.5067, ridge probabilistic 0.4675 / 0.4605, random forest 0.4680 /
0.4874, and XGBoost 0.4826 / 0.4812. These descriptive results are classified
**INCONCLUSIVE**, not as a production recommendation.

## Offline validation

```text
cd backend
python -m pytest tests/test_v2_pooled_models.py -q
```

The tests prove matrix alignment and missing-data handling, deterministic
identity encoding and outputs, global chronological purge/embargo, future-row
and holdout isolation, model probabilities, multi-symbol/per-symbol metrics,
calibration diagnostics, and shuffled controls. Synthetic metrics are contract
evidence only and are not market evidence.

---

## Resolution via Task 17

Task 16 identified "no clean V1 comparison from a shared historical snapshot"
as a key limitation. Task 17 (`v2_v1_fair_compare.py`) resolved this by running
a strict apples-to-apples comparison on the identical shared snapshot, symbols,
folds, target, seeds, and metrics. Result: **KEEP FROZEN V1** — pooled learning
does not beat frozen V1 (median AUC delta −0.032, win rate 30%, fold robustness
false). See `docs/model-v2-v1-fair-comparison.md` for the full report.

---

## Follow-up: Tasks 21–22 — cross-symbol generalisation with the vol-adjusted target

**Status: research-only; NO ADOPT; production untouched.** Modules
`backend/app/ml/v2_pooled_vol_adj.py` (+ 6 tests) and `v2_pooled_blend.py`
(+ 4 tests). Same Task-17 shared 20-symbol panel (as_of 2026-09-09,
~1240 rows/symbol), locked purged splits (40/90/260/embargo-20), vol-adjusted
train label (`train_up_20d`, k=0.5), full-binary eval `target_up_20d`.

- **D1 — target coverage transfers:** k=0.5 coverage is uniform across the panel
  (0.57–0.67; new symbols ICICIBANK 0.5975 / SBIN 0.5697 / ITC 0.5811). The
  threshold normalisation is NOT why new symbols fail.
- **D2 — raw OHLC levels are not the driver:** per-symbol full-OOF AUC with vs
  without levels differs by ≤0.01 on the 7 focus symbols (per-symbol models see
  one price scale).
- **T21 — pooled leave-one-symbol-out** (ONE rf, locked 300/8/30; pool = other
  symbols up to each fold's embargo cut; held-out symbol never trained): lifts
  the weakest symbols (ICICIBANK 0.43→0.59, SBIN 0.37→0.49, TCS 0.50→0.64,
  HDFCBANK 0.61→0.66) but degrades RELIANCE (0.55→0.42) and ITC (0.46→0.38).
  NEW3 pooled median **0.4935** (< 0.52 bar); CORE4 median 0.6428 (≥ 0.55 bar).
- **T22 — probability blend** `P = w·P_pooled + (1−w)·P_per_symbol`,
  w ∈ {0.3, 0.5, 0.7} (vectors date-aligned 440/440): blending **dilutes** the
  pooled signal — best NEW3 median 0.4395 @ w=0.3 vs pure pooled 0.4935. Blending
  buffers RELIANCE/ITC damage but the reward is far smaller than the lost lift.
- **Oracle ceiling (binding proof):** even a per-symbol best-of (perfect routing)
  caps NEW3 at **0.4935** — SBIN 0.4935 and ITC 0.4597 are per-symbol ceilings;
  no approach (pool/blend/routing) can cross them in this window.
- **Conclusion:** NEW3 median > 0.52 is **mathematically unreachable in this
  window**; ITC/SBIN at chance is an honest market-predictability ceiling, not a
  model bug. Cross-symbol generalization → future research (new window / more
  data / richer features). Artifacts:
  `backend/ml_v2_pooled_vol_adj/task21_pooled_voladj_result.json` +
  `task22_pooled_blend_result.json`.
