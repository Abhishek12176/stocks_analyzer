# Model V2 target research (Task 15)

**Status: research-only.** `backend/app/ml/v2_target_research.py` is not
imported by the production pipeline, API, calibration, or backtest. V1's
`target_ret_20d` and `target_up_20d` remain unchanged.

## Pre-registered candidates

The horizon defaults to 20 observed rows. V1 is the frozen comparison:
`r(T) = Close[T+20] / Close[T] - 1`, with `UP = r > 0` and zero treated as
DOWN. The meaningful-return candidate uses one fixed `r > 0.01` hurdle. One
percent is a deliberately conservative 20-day noise/round-trip-cost screen;
it is documented before evaluation and is **not** selected or searched from
the data. The three-class candidate uses strict boundaries:

* `DOWN` (`-1`) if `r < -0.01`;
* `NEUTRAL` (`0`) if `-0.01 <= r <= 0.01`;
* `UP` (`1`) if `r > 0.01`.

The volatility-adjusted candidate uses `r / sigma_past > 1` as its binary
meaningful-gain label. `sigma_past` is sample standard deviation of the
previous 20 close-to-close returns, ending at `T-1`; the return at `T` and all
future returns are excluded. Undefined warm-up/zero-volatility values remain
`NaN`. The regression candidate is the continuous V1 forward return.

## Evaluation contract

`run_target_research()` uses the same supplied numeric features, seed, global
chronological purged/embargoed folds, and final newest-date holdout for every candidate.
Logistic regression is used for binary/three-class labels and ridge regression
for the continuous return, with training-fold-only median imputation. Results
include label counts, near-zero return buckets, pairwise agreement/correlation,
per-symbol/per-fold/aggregate metrics, holdout boundaries, configuration, and
a SHA-256 fingerprint of the exact panel plus configuration. The target horizon
is used as the fold embargo, and the final training side is purged by the same
horizon before the holdout. `as_of` truncates the panel **before** targets are
built.

## Evidence and classifications

No live market benchmark is claimed here: network-free deterministic tests are
the evidence for this task, and no synthetic accuracy is presented as a market
result. The offline contract suite is:

```
cd backend
python -m pytest tests/test_v2_target_research.py -q
```

It verifies exact threshold boundaries, symbol-local forward windows, PIT
truncation, strict past-volatility isolation, zero/NaN handling, class counts,
diagnostics, reproducibility, and final-holdout isolation. A synthetic frame
is suitable for a smoke run, but its metrics are illustrative only and must
not be used to promote a target.

The recorded deterministic smoke used two symbols and 80 business dates
(160 rows), horizon 3, volatility window 5, feature `feature`, seed 17, eight
walk-forward folds, and 32 final-holdout rows. Its fingerprint was
`2b8309ace6e431ee1ac45c14529a5c5e11718f33f3fa92e865c043c0d8eae678`; no
accuracy or return number from that synthetic frame is treated as market
evidence.

| Target | Classification | Evidence |
|---|---|---|
| Frozen V1 binary | **REFERENCE** | Existing production contract; unchanged. |
| Fixed meaningful-return binary | **INCONCLUSIVE** | Formula and boundary tests pass; no live out-of-sample evidence in this task. |
| Causal volatility-adjusted direction | **INCONCLUSIVE** | Past-only isolation passes; no live out-of-sample evidence in this task. |
| Fixed DOWN/NEUTRAL/UP | **INCONCLUSIVE** | Exact class boundaries and diagnostics pass; no live calibration/economic evidence. |
| Continuous return regression | **INCONCLUSIVE** | Exact continuous target and deterministic ridge comparison are implemented; no live evidence. |

No candidate is `PROMISING` or `REJECTED` without the registered pooled
dataset, purged/embargoed evaluation, calibration, shuffled controls, and
economic accounting from later tasks. In particular, this task does not tune
thresholds, alter the V1 backtest, or select a production target.
