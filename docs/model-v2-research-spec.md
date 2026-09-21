# Task 12 — V2 Model Research & Design Specification

**Status:** Design only; no V2 production implementation  
**Date:** 08-Sep-2026  
**Scope:** Research and controlled future-task design for the AVORA / EquityLens 20-trading-day forecast

## 1. Executive summary

V1 is a complete, reproducible production benchmark, but its Task 11 evidence does not justify
shipping a new model, changing weights, lowering thresholds, or adding a regime gate. The central
finding is not simply “the model is inaccurate.” Three distinct layers must be separated:

1. **Predictive information:** aggregate holdout discrimination is near chance and varies materially
   by symbol and snapshot.
2. **Probability quality:** probabilities can collapse near 0.50 or become calibration-boundary
   values, limiting their use as confidence.
3. **Decision translation:** the current long-only policy turns weakly separated probabilities into
   effectively all-long books, which can lose in a down-window even when the model has little edge.

V2 should therefore begin with pooled, point-in-time-safe research infrastructure and richer
relative features, while keeping V1 frozen. Multi-output forecasts and a selective decision layer
should be evaluated only after those foundations demonstrate cross-symbol robustness. Every future
experiment must use chronological, purged, embargoed evaluation and the same frozen benchmark
semantics.

## 2. Evidence from V1

Task 11 attempted 17 symbols: 16 completed and TATAMOTORS was unavailable from the upstream
provider. Median holdout ensemble ROC-AUC was approximately 0.507; 10/16 symbols exceeded 0.50
and 9/16 exceeded 0.55. Fourteen of 16 symbols had negative cumulative return and 14/16 produced
effectively all-long holdout books. In the down-market holdout, median always-up was about -30.5%
while always-down was about +1.4%.

The full-OOF logistic median AUC was approximately 0.587, while voting median AUC was about 0.475.
This is descriptive evidence, not a reason to replace the production ensemble: an earlier controlled
study found L+V winning TCS on one holdout (about 0.701 versus about 0.415) but winning only 3/7
symbols across the study. Fold-level production-ensemble wins were substantially more frequent.

Probability concentration and calibration-boundary behavior are real. ITC triggered a collapse
diagnostic. Calibration stability work showed that degeneracy can be reported and a Platt fallback
can stabilize the mapping, but it cannot create missing predictive information. Fingerprinting
showed that same-code results must be interpreted with the exact upstream input snapshot.

## 3. Confirmed problems

### 3.1 Predictive-information problems

- Aggregate holdout discrimination is near chance.
- Performance is inconsistent across symbols; a single ticker is not representative.
- Results are snapshot-sensitive and likely regime-sensitive.
- Components disagree, but disagreement alone does not identify the best component.
- The observed logistic signal is promising enough to test, not strong enough to promote globally.

### 3.2 Probability problems

- Raw or calibrated probabilities may cluster near 0.50.
- Isotonic maps can have few distinct levels or a boundary collapse.
- Calibration stability does not imply useful ranking or separation.
- Current probabilities do not reliably distinguish actionable edge from uncertainty.

### 3.3 Decision and trading problems

- The current long-only decision path converts borderline probabilities into positions.
- Effectively all-long books are exposed to the market direction and transaction costs.
- Negative trading results can arise from the decision layer even when classification is ambiguous.
- Trade count and cumulative return must not be used as substitutes for predictive skill.

## 4. Findings that must not trigger changes

- One-stock improvements do not justify global model or weight changes.
- A single negative snapshot does not prove the features are useless.
- A higher full-OOF score is not evidence of holdout improvement.
- A regime-aware drift gate is not justified by saturated/inconsistent PSI alarms.
- Probability smoothing, arbitrary threshold lowering, or changing costs to improve returns is not
  a valid fix.
- The TCS 88-versus-67/69 trade-count difference was input-snapshot drift, not a code defect.
- TATAMOTORS' upstream 404 is a data-availability record, not fabricated training evidence.

## 5. V2 design principles

V2 must:

1. Use only information available at prediction time; no lookahead.
2. Enforce point-in-time availability for prices, fundamentals, news, events, and universes.
3. Keep the final chronological holdout evaluation-only.
4. Persist fingerprint, seed, versions, data sources, and exact evaluation windows.
5. Represent unavailable data explicitly; never fabricate or silently backfill.
6. Prohibit final-holdout selection, feature selection, threshold selection, and calibration fitting.
7. Prefer cross-symbol and cross-period robustness over one-stock gains.
8. Treat calibration quality and separation as first-class outcomes.
9. Judge decisions economically, with costs, exposure, drawdown, and uncertainty.
10. Preserve V1 as an immutable benchmark and run it beside every V2 candidate.

## 6. Candidate architecture A — Cross-symbol pooled learning

Train one panel model on dated stock observations instead of one independent model per stock.
The panel row is `(symbol, date)` and contains stock, market, sector, and relative features.

### Representation

- Include a stable symbol identifier only as a controlled categorical or learned effect; compare
  with no-identity and sector-only variants.
- Encode sector using a point-in-time classification table, not today's sector membership.
- Prefer market-relative and sector-relative features so the model can learn portable relationships.
- Include liquidity/history masks and row availability indicators where missingness is informative.

### Splitting and data quality

- Split dates globally: all symbols on a test date remain in the test side.
- Train only on dates strictly before the test window, then purge labels overlapping the test period
  and embargo at least the 20-day horizon.
- Permit unequal histories; require minimum per-symbol observations and report coverage.
- Construct the universe point-in-time to avoid survivorship bias. Symbols entering later must not
  appear before listing/coverage dates; delisted names should remain when historical data exists.
- Never let a future symbol label, sector membership, or revised fundamental enter an earlier row.

### Why it may help

Pooling increases effective sample size: one symbol contributes only a limited number of 20-day
labels, while a panel can expose the learner to many independent-ish stock-date examples across
sectors and regimes. It may learn common market/relative relationships while retaining symbol
effects.

### Risks and complexity

Cross-sectional dependence reduces the nominal sample size; a broad market shock can make many rows
share one label. Dominant symbols or sectors can overwhelm smaller groups. Identity encoding can
memorize symbols, and an improperly time-split panel leaks same-date information. Complexity is
high: point-in-time universe construction, panel alignment, grouped metrics, and missingness tests
are prerequisites. This is the highest-priority architectural experiment, not an automatic
replacement.

## 7. Candidate architecture B — Relative and cross-sectional features

These are candidates only; none are implemented by Task 12.

| Family | Source and expected signal | PIT requirements | Redundancy/cost | Failure mode |
|---|---|---|---|---|
| Stock vs NIFTY | Stock excess return and beta; separates market direction from idiosyncratic strength | Same-date aligned closes; no future index revisions | Existing relative returns/beta overlap; low incremental cost | Correlation changes by regime |
| Stock vs sector | Excess return against sector index or point-in-time peer basket | Sector membership and index constituents must be historical | Medium data and alignment cost | Sector proxy may be stale or unavailable |
| Peer rank | Cross-sectional return/risk rank among eligible peers | Universe must be known at each date; compute ranks using that date only | Medium; overlaps momentum | Survivorship and sector concentration |
| Momentum rank | 20/60/120-day percentile across stocks | Require common as-of date and minimum history | Medium; likely overlaps alpha momentum | Rank instability in thin panels |
| Relative volatility | Stock volatility divided by market/sector/peer volatility | Both windows must end at T | Low/medium; overlaps vol ratio | Denominator near zero |
| Relative drawdown | Stock drawdown versus sector/market drawdown | Trailing-only windows | Medium; may duplicate max drawdown | Different calendars and recovery speeds |
| Relative strength | Price ratio and trend of stock/benchmark | Only prices through T | Low/medium | Trend reversal and nonstationarity |
| Sector momentum | Sector return, breadth, and dispersion | Historical constituents/weights | Medium | Sector ETF/index proxy bias |

Each family needs truncation-equivalence tests, missing-source tests, and an ablation that measures
incremental value rather than accepting all correlated columns.

## 8. Candidate architecture C — Multi-output forecasting

V2 should conceptually expose three distinct quantities:

- **Direction:** `P(return_20d > 0)`.
- **Expected return:** `E(return_20d)`.
- **Risk/uncertainty:** volatility, downside risk, and/or predictive interval.

Separate models are the safest first experiment: classification, regression, and risk estimation
have different losses and calibration requirements. A shared representation with separate heads can
be tested later, but it increases coupling and leakage risk during model selection. A sequential
probability-plus-regression design is interpretable but must not condition later targets on
holdout-selected predictions.

Regression should model the same 20-day close-to-close return and use robust losses or winsorization
rules defined on training data only. Risk should be evaluated as a forecast (for example, interval
coverage or downside calibration), not merely as a feature. Combining probability and expected
return is useful only when both are independently validated; trading return alone cannot choose the
architecture.

## 9. Candidate architecture D — Better decision layer

The decision layer should not simply lower the current threshold. It should require sufficient
estimated edge after costs and uncertainty:

1. Estimate direction probability, expected return, and risk from training-only-fitted components.
2. Convert probability to directional edge relative to a neutral prior.
3. Subtract estimated round-trip transaction costs and a volatility/uncertainty penalty.
4. Apply a minimum evidence rule: ambiguous probabilities near 0.50 remain HOLD.
5. Enforce exposure, turnover, liquidity, and concurrent-position limits.
6. Use regime only as a declared conditioning input if out-of-sample validation proves benefit;
   never as an unvalidated emergency gate.

BUY requires positive net expected value and a separation/uncertainty condition. SELL in the
long-only product remains flat unless shorting is explicitly designed and separately validated.
HOLD is the correct outcome for low edge, high uncertainty, unavailable inputs, or insufficient
history. All rules and parameters must be selected inside training/validation folds, never from the
final holdout. The required acceptance test is a reduction in borderline all-long exposure without
destroying cross-symbol skill or creating excessive turnover.

## 10. Candidate architecture E — Ensemble options

- **Static weighted ensemble:** simplest and robust, but cannot respond to component quality.
- **Validation-derived weights:** potentially useful; weights must be fit in inner chronological
  folds and regularized toward equal weights.
- **Stacking/meta-model:** can learn interactions, but has the highest leakage and overfitting risk;
  meta-features must be strictly out-of-fold.
- **Conditional weighting:** may handle symbol/sector/regime differences, but multiplies selection
  degrees of freedom and needs enough samples per condition.
- **Regime-aware weighting:** currently unsupported by evidence because drift monitors are saturated;
  remains exploratory only after a predefined regime experiment.

No new weights are selected in Task 12. Any future ensemble change must beat frozen V1 on the
cross-symbol scorecard, survive shuffled controls, and show stable gains across chronological folds.

## 11. Candidate targets

The production target remains:

`target_ret_20d = Close[T+20] / Close[T] - 1`  
`target_up_20d = target_ret_20d > 0`

Alternatives for controlled research:

| Target | Solves | Information lost / balance | Calibration and business meaning | Leakage risk |
|---|---|---|---|---|
| Positive threshold `r > c` | Ignores noise and tiny moves | Discards small positive/negative distinctions; class balance changes with c | Probability means chance of meaningful gain; c must be training-fold defined | Threshold chosen on holdout leaks |
| Volatility-adjusted `r/vol > c` | Makes outcomes comparable across risk levels | Less direct rupee-return meaning; unstable vol estimates | Probability is risk-normalized direction; needs careful calibration | Future volatility must not enter T |
| Multi-class buckets | Separates loss/flat/gain | More labels and lower per-class sample size | Requires multiclass calibration and clearer action mapping | Bucket cutoffs must be fixed or fold-fit |
| Direction + magnitude | Preserves sign and size | More complex joint evaluation | Supports expected-return decisions | Joint labels must use the same horizon |
| Return regression | Directly estimates expected return | Sensitive to outliers and weakly calibrated tails | Natural for net-edge decisions | Robust scaling/loss fit only on train |

No alternative is promoted without class-balance, calibration, and economic evaluation against the
unchanged binary target.

## 12. Validation methodology

Every V2 candidate uses expanding chronological walk-forward folds. For each fold, training dates
precede validation/test dates; labels whose forward windows overlap the test start are purged; an
embargo of at least the 20-day horizon follows training. Any hyperparameter, feature choice,
calibration, threshold, or ensemble weight is fit inside training/inner-validation data only.

The newest predeclared 20% of OOF observations is the final holdout. It is touched once for the
headline report and never for selection. Cross-symbol tests use global date splits and also report
per-symbol coverage, with no same-date leakage through ranks or pooled normalization.

Report per symbol and aggregate median, mean, standard deviation, interquartile range, and pass
counts. Include ROC-AUC/rank metrics, Brier, ECE, calibration curves, distinct probability levels,
boundary-collapse share, expected calibration error by probability bin, and uncertainty intervals
where reasonable (bootstrap by date block or symbol, not iid rows when dependence matters).

Economic reports must use the frozen marked-to-market ledger, cost model, exposure rules, and
metric-group provenance. Include buy-and-hold, always-up, always-down, seeded random, and
within-training-fold shuffled-target controls on the same holdout. Include installed model versus
model comparisons, but do not rank candidates by cumulative return alone.

Selection must require evidence across symbols, chronological folds, and regimes. Trade count is a
diagnostic and coverage constraint, never a success metric by itself.

## 13. V2 success criteria

A candidate is acceptable only if the predeclared balanced scorecard improves or preserves all
critical dimensions:

- **Predictive skill:** better or non-inferior holdout AUC/rank ordering, positive median
  cross-symbol improvement, and no concentration of gains in one ticker.
- **Calibration:** lower or non-inferior Brier/ECE, more useful separation, lower collapse frequency,
  and stable seed/fold behavior.
- **Economic usefulness:** improved median per-bet return/profit factor and risk-adjusted metrics,
  controlled drawdown, and sensible comparison with buy-and-hold and baselines.
- **Robustness:** symbol coverage, fold consistency, regime coverage, seed sensitivity, and exact
  fingerprint reproducibility.
- **Decision quality:** materially fewer borderline all-long positions, no hidden leverage, no
  unacceptable turnover, and HOLD used for genuinely uncertain cases.

Failure of any critical safety rule, holdout isolation, reproducibility, or no-fabrication rule
rejects the candidate regardless of an attractive single metric.

## 14. Experiment priority

### Tier 1

1. **Pooled panel infrastructure.** Hypothesis: shared samples improve cross-symbol skill. Data:
   point-in-time universe, existing features, fingerprints. Benefit: sample size and portability.
   Risk: dependence and dominant symbols. Leakage: global date/rank leakage. Complexity: high.
   Success: robust median/dispersion improvement over V1; failure: one-symbol or one-regime gain.
2. **Relative/cross-sectional feature blocks.** Hypothesis: excess strength separates market beta
   from stock alpha. Data: historical index/sector/peer panels. Benefit: portable signal. Risk:
   redundancy. Leakage: future membership. Complexity: medium/high. Success: incremental OOF value
   across symbols; failure: no incremental value or unstable ranks.
3. **Decision-layer simulation using frozen predictions.** Hypothesis: selective HOLD reduces
   all-long exposure without changing model information. Data: existing OOF probabilities and
   expected costs. Benefit: isolates translation problem. Risk: overfitting policy. Leakage:
   holdout threshold selection. Complexity: medium. Success: fewer borderline positions with
   non-inferior calibrated/economic scorecard; failure: return-only improvement.

### Tier 2

4. **Expected-return regression and risk estimate.** Hypothesis: magnitude and uncertainty improve
   net-edge decisions. Risk: noisy tails and incompatible losses. Complexity: medium.
5. **Regularized validation-derived ensemble weights.** Hypothesis: stable component quality differs
   by context. Risk: weight overfit. Complexity: medium/high. Success requires cross-symbol stability.
6. **Target sensitivity study.** Compare fixed meaningful-return and multiclass targets without
   changing production. Risk: class imbalance and business ambiguity. Complexity: medium.

### Tier 3

7. Conditional/regime-aware weighting, stacking/meta-models, learned symbol embeddings, and shared
   multi-task heads. These may be valuable, but current evidence does not justify their complexity.
   They require larger panel data and stronger inner-validation controls.

## 15. Recommended future task sequence

1. **Task 13:** Build point-in-time pooled dataset and universe manifest; no production wiring.
2. **Task 14:** Add relative/cross-sectional feature research blocks with causal tests and ablations.
3. **Task 15:** Run target alternatives as isolated offline experiments.
4. **Task 16:** Evaluate pooled candidate models with frozen V1 comparison.
5. **Task 17:** Evaluate expected-return and risk heads. **Complete; KEEP FROZEN V1.**
6. **Task 18:** Design and validate selective probability/edge decision policies.
   **Complete as research-only; production remains frozen V1.**
7. **Task 19:** Validate the frozen entropy selective-forecast candidate across symbols,
   chronological folds, seeds, calibration, economic metrics, and shuffled controls.
   **Complete as research-only; classification is EVIDENCE STILL INSUFFICIENT.**
8. **Future Task:** Evaluate regularized ensemble/meta-model options with nested walk-forward
   validation only with explicit approval.
9. **Future Task:** Run complete V2 versus frozen V1 report and shadow evaluation; only then consider
   separately approved production implementation tasks.

Each task must be independently testable, update the handoff, and stop before the next task starts.

## 16. Frozen V1 baseline

V1 is permanently defined by the Task 11 benchmark artifacts and current production implementation.
Every V2 comparison must use the same symbol universe manifest, date windows, final-holdout
procedure, target (where applicable), purging/embargo, marked-to-market accounting, transaction
cost model, rounding/provenance semantics, and input-fingerprint metadata. Benchmark code and
artifacts must be versioned rather than silently regenerated under changed defaults.

The comparison must answer `V2 vs V1` on identical snapshots whenever possible. If upstream data
differs, the report must state both fingerprints and must not present the result as a clean model
comparison.

## 17. Risks and failure modes

- Cross-sectional leakage through future constituents, revised fundamentals, ranks, or same-date
  normalization.
- Survivorship bias and missing/delisted stocks.
- Panel dependence overstating effective sample size.
- Symbol identity memorization and sector imbalance.
- Calibration collapse being hidden by smoothing.
- Decision rules overfitting one bearish or bullish window.
- Regression tails and risk estimates destabilizing the decision layer.
- More flexible ensembles converting fold noise into apparent skill.
- Data-provider drift making non-identical snapshots look comparable.

Every failure must be reported with availability, coverage, warning, and fingerprint metadata.

## 18. Explicit non-goals

Task 12 does not implement V2, change production features/models/weights/thresholds/calibration,
change targets, modify backtest accounting or monitoring, alter API behavior, add new selection
experiments, or claim improved accuracy. It does not remove RF/XGB, promote logistic, add a regime
gate, lower thresholds, or replace the current V1 forecast.
