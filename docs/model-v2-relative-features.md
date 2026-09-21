# V2 Relative Features (research-only)

**Feature version:** `v2-relative-research-1`  
**Schema:** `v2-relative-panel-1`  
**Manifest:** `v2-relative-manifest-1`

`backend/app/ml/v2_relative_features.py` builds candidate features for a
long `(symbol, date)` panel. It is deliberately not imported by the V1
forecast pipeline, settings, models, API, or frontend. It does not train a
model, choose a holdout, or claim predictive value.

## Contract and mappings

`build_relative_features(panel, ...)` requires `symbol`, `date`, and `Close`.
`market_close` is an optional same-date market close. `sector` may be on each
row, supplied as a static `{symbol: sector}` mapping, or supplied by a
point-in-time mapping DataFrame. Mapping rows can have inclusive
`valid_from`/`valid_to`; a manifest uses the same fields. Rows outside a
manifest validity window are retained for audit but all relative features are
`NaN`. This prevents a current constituent or current sector from being
silently projected into history.

The output is a copy, sorted by normalized date then symbol, with unique
`(symbol, date)` keys. Its `attrs` contain the feature/schema versions,
manifest, minimum group size, coverage report, and a deterministic input/output
fingerprint.

## Formulas

For symbol `i` and horizon `h`, `R_i(T,h) = Close_i(T) /
Close_i(T-h) - 1`, where `T-h` means the h-th prior **observed trading row**
for that symbol. Missing closes produce missing returns.

* **Market excess:** `R_i(T,h) - R_m(T,h)`.
* **Market relative strength:** `(1 + R_i(T,h)) / (1 + R_m(T,h)) - 1`.
* **Sector return:** the arithmetic mean of eligible, non-missing
  `R_i(T,h)` values in the same `(T, sector)` group.
* **Sector excess:** `R_i(T,h) - sector_return(T,h)`.
* **Peer median divergence:** `R_i(T,h) - median(R_j(T,h))`, excluding `i`
  and using same-date peers in the same sector by default. An explicit
  `peer_group_col` can select an industry or basket instead. This is
  intentionally different from sector excess (median versus mean and
  leave-one-out).
* **Cross-sectional/sector rank:** midrank percentile
  `(1 + #less + 0.5 * #ties) / n`. **Peer rank** excludes the target's value
  set and uses `(1 + #less + 0.5 * #ties) / (n_peers + 1)`, keeping the
  target's percentile on `[0, 1]`. Ties therefore receive the same value and
  are deterministic; there is no arbitrary symbol-order tie break.

The default horizons are 1, 5, and 20 observations. A group must have at
least three valid members by default. Peer calculations exclude the target
and consequently require at least two valid peers with the default. The
threshold is configurable and recorded in the output metadata.

## Point-in-time and missing-data guarantees

* Every rolling/trailing operation is backward-looking and uses `pct_change`
  with `fill_method=None`; no forward-fill, backfill, interpolation, or
  imputation is performed.
* A cross-sectional rank and group statistic uses only valid values on the
  row's date. A later stock or later date cannot affect an earlier rank.
  A same-date close change can legitimately change that date's rank.
* Market features use the existing causal `market_features.excess_return` and
  `relative_strength_ratio` helpers, aligned on common non-missing dates and
  reindexed without fabrication.
* Weekends/exchange holidays are absent rows, not zero-return rows. Horizons
  count observations, so truncating a panel at date `T` produces exactly the
  same features for all rows through `T`.
* Insufficient history, missing benchmark, missing sector, missing peer, and
  thin groups remain `NaN`. Coverage diagnostics report the resulting loss;
  they do not hide it.

## Redundancy and research use

Market excess/strength overlap existing V1 market-relative returns and beta;
sector excess overlaps sector momentum; ranks and peer divergence overlap
alpha momentum and can be unstable in thin universes. These columns are
therefore candidates for an explicitly pre-registered ablation, not an
automatic feature expansion. The module contains no fitting or holdout
selection and must be evaluated with global chronological, purged, embargoed
panel splits and shuffled-target controls before any production discussion.
