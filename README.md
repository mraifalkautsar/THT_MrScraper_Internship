# MrScraper Price Intelligence Take-Home

This repository reconstructs missing marketplace prices during a simulated scraping outage. The pipeline uses historical product/shop/category behavior, a global CatBoost model, entity-level price priors, and optional anchor-based day calibration.

## Problem Setup

The target column is `price`. Training data contains historical prices. The test file contains outage-day rows where most prices are missing, plus a small set of known anchor prices. The pipeline must fill only the missing prices while preserving known anchor prices.

## Approach

The production-style pipeline is implemented in `src/` and can be run from `scripts/`.

1. Load train and test data.
2. Create row-level features from time, discount, stock, ratings, comments, and shop metadata.
3. Create leakage-safe historical price aggregates from training history only.
4. Train a global CatBoost regressor on `log1p(price)`.
5. Predict log prices for outage rows.
6. Blend the global model prediction with a fallback entity historical median.
7. Use known same-day anchor prices to estimate calibration residuals.
8. Fill missing prices and run output sanity checks.

The notebook remains useful as an exploratory/report artifact, but the runnable pipeline is now script-based.

## Feature Groups

Raw categorical features:

- `shopId`
- `itemId`
- `modelId`
- `promotionId`
- `cat_id`
- `brand`

Boolean features:

- `is_free_shipping`
- `is_pre_order`
- `is_official_shop`
- `is_verified`
- `is_preferred_plus_seller`

Row-level engineered features:

- time features: `hour`, `dayofweek`, `day`, `month`, `is_weekend`
- discount features: `has_discount`, `discount_ratio`, `price_before_log`, `raw_discount_log`, `show_discount_ratio`
- stock features: `stock_ratio`, `stock_gap`, `is_low_stock`
- engagement features: `total_rating_count_log`, `cmt_count_log`, `shop_follower_count_log`, `review_strength`, `shop_strength`

Historical aggregate features are computed for:

- `modelId`
- `itemId`
- `shopId`
- `cat_id`
- `brand`

For each key, the pipeline computes historical log-price:

- count
- mean
- median
- standard deviation
- minimum
- maximum

The pipeline also creates:

- historical metadata imputations for slow-moving blank test fields: product metadata
  (`cat_id`, `brand`, shipping/pre-order flags, review/comment counts) is recovered from
  recent `modelId`/`itemId` history, and shop metadata (`shop_rating`,
  `shop_response_rate`, `shop_follower_count`, seller flags) is recovered from recent
  `shopId` history.
- recent-price features for `modelId` and `itemId`: last observed price, last 3-observation median, last 7-observation median, prior count, and hours since last seen.
- `fallback_entity_price_log`: most specific available recent/entity prior, falling back from `modelId` last price to `itemId` last price, then historical medians from broader groups.
- `fallback_entity_history_count`: matching historical evidence count, used to decide how strongly to trust the fallback entity price.

The CatBoost model is trained on fields available in hidden test rows plus leakage-safe
recent-history features and historically recoverable metadata. Volatile same-day fields
such as promotion, discount, stock, and item min/max price remain excluded. Target-derived
broad historical aggregates are still used by the explicit fallback strategy, but they are
excluded from CatBoost features to avoid training self-leakage.

## Leakage Prevention

Historical features and metadata imputations are fitted only on history available before the
validation or prediction period. During validation, held-out validation days are excluded
before computing these features. This prevents hidden validation prices and same-day hidden
metadata from leaking into their own features.

## Entity Blending

Entity blending combines two different signals:

- the global CatBoost prediction, which can generalize from row features, shop behavior,
  category/brand behavior, and broader marketplace patterns
- the entity prior, which is the most specific historical price evidence available for the
  same product/listing

The entity prior is stored as `fallback_entity_price_log`. It is selected in order of
specificity: recent `modelId` price first, then recent `itemId` price, then broader
historical medians if exact recent evidence is unavailable. The matching amount of evidence
is stored in `fallback_entity_history_count`.

The blend is computed in log-price space:

```text
entity_weight = count / (count + entity_smoothing)
blended_pred_log =
    (1 - entity_weight) * model_pred_log
    + entity_weight * fallback_entity_price_log
```

The default `entity_smoothing` is `20`, meaning 20 historical observations gives the
entity prior 50% weight. More history pushes the prediction closer to the entity prior;
little or no history leaves the prediction closer to the global model. For example:

```text
count = 0   -> entity_weight = 0.00
count = 5   -> entity_weight = 0.20
count = 20  -> entity_weight = 0.50
count = 100 -> entity_weight = 0.83
```

This matters because the test set is mostly repeated products. In that setting, exact recent
entity prices are often stronger than a learned model. The blend still keeps the model useful
for sparse-history rows, but it avoids over-trusting the model when a product has substantial
price history.

## Anchor Calibration

Anchor calibration uses the known prices inside the outage/test file as same-day reference
points. These anchors are never used as training labels for the hidden rows. They are used
after prediction to ask whether a candidate prediction family is systematically too high or
too low on that outage day.

For each outage day, known anchor prices are compared against candidate predictions in
log-price space:

```text
residual_log = log1p(anchor_price) - predicted_log_price
```

The median residual is used as the global day-level correction. If the anchors show that
the model is generally 2% too low on a given date, the calibrated prediction can shift the
hidden rows upward by roughly that amount in log space.

The calibration also estimates residuals by `cat_id`, `shopId`, `brand`, and `promotionId`,
with shrinkage toward the global day residual:

```text
weight = anchor_count / (anchor_count + calibration_smoothing)
segment_delta = weight * segment_median_residual
              + (1 - weight) * global_day_residual
```

The default `calibration_smoothing` is `8`, also tunable. A segment with many anchors can
move toward its own residual estimate. A segment with only one or two anchors is pulled back
toward the broader day-level residual so that a small noisy anchor group does not dominate.

Calibration is intentionally applied cautiously. The validation results show that broad
calibration improves weaker global/model fallback variants slightly, but it can damage rows
where the latest entity price is already extremely accurate. For that reason, the final
strategy keeps strong recent-entity rows unchanged and uses anchors only to decide fallback
behavior for weaker-history rows.

The validation pipeline keeps the following strategy families separate:

**Approach 1: Global Marketplace Model**

- `global_no_calibration`: pure CatBoost global model trained across all shops/items/models.
- `global_calibrated`: global CatBoost model plus anchor-based day/segment calibration.

**Approach 2: Shop/Product-Level Model**

Approach 2 is not a separate model per product. It is an entity-aware family of strategies
that conditions predictions on `modelId`, `itemId`, and `shopId` history. This matches the
assignment setting because test products are expected to have prior training history.

- `last_price_baseline`
  - Uses the most specific latest historical price available: `modelId` last price first,
    then `itemId` last price, then broader historical medians.
  - Ignores CatBoost and anchors.
  - Strongest validation baseline because most tracked product prices are stable across adjacent scrape days.
  - Most vulnerable when a product has a real same-day promotion, price correction, or very sparse history.

- `entity_blend_no_calibration`
  - Combines the global CatBoost prediction with the historical entity prior.
  - The blend weight is based on entity history count:
    `entity_weight = count / (count + entity_smoothing)`.
  - Rows with many prior observations lean toward the entity prior; sparse rows lean more toward the global model.
  - This is the main explicit Tier 2 model because it conditions predictions on product/shop history while retaining a global fallback.

- `entity_blend_calibrated`
  - Starts from `entity_blend_no_calibration`.
  - Uses same-day anchors to estimate residual corrections globally and by `cat_id`,
    `shopId`, `brand`, and `promotionId`.
  - Tests whether anchors reveal a systematic outage-day shift.
  - Validation shows calibration helps weaker model variants slightly, but it still does not
    beat direct recent entity prices.

- `hybrid_last_price_uncalibrated_fallback`
  - Uses latest entity price directly whenever prior `modelId` or `itemId` history exists.
  - Uses the entity-blend model only for fallback rows.
  - Protects stable products from being moved away from a very strong last-known-price prior.
  - Appropriate when product history is frequent and prices are sticky.

- `hybrid_last_price_calibrated_fallback`
  - Same as `hybrid_last_price_uncalibrated_fallback` for strong-history rows.
  - Applies anchor calibration only to fallback rows where the model/entity blend is used.
  - Avoids the main failure mode of broad calibration: degrading rows whose recent entity price was already accurate.
  - Basically like the previous hybrid strategy, but calibrated with day-level correction.

- `anchor_gated_fallback_calibration`
  - Current default final strategy.
  - Keeps recent entity prices unchanged.
  - For weak-history fallback rows, uses fallback-like anchors to choose whether calibrated or uncalibrated fallback predictions are more reliable on that outage day.
  - This is the most conservative anchor-use strategy: anchors influence fallback behavior without overriding strong entity priors.

- `second_stage_residual`
  - It learns a same-day residual correction from anchor residual
  features. 
  - Details are in the section below because this strategy is more complex than
  the median-residual calibration variants.

**Meta / Diagnostic Strategy**

- `anchor_model_selection`: chooses the best candidate strategy per day based on anchor MAE. It can select either global or entity variants, so it is treated as a comparison/meta-strategy rather than a pure Approach 1 or Approach 2 model.
- Selective calibration can be enabled with `--selective-calibration` to skip calibration when anchor evidence is weak or noisy.

The main strategy is now gated fallback calibration, because validation and anchor EDA show
that same-entity recent prices are much stronger than broad model correction for most outage
rows. Anchors are therefore used conservatively: they calibrate or select fallback behavior
instead of overriding strong recent entity priors.

## Second-Stage Residual Details

`second_stage_residual` is one of the strategies above. It is a learned
version of anchor calibration: instead of applying only a median day/segment residual, it
fits a small model that predicts a residual correction for each hidden row.

The first-stage prediction is still `blended_pred_log`. The second stage learns this target
on same-day anchor rows:

```text
target_residual_log = log1p(anchor_price) - blended_pred_log
second_stage_pred_log = blended_pred_log + predicted_residual_log
```

The model is:

```text
StandardScaler() + Ridge(alpha=second_stage_alpha)
```

It is trained separately for each outage day using only that day's known anchor rows. It is
not trained on hidden rows, and hidden validation prices are never available during fitting.
The default `second_stage_alpha` is `10.0`; larger values make the residual model more
conservative.

The second-stage feature set is limited to information available for both anchors and hidden
rows on the same outage day:

- base prediction features: `model_pred_log`, `blended_pred_log`, `fallback_entity_price_log`
- entity confidence features: `fallback_entity_history_count`, `entity_weight`, recent
  prior counts, recent last-price features, and hours since last seen
- global anchor residual features: median residual, mean residual, residual standard
  deviation, residual IQR, anchor count, and log anchor count
- segment anchor residual features for `cat_id`, `shopId`, `brand`, and `promotionId`:
  segment median residual, segment anchor count, log count, shrinkage weight, shrunk
  residual, and segment-minus-global residual

The segment residual features use the same shrinkage idea as day-level calibration:

```text
segment_weight = segment_anchor_count / (segment_anchor_count + calibration_smoothing)
shrunk_segment_delta =
    segment_weight * segment_median_residual
    + (1 - segment_weight) * global_day_residual
```

This lets the Ridge model learn patterns such as "shop-level anchor residuals are useful
when there are enough shop anchors" while still seeing the broader day-level residual.

Safeguards:

- If a day has fewer than `second_stage_min_anchors` valid anchors, the model skips fitting
  and returns the first-stage prediction unchanged.
- If `calibration_delta_cap` is set, predicted residual corrections are clipped to that
  absolute log-delta cap.
- The model is included in validation as a diagnostic strategy, but it is not the default
  prediction strategy because current validation still favors recent entity prices.

## Run Validation

Install dependencies first:

```bash
pip install -r requirements.txt
```

Run the full validation:

```bash
python scripts/run_validation.py
```

Run validation with safer calibration controls:

```bash
python scripts/run_validation.py \
  --calibration-delta-cap 0.05 \
  --selective-calibration \
  --calibration-min-anchors 50 \
  --calibration-max-residual-iqr 0.10 \
  --calibration-min-abs-global-delta 0.01
```

Run validation while changing the learned second-stage residual settings:

```bash
python scripts/run_validation.py \
  --second-stage-alpha 10 \
  --second-stage-min-anchors 10
```

Run assignment-aligned validation that keeps only rows with prior `modelId` or `itemId` history before the validation day:

```bash
python scripts/run_validation.py \
  --validation-days 16 \
  --anchors-per-day 100 \
  --test-like-only \
  --output-dir outputs/validation_test_like
```

For a quick smoke test:

```bash
python scripts/run_validation.py --sample-rows 5000 --validation-days 1 --anchors-per-day 20 --iterations 20
```

Validation simulates the production outage with a chronological split:

1. The most recent validation dates are held out from training.
2. The model and all historical features are built only from dates before the validation
   window.
3. On each validation date, a small number of rows are sampled as known anchors.
4. The remaining rows are treated like hidden outage rows: their prices are hidden from
   feature generation, calibration, and prediction.
5. Predictions are evaluated against the hidden true prices only after the prediction step.

This setup is intentionally stricter than a random split. A random split would let future or
same-day prices influence historical aggregates for nearby rows. The chronological split
matches the real task: at prediction time, only prior training history plus known same-day
anchors are available.

By default, validation also masks hidden-row fields that are blank in the real test file,
such as discount, category, brand, promotion, rating, and item min/max price fields. The
feature builder then recovers only slow-moving historical metadata from pre-validation
history. Volatile same-day fields remain unavailable because they cannot be trusted for real
hidden test rows.

The `--test-like-only` option keeps only validation rows with prior `modelId` or `itemId`
history before the validation day. This better matches the assignment statement that the
remaining test products have appeared in training before, and should be used for the main
interview-facing validation check.

Outputs:

- `outputs/validation_results.csv`
- `outputs/validation_summary.csv`
- `outputs/validation_segments.csv`
- `outputs/validation_predictions.csv`

`validation_results.csv` contains one row per date/strategy evaluation. `validation_summary.csv`
aggregates those rows by strategy. `validation_segments.csv` reports metrics by date, price
bucket, history-count bucket, and calibration status. The segment file is the main diagnostic
artifact for understanding whether errors come from sparse history, expensive products, or a
specific calibration decision. `validation_predictions.csv` contains row-level validation
predictions and errors by strategy, which is used by the EDA notebook to inspect worst
misses and compare which strategy wins on each hidden row.

## Tune Hyperparameters

Tune CatBoost settings plus the two post-processing smoothing constants:

```bash
python scripts/tune_hyperparameters.py
```

For a quick smoke test:

```bash
python scripts/tune_hyperparameters.py \
  --sample-rows 5000 \
  --validation-days 1 \
  --anchors-per-day 20 \
  --learning-rates 0.05 \
  --depths 6 \
  --iterations 5 \
  --entity-smoothing 10,20 \
  --calibration-smoothing 4,8 \
  --output-dir outputs/smoke_tuning
```

The tuning script writes:

- `outputs/tuning/tuning_summary.csv`
- `outputs/tuning/tuning_details.csv`

The grid can be controlled from the command line:

```bash
python scripts/tune_hyperparameters.py \
  --learning-rates 0.03,0.05,0.08 \
  --depths 6,8,10 \
  --iterations 800,1200 \
  --entity-smoothing 5,10,20,50,100 \
  --calibration-smoothing 2,4,8,16,32
```

For faster random-search tuning over a large grid, cap the number of combinations:

```bash
python scripts/tune_hyperparameters.py \
  --learning-rates 0.03,0.05,0.08 \
  --depths 6,8,10 \
  --iterations 400,800,1200 \
  --entity-smoothing 5,10,20,50,100 \
  --calibration-smoothing 2,4,8,16,32 \
  --max-trials 12
```

This makes `entity_smoothing` and `calibration_smoothing` validation-tuned instead of purely hand-picked. Calibration caps and non-MAE selection metrics are still available as experimental options, but they are disabled in the default workflow because the current validation did not justify using them.

## Predict Test Prices

Run:

```bash
python scripts/predict_test.py
```

Choose a final prediction variant:

```bash
python scripts/predict_test.py --prediction-variant hybrid_last_price_uncalibrated_fallback
python scripts/predict_test.py --prediction-variant hybrid_last_price_calibrated_fallback
python scripts/predict_test.py --prediction-variant anchor_gated_fallback_calibration
python scripts/predict_test.py --prediction-variant last_price_baseline
python scripts/predict_test.py --prediction-variant entity_blend_no_calibration
python scripts/predict_test.py --prediction-variant entity_blend_calibrated
python scripts/predict_test.py --prediction-variant anchor_model_selection
python scripts/predict_test.py --prediction-variant second_stage_residual
```

For a faster smoke run:

```bash
python scripts/predict_test.py --iterations 20 --prediction-variant anchor_gated_fallback_calibration
```

Output:

- `outputs/completed_test_predictions.csv`

The script checks that:

- row count matches the input test file
- all prices are filled
- no prices are negative
- known anchor prices are preserved

## Current Validation Summary

The current full validation run in `outputs/validation_summary.csv` simulates an outage on
the last three available training dates (`2025-03-20`, `2025-03-21`, `2025-03-22`) with
100 anchors sampled per day and hidden-row fields masked to match the real test file.

Average hidden-row metrics:

```text
anchor_gated_fallback_calibration: MAE 38,129  | RMSE 0.96M  | MAPE 0.142%
last_price_baseline:               MAE 38,129  | RMSE 0.96M  | MAPE 0.142%
hybrid_last_price_uncalibrated..:  MAE 38,129  | RMSE 0.96M  | MAPE 0.142%
entity_blend_calibrated:           MAE 332,932 | RMSE 3.45M  | MAPE 0.559%
entity_blend_no_calibration:       MAE 336,883 | RMSE 3.52M  | MAPE 0.580%
global_calibrated:                 MAE 1.00M   | RMSE 10.77M | MAPE 1.648%
global_no_calibration:             MAE 1.01M   | RMSE 10.95M | MAPE 1.707%
```

MAPE is reported as a percentage in the code. For example, `0.142` means `0.142%`
average percentage error, not `14.2%`.

Per-day MAE for the final gated strategy:

```text
2025-03-20: 17,203
2025-03-21: 71,138
2025-03-22: 26,046
```

### Approach Comparison

**Approach 1: Global marketplace model.** The global CatBoost model is a useful baseline and sparse-history fallback, but it is not competitive when exact entity history exists. Anchor calibration improves it slightly (`1.01M` MAE to `1.00M` MAE), which confirms anchors contain some same-day correction signal, but the remaining error is much larger than entity-history methods.

**Approach 2: Shop/product/entity model.** Entity-level history dominates. The best-performing methods use the latest available `modelId`/`itemId` price directly and fall back only when recent entity evidence is weak. This is expected for tracked marketplace products with sticky prices and frequent historical observations.

**Anchor strategy.** Broad anchor calibration is not applied to strong recent-entity prices
because validation shows those prices are already more accurate than calibrated model outputs.
The default `anchor_gated_fallback_calibration` keeps recent entity prices unchanged and uses fallback-like anchors to choose calibrated or uncalibrated model fallback behavior for weaker history rows.

### Segment Insights

The best strategy remains accurate across price buckets:

```text
price_q1: MAE 596    | MAPE 0.010%
price_q2: MAE 39,940 | MAPE 0.287%
price_q3: MAE 32,340 | MAPE 0.131%
price_q4: MAE 89,089 | MAPE 0.134%
```

High-price rows naturally contribute more to RMSE and MAE, but their percentage error remains
low. This is why RMSE can look large even when MAPE is strong.

### Outliers And Missing Values

Prices are modeled in log space with `log1p(price)` to reduce the effect of extreme price
values. Predictions are converted back with `expm1`, clipped to non-negative values, and rounded
for the final CSV. I do not remove high-price rows aggressively because marketplace extremes can
be valid products rather than data errors; instead, validation reports price-bucket segments so
outlier impact is visible.

Hidden-row fields that are blank in the real test file are masked during validation. Slow-moving
metadata is recovered from prior history with time-aware as-of lookups; volatile same-day fields
such as discount, promotion, stock, and item min/max price remain unavailable.

### Reproducibility And Model Artifacts

No model artifact is committed. The prediction script retrains deterministically from the provided
CSV files using the fixed `random_seed` in `PipelineConfig`, then writes
`outputs/completed_test_predictions.csv`. This keeps the repository lightweight while still
satisfying reproducibility through code and pinned dependencies.

## Criticism And Next Improvements

The pipeline structure is defensible for this task, but there are clear improvements:

- Tune `entity_smoothing` and `calibration_smoothing` with rolling time validation.
- Add more robust fallback behavior for true cold-start variants that have no prior `modelId`
  history but do have `itemId` history.
- Evaluate and report by shop/category in addition to the existing date, price-bucket, and
  history-count segments.
- Promote or remove the learned second-stage residual model after more rolling validation.
- Compare CatBoost with LightGBM or XGBoost.
- Save model artifacts if repeated inference is required.
