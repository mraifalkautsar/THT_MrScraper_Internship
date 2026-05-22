import numpy as np
import pandas as pd


LAST_PRICE_BASELINE = "last_price_baseline"
HYBRID_LAST_PRICE_ENTITY = "hybrid_last_price_entity"
HYBRID_LAST_PRICE_CALIBRATED_FALLBACK = "hybrid_last_price_calibrated_fallback"
GLOBAL_NO_CALIBRATION = "global_no_calibration"
ENTITY_BLEND_NO_CALIBRATION = "entity_blend_no_calibration"
GLOBAL_CALIBRATED = "global_calibrated"
ENTITY_BLEND_CALIBRATED = "entity_blend_calibrated"
ANCHOR_MODEL_SELECTION = "anchor_model_selection"
SECOND_STAGE_RESIDUAL = "second_stage_residual"

CORE_PREDICTION_VARIANTS = [
    LAST_PRICE_BASELINE,
    HYBRID_LAST_PRICE_ENTITY,
    HYBRID_LAST_PRICE_CALIBRATED_FALLBACK,
    GLOBAL_NO_CALIBRATION,
    ENTITY_BLEND_NO_CALIBRATION,
    GLOBAL_CALIBRATED,
    ENTITY_BLEND_CALIBRATED,
    ANCHOR_MODEL_SELECTION,
]

EXPERIMENTAL_PREDICTION_VARIANTS = [
    SECOND_STAGE_RESIDUAL,
]

ALL_PREDICTION_VARIANTS = CORE_PREDICTION_VARIANTS + EXPERIMENTAL_PREDICTION_VARIANTS


def hybrid_last_price_then_log_prediction(
    df: pd.DataFrame,
    fallback_log_col: str = "blended_pred_log",
) -> pd.Series:
    """Use the latest known entity price when present, otherwise use a fallback log prediction."""
    has_recent_history = df["fallback_entity_history_count"].fillna(0) > 0
    return df["fallback_entity_price_log"].where(has_recent_history, df[fallback_log_col])


def choose_variant_by_anchor_mae(
    day_df: pd.DataFrame,
    target_col: str,
    candidate_log_predictions: dict[str, pd.Series],
    default_variant: str = HYBRID_LAST_PRICE_ENTITY,
) -> str:
    """Pick the candidate with the lowest anchor MAE for a single outage day."""
    anchors = day_df[day_df[target_col].notna()].copy()
    if len(anchors) == 0:
        return default_variant

    scores = []
    y_true = anchors[target_col].astype(float)
    for name, pred_log in candidate_log_predictions.items():
        pred = np.expm1(pred_log.loc[anchors.index]).clip(lower=0)
        mae = (y_true - pred.astype(float)).abs().mean()
        scores.append((name, mae))

    return sorted(scores, key=lambda x: x[1])[0][0]
