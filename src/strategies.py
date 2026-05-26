import numpy as np
import pandas as pd


LAST_PRICE_BASELINE = "last_price_baseline"
HYBRID_LAST_PRICE_UNCALIBRATED_FALLBACK = "hybrid_last_price_uncalibrated_fallback"
HYBRID_LAST_PRICE_ENTITY = "hybrid_last_price_entity"
HYBRID_LAST_PRICE_CALIBRATED_FALLBACK = "hybrid_last_price_calibrated_fallback"
ANCHOR_GATED_FALLBACK_CALIBRATION = "anchor_gated_fallback_calibration"
GLOBAL_NO_CALIBRATION = "global_no_calibration"
ENTITY_BLEND_NO_CALIBRATION = "entity_blend_no_calibration"
GLOBAL_CALIBRATED = "global_calibrated"
ENTITY_BLEND_CALIBRATED = "entity_blend_calibrated"
ANCHOR_MODEL_SELECTION = "anchor_model_selection"
SECOND_STAGE_RESIDUAL = "second_stage_residual"

CORE_PREDICTION_VARIANTS = [
    LAST_PRICE_BASELINE,
    HYBRID_LAST_PRICE_UNCALIBRATED_FALLBACK,
    HYBRID_LAST_PRICE_CALIBRATED_FALLBACK,
    ANCHOR_GATED_FALLBACK_CALIBRATION,
    GLOBAL_NO_CALIBRATION,
    ENTITY_BLEND_NO_CALIBRATION,
    GLOBAL_CALIBRATED,
    ENTITY_BLEND_CALIBRATED,
    ANCHOR_MODEL_SELECTION,
]

EXPERIMENTAL_PREDICTION_VARIANTS = [
    SECOND_STAGE_RESIDUAL,
]

LEGACY_PREDICTION_VARIANTS = [
    HYBRID_LAST_PRICE_ENTITY,
]

ALL_PREDICTION_VARIANTS = (
    CORE_PREDICTION_VARIANTS
    + EXPERIMENTAL_PREDICTION_VARIANTS
    + LEGACY_PREDICTION_VARIANTS
)


def log_predictions_to_price(pred_log: pd.Series) -> pd.Series:
    """Convert log-price predictions back to nonnegative raw prices."""
    return np.expm1(pred_log).clip(lower=0)


def hybrid_last_price_then_log_prediction(
    df: pd.DataFrame,
    fallback_log_col: str = "blended_pred_log",
) -> pd.Series:
    """Use the latest known entity price when present, otherwise use a fallback log prediction."""
    has_recent_history = df["fallback_entity_history_count"].fillna(0) > 0
    return df["fallback_entity_price_log"].where(has_recent_history, df[fallback_log_col])


def has_recent_entity_history(df: pd.DataFrame) -> pd.Series:
    """Return rows where the fallback entity prior is supported by history."""
    return df["fallback_entity_history_count"].fillna(0) > 0


def score_anchor_mae(
    day_df: pd.DataFrame,
    target_col: str,
    candidate_log_predictions: dict[str, pd.Series],
    anchor_mask: pd.Series | None = None,
) -> list[tuple[str, float]]:
    """Score candidate log predictions by MAE on visible anchor rows."""
    anchors = day_df[day_df[target_col].notna()].copy()
    if anchor_mask is not None:
        anchors = anchors[anchor_mask.reindex(anchors.index).fillna(False)]
    if len(anchors) == 0:
        return []

    y_true = anchors[target_col].astype(float)
    scores = []
    for name, pred_log in candidate_log_predictions.items():
        pred = log_predictions_to_price(pred_log.loc[anchors.index])
        mae = (y_true - pred.astype(float)).abs().mean()
        scores.append((name, mae))
    return sorted(scores, key=lambda x: x[1])


def choose_variant_by_anchor_mae(
    day_df: pd.DataFrame,
    target_col: str,
    candidate_log_predictions: dict[str, pd.Series],
    default_variant: str = HYBRID_LAST_PRICE_UNCALIBRATED_FALLBACK,
) -> str:
    """Pick the candidate with the lowest anchor MAE for a single outage day."""
    scores = score_anchor_mae(day_df, target_col, candidate_log_predictions)
    return scores[0][0] if scores else default_variant


def choose_fallback_variant_by_anchor_mae(
    day_df: pd.DataFrame,
    target_col: str,
    candidate_log_predictions: dict[str, pd.Series],
    fallback_anchor_mask: pd.Series,
    default_variant: str = ENTITY_BLEND_NO_CALIBRATION,
) -> str:
    """Pick the fallback-only candidate with the lowest MAE on fallback-like anchors."""
    scores = score_anchor_mae(
        day_df,
        target_col,
        candidate_log_predictions,
        anchor_mask=fallback_anchor_mask,
    )
    return scores[0][0] if scores else default_variant
