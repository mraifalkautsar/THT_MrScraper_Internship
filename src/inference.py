import numpy as np
import pandas as pd

from .calibration import (
    calibrate_all_days,
    second_stage_residual_calibrate_all_days,
)
from .config import PipelineConfig
from .data import assert_submission_valid, load_data
from .features import build_features, get_cat_feature_indices, get_feature_columns
from .model import add_model_predictions, train_global_model
from .strategies import (
    ALL_PREDICTION_VARIANTS,
    ANCHOR_GATED_FALLBACK_CALIBRATION,
    ANCHOR_MODEL_SELECTION,
    ENTITY_BLEND_CALIBRATED,
    ENTITY_BLEND_NO_CALIBRATION,
    GLOBAL_CALIBRATED,
    GLOBAL_NO_CALIBRATION,
    HYBRID_LAST_PRICE_CALIBRATED_FALLBACK,
    HYBRID_LAST_PRICE_ENTITY,
    HYBRID_LAST_PRICE_UNCALIBRATED_FALLBACK,
    LAST_PRICE_BASELINE,
    SECOND_STAGE_RESIDUAL,
    choose_fallback_variant_by_anchor_mae,
    choose_variant_by_anchor_mae,
    has_recent_entity_history,
    hybrid_last_price_then_log_prediction,
    log_predictions_to_price,
)


def _calibrated_candidate_logs(
    day_df: pd.DataFrame, config: PipelineConfig
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.Series]]:
    cal_global = calibrate_all_days(
        day_df.assign(blended_pred_log=day_df["model_pred_log"]),
        config,
        pred_log_col="blended_pred_log",
    )
    cal_blend = calibrate_all_days(day_df, config, pred_log_col="blended_pred_log")
    candidates = {
        LAST_PRICE_BASELINE: day_df["fallback_entity_price_log"],
        HYBRID_LAST_PRICE_UNCALIBRATED_FALLBACK: hybrid_last_price_then_log_prediction(
            day_df, "blended_pred_log"
        ),
        HYBRID_LAST_PRICE_CALIBRATED_FALLBACK: hybrid_last_price_then_log_prediction(
            cal_blend, "calibrated_pred_log"
        ),
        GLOBAL_NO_CALIBRATION: day_df["model_pred_log"],
        ENTITY_BLEND_NO_CALIBRATION: day_df["blended_pred_log"],
        GLOBAL_CALIBRATED: cal_global["calibrated_pred_log"],
        ENTITY_BLEND_CALIBRATED: cal_blend["calibrated_pred_log"],
    }
    return cal_global, cal_blend, candidates


def add_final_prediction_by_variant(test_feat: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    df = test_feat.copy()
    variant = config.prediction_variant

    if variant == LAST_PRICE_BASELINE:
        df["predicted_price"] = log_predictions_to_price(df["fallback_entity_price_log"])
        return df

    if variant in {HYBRID_LAST_PRICE_UNCALIBRATED_FALLBACK, HYBRID_LAST_PRICE_ENTITY}:
        pred_log = hybrid_last_price_then_log_prediction(df, "blended_pred_log")
        df["predicted_price"] = log_predictions_to_price(pred_log)
        return df

    if variant == GLOBAL_NO_CALIBRATION:
        df["predicted_price"] = log_predictions_to_price(df["model_pred_log"])
        return df

    if variant == ENTITY_BLEND_NO_CALIBRATION:
        df["predicted_price"] = log_predictions_to_price(df["blended_pred_log"])
        return df

    if variant == GLOBAL_CALIBRATED:
        calibrated = calibrate_all_days(
            df.assign(blended_pred_log=df["model_pred_log"]),
            config,
            pred_log_col="blended_pred_log",
        )
        calibrated["predicted_price"] = log_predictions_to_price(
            calibrated["calibrated_pred_log"]
        )
        return calibrated

    if variant == ENTITY_BLEND_CALIBRATED:
        calibrated = calibrate_all_days(df, config, pred_log_col="blended_pred_log")
        calibrated["predicted_price"] = log_predictions_to_price(
            calibrated["calibrated_pred_log"]
        )
        return calibrated

    if variant == HYBRID_LAST_PRICE_CALIBRATED_FALLBACK:
        calibrated = calibrate_all_days(df, config, pred_log_col="blended_pred_log")
        pred_log = hybrid_last_price_then_log_prediction(calibrated, "calibrated_pred_log")
        calibrated["predicted_price"] = log_predictions_to_price(pred_log)
        return calibrated

    if variant == ANCHOR_GATED_FALLBACK_CALIBRATION:
        selected_days = []
        for _, day_df in df.groupby("date"):
            day_df = day_df.copy()
            _, _, candidates = _calibrated_candidate_logs(day_df, config)
            fallback_candidates = {
                name: candidates[name]
                for name in [
                    ENTITY_BLEND_NO_CALIBRATION,
                    ENTITY_BLEND_CALIBRATED,
                    GLOBAL_NO_CALIBRATION,
                    GLOBAL_CALIBRATED,
                ]
            }
            recent_mask = has_recent_entity_history(day_df)
            selected = choose_fallback_variant_by_anchor_mae(
                day_df,
                config.target,
                fallback_candidates,
                fallback_anchor_mask=~recent_mask,
            )
            day_df["selected_fallback_variant"] = selected
            pred_log = day_df["fallback_entity_price_log"].where(
                recent_mask,
                fallback_candidates[selected].loc[day_df.index],
            )
            day_df["predicted_price"] = log_predictions_to_price(pred_log)
            selected_days.append(day_df)
        return pd.concat(selected_days, axis=0).sort_index()

    if variant == SECOND_STAGE_RESIDUAL:
        calibrated = second_stage_residual_calibrate_all_days(
            df, config, pred_log_col="blended_pred_log"
        )
        calibrated["predicted_price"] = log_predictions_to_price(
            calibrated["second_stage_pred_log"]
        )
        return calibrated

    if variant == ANCHOR_MODEL_SELECTION:
        selected_days = []
        for _, day_df in df.groupby("date"):
            day_df = day_df.copy()
            _, _, candidates = _calibrated_candidate_logs(day_df, config)
            selected = choose_variant_by_anchor_mae(day_df, config.target, candidates)
            day_df["selected_anchor_variant"] = selected
            day_df["predicted_price"] = log_predictions_to_price(
                candidates[selected].loc[day_df.index]
            )
            selected_days.append(day_df)
        return pd.concat(selected_days, axis=0).sort_index()

    raise ValueError(
        "prediction_variant must be one of: " + ", ".join(ALL_PREDICTION_VARIANTS)
    )


def train_and_predict(
    train_df: pd.DataFrame, test_df: pd.DataFrame, config: PipelineConfig
) -> pd.DataFrame:
    if config.prediction_variant == LAST_PRICE_BASELINE:
        test_feat = build_features(train_df, test_df, config)
    else:
        train_feat = build_features(train_df, train_df, config)
        feature_cols = get_feature_columns(train_feat, config)
        cat_idx = get_cat_feature_indices(train_feat, feature_cols, config)
        model = train_global_model(train_feat, feature_cols, cat_idx, config)

        test_feat = build_features(train_df, test_df, config)
        test_feat = add_model_predictions(model, test_feat, feature_cols, config)

    test_feat["_row_id"] = np.arange(len(test_feat))
    test_predicted = add_final_prediction_by_variant(test_feat, config)
    test_predicted = test_predicted.sort_values("_row_id").reset_index(drop=True)

    submission = test_df.copy()
    missing_mask = submission[config.target].isna()
    submission.loc[missing_mask, config.target] = test_predicted.loc[
        missing_mask.to_numpy(), "predicted_price"
    ].to_numpy()
    submission[config.target] = submission[config.target].round().astype("int64")
    assert_submission_valid(test_df, submission, config)
    return submission


def run_prediction(config: PipelineConfig) -> pd.DataFrame:
    train_df, test_df = load_data(config.train_path, config.test_path)
    submission = train_and_predict(train_df, test_df, config)
    config.output_dir.mkdir(exist_ok=True)
    output_path = config.output_dir / "completed_test_predictions.csv"
    submission.to_csv(output_path, index=False)
    return submission
