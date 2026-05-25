import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .calibration import (
    calibrate_day_from_anchor_residuals,
    second_stage_residual_calibrate_day,
)
from .config import PipelineConfig
from .data import basic_preprocess
from .features import (
    HIDDEN_UNAVAILABLE_RAW_COLS,
    build_features,
    get_cat_feature_indices,
    get_feature_columns,
)
from .model import add_model_predictions, train_global_model
from .strategies import (
    ANCHOR_GATED_FALLBACK_CALIBRATION,
    ENTITY_BLEND_CALIBRATED,
    ENTITY_BLEND_NO_CALIBRATION,
    GLOBAL_CALIBRATED,
    GLOBAL_NO_CALIBRATION,
    HYBRID_LAST_PRICE_CALIBRATED_FALLBACK,
    HYBRID_LAST_PRICE_UNCALIBRATED_FALLBACK,
    LAST_PRICE_BASELINE,
    SECOND_STAGE_RESIDUAL,
    choose_fallback_variant_by_anchor_mae,
    choose_variant_by_anchor_mae,
    has_recent_entity_history,
    hybrid_last_price_then_log_prediction,
    log_predictions_to_price,
)


def evaluate_predictions(
    y_true: pd.Series, y_pred: pd.Series, name: str
) -> dict[str, float | int | str]:
    eval_df = pd.DataFrame({"y_true": y_true.astype(float), "y_pred": y_pred.astype(float)})
    eval_df = eval_df.replace([np.inf, -np.inf], np.nan).dropna()

    if len(eval_df) == 0:
        return {"model": name, "MAE": np.nan, "RMSE": np.nan, "MAPE": np.nan, "n_eval": 0}

    y_true_clean = eval_df["y_true"].values
    y_pred_clean = eval_df["y_pred"].values
    nonzero = y_true_clean != 0
    mape = (
        np.mean(np.abs((y_true_clean[nonzero] - y_pred_clean[nonzero]) / y_true_clean[nonzero]))
        * 100
        if nonzero.sum() > 0
        else np.nan
    )

    return {
        "model": name,
        "MAE": mean_absolute_error(y_true_clean, y_pred_clean),
        "RMSE": np.sqrt(mean_squared_error(y_true_clean, y_pred_clean)),
        "MAPE": mape,
        "n_eval": len(eval_df),
    }


def make_prediction_frame(
    rows: pd.DataFrame,
    y_pred: pd.Series,
    variant: str,
    date: str,
    target_col: str,
    calibration_meta: pd.DataFrame | None = None,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "date": date,
            "variant": variant,
            "y_true": rows[target_col].astype(float).values,
            "y_pred": pd.Series(y_pred).astype(float).values,
            "fallback_entity_history_count": rows["fallback_entity_history_count"].values,
        },
        index=rows.index,
    )
    id_cols = [
        "shopId",
        "itemId",
        "modelId",
        "cat_id",
        "brand",
        "model_pred_log",
        "blended_pred_log",
        "fallback_entity_price_log",
        "entity_weight",
        "modelId_recent_prior_count",
        "itemId_recent_prior_count",
        "modelId_hours_since_last_seen",
        "itemId_hours_since_last_seen",
    ]
    for col in id_cols:
        if col in rows.columns:
            frame[col] = rows[col].values
    frame["abs_error"] = (frame["y_true"] - frame["y_pred"]).abs()
    frame["pct_error"] = np.where(
        frame["y_true"] != 0, frame["abs_error"] / frame["y_true"] * 100, np.nan
    )

    if calibration_meta is not None:
        meta_cols = [
            "calibration_applied",
            "calibration_skip_reason",
            "calibration_delta_log",
            "second_stage_applied",
            "second_stage_skip_reason",
            "second_stage_delta_log",
            "anchor_count",
            "anchor_residual_iqr",
            "anchor_global_delta",
        ]
        available_meta_cols = [col for col in meta_cols if col in calibration_meta.columns]
        frame = frame.join(calibration_meta[available_meta_cols], how="left")
    if "calibration_applied" not in frame.columns:
        frame["calibration_applied"] = False
    if "calibration_skip_reason" not in frame.columns:
        frame["calibration_skip_reason"] = "not_calibrated"
    if "calibration_delta_log" not in frame.columns:
        frame["calibration_delta_log"] = 0.0
    if "second_stage_applied" not in frame.columns:
        frame["second_stage_applied"] = False
    if "second_stage_skip_reason" not in frame.columns:
        frame["second_stage_skip_reason"] = "not_second_stage"
    if "second_stage_delta_log" not in frame.columns:
        frame["second_stage_delta_log"] = 0.0
    if "anchor_count" not in frame.columns:
        frame["anchor_count"] = np.nan
    if "anchor_residual_iqr" not in frame.columns:
        frame["anchor_residual_iqr"] = np.nan
    if "anchor_global_delta" not in frame.columns:
        frame["anchor_global_delta"] = np.nan

    return frame.reset_index(drop=True)


def add_validation_segments(prediction_rows: pd.DataFrame) -> pd.DataFrame:
    df = prediction_rows.copy()
    df["price_bucket"] = "unknown"
    valid_price = df["y_true"].notna()
    if valid_price.sum() > 0:
        try:
            df.loc[valid_price, "price_bucket"] = pd.qcut(
                df.loc[valid_price, "y_true"],
                q=4,
                labels=["price_q1", "price_q2", "price_q3", "price_q4"],
                duplicates="drop",
            ).astype(str)
        except ValueError:
            df.loc[valid_price, "price_bucket"] = "all_prices"

    count = df["fallback_entity_history_count"].fillna(0)
    df["history_count_bucket"] = pd.cut(
        count,
        bins=[-0.1, 0, 5, 20, 100, np.inf],
        labels=["0", "1-5", "6-20", "21-100", "100+"],
    ).astype(str)
    df["calibration_status"] = np.where(
        df["second_stage_applied"].fillna(False),
        "second_stage_applied",
        np.where(
            df["calibration_applied"].fillna(False),
            "applied",
            df["calibration_skip_reason"].fillna("not_calibrated"),
        ),
    )
    df["calibration_status"] = np.where(
        df["variant"].eq(SECOND_STAGE_RESIDUAL) & ~df["second_stage_applied"].fillna(False),
        df["second_stage_skip_reason"].fillna("second_stage_not_applied"),
        df["calibration_status"],
    )
    return df


def aggregate_prediction_metrics(
    prediction_rows: pd.DataFrame, group_cols: list[str]
) -> pd.DataFrame:
    rows = []
    for keys, group in prediction_rows.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        metrics = evaluate_predictions(group["y_true"], group["y_pred"], "segment")
        row = dict(zip(group_cols, keys))
        row.update(
            {
                "MAE": metrics["MAE"],
                "RMSE": metrics["RMSE"],
                "MAPE": metrics["MAPE"],
                "n_eval": metrics["n_eval"],
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def build_segment_summary(prediction_rows: pd.DataFrame) -> pd.DataFrame:
    segmented = add_validation_segments(prediction_rows)
    frames = []
    segment_specs = [
        ("date", ["variant", "date"]),
        ("price_bucket", ["variant", "price_bucket"]),
        ("history_count_bucket", ["variant", "history_count_bucket"]),
        ("calibration_status", ["variant", "calibration_status"]),
    ]
    for segment_name, group_cols in segment_specs:
        frame = aggregate_prediction_metrics(segmented, group_cols)
        value_col = group_cols[1]
        frame = frame.rename(columns={value_col: "segment_value"})
        frame.insert(0, "segment", segment_name)
        frame = frame[
            ["segment", "segment_value", "variant", "MAE", "RMSE", "MAPE", "n_eval"]
        ]
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def filter_to_test_like_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Keep validation rows that have prior product evidence before the outage day."""
    has_model_history = (
        df["modelId_recent_prior_count"].fillna(0) > 0
        if "modelId_recent_prior_count" in df.columns
        else pd.Series(False, index=df.index)
    )
    has_item_history = (
        df["itemId_recent_prior_count"].fillna(0) > 0
        if "itemId_recent_prior_count" in df.columns
        else pd.Series(False, index=df.index)
    )
    return df[has_model_history | has_item_history].copy()

def append_prediction_eval(
    eval_rows: list[dict[str, float | int | str]],
    prediction_rows: list[pd.DataFrame],
    hidden: pd.DataFrame,
    y_pred: pd.Series,
    variant: str,
    date: str,
    config: PipelineConfig,
    calibration_meta: pd.DataFrame | None = None,
) -> None:
    eval_rows.append(
        evaluate_predictions(hidden[config.target], y_pred, f"{date} {variant}")
    )
    prediction_rows.append(
        make_prediction_frame(
            hidden,
            y_pred,
            variant,
            date,
            config.target,
            calibration_meta=calibration_meta,
        )
    )

def calibrated_candidate_logs(
    day_df: pd.DataFrame, config: PipelineConfig
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.Series]]:
    cal_global = calibrate_day_from_anchor_residuals(
        day_df.assign(blended_pred_log=day_df["model_pred_log"]),
        config,
        pred_log_col="blended_pred_log",
    )
    cal_blend = calibrate_day_from_anchor_residuals(
        day_df, config, pred_log_col="blended_pred_log"
    )
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

def add_validation_anchor_split(df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    df = df.copy()
    if "_validation_row_id" not in df.columns:
        df["_validation_row_id"] = np.arange(len(df))
    df["_validation_is_anchor"] = False
    for _, day_df in df.groupby("date"):
        anchor_idx = day_df.sample(
            n=min(config.anchors_per_day, len(day_df)),
            random_state=config.random_seed,
        ).index
        df.loc[anchor_idx, "_validation_is_anchor"] = True
    return df

def mask_hidden_validation_columns(df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """Mask held-out validation fields that are unavailable on real hidden test rows."""
    df = df.copy()
    hidden_mask = ~df["_validation_is_anchor"]
    cols_to_mask = [
        col
        for col in HIDDEN_UNAVAILABLE_RAW_COLS
        if col in df.columns and col != config.target
    ]
    df.loc[hidden_mask, cols_to_mask] = np.nan
    return df

def run_outage_validation(
    train_df: pd.DataFrame, config: PipelineConfig
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = basic_preprocess(train_df, config)
    unique_dates = sorted(df["date"].unique())
    val_dates = unique_dates[-config.validation_days :]
    train_dates = unique_dates[: -config.validation_days]

    history_df = df[df["date"].isin(train_dates)].copy()
    val_raw = df[df["date"].isin(val_dates)].copy()
    val_raw["_validation_row_id"] = np.arange(len(val_raw))
    if config.validation_test_like_only:
        prelim_val_feat = build_features(history_df, val_raw, config)
        test_like_row_ids = filter_to_test_like_rows(prelim_val_feat)[
            "_validation_row_id"
        ]
        val_raw = val_raw[val_raw["_validation_row_id"].isin(test_like_row_ids)].copy()

    val_raw = add_validation_anchor_split(val_raw, config)
    if config.validation_mask_hidden_like_test:
        val_raw = mask_hidden_validation_columns(val_raw, config)

    train_feat = build_features(history_df, history_df, config)
    val_feat = build_features(history_df, val_raw, config)
    feature_cols = get_feature_columns(train_feat, config)
    cat_idx = get_cat_feature_indices(train_feat, feature_cols, config)
    model = train_global_model(train_feat, feature_cols, cat_idx, config)
    val_feat = add_model_predictions(model, val_feat, feature_cols, config)

    eval_rows = []
    prediction_rows = []
    for date, day_df in val_feat.groupby("date"):
        day_df = day_df.copy()
        anchor_idx = day_df[day_df["_validation_is_anchor"]].index
        hidden_idx = day_df.index.difference(anchor_idx)
        hidden = day_df.loc[hidden_idx].copy()

        day_for_cal = day_df.copy()
        day_for_cal.loc[hidden_idx, config.target] = np.nan
        cal_global, cal_blend, candidate_logs = calibrated_candidate_logs(
            day_for_cal, config
        )

        calibration_meta_by_variant = {
            GLOBAL_CALIBRATED: cal_global.loc[hidden_idx],
            ENTITY_BLEND_CALIBRATED: cal_blend.loc[hidden_idx],
            HYBRID_LAST_PRICE_CALIBRATED_FALLBACK: cal_blend.loc[hidden_idx],
        }
        for variant in [
            LAST_PRICE_BASELINE,
            HYBRID_LAST_PRICE_UNCALIBRATED_FALLBACK,
            GLOBAL_NO_CALIBRATION,
            ENTITY_BLEND_NO_CALIBRATION,
            GLOBAL_CALIBRATED,
            ENTITY_BLEND_CALIBRATED,
            HYBRID_LAST_PRICE_CALIBRATED_FALLBACK,
        ]:
            append_prediction_eval(
                eval_rows,
                prediction_rows,
                hidden,
                log_predictions_to_price(candidate_logs[variant].loc[hidden_idx]),
                variant,
                date,
                config,
                calibration_meta=calibration_meta_by_variant.get(variant),
            )

        fallback_candidates = {
            name: candidate_logs[name]
            for name in [
                ENTITY_BLEND_NO_CALIBRATION,
                ENTITY_BLEND_CALIBRATED,
                GLOBAL_NO_CALIBRATION,
                GLOBAL_CALIBRATED,
            ]
        }
        recent_mask = has_recent_entity_history(day_df)
        selected_fallback_variant = choose_fallback_variant_by_anchor_mae(
            day_for_cal,
            config.target,
            fallback_candidates,
            fallback_anchor_mask=~recent_mask,
        )
        gated_log = day_df["fallback_entity_price_log"].where(
            recent_mask,
            fallback_candidates[selected_fallback_variant].loc[day_df.index],
        )
        gated_pred = log_predictions_to_price(gated_log.loc[hidden_idx])
        gated_meta = cal_blend.loc[hidden_idx].copy()
        gated_meta["calibration_applied"] = (
            selected_fallback_variant in {ENTITY_BLEND_CALIBRATED, GLOBAL_CALIBRATED}
        )
        gated_meta["calibration_skip_reason"] = selected_fallback_variant
        append_prediction_eval(
            eval_rows,
            prediction_rows,
            hidden,
            gated_pred,
            f"{ANCHOR_GATED_FALLBACK_CALIBRATION}_{selected_fallback_variant}",
            date,
            config,
            calibration_meta=gated_meta,
        )

        selected_variant = choose_variant_by_anchor_mae(
            day_for_cal, config.target, candidate_logs
        )
        append_prediction_eval(
            eval_rows,
            prediction_rows,
            hidden,
            log_predictions_to_price(candidate_logs[selected_variant].loc[hidden_idx]),
            f"anchor_model_selected_{selected_variant}",
            date,
            config,
        )

        second_stage = second_stage_residual_calibrate_day(
            day_for_cal, config, pred_log_col="blended_pred_log"
        )
        second_stage_pred = np.expm1(
            second_stage.loc[hidden_idx, "second_stage_pred_log"]
        ).clip(lower=0)
        append_prediction_eval(
            eval_rows,
            prediction_rows,
            hidden,
            second_stage_pred,
            SECOND_STAGE_RESIDUAL,
            date,
            config,
            calibration_meta=second_stage.loc[hidden_idx],
        )

    results = pd.DataFrame(eval_rows)
    summary = (
        results.assign(
            base_model=lambda x: x["model"].str.replace(
                r"^\d{4}-\d{2}-\d{2} ", "", regex=True
            )
        )
        .groupby("base_model")[["MAE", "RMSE", "MAPE"]]
        .mean()
        .sort_values("MAE")
    )
    prediction_frame = pd.concat(prediction_rows, ignore_index=True)
    segment_summary = build_segment_summary(prediction_frame)
    return results, summary, segment_summary, prediction_frame