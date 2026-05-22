import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from .config import PipelineConfig


def _empty_calibration_result(
    df: pd.DataFrame, pred_log_col: str, reason: str
) -> pd.DataFrame:
    df["calibration_delta_log"] = 0.0
    df["calibration_applied"] = False
    df["calibration_skip_reason"] = reason
    df["anchor_count"] = 0
    df["anchor_residual_iqr"] = np.nan
    df["anchor_global_delta"] = 0.0
    df["calibrated_pred_log"] = df[pred_log_col]
    return df.set_index("_orig_index")


def calibrate_day_from_anchor_residuals(
    day_df: pd.DataFrame,
    config: PipelineConfig,
    pred_log_col: str = "blended_pred_log",
) -> pd.DataFrame:
    df = day_df.copy()
    df["_orig_index"] = df.index

    if pred_log_col not in df.columns:
        raise ValueError(f"{pred_log_col} not found in dataframe.")

    df[pred_log_col] = df[pred_log_col].fillna(df["fallback_entity_price_log"])
    df[pred_log_col] = df[pred_log_col].fillna(df[pred_log_col].median())
    anchors = df[df[config.target].notna()].copy()

    if len(anchors) == 0:
        return _empty_calibration_result(df, pred_log_col, "no_anchors")

    anchors["anchor_true_log"] = np.log1p(anchors[config.target].clip(lower=0))
    anchors["residual_log"] = anchors["anchor_true_log"] - anchors[pred_log_col]
    anchors = anchors.replace([np.inf, -np.inf], np.nan).dropna(subset=["residual_log"])

    if len(anchors) == 0:
        return _empty_calibration_result(df, pred_log_col, "no_valid_anchor_residuals")

    global_delta = anchors["residual_log"].median()
    if pd.isna(global_delta):
        global_delta = 0.0
    residual_q1 = anchors["residual_log"].quantile(0.25)
    residual_q3 = anchors["residual_log"].quantile(0.75)
    residual_iqr = residual_q3 - residual_q1

    skip_reason = None
    if config.selective_calibration:
        if len(anchors) < config.calibration_min_anchors:
            skip_reason = "too_few_anchors"
        elif (
            config.calibration_max_residual_iqr is not None
            and residual_iqr > config.calibration_max_residual_iqr
        ):
            skip_reason = "anchor_residuals_too_noisy"
        elif abs(global_delta) < config.calibration_min_abs_global_delta:
            skip_reason = "global_delta_too_small"

    df["anchor_count"] = len(anchors)
    df["anchor_residual_iqr"] = residual_iqr
    df["anchor_global_delta"] = global_delta

    if skip_reason is not None:
        df["calibration_delta_log"] = 0.0
        df["calibration_applied"] = False
        df["calibration_skip_reason"] = skip_reason
        df["calibrated_pred_log"] = df[pred_log_col]
        return df.set_index("_orig_index")

    df["delta_numerator"] = global_delta
    df["delta_denominator"] = 1.0

    for key in config.calibration_keys:
        if key not in df.columns:
            continue

        stats = anchors.groupby(key)["residual_log"].agg(["median", "count"]).reset_index()
        stats.columns = [key, f"{key}_delta", f"{key}_anchor_count"]
        df = df.merge(stats, on=key, how="left")

        count = df[f"{key}_anchor_count"].fillna(0)
        delta = df[f"{key}_delta"].fillna(global_delta)
        weight = count / (count + config.calibration_smoothing)
        df["delta_numerator"] += delta * weight
        df["delta_denominator"] += weight

    df["calibration_delta_log"] = df["delta_numerator"] / df["delta_denominator"]
    df["calibration_delta_log"] = df["calibration_delta_log"].fillna(global_delta).fillna(0)
    if config.calibration_delta_cap is not None:
        cap = abs(config.calibration_delta_cap)
        df["calibration_delta_log"] = df["calibration_delta_log"].clip(-cap, cap)

    df["calibration_applied"] = True
    df["calibration_skip_reason"] = "applied"
    df[pred_log_col] = df[pred_log_col].fillna(df[pred_log_col].median()).fillna(0)
    df["calibrated_pred_log"] = df[pred_log_col] + df["calibration_delta_log"]
    df["calibrated_pred_log"] = (
        df["calibrated_pred_log"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(df[pred_log_col])
        .fillna(0)
    )
    return df.set_index("_orig_index")


def calibrate_all_days(
    df: pd.DataFrame, config: PipelineConfig, pred_log_col: str = "blended_pred_log"
) -> pd.DataFrame:
    calibrated = [
        calibrate_day_from_anchor_residuals(day_df, config, pred_log_col=pred_log_col)
        for _, day_df in df.groupby("date")
    ]
    return pd.concat(calibrated, axis=0).sort_index()


def add_anchor_residual_features(
    day_df: pd.DataFrame,
    config: PipelineConfig,
    pred_log_col: str = "blended_pred_log",
) -> pd.DataFrame:
    df = day_df.copy()
    df["_anchor_orig_index"] = df.index
    anchors = df[df[config.target].notna()].copy()

    if len(anchors) == 0:
        df["anchor_global_delta"] = 0.0
        df["anchor_residual_iqr"] = np.nan
        df["anchor_count"] = 0
        for key in config.calibration_keys:
            df[f"{key}_stage_delta"] = 0.0
            df[f"{key}_stage_anchor_count"] = 0
        return df

    anchors["anchor_true_log"] = np.log1p(anchors[config.target].clip(lower=0))
    anchors["residual_log"] = anchors["anchor_true_log"] - anchors[pred_log_col]
    anchors = anchors.replace([np.inf, -np.inf], np.nan).dropna(subset=["residual_log"])

    if len(anchors) == 0:
        df["anchor_global_delta"] = 0.0
        df["anchor_residual_iqr"] = np.nan
        df["anchor_count"] = 0
        for key in config.calibration_keys:
            df[f"{key}_stage_delta"] = 0.0
            df[f"{key}_stage_anchor_count"] = 0
        return df

    global_delta = anchors["residual_log"].median()
    residual_iqr = anchors["residual_log"].quantile(0.75) - anchors["residual_log"].quantile(0.25)
    df["anchor_global_delta"] = global_delta
    df["anchor_residual_iqr"] = residual_iqr
    df["anchor_count"] = len(anchors)

    for key in config.calibration_keys:
        if key not in df.columns:
            continue
        stats = anchors.groupby(key)["residual_log"].agg(["median", "count"]).reset_index()
        stats.columns = [key, f"{key}_stage_delta", f"{key}_stage_anchor_count"]
        df = df.merge(stats, on=key, how="left")
        df[f"{key}_stage_delta"] = df[f"{key}_stage_delta"].fillna(global_delta)
        df[f"{key}_stage_anchor_count"] = df[f"{key}_stage_anchor_count"].fillna(0)

    return df


def second_stage_residual_calibrate_day(
    day_df: pd.DataFrame,
    config: PipelineConfig,
    pred_log_col: str = "blended_pred_log",
) -> pd.DataFrame:
    df = add_anchor_residual_features(day_df, config, pred_log_col=pred_log_col)
    df["_orig_index"] = df["_anchor_orig_index"]
    df[pred_log_col] = df[pred_log_col].fillna(df["fallback_entity_price_log"])
    df[pred_log_col] = df[pred_log_col].fillna(df[pred_log_col].median()).fillna(0)

    anchors = df[df[config.target].notna()].copy()
    if len(anchors) < 2:
        df["second_stage_delta_log"] = 0.0
        df["second_stage_applied"] = False
        df["second_stage_pred_log"] = df[pred_log_col]
        return df.set_index("_orig_index")

    anchors["target_residual_log"] = np.log1p(anchors[config.target].clip(lower=0)) - anchors[
        pred_log_col
    ]
    anchors = anchors.replace([np.inf, -np.inf], np.nan).dropna(subset=["target_residual_log"])
    if len(anchors) < 2:
        df["second_stage_delta_log"] = 0.0
        df["second_stage_applied"] = False
        df["second_stage_pred_log"] = df[pred_log_col]
        return df.set_index("_orig_index")

    feature_cols = [
        "model_pred_log",
        "blended_pred_log",
        "fallback_entity_price_log",
        "fallback_entity_history_count",
        "entity_weight",
        "anchor_global_delta",
        "anchor_residual_iqr",
        "anchor_count",
    ]
    for key in config.calibration_keys:
        for suffix in ["stage_delta", "stage_anchor_count"]:
            col = f"{key}_{suffix}"
            if col in df.columns:
                feature_cols.append(col)

    feature_cols = [col for col in feature_cols if col in df.columns]
    X_anchor = anchors[feature_cols].copy()
    X_all = df[feature_cols].copy()

    medians = X_anchor.median(numeric_only=True).fillna(0)
    X_anchor = X_anchor.fillna(medians).fillna(0)
    X_all = X_all.fillna(medians).fillna(0)

    model = Ridge(alpha=config.second_stage_alpha)
    model.fit(X_anchor, anchors["target_residual_log"])
    delta = pd.Series(model.predict(X_all), index=df.index)

    if config.calibration_delta_cap is not None:
        cap = abs(config.calibration_delta_cap)
        delta = delta.clip(-cap, cap)

    df["second_stage_delta_log"] = delta
    df["second_stage_applied"] = True
    df["second_stage_pred_log"] = df[pred_log_col] + df["second_stage_delta_log"]
    df["second_stage_pred_log"] = (
        df["second_stage_pred_log"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(df[pred_log_col])
        .fillna(0)
    )
    return df.set_index("_orig_index")


def second_stage_residual_calibrate_all_days(
    df: pd.DataFrame,
    config: PipelineConfig,
    pred_log_col: str = "blended_pred_log",
) -> pd.DataFrame:
    calibrated = [
        second_stage_residual_calibrate_day(day_df, config, pred_log_col=pred_log_col)
        for _, day_df in df.groupby("date")
    ]
    return pd.concat(calibrated, axis=0).sort_index()
