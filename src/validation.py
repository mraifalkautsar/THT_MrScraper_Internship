import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .calibration import (
    calibrate_day_from_anchor_residuals,
    second_stage_residual_calibrate_day,
)
from .config import PipelineConfig
from .data import basic_preprocess
from .features import build_features, get_cat_feature_indices, get_feature_columns
from .model import add_model_predictions, train_global_model
from .strategies import (
    ENTITY_BLEND_CALIBRATED,
    ENTITY_BLEND_NO_CALIBRATION,
    GLOBAL_CALIBRATED,
    GLOBAL_NO_CALIBRATION,
    HYBRID_LAST_PRICE_CALIBRATED_FALLBACK,
    HYBRID_LAST_PRICE_ENTITY,
    LAST_PRICE_BASELINE,
    SECOND_STAGE_RESIDUAL,
    choose_variant_by_anchor_mae,
    hybrid_last_price_then_log_prediction,
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
    frame["abs_error"] = (frame["y_true"] - frame["y_pred"]).abs()
    frame["pct_error"] = np.where(
        frame["y_true"] != 0, frame["abs_error"] / frame["y_true"] * 100, np.nan
    )

    if calibration_meta is not None:
        meta_cols = [
            "calibration_applied",
            "calibration_skip_reason",
            "calibration_delta_log",
            "anchor_count",
            "anchor_residual_iqr",
            "anchor_global_delta",
        ]
        available_meta_cols = [col for col in meta_cols if col in calibration_meta.columns]
        frame = frame.join(calibration_meta[available_meta_cols], how="left")
    else:
        frame["calibration_applied"] = False
        frame["calibration_skip_reason"] = "not_calibrated"
        frame["calibration_delta_log"] = 0.0
        frame["anchor_count"] = np.nan
        frame["anchor_residual_iqr"] = np.nan
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
        df["calibration_applied"].fillna(False),
        "applied",
        df["calibration_skip_reason"].fillna("not_calibrated"),
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


def run_outage_validation(
    train_df: pd.DataFrame, config: PipelineConfig
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = basic_preprocess(train_df, config)
    unique_dates = sorted(df["date"].unique())
    val_dates = unique_dates[-config.validation_days :]
    train_dates = unique_dates[: -config.validation_days]

    history_df = df[df["date"].isin(train_dates)].copy()
    val_raw = df[df["date"].isin(val_dates)].copy()

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
        anchor_idx = day_df.sample(
            n=min(config.anchors_per_day, len(day_df)),
            random_state=config.random_seed,
        ).index
        hidden_idx = day_df.index.difference(anchor_idx)
        hidden = day_df.loc[hidden_idx].copy()

        model_pred = np.expm1(hidden["model_pred_log"]).clip(lower=0)
        blend_pred = np.expm1(hidden["blended_pred_log"]).clip(lower=0)
        last_price_baseline_pred = np.expm1(hidden["fallback_entity_price_log"]).clip(lower=0)
        hybrid_log = hybrid_last_price_then_log_prediction(hidden, "blended_pred_log")
        hybrid_pred = np.expm1(hybrid_log).clip(lower=0)
        eval_rows.append(
            evaluate_predictions(
                hidden[config.target], last_price_baseline_pred, f"{date} {LAST_PRICE_BASELINE}"
            )
        )
        prediction_rows.append(
            make_prediction_frame(
                hidden,
                last_price_baseline_pred,
                LAST_PRICE_BASELINE,
                date,
                config.target,
            )
        )
        eval_rows.append(
            evaluate_predictions(
                hidden[config.target], hybrid_pred, f"{date} {HYBRID_LAST_PRICE_ENTITY}"
            )
        )
        prediction_rows.append(
            make_prediction_frame(
                hidden,
                hybrid_pred,
                HYBRID_LAST_PRICE_ENTITY,
                date,
                config.target,
            )
        )
        eval_rows.append(
            evaluate_predictions(hidden[config.target], model_pred, f"{date} {GLOBAL_NO_CALIBRATION}")
        )
        prediction_rows.append(
            make_prediction_frame(
                hidden, model_pred, GLOBAL_NO_CALIBRATION, date, config.target
            )
        )
        eval_rows.append(
            evaluate_predictions(
                hidden[config.target], blend_pred, f"{date} {ENTITY_BLEND_NO_CALIBRATION}"
            )
        )
        prediction_rows.append(
            make_prediction_frame(
                hidden, blend_pred, ENTITY_BLEND_NO_CALIBRATION, date, config.target
            )
        )

        day_for_cal = day_df.copy()
        day_for_cal.loc[hidden_idx, config.target] = np.nan

        cal_global = calibrate_day_from_anchor_residuals(
            day_for_cal.assign(blended_pred_log=day_for_cal["model_pred_log"]),
            config,
            pred_log_col="blended_pred_log",
        )
        cal_global_pred = np.expm1(cal_global.loc[hidden_idx, "calibrated_pred_log"]).clip(lower=0)
        eval_rows.append(
            evaluate_predictions(hidden[config.target], cal_global_pred, f"{date} {GLOBAL_CALIBRATED}")
        )
        prediction_rows.append(
            make_prediction_frame(
                hidden,
                cal_global_pred,
                GLOBAL_CALIBRATED,
                date,
                config.target,
                calibration_meta=cal_global.loc[hidden_idx],
            )
        )

        cal_blend = calibrate_day_from_anchor_residuals(
            day_for_cal, config, pred_log_col="blended_pred_log"
        )
        cal_blend_pred = np.expm1(cal_blend.loc[hidden_idx, "calibrated_pred_log"]).clip(lower=0)
        eval_rows.append(
            evaluate_predictions(
                hidden[config.target], cal_blend_pred, f"{date} {ENTITY_BLEND_CALIBRATED}"
            )
        )
        prediction_rows.append(
            make_prediction_frame(
                hidden,
                cal_blend_pred,
                ENTITY_BLEND_CALIBRATED,
                date,
                config.target,
                calibration_meta=cal_blend.loc[hidden_idx],
            )
        )
        hybrid_calibrated_log = hybrid_last_price_then_log_prediction(
            cal_blend.loc[hidden_idx], "calibrated_pred_log"
        )
        hybrid_calibrated_pred = np.expm1(hybrid_calibrated_log).clip(lower=0)
        eval_rows.append(
            evaluate_predictions(
                hidden[config.target],
                hybrid_calibrated_pred,
                f"{date} {HYBRID_LAST_PRICE_CALIBRATED_FALLBACK}",
            )
        )
        prediction_rows.append(
            make_prediction_frame(
                hidden,
                hybrid_calibrated_pred,
                HYBRID_LAST_PRICE_CALIBRATED_FALLBACK,
                date,
                config.target,
                calibration_meta=cal_blend.loc[hidden_idx],
            )
        )

        candidate_log_predictions = {
            LAST_PRICE_BASELINE: day_df["fallback_entity_price_log"],
            HYBRID_LAST_PRICE_ENTITY: hybrid_last_price_then_log_prediction(
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
        selected_variant = choose_variant_by_anchor_mae(
            day_for_cal, config.target, candidate_log_predictions
        )
        selected_pred = np.expm1(
            candidate_log_predictions[selected_variant].loc[hidden_idx]
        ).clip(lower=0)
        eval_rows.append(
            evaluate_predictions(
                hidden[config.target],
                selected_pred,
                f"{date} anchor_model_selected_{selected_variant}",
            )
        )
        prediction_rows.append(
            make_prediction_frame(
                hidden,
                selected_pred,
                f"anchor_model_selected_{selected_variant}",
                date,
                config.target,
            )
        )

        if config.include_experimental_variants:
            second_stage = second_stage_residual_calibrate_day(
                day_for_cal, config, pred_log_col="blended_pred_log"
            )
            second_stage_pred = np.expm1(
                second_stage.loc[hidden_idx, "second_stage_pred_log"]
            ).clip(lower=0)
            eval_rows.append(
                evaluate_predictions(
                    hidden[config.target], second_stage_pred, f"{date} {SECOND_STAGE_RESIDUAL}"
                )
            )
            prediction_rows.append(
                make_prediction_frame(
                    hidden,
                    second_stage_pred,
                    SECOND_STAGE_RESIDUAL,
                    date,
                    config.target,
                )
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
    return results, summary, segment_summary
