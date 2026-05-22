import numpy as np
import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_categorical_dtype,
    is_numeric_dtype,
    is_string_dtype,
)

from .config import PipelineConfig
from .data import basic_preprocess


DROP_COLS = {"target_log"}


def add_row_features(df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    df = df.copy()
    dt = df[config.date_col]

    df["hour"] = dt.dt.hour
    df["dayofweek"] = dt.dt.dayofweek
    df["day"] = dt.dt.day
    df["month"] = dt.dt.month
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)

    if "priceBeforeDiscount" in df.columns:
        df["has_discount"] = (df["priceBeforeDiscount"] > 0).astype(int)
        df["discount_ratio"] = np.where(
            df["priceBeforeDiscount"] > 0,
            df.get("raw_discount", 0) / (df["priceBeforeDiscount"] + 1),
            0,
        )
        df["price_before_log"] = np.log1p(df["priceBeforeDiscount"].clip(lower=0))

    if "raw_discount" in df.columns:
        df["raw_discount_log"] = np.log1p(df["raw_discount"].clip(lower=0))

    if "show_discount" in df.columns:
        df["show_discount"] = df["show_discount"].fillna(0)
        df["show_discount_ratio"] = df["show_discount"] / 100.0

    if "stock" in df.columns and "normal_stock" in df.columns:
        df["stock_ratio"] = df["stock"] / (df["normal_stock"] + 1)
        df["stock_gap"] = df["normal_stock"] - df["stock"]
        df["is_low_stock"] = (df["stock_ratio"] < 0.2).astype(int)

    for col in ["total_rating_count", "cmt_count", "shop_follower_count"]:
        if col in df.columns:
            df[f"{col}_log"] = np.log1p(df[col].clip(lower=0))

    if "review_rating" in df.columns and "total_rating_count" in df.columns:
        df["review_strength"] = df["review_rating"] * np.log1p(
            df["total_rating_count"].clip(lower=0)
        )

    if "shop_rating" in df.columns and "shop_follower_count" in df.columns:
        df["shop_strength"] = df["shop_rating"] * np.log1p(
            df["shop_follower_count"].clip(lower=0)
        )

    return df


def compute_historical_price_stats(
    history_df: pd.DataFrame, config: PipelineConfig
) -> list[tuple[str, pd.DataFrame]]:
    history = history_df.dropna(subset=[config.target]).copy()
    history["target_log"] = np.log1p(history[config.target].clip(lower=0))
    agg_frames = []

    for key in config.history_keys:
        group = (
            history.groupby(key)["target_log"]
            .agg(["count", "mean", "median", "std", "min", "max"])
            .reset_index()
        )
        group.columns = [
            key,
            f"{key}_price_count",
            f"{key}_price_mean_log",
            f"{key}_price_median_log",
            f"{key}_price_std_log",
            f"{key}_price_min_log",
            f"{key}_price_max_log",
        ]
        agg_frames.append((key, group))

    return agg_frames


def add_historical_price_features(
    history_df: pd.DataFrame, df: pd.DataFrame, config: PipelineConfig
) -> pd.DataFrame:
    df = df.copy()
    for key, agg in compute_historical_price_stats(history_df, config):
        df = df.merge(agg, on=key, how="left")

    global_log_median = np.log1p(history_df[config.target].median())
    df["fallback_entity_price_log"] = df.get(
        "modelId_last_price_log", pd.Series(np.nan, index=df.index)
    )

    if "itemId_last_price_log" in df.columns:
        df["fallback_entity_price_log"] = df["fallback_entity_price_log"].fillna(
            df["itemId_last_price_log"]
        )

    for col in [
        "modelId_price_median_log",
        "itemId_price_median_log",
        "shopId_price_median_log",
        "cat_id_price_median_log",
        "brand_price_median_log",
    ]:
        df["fallback_entity_price_log"] = df["fallback_entity_price_log"].fillna(df[col])

    df["fallback_entity_price_log"] = df["fallback_entity_price_log"].fillna(
        global_log_median
    )

    df["fallback_entity_history_count"] = df.get(
        "modelId_recent_prior_count", pd.Series(np.nan, index=df.index)
    )

    for col in [
        "itemId_recent_prior_count",
        "modelId_price_count",
        "itemId_price_count",
        "shopId_price_count",
        "cat_id_price_count",
    ]:
        df["fallback_entity_history_count"] = df[
            "fallback_entity_history_count"
        ].fillna(df[col])
    df["fallback_entity_history_count"] = df["fallback_entity_history_count"].fillna(0)

    return df


def _recent_stats_for_key(
    history_df: pd.DataFrame,
    df: pd.DataFrame,
    key: str,
    config: PipelineConfig,
) -> pd.DataFrame:
    result = pd.DataFrame(index=df.index)
    prefix = key
    for col in [
        f"{prefix}_recent_prior_count",
        f"{prefix}_last_price",
        f"{prefix}_last_price_log",
        f"{prefix}_last_seen_hours_ago",
        f"{prefix}_last_3_median_price_log",
        f"{prefix}_last_7_median_price_log",
    ]:
        result[col] = np.nan

    history = (
        history_df.dropna(subset=[config.target])
        [[key, config.date_col, config.target]]
        .copy()
    )
    if history.empty or key not in df.columns:
        return result

    history[key] = history[key].astype(str)
    history = history.sort_values([key, config.date_col])
    history_groups = {key_value: group for key_value, group in history.groupby(key, sort=False)}

    target_keys = df[key].astype(str)
    target_times = df[config.date_col]
    one_hour_ns = 3_600_000_000_000

    for key_value, target_idx in target_keys.groupby(target_keys).groups.items():
        hist_group = history_groups.get(key_value)
        if hist_group is None or hist_group.empty:
            continue

        hist_times = hist_group[config.date_col].astype("int64").to_numpy()
        hist_prices = hist_group[config.target].astype(float).to_numpy()
        row_times = target_times.loc[target_idx].astype("int64").to_numpy()
        positions = np.searchsorted(hist_times, row_times, side="left") - 1

        valid = positions >= 0
        if not valid.any():
            result.loc[target_idx, f"{prefix}_recent_prior_count"] = 0
            continue

        target_index = pd.Index(target_idx)
        valid_index = target_index[valid]
        valid_positions = positions[valid]
        last_prices = hist_prices[valid_positions]
        last_times = hist_times[valid_positions]

        result.loc[target_idx, f"{prefix}_recent_prior_count"] = positions + 1
        result.loc[valid_index, f"{prefix}_last_price"] = last_prices
        result.loc[valid_index, f"{prefix}_last_price_log"] = np.log1p(
            np.clip(last_prices, a_min=0, a_max=None)
        )
        result.loc[valid_index, f"{prefix}_last_seen_hours_ago"] = (
            row_times[valid] - last_times
        ) / one_hour_ns

        last_3 = []
        last_7 = []
        for pos in valid_positions:
            last_3.append(np.median(hist_prices[max(0, pos - 2) : pos + 1]))
            last_7.append(np.median(hist_prices[max(0, pos - 6) : pos + 1]))

        result.loc[valid_index, f"{prefix}_last_3_median_price_log"] = np.log1p(
            np.clip(last_3, a_min=0, a_max=None)
        )
        result.loc[valid_index, f"{prefix}_last_7_median_price_log"] = np.log1p(
            np.clip(last_7, a_min=0, a_max=None)
        )

    result[f"{prefix}_recent_prior_count"] = result[
        f"{prefix}_recent_prior_count"
    ].fillna(0)
    return result


def add_recent_price_features(
    history_df: pd.DataFrame, df: pd.DataFrame, config: PipelineConfig
) -> pd.DataFrame:
    df = df.copy()
    for key in ["modelId", "itemId"]:
        recent = _recent_stats_for_key(history_df, df, key, config)
        df = pd.concat([df, recent], axis=1)
    return df


def build_features(
    history_df: pd.DataFrame, df: pd.DataFrame, config: PipelineConfig
) -> pd.DataFrame:
    history = basic_preprocess(history_df, config)
    features = basic_preprocess(df, config)
    features = add_row_features(features, config)
    features = add_recent_price_features(history, features, config)
    features = add_historical_price_features(history, features, config)
    return features


def get_feature_columns(df: pd.DataFrame, config: PipelineConfig) -> list[str]:
    excluded = {config.target, config.date_col, "date"} | DROP_COLS
    valid_cols = []

    for col in [c for c in df.columns if c not in excluded]:
        dtype = df[col].dtype
        if (
            is_numeric_dtype(dtype)
            or is_bool_dtype(dtype)
            or is_string_dtype(dtype)
            or is_categorical_dtype(dtype)
        ):
            valid_cols.append(col)

    return valid_cols


def get_cat_feature_indices(
    df: pd.DataFrame, feature_cols: list[str], config: PipelineConfig
) -> list[int]:
    cat_cols = [
        col
        for col in feature_cols
        if col in config.cat_cols
        or is_string_dtype(df[col].dtype)
        or is_categorical_dtype(df[col].dtype)
    ]
    return [feature_cols.index(col) for col in cat_cols]
