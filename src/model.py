import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from .config import PipelineConfig


def train_global_model(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    cat_feature_indices: list[int],
    config: PipelineConfig,
) -> CatBoostRegressor:
    train_df = train_df.dropna(subset=[config.target]).copy()
    train_df["target_log"] = np.log1p(train_df[config.target].clip(lower=0))

    model = CatBoostRegressor(
        loss_function="MAE",
        eval_metric="MAE",
        iterations=config.catboost_iterations,
        learning_rate=config.catboost_learning_rate,
        depth=config.catboost_depth,
        random_seed=config.random_seed,
        verbose=200,
        allow_writing_files=False,
    )
    model.fit(
        train_df[feature_cols],
        train_df["target_log"],
        cat_features=cat_feature_indices,
    )
    return model


def add_model_predictions(
    model: CatBoostRegressor,
    df: pd.DataFrame,
    feature_cols: list[str],
    config: PipelineConfig,
) -> pd.DataFrame:
    df = df.copy()
    for col in feature_cols:
        if col not in df.columns:
            df[col] = np.nan

    df["model_pred_log"] = model.predict(df[feature_cols])
    count = df["fallback_entity_history_count"].fillna(0)
    entity_weight = count / (count + config.entity_smoothing)
    df["entity_weight"] = entity_weight
    df["blended_pred_log"] = (
        (1 - entity_weight) * df["model_pred_log"]
        + entity_weight * df["fallback_entity_price_log"]
    )
    return df

