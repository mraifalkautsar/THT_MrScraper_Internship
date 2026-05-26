from dataclasses import replace
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import PipelineConfig
from .validation import run_outage_validation


def parse_grid(values: str, cast: type) -> list[Any]:
    """Parse a comma-separated CLI grid into typed values."""
    return [cast(value.strip()) for value in values.split(",") if value.strip()]


def parse_optional_float_grid(values: str) -> list[float | None]:
    """Parse float grids where none/null/no_cap disable a calibration cap."""
    parsed = []
    for value in values.split(","):
        value = value.strip()
        if not value:
            continue
        if value.lower() in {"none", "null", "no_cap"}:
            parsed.append(None)
        else:
            parsed.append(float(value))
    return parsed


def choose_best_variant(summary: pd.DataFrame, selection_metric: str) -> pd.Series:
    """Select the best validation strategy by one metric or a simple composite rank."""
    candidates = summary.reset_index().copy()
    metric = selection_metric.upper()
    if metric in {"MAE", "RMSE", "MAPE"}:
        return candidates.sort_values(metric).iloc[0]
    if metric == "COMPOSITE":
        candidates["variant_selection_score"] = (
            candidates["MAE"].rank(method="min")
            + candidates["MAPE"].rank(method="min")
            + 0.5 * candidates["RMSE"].rank(method="min")
        )
        return candidates.sort_values("variant_selection_score").iloc[0]
    raise ValueError("selection_metric must be one of: MAE, RMSE, MAPE, composite")


def tune_pipeline(
    train_df: pd.DataFrame,
    base_config: PipelineConfig,
    learning_rates: list[float],
    depths: list[int],
    iterations: list[int],
    entity_smoothing_values: list[float],
    calibration_smoothing_values: list[float],
    calibration_delta_caps: list[float | None],
    selection_metric: str = "MAE",
    max_trials: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run outage validation across a hyperparameter grid and collect best variants."""
    tuning_rows = []
    detailed_results = []

    grid = list(product(
        learning_rates,
        depths,
        iterations,
        entity_smoothing_values,
        calibration_smoothing_values,
        calibration_delta_caps,
    ))

    if max_trials is not None and max_trials > 0 and max_trials < len(grid):
        rng = np.random.default_rng(base_config.random_seed)
        selected_indices = rng.choice(len(grid), size=max_trials, replace=False)
        grid = [grid[i] for i in selected_indices]

    for (
        learning_rate,
        depth,
        n_iterations,
        entity_smoothing,
        calibration_smoothing,
        calibration_delta_cap,
    ) in grid:
        config = replace(
            base_config,
            catboost_learning_rate=learning_rate,
            catboost_depth=depth,
            catboost_iterations=n_iterations,
            entity_smoothing=entity_smoothing,
            calibration_smoothing=calibration_smoothing,
            calibration_delta_cap=calibration_delta_cap,
        )

        results, summary, _, _ = run_outage_validation(train_df, config)
        for _, row in results.iterrows():
            detailed = row.to_dict()
            detailed.update(
                {
                    "learning_rate": learning_rate,
                    "depth": depth,
                    "iterations": n_iterations,
                    "entity_smoothing": entity_smoothing,
                    "calibration_smoothing": calibration_smoothing,
                    "calibration_delta_cap": calibration_delta_cap,
                }
            )
            detailed_results.append(detailed)

        best_variant = choose_best_variant(summary, selection_metric)
        tuning_rows.append(
            {
                "learning_rate": learning_rate,
                "depth": depth,
                "iterations": n_iterations,
                "entity_smoothing": entity_smoothing,
                "calibration_smoothing": calibration_smoothing,
                "calibration_delta_cap": calibration_delta_cap,
                "selection_metric": selection_metric,
                "best_variant": best_variant["base_model"],
                "best_MAE": best_variant["MAE"],
                "best_RMSE": best_variant["RMSE"],
                "best_MAPE": best_variant["MAPE"],
            }
        )

    tuning_summary = pd.DataFrame(tuning_rows)
    metric = selection_metric.upper()
    if metric == "COMPOSITE":
        tuning_summary["selection_score"] = (
            tuning_summary["best_MAE"].rank(method="min")
            + tuning_summary["best_MAPE"].rank(method="min")
            + 0.5 * tuning_summary["best_RMSE"].rank(method="min")
        )
        tuning_summary = tuning_summary.sort_values(
            ["selection_score", "best_MAE", "best_MAPE", "best_RMSE"]
        )
    elif metric in {"MAE", "RMSE", "MAPE"}:
        tuning_summary["selection_score"] = tuning_summary[f"best_{metric}"]
        tuning_summary = tuning_summary.sort_values(
            ["selection_score", "best_MAE", "best_RMSE", "best_MAPE"]
        )
    else:
        raise ValueError("selection_metric must be one of: MAE, RMSE, MAPE, composite")

    tuning_details = pd.DataFrame(detailed_results)
    return tuning_summary, tuning_details


def save_tuning_outputs(
    tuning_summary: pd.DataFrame,
    tuning_details: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Write tuning summary and per-run details to the configured output directory."""
    output_dir.mkdir(exist_ok=True)
    tuning_summary.to_csv(output_dir / "tuning_summary.csv", index=False)
    tuning_details.to_csv(output_dir / "tuning_details.csv", index=False)
