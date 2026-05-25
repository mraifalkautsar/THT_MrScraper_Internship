import argparse
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import PipelineConfig
from src.tuning import (
    parse_grid,
    parse_optional_float_grid,
    save_tuning_outputs,
    tune_pipeline,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune CatBoost and post-processing hyperparameters."
    )
    parser.add_argument("--train-path", default="ecommerce_price_prediction-train.csv")
    parser.add_argument("--output-dir", default="outputs/tuning")
    parser.add_argument("--validation-days", type=int, default=3)
    parser.add_argument("--anchors-per-day", type=int, default=100)
    parser.add_argument(
        "--test-like-only",
        action="store_true",
        help="Tune on validation rows with prior modelId or itemId history only.",
    )
    parser.add_argument(
        "--no-mask-hidden-like-test",
        action="store_true",
        help="Do not mask validation hidden-row columns to match real hidden test missingness.",
    )
    parser.add_argument("--learning-rates", default="0.03,0.05")
    parser.add_argument("--depths", default="6,8")
    parser.add_argument("--iterations", default="800,1200")
    parser.add_argument("--entity-smoothing", default="10,20,50")
    parser.add_argument("--calibration-smoothing", default="4,8,16")
    parser.add_argument("--calibration-delta-caps", default="none")
    parser.add_argument(
        "--selection-metric",
        default="MAE",
        choices=["MAE", "RMSE", "MAPE", "composite"],
    )
    parser.add_argument("--selective-calibration", action="store_true")
    parser.add_argument("--calibration-min-anchors", type=int, default=20)
    parser.add_argument("--calibration-max-residual-iqr", type=float, default=None)
    parser.add_argument("--calibration-min-abs-global-delta", type=float, default=0.0)
    parser.add_argument(
        "--max-trials",
        type=int,
        default=None,
        help="Randomly sample at most this many grid combinations.",
    )
    parser.add_argument("--sample-rows", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = replace(
        PipelineConfig(),
        train_path=Path(args.train_path),
        output_dir=Path(args.output_dir),
        validation_days=args.validation_days,
        anchors_per_day=args.anchors_per_day,
        validation_test_like_only=args.test_like_only,
        validation_mask_hidden_like_test=not args.no_mask_hidden_like_test,
        selective_calibration=args.selective_calibration,
        calibration_min_anchors=args.calibration_min_anchors,
        calibration_max_residual_iqr=args.calibration_max_residual_iqr,
        calibration_min_abs_global_delta=args.calibration_min_abs_global_delta,
    )

    train_df = pd.read_csv(config.train_path)
    if args.sample_rows:
        train_df = (
            train_df.sample(args.sample_rows, random_state=config.random_seed)
            .sort_values(config.date_col)
            .reset_index(drop=True)
        )

    tuning_summary, tuning_details = tune_pipeline(
        train_df=train_df,
        base_config=config,
        learning_rates=parse_grid(args.learning_rates, float),
        depths=parse_grid(args.depths, int),
        iterations=parse_grid(args.iterations, int),
        entity_smoothing_values=parse_grid(args.entity_smoothing, float),
        calibration_smoothing_values=parse_grid(args.calibration_smoothing, float),
        calibration_delta_caps=parse_optional_float_grid(args.calibration_delta_caps),
        selection_metric=args.selection_metric,
        max_trials=args.max_trials,
    )
    save_tuning_outputs(tuning_summary, tuning_details, config.output_dir)

    print(tuning_summary.head(20).to_string(index=False))
    print()
    best = tuning_summary.iloc[0]
    print("Best configuration:")
    print(best.to_string())


if __name__ == "__main__":
    main()
