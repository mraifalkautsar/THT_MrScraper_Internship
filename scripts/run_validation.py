import argparse
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import PipelineConfig
from src.validation import run_outage_validation


def parse_args() -> argparse.Namespace:
    """Parse CLI options for outage-style validation."""
    parser = argparse.ArgumentParser(description="Run outage-style validation.")
    parser.add_argument("--train-path", default="ecommerce_price_prediction-train.csv")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--validation-days", type=int, default=3)
    parser.add_argument("--anchors-per-day", type=int, default=100)
    parser.add_argument(
        "--test-like-only",
        action="store_true",
        help="Evaluate only rows with prior modelId or itemId history before the validation day.",
    )
    parser.add_argument(
        "--no-mask-hidden-like-test",
        action="store_true",
        help="Do not mask validation hidden-row columns to match real hidden test missingness.",
    )
    parser.add_argument("--iterations", type=int, default=1200)
    parser.add_argument("--sample-rows", type=int, default=None)
    parser.add_argument("--calibration-delta-cap", type=float, default=None)
    parser.add_argument("--selective-calibration", action="store_true")
    parser.add_argument("--calibration-min-anchors", type=int, default=20)
    parser.add_argument("--calibration-max-residual-iqr", type=float, default=None)
    parser.add_argument("--calibration-min-abs-global-delta", type=float, default=0.0)
    parser.add_argument("--second-stage-alpha", type=float, default=10.0)
    parser.add_argument("--second-stage-min-anchors", type=int, default=10)
    parser.add_argument(
        "--include-experimental-variants",
        action="store_true",
        help="Reserved for additional experimental variants. second_stage_residual is included by default.",
    )
    return parser.parse_args()


def main() -> None:
    """Load data, run validation, and save summary/segment/row-level outputs."""
    args = parse_args()
    config = replace(
        PipelineConfig(),
        train_path=Path(args.train_path),
        output_dir=Path(args.output_dir),
        validation_days=args.validation_days,
        anchors_per_day=args.anchors_per_day,
        validation_test_like_only=args.test_like_only,
        validation_mask_hidden_like_test=not args.no_mask_hidden_like_test,
        catboost_iterations=args.iterations,
        calibration_delta_cap=args.calibration_delta_cap,
        selective_calibration=args.selective_calibration,
        calibration_min_anchors=args.calibration_min_anchors,
        calibration_max_residual_iqr=args.calibration_max_residual_iqr,
        calibration_min_abs_global_delta=args.calibration_min_abs_global_delta,
        second_stage_alpha=args.second_stage_alpha,
        second_stage_min_anchors=args.second_stage_min_anchors,
        include_experimental_variants=args.include_experimental_variants,
    )

    train_df = pd.read_csv(config.train_path)
    if args.sample_rows:
        train_df = (
            train_df.sample(args.sample_rows, random_state=config.random_seed)
            .sort_values(config.date_col)
            .reset_index(drop=True)
        )

    results, summary, segments, predictions = run_outage_validation(train_df, config)
    config.output_dir.mkdir(exist_ok=True)
    results.to_csv(config.output_dir / "validation_results.csv", index=False)
    summary.to_csv(config.output_dir / "validation_summary.csv")
    segments.to_csv(config.output_dir / "validation_segments.csv", index=False)
    predictions.to_csv(config.output_dir / "validation_predictions.csv", index=False)
    print(results.to_string(index=False))
    print()
    print(summary.to_string())
    print()
    print("Saved segmented validation:", config.output_dir / "validation_segments.csv")
    print("Saved row-level validation:", config.output_dir / "validation_predictions.csv")


if __name__ == "__main__":
    main()
