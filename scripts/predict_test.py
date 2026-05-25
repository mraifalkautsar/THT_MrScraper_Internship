import argparse
import sys
from dataclasses import replace
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import PipelineConfig
from src.inference import run_prediction
from src.strategies import ALL_PREDICTION_VARIANTS, ANCHOR_GATED_FALLBACK_CALIBRATION


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fill missing prices in the test file.")
    parser.add_argument("--train-path", default="ecommerce_price_prediction-train.csv")
    parser.add_argument("--test-path", default="ecommerce_price_prediction-test-3-days.csv")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--iterations", type=int, default=1200)
    parser.add_argument("--second-stage-alpha", type=float, default=10.0)
    parser.add_argument("--second-stage-min-anchors", type=int, default=10)
    parser.add_argument(
        "--prediction-variant",
        default=ANCHOR_GATED_FALLBACK_CALIBRATION,
        choices=ALL_PREDICTION_VARIANTS,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = replace(
        PipelineConfig(),
        train_path=Path(args.train_path),
        test_path=Path(args.test_path),
        output_dir=Path(args.output_dir),
        catboost_iterations=args.iterations,
        prediction_variant=args.prediction_variant,
        second_stage_alpha=args.second_stage_alpha,
        second_stage_min_anchors=args.second_stage_min_anchors,
    )
    submission = run_prediction(config)
    output_path = config.output_dir / "completed_test_predictions.csv"
    print(f"Saved {output_path}")
    print(f"Rows: {len(submission)}")
    print(f"Missing prices remaining: {submission[config.target].isna().sum()}")


if __name__ == "__main__":
    main()
