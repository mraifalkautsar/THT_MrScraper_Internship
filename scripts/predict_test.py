import argparse
import sys
from dataclasses import replace
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import PipelineConfig
from src.inference import run_prediction
from src.strategies import ALL_PREDICTION_VARIANTS, HYBRID_LAST_PRICE_ENTITY


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fill missing prices in the test file.")
    parser.add_argument("--train-path", default="ecommerce_price_prediction-train.csv")
    parser.add_argument("--test-path", default="ecommerce_price_prediction-test-3-days.csv")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--iterations", type=int, default=1200)
    parser.add_argument(
        "--prediction-variant",
        default=HYBRID_LAST_PRICE_ENTITY,
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
    )
    submission = run_prediction(config)
    output_path = config.output_dir / "completed_test_predictions.csv"
    print(f"Saved {output_path}")
    print(f"Rows: {len(submission)}")
    print(f"Missing prices remaining: {submission[config.target].isna().sum()}")


if __name__ == "__main__":
    main()
