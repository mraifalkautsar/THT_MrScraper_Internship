from pathlib import Path

import pandas as pd

from .config import PipelineConfig


def load_data(train_path: str | Path, test_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the historical train data and outage-style test data from CSV files."""
    return pd.read_csv(train_path), pd.read_csv(test_path)


def basic_preprocess(df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """Normalize timestamps, date keys, booleans, and categorical missing values."""
    df = df.copy()
    df[config.date_col] = pd.to_datetime(df[config.date_col], errors="coerce")
    df["date"] = df[config.date_col].dt.date.astype(str)

    for col in config.bool_cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.lower()
                .map(
                    {
                        "true": 1,
                        "t": 1,
                        "1": 1,
                        "yes": 1,
                        "false": 0,
                        "f": 0,
                        "0": 0,
                        "no": 0,
                    }
                )
                .fillna(0)
                .astype(int)
            )

    for col in config.cat_cols:
        if col in df.columns:
            df[col] = df[col].where(df[col].notna(), "missing").astype(str)

    return df


def assert_submission_valid(
    original_test: pd.DataFrame, submission: pd.DataFrame, config: PipelineConfig
) -> None:
    """Validate final output shape, filled prices, nonnegative values, and anchor preservation."""
    if len(original_test) != len(submission):
        raise ValueError("Submission row count does not match test row count.")
    if submission[config.target].isna().any():
        raise ValueError("Submission still contains missing prices.")
    if (submission[config.target] < 0).any():
        raise ValueError("Submission contains negative prices.")

    anchor_mask = original_test[config.target].notna()
    if anchor_mask.any():
        original_anchor = original_test.loc[anchor_mask, config.target].round().astype("int64")
        submitted_anchor = submission.loc[anchor_mask, config.target].astype("int64")
        if not original_anchor.equals(submitted_anchor):
            raise ValueError("Known anchor prices were modified in the submission.")
