"""Load, validate, clean, and save the raw Premier League match data."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src import config

logger = logging.getLogger(__name__)


def load_raw(path: Path) -> pd.DataFrame:
    """Read the raw CSV file."""
    if not path.exists():
        raise FileNotFoundError(f"Raw data not found: {path}")
    df = pd.read_csv(path)
    logger.info("Loaded %d rows and %d columns from %s", *df.shape, path.name)
    return df


def check_required_columns(df: pd.DataFrame) -> None:
    """Fail early if essential columns are missing."""
    missing = set(config.REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Parse dates, strip text, drop duplicates and incomplete rows, sort by time."""
    df = df.copy()

    df[config.DATE_COL] = pd.to_datetime(df[config.DATE_COL], errors="coerce")

    for col in ("HomeTeam", "AwayTeam", config.TARGET_COL):
        df[col] = df[col].astype(str).str.strip()

    n_before = len(df)
    df = df.drop_duplicates()
    logger.info("Removed %d duplicate/incomplete rows", n_before - len(df))

    # Chronological order is essential to avoid data leakage later
    df = df.sort_values(config.DATE_COL, kind="stable").reset_index(drop=True)
    return df


def validate(df: pd.DataFrame) -> None:
    """Run data-quality checks. Raise on critical errors, warn on minor ones."""
    # 1. Target values must be H, D, or A
    invalid = set(df[config.TARGET_COL].unique()) - config.VALID_RESULTS
    if invalid:
        raise ValueError(f"Invalid values in {config.TARGET_COL}: {invalid}")

    # 2. Result must be consistent with the goals scored
    expected = np.select(
        [df["FullTimeHomeGoals"] > df["FullTimeAwayGoals"],
         df["FullTimeHomeGoals"] < df["FullTimeAwayGoals"]],
        ["H", "A"],
        default="D",
    )
    mismatches = int((expected != df[config.TARGET_COL]).sum())
    if mismatches:
        raise ValueError(f"{mismatches} rows where result disagrees with goals")

    # 3. A team cannot play itself
    if (df["HomeTeam"] == df["AwayTeam"]).any():
        raise ValueError("Found matches where HomeTeam equals AwayTeam")

    # 4. Stats must be non-negative (only checks columns that exist)
    stats = [c for c in config.STAT_COLUMNS if c in df.columns]
    negatives = int((df[stats] < 0).sum().sum())
    if negatives:
        raise ValueError(f"Found {negatives} negative values in match statistics")

    # 5. Soft check: matches per season should be roughly 380
    per_season = df.groupby("Season").size()
    odd = per_season[(per_season < 300) | (per_season > 400)]
    if not odd.empty:
        logger.warning("Unusual number of matches in seasons:\n%s", odd.to_string())

    logger.info("All validation checks passed")


def summarize(df: pd.DataFrame) -> None:
    """Log key facts about the dataset, including the baseline accuracy."""
    dist = df[config.TARGET_COL].value_counts(normalize=True).round(3)
    baseline = dist.max()

    logger.info("Shape: %s", df.shape)
    logger.info("Date range: %s to %s",
                df[config.DATE_COL].min().date(), df[config.DATE_COL].max().date())
    logger.info("Seasons: %d | Teams: %d",
                df["Season"].nunique(), df["HomeTeam"].nunique())
    logger.info("Matches per season:\n%s",
                df.groupby("Season").size().to_string())
    logger.info("Result distribution:\n%s", dist.to_string())
    logger.info("Baseline accuracy (always predict '%s'): %.1f%%",
                dist.idxmax(), baseline * 100)
    logger.info("Average goals - home: %.2f | away: %.2f",
                df["FullTimeHomeGoals"].mean(), df["FullTimeAwayGoals"].mean())


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    df = load_raw(config.RAW_PATH)
    check_required_columns(df)
    df = clean(df)
    validate(df)
    summarize(df)

    df.to_csv(config.CLEAN_PATH, index=False)
    print(df.shape)
    logger.info("Saved clean data to %s", config.CLEAN_PATH)


if __name__ == "__main__":
    main()