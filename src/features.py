"""Build leakage-free rolling-form features for each match."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)

FORM_METRICS = ["points", "goals_for", "goals_against", "shots", "shots_on_target"]


def to_long_format(df: pd.DataFrame) -> pd.DataFrame:
    """Turn one row per match into one row per team per match."""

    def side(team: str, opp: str, prefix: str, opp_prefix: str, is_home: int) -> pd.DataFrame:
        out = pd.DataFrame({
            "match_id": df.index,
            "team": df[team],
            "is_home": is_home,
            "goals_for": df[f"FullTime{prefix}Goals"],
            "goals_against": df[f"FullTime{opp_prefix}Goals"],
            "shots": df[f"{prefix}Shots"],
            "shots_on_target": df[f"{prefix}ShotsOnTarget"],
            "ht_goals": df[f"HalfTime{prefix}Goals"],
        })
        out["points"] = np.select(
            [out["goals_for"] > out["goals_against"],
             out["goals_for"] == out["goals_against"]],
            [3, 1],
            default=0,
        )
        return out

    home = side("HomeTeam", "AwayTeam", "Home", "Away", 1)
    away = side("AwayTeam", "HomeTeam", "Away", "Home", 0)
    return pd.concat([home, away], ignore_index=True)


def add_rolling_form(long_df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Average of each team's PREVIOUS `window` matches (current match excluded)."""
    # match_id follows chronological order, so sorting by it keeps time order
    long_df = long_df.sort_values(["team", "match_id"]).reset_index(drop=True)

    grouped = long_df.groupby("team")[FORM_METRICS]
    form = grouped.transform(
        lambda s: s.shift(1).rolling(window, min_periods=3).mean()
    )
    form.columns = [f"form_{c}" for c in FORM_METRICS]
    return pd.concat([long_df[["match_id", "team", "is_home"]], form], axis=1)



def build_features(df: pd.DataFrame, window: int = config.ROLLING_WINDOW) -> pd.DataFrame:
    long_df = add_rolling_form(to_long_format(df), window)
    form_cols = [f"form_{c}" for c in FORM_METRICS]

    home = (long_df[long_df["is_home"] == 1]
            .set_index("match_id")[form_cols].add_prefix("home_"))
    away = (long_df[long_df["is_home"] == 0]
            .set_index("match_id")[form_cols].add_prefix("away_"))

    out = df.join(home).join(away)

    # Difference features: how much better is the home team's form?
    for c in form_cols:
        out[f"diff_{c}"] = out[f"home_{c}"] - out[f"away_{c}"]
    return out

def add_history_features(long_df: pd.DataFrame, min_matches: int = 5) -> pd.DataFrame:
    """Expanding-window ratios using ALL previous matches of each team (current excluded)."""
    long_df = long_df.sort_values(["team", "match_id"]).reset_index(drop=True)
    grouped = long_df.groupby("team")

    cols = ["goals_for", "ht_goals", "shots", "shots_on_target", "points"]
    for c in cols:
        # cumsum includes the current row, so subtract it to keep only the PAST
        long_df[f"cum_{c}"] = grouped[c].cumsum() - long_df[c]
    long_df["cum_matches"] = grouped.cumcount()

    def ratio(num: str, den: str) -> pd.Series:
        return long_df[num] / long_df[den].replace(0, np.nan)

    long_df["hist_points_per_match"] = ratio("cum_points", "cum_matches")
    long_df["hist_first_half_goal_share"] = ratio("cum_ht_goals", "cum_goals_for")
    long_df["hist_shot_accuracy"] = ratio("cum_shots_on_target", "cum_shots")
    long_df["hist_conversion_on_target"] = ratio("cum_goals_for", "cum_shots_on_target")
    long_df["hist_conversion_all_shots"] = ratio("cum_goals_for", "cum_shots")

    hist_cols = [c for c in long_df.columns if c.startswith("hist_")]
    long_df.loc[long_df["cum_matches"] < min_matches, hist_cols] = np.nan
    return long_df[["match_id", "team", "is_home"] + hist_cols]
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    df = pd.read_csv(config.CLEAN_PATH, parse_dates=[config.DATE_COL])
    df = df.sort_values(config.DATE_COL, kind="stable").reset_index(drop=True)

    features = build_features(df)

    n_before = len(features)
    feature_cols = [c for c in features.columns if c.startswith(("home_form", "away_form", "diff_"))]
    features = features.dropna(subset=feature_cols).reset_index(drop=True)
    logger.info("Dropped %d early matches without enough history", n_before - len(features))

    logger.info("Feature columns (%d): %s", len(feature_cols), feature_cols)
    logger.info("Final shape: %s", features.shape)

    features.to_csv(config.FEATURES_PATH, index=False)
    logger.info("Saved to %s", config.FEATURES_PATH)


if __name__ == "__main__":
    main()