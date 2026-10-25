"""Elo ratings computed chronologically. Only PRE-match ratings are stored."""
from __future__ import annotations

import pandas as pd

K = 20            # how fast ratings move
HOME_ADV = 65     # Elo points of home advantage
INITIAL = 1500    # starting rating for every team


def _run(df: pd.DataFrame, k: float, home_adv: float, initial: float):
    """Process matches in order. Returns pre-match ratings and the final ratings."""
    ratings: dict[str, float] = {}
    home_pre, away_pre = [], []

    cols = ["HomeTeam", "AwayTeam", "FullTimeHomeGoals", "FullTimeAwayGoals"]
    for home, away, hg, ag in df[cols].itertuples(index=False):
        r_home = ratings.get(home, initial)
        r_away = ratings.get(away, initial)

        # Store the rating BEFORE the match (this is what the model may use)
        home_pre.append(r_home)
        away_pre.append(r_away)

        # Expected score of the home team
        expected = 1 / (1 + 10 ** ((r_away - (r_home + home_adv)) / 400))
        actual = 1.0 if hg > ag else 0.5 if hg == ag else 0.0

        # Bigger wins move the rating more
        gd = abs(hg - ag)
        mult = 1.0 if gd <= 1 else 1.5 if gd == 2 else (11 + gd) / 8

        delta = k * mult * (actual - expected)
        ratings[home] = r_home + delta
        ratings[away] = r_away - delta

    return home_pre, away_pre, ratings


def add_elo(df: pd.DataFrame, k: float = K, home_adv: float = HOME_ADV,
            initial: float = INITIAL) -> pd.DataFrame:
    """Add home_elo, away_elo, diff_elo. df must be sorted by date."""
    home_pre, away_pre, _ = _run(df, k, home_adv, initial)
    out = df.copy()
    out["home_elo"] = home_pre
    out["away_elo"] = away_pre
    out["diff_elo"] = out["home_elo"] - out["away_elo"]
    return out


def current_ratings(df: pd.DataFrame, k: float = K, home_adv: float = HOME_ADV,
                    initial: float = INITIAL) -> dict[str, float]:
    """Each team's rating after the last match in df (used for future predictions)."""
    return _run(df, k, home_adv, initial)[2]