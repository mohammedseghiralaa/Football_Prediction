import logging 
import numpy as np
import pandas as pd
import config

logger = logging.getLogger(__name__)

TARGET = "result"  # 0 = away win, 1 = draw, 2 = home win

def add_target(df: pd.DataFrame) -> pd.DataFrame:
    hg, ag = df["FullTimeHomeGoals"], df["FullTimeAwayGoals"]
    df[TARGET] = np.select([hg > ag, hg == ag], [2, 1], default=0)
    return df

def get_feature_cols(df: pd.DataFrame) -> list[str]:
    # whitelist: only pre-match features, nothing from the match itself
    prefixes = ("home_form", "away_form", "diff_", "hist_")
    return [c for c in df.columns if c.startswith(prefixes)]

def time_split(df: pd.DataFrame, test_frac: float = 0.2):
    df = df.sort_values(config.DATE_COL, kind="stable").reset_index(drop=True)
    cut = int(len(df) * (1 - test_frac))
    return df.iloc[:cut], df.iloc[cut:]
def time_split(df: pd.DataFrame, test_frac: float = 0.2):
    df = df.sort_values(config.DATE_COL, kind="stable").reset_index(drop=True)
    cut = int(len(df) * (1 - test_frac))
    return df.iloc[:cut], df.iloc[cut:]
