"""Compare feature sets, then tune Elo's K and home advantage (walk-forward, test season untouched)."""
from __future__ import annotations

import logging

import pandas as pd

import config
from ablation import MODELS, group_columns, walk_forward
from elo import add_elo
from train import get_feature_columns

logger = logging.getLogger(__name__)

KEYS = [config.DATE_COL, "HomeTeam", "AwayTeam"]
ELO_COLS = ["home_elo", "away_elo", "diff_elo"]


def with_elo(feat: pd.DataFrame, clean: pd.DataFrame, k: float, home_adv: float) -> pd.DataFrame:
    """Recompute Elo on ALL matches, then attach it to the feature table."""
    elo = add_elo(clean, k=k, home_adv=home_adv)[KEYS + ELO_COLS]
    return feat.drop(columns=ELO_COLS).merge(elo, on=KEYS, how="left")


def score(df: pd.DataFrame, feats: list[str]) -> dict:
    return {m: walk_forward(df, feats, mk) for m, mk in MODELS.items()}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    feat = pd.read_csv(config.FEATURES_PATH, parse_dates=[config.DATE_COL])
    feat = feat.sort_values(config.DATE_COL, kind="stable").reset_index(drop=True)
    feat["y"] = feat[config.TARGET_COL].map(config.LABELS)

    clean = pd.read_csv(config.CLEAN_PATH, parse_dates=[config.DATE_COL])
    clean = clean.sort_values(config.DATE_COL, kind="stable").reset_index(drop=True)

    all_feats = get_feature_columns(feat)
    reduced = group_columns(all_feats, "elo") + group_columns(all_feats, "form_shots")
    logger.info("Reduced feature set (%d): %s", len(reduced), reduced)

    # 1. Does the reduced set match the full set?
    sets = pd.DataFrame({"all features": score(feat, all_feats),
                         "elo + shots": score(feat, reduced)}).T
    logger.info("Feature sets (walk-forward log loss):\n%s", sets.round(4).to_string())

    # 2. Tune Elo parameters using the reduced set
    rows = {}
    for k in (10, 20, 30, 40):
        for adv in (50, 65, 80):
            rows[f"K={k}, adv={adv}"] = score(with_elo(feat, clean, k, adv), reduced)
            logger.info("Done K=%d adv=%d", k, adv)

    grid = pd.DataFrame(rows).T
    grid["mean"] = grid.mean(axis=1)
    logger.info("Elo tuning (lower is better):\n%s",
                grid.sort_values("mean").round(4).to_string())


if __name__ == "__main__":
    main()