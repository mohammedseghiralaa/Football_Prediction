"""Tune Logistic Regression's C with walk-forward validation (test season untouched)."""
from __future__ import annotations

import logging

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config
from ablation import group_columns, walk_forward
from train import get_feature_columns

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    df = pd.read_csv(config.FEATURES_PATH, parse_dates=[config.DATE_COL])
    df = df.sort_values(config.DATE_COL, kind="stable").reset_index(drop=True)
    df["y"] = df[config.TARGET_COL].map(config.LABELS).astype(int)

    all_feats = get_feature_columns(df)
    feats = group_columns(all_feats, "elo") + group_columns(all_feats, "form_shots")

    rows = {}
    for c in (0.003, 0.01, 0.03, 0.1, 0.3, 1.0):
        make = lambda c=c: make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, C=c))
        rows[f"C={c}"] = walk_forward(df, feats, make)

    table = pd.Series(rows, name="log_loss").sort_values()
    logger.info("Walk-forward log loss by C (lower is better):\n%s", table.round(4).to_string())


if __name__ == "__main__":
    main()