"""Walk-forward ablation: which feature groups really help? Test season is never used."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config
from train import get_feature_columns

logger = logging.getLogger(__name__)

# 2015/16 ... 2023/24 (2024/25 is the locked test season)
EVAL_SEASONS = [f"{y}/{str(y + 1)[-2:]}" for y in range(2015, 2024)]

# Feature groups, matched by substring in the column name
GROUPS = {
    "elo": ["elo"],
    "hist_points": ["hist_points_per_match"],
    "form_points_goals": ["form_points", "form_goals_for", "form_goals_against"],
    "form_shots": ["form_shots"],  # also matches form_shots_on_target
    "shot_ratios": ["hist_shot_accuracy", "hist_conversion"],
    "first_half_share": ["first_half_goal_share"],
}


def group_columns(all_feats: list[str], group: str) -> list[str]:
    keys = GROUPS[group]
    return [c for c in all_feats if any(k in c for k in keys)]


MODELS = {
    "LightGBM": lambda: LGBMClassifier(
        n_estimators=200, learning_rate=0.03, max_depth=3,
        subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1),
    "LogReg": lambda: make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=2000, C=0.1)),
}


def walk_forward(df: pd.DataFrame, features: list[str], make_model) -> float:
    """Pooled log loss over all evaluation seasons (always train on the past only)."""
    probas, ys = [], []
    for season in EVAL_SEASONS:
        train = df[df["Season"] < season]
        test = df[df["Season"] == season]
        model = make_model()
        model.fit(train[features], train["y"])
        probas.append(model.predict_proba(test[features]))
        ys.append(test["y"].to_numpy())
    return log_loss(np.concatenate(ys), np.vstack(probas), labels=[0, 1, 2])


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    df = pd.read_csv(config.FEATURES_PATH, parse_dates=[config.DATE_COL])
    df = df.sort_values(config.DATE_COL, kind="stable").reset_index(drop=True)
    df["y"] = df[config.TARGET_COL].map(config.LABELS)

    all_feats = get_feature_columns(df)

    # Safety check: every feature belongs to exactly one group
    covered = [c for g in GROUPS for c in group_columns(all_feats, g)]
    assert sorted(covered) == sorted(all_feats), "Groups do not cover all features"

    experiments = {"all features": all_feats,
                   "only Elo": group_columns(all_feats, "elo")}
    for g in GROUPS:
        drop = set(group_columns(all_feats, g))
        experiments[f"without {g}"] = [c for c in all_feats if c not in drop]

    rows = {}
    for name, feats in experiments.items():
        rows[name] = {m: walk_forward(df, feats, mk) for m, mk in MODELS.items()}
        logger.info("Done: %s (%d features)", name, len(feats))

    table = pd.DataFrame(rows).T
    for m in MODELS:
        table[f"{m}_delta"] = table[m] - table.loc["all features", m]

    logger.info("Walk-forward log loss (lower is better; delta > 0 means worse "
                "than all features):\n%s", table.round(4).to_string())


if __name__ == "__main__":
    main()