"""Time-based split and model comparison on the VALIDATION set only."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config

logger = logging.getLogger(__name__)


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith(("home_", "away_", "diff_"))]


def time_split(df: pd.DataFrame):
    """Split by season so that training data is always older than evaluation data."""
    is_test = df["Season"].isin(config.TEST_SEASONS)
    is_val = df["Season"].isin(config.VAL_SEASONS)
    return df[~is_test & ~is_val], df[is_val], df[is_test]


def get_models() -> dict:
    return {
        "Baseline (majority class)": DummyClassifier(strategy="prior"),
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, C=0.1)
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=20, n_jobs=-1, random_state=42
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=200, learning_rate=0.03, max_depth=3,
            subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1,
        ),
    }


def evaluate(model, X, y) -> dict:
    proba = model.predict_proba(X)
    pred = proba.argmax(axis=1)
    return {
        "accuracy": accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro"),
        "log_loss": log_loss(y, proba, labels=[0, 1, 2]),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    df = pd.read_csv(config.FEATURES_PATH, parse_dates=[config.DATE_COL])
    df = df.sort_values(config.DATE_COL, kind="stable").reset_index(drop=True)

    features = get_feature_columns(df)
    train, val, test = time_split(df)
    logger.info("Train: %d | Validation: %d | Test (locked): %d",
                len(train), len(val), len(test))

    X_train, X_val = train[features], val[features]
    y_train = train[config.TARGET_COL].map(config.LABELS)
    y_val = val[config.TARGET_COL].map(config.LABELS)

    results, fitted = {}, {}
    for name, model in get_models().items():
        model.fit(X_train, y_train)
        results[name] = evaluate(model, X_val, y_val)
        fitted[name] = model

    table = pd.DataFrame(results).T.sort_values("log_loss")
    logger.info("Validation results:\n%s", table.round(4).to_string())

    # Which features does LightGBM find most useful?
    importance = pd.Series(
        fitted["LightGBM"].booster_.feature_importance(importance_type="gain"),
        index=features,
    ).sort_values(ascending=False)
    logger.info("Top 10 features (LightGBM gain):\n%s", importance.head(10).round(1).to_string())


if __name__ == "__main__":
    main()