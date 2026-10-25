"""Final model: check calibration, train on all pre-test seasons, evaluate ONCE on the test season."""
from __future__ import annotations

import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config
from ablation import EVAL_SEASONS, group_columns
from train import get_feature_columns

logger = logging.getLogger(__name__)

LABEL_NAMES = {0: "Away", 1: "Draw", 2: "Home"}


def make_model():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.1))


def walk_forward_probas(df: pd.DataFrame, feats: list[str]):
    """Out-of-sample probabilities for every evaluation season (test season excluded)."""
    probas, ys = [], []
    for season in EVAL_SEASONS:
        train = df[df["Season"] < season]
        test = df[df["Season"] == season]
        model = make_model().fit(train[feats], train["y"])
        probas.append(model.predict_proba(test[feats]))
        ys.append(test["y"].to_numpy())
    return np.vstack(probas), np.concatenate(ys)


def calibration_table(proba: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    """For each class: predicted probability vs how often it really happened, by quartile."""
    out = []
    for cls, name in LABEL_NAMES.items():
        d = pd.DataFrame({"predicted": proba[:, cls], "actual": (y == cls).astype(float)})
        d["quartile"] = pd.qcut(d["predicted"], 4, labels=False, duplicates="drop")
        g = d.groupby("quartile")[["predicted", "actual"]].mean()
        g.insert(0, "class", name)
        out.append(g)
    return pd.concat(out).round(3)


def per_match_loss(proba: np.ndarray, y: np.ndarray) -> np.ndarray:
    return -np.log(np.clip(proba[np.arange(len(y)), y], 1e-15, 1.0))


def bootstrap_ci(values: np.ndarray, n: int = 5000, seed: int = 42):
    rng = np.random.default_rng(seed)
    means = [rng.choice(values, len(values)).mean() for _ in range(n)]
    return np.percentile(means, [2.5, 97.5])


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    df = pd.read_csv(config.FEATURES_PATH, parse_dates=[config.DATE_COL])
    df = df.sort_values(config.DATE_COL, kind="stable").reset_index(drop=True)
    df["y"] = df[config.TARGET_COL].map(config.LABELS).astype(int)

    all_feats = get_feature_columns(df)
    feats = group_columns(all_feats, "elo") + group_columns(all_feats, "form_shots")
    logger.info("Using %d features: %s", len(feats), feats)

    # 1. Calibration check on walk-forward predictions (no test data involved)
    oof_proba, oof_y = walk_forward_probas(df, feats)
    logger.info("Calibration (walk-forward). 'predicted' should be close to 'actual':\n%s",
                calibration_table(oof_proba, oof_y).to_string())

    # 2. Train on every season before the test season
    is_test = df["Season"].isin(config.TEST_SEASONS)
    train, test = df[~is_test], df[is_test]
    model = make_model().fit(train[feats], train["y"])

    # 3. Evaluate ONCE on the test season
    y = test["y"].to_numpy()
    proba = model.predict_proba(test[feats])
    pred = proba.argmax(axis=1)

    priors = train["y"].value_counts(normalize=True).sort_index().to_numpy()
    base_proba = np.tile(priors, (len(test), 1))
    base_pred = base_proba.argmax(axis=1)

    results = pd.DataFrame({
        "accuracy": [accuracy_score(y, pred), accuracy_score(y, base_pred)],
        "macro_f1": [f1_score(y, pred, average="macro"),
                     f1_score(y, base_pred, average="macro")],
        "log_loss": [log_loss(y, proba, labels=[0, 1, 2]),
                     log_loss(y, base_proba, labels=[0, 1, 2])],
    }, index=["Final model", "Baseline"])
    logger.info("TEST RESULTS (%s, %d matches):\n%s",
                config.TEST_SEASONS, len(test), results.round(4).to_string())

    cm = pd.DataFrame(confusion_matrix(y, pred, labels=[0, 1, 2]),
                      index=[f"true {n}" for n in LABEL_NAMES.values()],
                      columns=[f"pred {n}" for n in LABEL_NAMES.values()])
    logger.info("Confusion matrix:\n%s", cm.to_string())

    # 4. How much can we trust a 350-match test? Bootstrap 95% intervals
    gain = per_match_loss(base_proba, y) - per_match_loss(proba, y)
    lo, hi = bootstrap_ci(gain)
    logger.info("Log-loss improvement over baseline: %.4f (95%% CI %.4f to %.4f)",
                gain.mean(), lo, hi)
    lo, hi = bootstrap_ci((pred == y).astype(float))
    logger.info("Accuracy: %.3f (95%% CI %.3f to %.3f)", (pred == y).mean(), lo, hi)

    # 5. Save the model for the demo app
    config.MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump({"model": model, "features": feats}, config.MODEL_PATH)
    logger.info("Saved model to %s", config.MODEL_PATH)


if __name__ == "__main__":
    main()