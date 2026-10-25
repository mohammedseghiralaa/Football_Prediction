"""Streamlit demo: predict a Premier League match from Elo and recent shot form."""
import sys
from pathlib import Path

# Make the project root importable when running `streamlit run app/app.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import pandas as pd
import streamlit as st

from src import config
from src.elo import current_ratings

FORM_WINDOW = 5


@st.cache_resource
def load_model():
    bundle = joblib.load(config.MODEL_PATH)
    return bundle["model"], bundle["features"]


@st.cache_data
def load_state():
    df = pd.read_csv(config.CLEAN_PATH, parse_dates=[config.DATE_COL])
    df = df.sort_values(config.DATE_COL, kind="stable").reset_index(drop=True)
    ratings = current_ratings(df)
    last = df[df["Season"] == df["Season"].iloc[-1]]
    teams = sorted(set(last["HomeTeam"]) | set(last["AwayTeam"]))
    return df, ratings, teams, df[config.DATE_COL].max()


def recent_shots(df: pd.DataFrame, team: str, n: int = FORM_WINDOW):
    """Average shots and shots on target over the team's last n matches."""
    home = (df[df["HomeTeam"] == team]
            [[config.DATE_COL, "HomeShots", "HomeShotsOnTarget"]]
            .rename(columns={"HomeShots": "shots", "HomeShotsOnTarget": "sot"}))
    away = (df[df["AwayTeam"] == team]
            [[config.DATE_COL, "AwayShots", "AwayShotsOnTarget"]]
            .rename(columns={"AwayShots": "shots", "AwayShotsOnTarget": "sot"}))
    last = pd.concat([home, away]).sort_values(config.DATE_COL).tail(n)
    return last["shots"].mean(), last["sot"].mean()


def build_row(df, ratings, home: str, away: str) -> dict:
    h_shots, h_sot = recent_shots(df, home)
    a_shots, a_sot = recent_shots(df, away)
    return {
        "home_elo": ratings[home], "away_elo": ratings[away],
        "diff_elo": ratings[home] - ratings[away],
        "home_form_shots": h_shots, "home_form_shots_on_target": h_sot,
        "away_form_shots": a_shots, "away_form_shots_on_target": a_sot,
        "diff_form_shots": h_shots - a_shots,
        "diff_form_shots_on_target": h_sot - a_sot,
    }


st.set_page_config(page_title="Football Match Predictor", page_icon="⚽")
st.title("⚽ Premier League Match Predictor")
st.caption("Logistic Regression on Elo ratings and recent shot form. "
           "Walk-forward tested; accuracy about 53% on the 2024/25 test season.")

model, feats = load_model()
df, ratings, teams, last_date = load_state()

col1, col2 = st.columns(2)
home = col1.selectbox("Home team", teams, index=0)
away = col2.selectbox("Away team", teams, index=1)

if home == away:
    st.warning("Choose two different teams.")
else:
    row = build_row(df, ratings, home, away)
    X = pd.DataFrame([row])[feats]
    p_away, p_draw, p_home = model.predict_proba(X)[0]  # classes: 0=A, 1=D, 2=H

    c1, c2, c3 = st.columns(3)
    c1.metric(f"{home} win", f"{p_home:.1%}")
    c2.metric("Draw", f"{p_draw:.1%}")
    c3.metric(f"{away} win", f"{p_away:.1%}")

    st.bar_chart(pd.Series({f"{home} win": p_home, "Draw": p_draw,
                            f"{away} win": p_away}))

    with st.expander("Inputs used by the model"):
        st.write(f"Elo: {home} {row['home_elo']:.0f} vs {away} {row['away_elo']:.0f}")
        st.write(f"Average shots (last {FORM_WINDOW}): "
                 f"{home} {row['home_form_shots']:.1f}, {away} {row['away_form_shots']:.1f}")

st.info(f"Data ends on {last_date.date()}. Ratings reflect results up to that date. "
        "This is a probability model for learning, not betting advice.")