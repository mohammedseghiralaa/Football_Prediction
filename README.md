# Premier League Match Outcome Prediction

An end-to-end ML project that predicts Premier League results (home win / draw / away win)
from Elo ratings and recent shot form, with a leakage-free pipeline, walk-forward
validation, and a Streamlit demo.

## Results (2024/25 test season, 350 matches, evaluated once)

|                                       | Accuracy                     | Log loss  |
| ------------------------------------- | ---------------------------- | --------- |
| **Final model (Logistic Regression)** | **53.4%** (95% CI 48.3-58.9) | **0.988** |
| Baseline (always predict home win)    | 41.1%                        | 1.085     |

- Log-loss improvement over baseline: 0.097 (95% CI 0.051 to 0.143), so the gain is statistically real.
- Probabilities are well calibrated: predicted and observed frequencies are within about 1-4 points.
- Walk-forward validation (2015/16 to 2023/24, about 3,400 matches): log loss about 0.966.

## Data

- [Dataset name and link], 9,380 Premier League matches, 2000/01 to 2024/25.
- Columns: goals, half-time goals, shots, shots on target, corners, fouls, cards.
- Not included: bookmaker odds.

## Method

1. **Cleaning and validation:** date parsing, duplicate removal, and consistency checks (result vs. goals).
2. **Leakage-free features:** every feature uses only matches played _before_ the one being predicted.
3. **Elo rating** built from scratch (K=20, home advantage 65, goal-difference multiplier).
4. **Time-based evaluation:** train on the past, test on the future. The test season was locked until the end and used once.
5. **Ablation study:** removed feature groups one at a time using walk-forward validation.
6. **Final model:** Logistic Regression (C=0.1) on 9 features: Elo (3) and recent shots / shots on target (6).

## Key findings

- **Elo is the dominant feature.** Removing it costs the most, and Elo alone nearly matches the full 33-feature set.
- **Recent shots add real signal** beyond Elo (they describe how a team plays, not just results).
- **Ideas tested and rejected:** shot accuracy, conversion rate, first-half goal share, and expanding points per match. None improved results once Elo was included.
- **A simple model matches LightGBM and Random Forest.** Better features mattered more than a more complex algorithm.

## Limitations

- The model never predicts a draw as the single most likely outcome (typical for football models), but its draw probabilities are calibrated.
- The test season has only 350 matches, so uncertainty is wide (about +/-5 points of accuracy).
- Some matches are missing from the dataset (335-350 instead of 380 in a few seasons).
- No bookmaker odds, so **no claim is made about beating the betting market.**
- Data ends in May 2025; the demo's ratings reflect that date.

## Run it

```bash
pip install -r requirements.txt
python -m src.clean_data
python -m src.features
python -m src.train
python -m src.ablation
python -m src.final
python -m streamlit run app/app.py
```

## Project structure

```
data/      datasets (not tracked)
src/       config, cleaning, features, Elo, training, ablation, final evaluation
app/       Streamlit demo
models/    saved model (not tracked)
```
