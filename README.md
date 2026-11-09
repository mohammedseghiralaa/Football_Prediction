# Premier League Match Outcome Prediction

> Predicts Premier League results (home win / draw / away win) with **53.4% accuracy** and **0.988 log loss** on a held-out season, beating the "always home win" baseline (41.1% / 1.085). Built around a leakage-free pipeline, walk-forward validation, and a test set used exactly once.

**Python:** 3.10+ | **License:** MIT | **Random seed:** 42 (set in `src/config.py`)

---

## Why this project is different

Most football-prediction repos report a good-looking accuracy and stop there. This one focuses on *whether the number can be trusted*:

- **No leakage:** every feature uses only matches played *before* the one being predicted (enforced by unit tests).
- **Time-based evaluation:** train on the past, test on the future. The 2024/25 season was locked until the end and evaluated once.
- **Ablation study:** feature groups removed one at a time to see what actually matters.
- **Honest claims:** no bookmaker odds in the data, so no claim of beating the market.

## Results (2024/25 test season, 350 matches, evaluated once)

| | Accuracy | Log loss | RPS |
| --- | --- | --- | --- |
| **Final model (Logistic Regression)** | **53.4%** (95% CI 48.3-58.9) | **0.988** | _TODO_ |
| Baseline (always predict home win) | 41.1% | 1.085 | _TODO_ |

- Log-loss improvement over baseline: **0.097** (95% CI 0.051 to 0.143, paired bootstrap over matches), so the gain is statistically real.
- Probabilities are well calibrated: predicted and observed frequencies are within about 1-4 points.
- Walk-forward validation (2015/16 to 2023/24, about 3,400 matches): log loss about **0.966**.
- The test log loss (0.988) is slightly higher than the walk-forward average (0.966). With only 350 test matches, season-to-season variance of this size is expected, and the gap is well inside the uncertainty range.

> **RPS (Ranked Probability Score)** is the standard football metric because it respects the ordering home > draw > away. Lower is better.

## Data

- **Source:** [DATASET NAME](DATASET_LINK), 9,380 Premier League matches, 2000/01 to 2024/25.
- **Columns:** goals, half-time goals, shots, shots on target, corners, fouls, cards.
- **Not included:** bookmaker odds.
- **Getting the data:** `data/` is not tracked. Download the dataset from the link above and place the file at `data/raw/matches.csv` (or run `python -m src.download_data` if you add that script).
- **Known gaps:** a few seasons have 335-350 matches instead of 380. A missing match means one Elo update never happens, so ratings after that point can be slightly off. _TODO: state whether you measured the effect (for example, by comparing Elo with and without the affected seasons)._

## Method

1. **Cleaning and validation:** date parsing, duplicate removal, and consistency checks (result vs. goals).
2. **Leakage-free features:** every feature uses only matches played _before_ the one being predicted.
3. **Elo rating** built from scratch:
   - K = 20, home advantage = 65, goal-difference multiplier.
   - _TODO: say how these were chosen (literature defaults, or tuned on which seasons)._
   - _TODO: say how Elo is initialized for the first season and for newly promoted teams._
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
- No bookmaker odds, so **no claim is made about beating the betting market.**
- Data ends in May 2025; the demo's ratings reflect that date.

## Next steps

- Compare against bookmaker closing odds, the real benchmark for any football model.
- Add expected goals (xG) and player-level features.
- Try an ordinal model (home > draw > away) to handle draws better.
- Update the data and Elo ratings automatically each matchday.

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

Run the tests (including the leakage check):

```bash
pytest
```

## Project structure

```
data/      datasets (not tracked)
src/
  config.py       paths, seeds, hyperparameters
  clean_data.py   parsing, deduplication, consistency checks
  elo.py          Elo rating implementation
  features.py     leakage-free feature construction
  train.py        walk-forward training and validation
  ablation.py     feature-group ablation study
  final.py        one-time evaluation on the locked test season
app/       Streamlit demo
models/    saved model (not tracked)
tests/     unit tests (leakage, Elo)
```
