from pathlib import Path

PROJECT_ROOT=Path(__file__).resolve().parent.parent
DATA_DIR=PROJECT_ROOT/"data"

RAW_PATH=DATA_DIR/"epl_final.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "model.joblib"
DATE_COL="MatchDate"
TARGET_COL="FullTimeResult"
FEATURES_PATH = DATA_DIR / "features.csv"
ROLLING_WINDOW = 5
VALID_RESULTS={"H","D","A"}
CLEAN_PATH = DATA_DIR / "epl_clean.csv"
REQUIRED_COLUMNS = [
    "Season", "MatchDate", "HomeTeam", "AwayTeam",
    "FullTimeHomeGoals", "FullTimeAwayGoals", "FullTimeResult",
]

STAT_COLUMNS = [
    "HalfTimeHomeGoals", "HalfTimeAwayGoals",
    "HomeShots", "AwayShots", "HomeShotsOnTarget", "AwayShotsOnTarget",
    "HomeCorners", "AwayCorners", "HomeFouls", "AwayFouls",
    "HomeYellowCards", "AwayYellowCards", "HomeRedCards", "AwayRedCards",
]
VAL_SEASONS = ["2022/23", "2023/24"]
TEST_SEASONS = ["2024/25"]
LABELS = {"A": 0, "D": 1, "H": 2}