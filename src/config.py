from pathlib import Path

PROJECT_ROOT=Path(__file__).resolve().parent.parent
DATA_DIR=PROJECT_ROOT/"data"

RAW_PATH=DATA_DIR/"epl_final.csv"

DATE_COL="MatchDate"
TARGET_COL="FullTimeResult"
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
