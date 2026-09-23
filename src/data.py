"""Downloads Premier League results and caches them in data/raw/."""
import pandas as pd

from src.config import CURRENT_SEASON, DATA_DIR, RESULTS_URL, SEASONS

COLUMNS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
ODDS_COLUMNS = ["OddsH", "OddsD", "OddsA"]

# Pre-match odds, best source first. Column names changed over the years:
# Avg* = market average (2019/20+), BbAv* = market average (older), then single bookmakers.
ODDS_SOURCES = {
    "OddsH": ["AvgH", "BbAvH", "B365H", "PSH"],
    "OddsD": ["AvgD", "BbAvD", "B365D", "PSD"],
    "OddsA": ["AvgA", "BbAvA", "B365A", "PSA"],
}


def extract_odds(df):
    """One clean set of odds columns, using the best source available in each row."""
    out = df[COLUMNS].copy()
    for col, sources in ODDS_SOURCES.items():
        available = [c for c in sources if c in df.columns]
        out[col] = df[available].bfill(axis=1).iloc[:, 0] if available else float("nan")
    return out


def download_season(season):
    return pd.read_csv(RESULTS_URL.format(season), encoding="latin-1")


def get_season(season, refresh=False):
    """Use the saved copy if we have one; finished seasons never change."""
    path = DATA_DIR / f"E0_{season}.csv"
    if path.exists() and not refresh:
        cached = pd.read_csv(path)
        if all(c in cached.columns for c in ODDS_COLUMNS):   # old copies have no odds
            return cached
    df = extract_odds(download_season(season)).dropna(subset=["FTR"])
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def load_data():
    frames = []
    for s in SEASONS:
        try:
            df = get_season(s, refresh=(s == CURRENT_SEASON))
            df["Season"] = s
            frames.append(df)
            print(f"Loaded season {s}: {len(df)} matches")
        except Exception as e:
            print(f"Skipping season {s}: {e}")
    df = pd.concat(frames, ignore_index=True)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, format="mixed")
    return df.sort_values("Date").reset_index(drop=True)
