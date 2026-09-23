"""Downloads Premier League results and caches them in data/raw/."""
import pandas as pd

from src.config import CURRENT_SEASON, DATA_DIR, RESULTS_URL, SEASONS

COLUMNS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]


def download_season(season):
    return pd.read_csv(RESULTS_URL.format(season), encoding="latin-1")


def get_season(season, refresh=False):
    """Use the saved copy if we have one; finished seasons never change."""
    path = DATA_DIR / f"E0_{season}.csv"
    if path.exists() and not refresh:
        return pd.read_csv(path)
    df = download_season(season)[COLUMNS].dropna(subset=["FTR"])
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
