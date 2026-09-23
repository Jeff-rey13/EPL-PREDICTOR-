"""Turns raw results into model features: Elo ratings and recent form."""
import numpy as np
import pandas as pd

from src.config import ELO_HOME_ADV, ELO_K, ELO_NEWCOMER, ELO_START, FORM_WINDOW


def add_elo(df, k=ELO_K, home_adv=ELO_HOME_ADV):
    """Stores each team's rating BEFORE the match, so there is no leakage."""
    elo = {}
    home_pre, away_pre = [], []
    first_season = df["Season"].iloc[0]
    for row in df.itertuples():
        default = ELO_START if row.Season == first_season else ELO_NEWCOMER
        h = elo.get(row.HomeTeam, default)
        a = elo.get(row.AwayTeam, default)
        home_pre.append(h)
        away_pre.append(a)

        expected_home = 1 / (1 + 10 ** ((a - (h + home_adv)) / 400))
        actual_home = {"H": 1.0, "D": 0.5, "A": 0.0}[row.FTR]
        margin = np.log(abs(row.FTHG - row.FTAG) + 1) + 1   # bigger wins count more
        delta = k * margin * (actual_home - expected_home)
        elo[row.HomeTeam] = h + delta
        elo[row.AwayTeam] = a - delta

    df["HomeElo"] = home_pre
    df["AwayElo"] = away_pre
    df["EloDiff"] = df["HomeElo"] - df["AwayElo"]
    return df, elo


def team_long_table(df):
    """One row per team per match (each match appears twice)."""
    home = pd.DataFrame({
        "match_id": df.index, "Date": df["Date"], "Team": df["HomeTeam"], "side": "H",
        "GF": df["FTHG"], "GA": df["FTAG"], "Pts": df["FTR"].map({"H": 3, "D": 1, "A": 0}),
    })
    away = pd.DataFrame({
        "match_id": df.index, "Date": df["Date"], "Team": df["AwayTeam"], "side": "A",
        "GF": df["FTAG"], "GA": df["FTHG"], "Pts": df["FTR"].map({"A": 3, "D": 1, "H": 0}),
    })
    return pd.concat([home, away]).sort_values(["Date", "match_id"]).reset_index(drop=True)


def add_form(df, n=FORM_WINDOW):
    long = team_long_table(df)
    for col in ["GF", "GA", "Pts"]:
        # shift(1) = only use matches BEFORE this one (prevents leakage)
        long[f"{col}_form"] = long.groupby("Team")[col].transform(
            lambda s: s.shift(1).rolling(n, min_periods=1).mean()
        )
    cols = ["GF_form", "GA_form", "Pts_form"]
    h = long[long.side == "H"].set_index("match_id")[cols].add_prefix("H_")
    a = long[long.side == "A"].set_index("match_id")[cols].add_prefix("A_")
    return df.join(h).join(a), long


def current_form(long, n=FORM_WINDOW):
    """Each team's form over its last n games, for predicting upcoming matches."""
    recent = long.groupby("Team").tail(n)
    return {team: (float(g["GF"].mean()), float(g["GA"].mean()), float(g["Pts"].mean()))
            for team, g in recent.groupby("Team")}


def build_features(df, form_window=FORM_WINDOW, elo_k=ELO_K, elo_home_adv=ELO_HOME_ADV):
    """Settings default to config.py; the experiment runner passes other values."""
    df = df.copy()
    df, elo = add_elo(df, k=elo_k, home_adv=elo_home_adv)
    df, long = add_form(df, n=form_window)
    return df, elo, current_form(long, n=form_window)
