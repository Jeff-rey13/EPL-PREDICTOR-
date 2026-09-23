"""
Player-based features from Fantasy Premier League (FPL) data.

Source: https://github.com/vaastav/Fantasy-Premier-League (free, every player, every match)

For every match, using ONLY games played before kickoff (no leakage):
  XIValue    total FPL price (in GBP millions) of the team's starting XI,
             averaged over its last 3 games. FPL prices are a decent proxy for
             player quality, so this drops when star players are injured or rested.
  xGF_form   average expected goals FOR over the last 5 games      (2022/23 onwards)
  xGA_form   average expected goals AGAINST over the last 5 games  (2022/23 onwards)

Note: this dataset is updated every few weeks, so the current season can lag
behind the results data. Matches without player data are simply left blank.
"""
import numpy as np
import pandas as pd

from src.config import CURRENT_SEASON, DATA_DIR, SEASONS

FPL_URL = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/{}/{}"
FPL_DIR = DATA_DIR / "fpl"

# 2018/19 has no team list in this dataset (and is only a warm-up season anyway)
PLAYER_SEASONS = [s for s in SEASONS if s != "1819"]

XI_WINDOW = 3     # games averaged for squad value
XG_WINDOW = 5     # games averaged for xG form

# FPL team names -> football-data.co.uk team names
FPL_NAMES = {
    "Man Utd": "Man United", "Spurs": "Tottenham", "Sheffield Utd": "Sheffield United",
    "Coventry City": "Coventry", "Hull City": "Hull", "Ipswich Town": "Ipswich",
}

SQUAD_FEATURES = ["H_XIValue", "A_XIValue", "XIValueDiff"]
XG_FEATURES = ["H_xGF_form", "H_xGA_form", "A_xGF_form", "A_xGA_form"]
PLAYER_FEATURES = SQUAD_FEATURES + XG_FEATURES

GW_COLUMNS = ["fixture", "was_home", "minutes", "value", "expected_goals"]


def fpl_season_name(code):
    """'1920' -> '2019-20' (the folder names used by the FPL dataset)."""
    return f"20{code[:2]}-{code[2:]}"


def get_fpl_file(season, filename, refresh=False):
    """Download once and keep a small local copy in data/raw/fpl/."""
    path = FPL_DIR / season / filename.replace("/", "_")
    if path.exists() and not refresh:
        return pd.read_csv(path)
    url = FPL_URL.format(fpl_season_name(season), filename)
    try:
        df = pd.read_csv(url, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(url, encoding="latin-1")
    if filename.endswith("merged_gw.csv"):
        df = df[[c for c in GW_COLUMNS if c in df.columns]]   # keep the file small
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def team_match_table(season):
    """One row per team per match: starting XI value and xG for that game."""
    refresh = season == CURRENT_SEASON
    gw = get_fpl_file(season, "gws/merged_gw.csv", refresh)
    fixtures = get_fpl_file(season, "fixtures.csv", refresh)
    teams = get_fpl_file(season, "teams.csv", refresh)

    names = teams.set_index("id")["name"].map(lambda n: FPL_NAMES.get(n, n))
    fixtures = fixtures[["id", "team_h", "team_a", "kickoff_time"]].copy()
    fixtures["home"] = fixtures["team_h"].map(names)
    fixtures["away"] = fixtures["team_a"].map(names)

    if "expected_goals" not in gw.columns:
        gw["expected_goals"] = np.nan          # xG only exists from 2022/23
    gw["was_home"] = gw["was_home"].astype(str).str.lower().eq("true")
    gw = gw.merge(fixtures[["id", "home", "away", "kickoff_time"]],
                  left_on="fixture", right_on="id")
    gw["Team"] = np.where(gw["was_home"], gw["home"], gw["away"])

    keys = ["fixture", "Team", "was_home", "home", "away", "kickoff_time"]
    played = gw[gw["minutes"] > 0]
    top11 = played.sort_values("minutes", ascending=False).groupby(["fixture", "Team"]).head(11)
    xi = (top11.groupby(keys)["value"].sum() / 10).rename("XIValue")      # price is in 0.1m units
    xg = gw.groupby(keys)["expected_goals"].sum(min_count=1).rename("xGF")
    tm = pd.concat([xi, xg], axis=1).reset_index()

    # xG against = the opponent's xG for in the same match
    opp = tm[["fixture", "Team", "xGF"]].rename(columns={"Team": "Opp", "xGF": "xGA"})
    tm["Opp"] = np.where(tm["was_home"], tm["away"], tm["home"])
    tm = tm.merge(opp, on=["fixture", "Opp"], how="left")
    tm["Season"] = season
    return tm


def build_player_features():
    """Returns one row per match, keyed by Season + HomeTeam + AwayTeam."""
    frames = []
    for s in PLAYER_SEASONS:
        try:
            frames.append(team_match_table(s))
            print(f"Loaded player data {fpl_season_name(s)}")
        except Exception as e:
            print(f"Skipping player data {fpl_season_name(s)}: {e}")
    tm = pd.concat(frames, ignore_index=True)
    tm["kickoff_time"] = pd.to_datetime(tm["kickoff_time"], utc=True)
    tm = tm.sort_values("kickoff_time").reset_index(drop=True)

    # shift(1) = only games BEFORE this one (prevents leakage)
    g = tm.groupby("Team")
    tm["XIValue_form"] = g["XIValue"].transform(
        lambda s: s.shift(1).rolling(XI_WINDOW, min_periods=1).mean())
    for col in ["xGF", "xGA"]:
        tm[f"{col}_form"] = g[col].transform(
            lambda s: s.shift(1).rolling(XG_WINDOW, min_periods=1).mean())

    cols = {"XIValue_form": "XIValue", "xGF_form": "xGF_form", "xGA_form": "xGA_form"}
    home = tm[tm["was_home"]].rename(columns={k: f"H_{v}" for k, v in cols.items()})
    away = tm[~tm["was_home"]].rename(columns={k: f"A_{v}" for k, v in cols.items()})
    matches = home[["Season", "home", "away", "fixture"] + [f"H_{v}" for v in cols.values()]].merge(
        away[["Season", "fixture"] + [f"A_{v}" for v in cols.values()]], on=["Season", "fixture"])
    matches = matches.rename(columns={"home": "HomeTeam", "away": "AwayTeam"})
    matches["XIValueDiff"] = matches["H_XIValue"] - matches["A_XIValue"]
    return matches[["Season", "HomeTeam", "AwayTeam"] + PLAYER_FEATURES]


def add_player_features(results):
    """Attach player features to the results table (blank where no player data)."""
    players = build_player_features()
    merged = results.merge(players, on=["Season", "HomeTeam", "AwayTeam"], how="left")
    matched = merged["H_XIValue"].notna().sum()
    print(f"Player data matched for {matched} of {len(merged)} matches")
    unmatched = set(players["HomeTeam"]) - set(results["HomeTeam"])
    if unmatched:
        print(f"  Warning: FPL team names not found in results: {sorted(unmatched)}. "
              f"Add them to FPL_NAMES in src/players.py.")
    return merged
