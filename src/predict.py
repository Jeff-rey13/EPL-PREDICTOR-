"""
Predicts a team's next Premier League game using the saved model.

Run from the project folder (after training once):
    python -m src.predict "man utd"
    python -m src.predict               (asks you for a team)
"""
import difflib
import os
import sys

import joblib
import pandas as pd
import requests
from dotenv import load_dotenv

from src.config import (ALIASES, ELO_NEWCOMER, FIXTURES_URL, MODEL_PATH, ODDS_API_URL,
                        ODDS_FIXTURES_URL)
from src.data import ODDS_SOURCES
from src.features import odds_to_log_probs


def load_bundle():
    if not MODEL_PATH.exists():
        sys.exit("No trained model found. Run this first:  python -m src.train")
    bundle = joblib.load(MODEL_PATH)
    if "fallback_model" not in bundle:
        sys.exit("Your saved model is from an older version. Run:  python -m src.train")
    return bundle


def resolve_team(name, known_teams):
    """Turn whatever the user (or the API) typed into our dataset's team name."""
    cleaned = name.lower().strip()
    for suffix in [" fc", " afc"]:
        cleaned = cleaned.removesuffix(suffix)
    cleaned = cleaned.removeprefix("afc ").strip()
    if cleaned in ALIASES:
        return ALIASES[cleaned]
    lookup = {t.lower(): t for t in known_teams}
    if cleaned in lookup:
        return lookup[cleaned]
    close = difflib.get_close_matches(cleaned, lookup.keys(), n=1, cutoff=0.6)
    return lookup[close[0]] if close else None


def fetch_next_fixture(team, known_teams, api_key):
    """Returns (home, away, kickoff_utc) for the team's next league game, or None."""
    resp = requests.get(FIXTURES_URL, headers={"X-Auth-Token": api_key}, timeout=15)
    resp.raise_for_status()
    upcoming = [m for m in resp.json()["matches"] if m["status"] in ("SCHEDULED", "TIMED")]
    for m in sorted(upcoming, key=lambda m: m["utcDate"]):
        home = resolve_team(m["homeTeam"]["name"], known_teams)
        away = resolve_team(m["awayTeam"]["name"], known_teams)
        if home is None or away is None:
            print(f"  Warning: couldn't match '{m['homeTeam']['name']}' or "
                  f"'{m['awayTeam']['name']}'. Add it to ALIASES in src/config.py.")
            continue
        if team in (home, away):
            return home, away, m["utcDate"]
    return None


def ask_for_fixture(team, known_teams):
    """Fallback when there's no API key: ask the user."""
    while True:
        opp = resolve_team(input(f"Who do {team} play next? "), known_teams)
        if opp and opp != team:
            break
        print("  Didn't recognise that team, try again.")
    venue = input("Home or away? [h/a] ").strip().lower()
    return (team, opp, None) if venue.startswith("h") else (opp, team, None)


def fetch_football_data_odds():
    """Source 1 (no key needed): this week's games from football-data.co.uk.
    Returns {(home, away): (odds_h, odds_d, odds_a)}."""
    try:
        fixtures = pd.read_csv(ODDS_FIXTURES_URL, encoding="utf-8-sig")
    except Exception as e:
        print(f"(Couldn't download odds from football-data.co.uk: {e})")
        return {}
    fixtures = fixtures[fixtures["Div"] == "E0"]
    table = {}
    for _, row in fixtures.iterrows():
        odds = []
        for col in ["OddsH", "OddsD", "OddsA"]:
            # same source priority as the training data: market average first
            value = next((row[c] for c in ODDS_SOURCES[col]
                          if c in fixtures.columns and pd.notna(row[c])), None)
            odds.append(value)
        if None not in odds:
            table[(str(row["HomeTeam"]).strip(), str(row["AwayTeam"]).strip())] = \
                tuple(float(o) for o in odds)
    return table


def fetch_odds_api_odds(api_key, known_teams):
    """Source 2 (free key): upcoming games further ahead from The Odds API.
    Averages each outcome across UK bookmakers, like the market average used in training.
    Each call uses 1 request of the free 500/month."""
    try:
        resp = requests.get(ODDS_API_URL, timeout=15, params={
            "apiKey": api_key, "regions": "uk", "markets": "h2h", "oddsFormat": "decimal"})
        resp.raise_for_status()
    except Exception as e:
        print(f"(Couldn't download odds from The Odds API: {e})")
        return {}
    remaining = resp.headers.get("x-requests-remaining")
    if remaining is not None:
        print(f"(The Odds API: {remaining} free requests left this month)")

    table = {}
    for event in resp.json():
        home = resolve_team(event["home_team"], known_teams)
        away = resolve_team(event["away_team"], known_teams)
        if home is None or away is None:
            print(f"  Warning: couldn't match '{event['home_team']}' or '{event['away_team']}'. "
                  f"Add it to ALIASES in src/config.py.")
            continue
        prices = {"H": [], "D": [], "A": []}
        for bookmaker in event.get("bookmakers", []):
            for market in bookmaker.get("markets", []):
                if market.get("key") != "h2h":
                    continue
                for outcome in market.get("outcomes", []):
                    if outcome["name"] == event["home_team"]:
                        prices["H"].append(outcome["price"])
                    elif outcome["name"] == event["away_team"]:
                        prices["A"].append(outcome["price"])
                    elif outcome["name"] == "Draw":
                        prices["D"].append(outcome["price"])
        if all(prices.values()):
            table[(home, away)] = tuple(sum(v) / len(v) for v in
                                        (prices["H"], prices["D"], prices["A"]))
    return table


def find_odds(home, away, known_teams):
    """Look for odds automatically. Returns (odds or None, where they came from)."""
    odds = fetch_football_data_odds().get((home, away))
    if odds:
        return odds, "football-data.co.uk"
    api_key = os.environ.get("ODDS_API_KEY")
    if api_key:
        odds = fetch_odds_api_odds(api_key, known_teams).get((home, away))
        if odds:
            return odds, "The Odds API"
    return None, None


def predict_match(bundle, home, away, odds=None):
    """Returns ({'H': p, 'D': p, 'A': p}, name of the model used)."""
    elo = bundle["elo"]
    h_elo, a_elo = elo.get(home, ELO_NEWCOMER), elo.get(away, ELO_NEWCOMER)
    elo_features = [h_elo, a_elo, h_elo - a_elo]
    if odds:
        row = pd.DataFrame([elo_features + odds_to_log_probs(*odds)],
                           columns=bundle["features"])
        model, used = bundle["model"], "Elo + bookmaker odds"
    else:
        row = pd.DataFrame([elo_features], columns=bundle["fallback_features"])
        model, used = bundle["fallback_model"], "Elo only (no odds available)"
    return dict(zip(model.classes_, model.predict_proba(row)[0])), used


def team_view(team, home, away, probs):
    """Flip the probabilities so they're from the chosen team's point of view."""
    at_home = team == home
    return {
        "team": team,
        "opponent": away if at_home else home,
        "venue": "home" if at_home else "away",
        "win": probs["H"] if at_home else probs["A"],
        "draw": probs["D"],
        "loss": probs["A"] if at_home else probs["H"],
    }


def report(view, home, away, kickoff=None, used="", odds=None, source=None):
    when = f" ({kickoff[:10]})" if kickoff else ""
    print(f"\nNext game: {home} vs {away}{when}")
    print(f"  Model: {used}")
    if odds:
        print(f"  Odds:  {odds[0]:.2f} / {odds[1]:.2f} / {odds[2]:.2f}  (from {source})")
    print(f"  {view['team']} ({view['venue']}) vs {view['opponent']}")
    print(f"  Win:  {view['win']:.1%}")
    print(f"  Draw: {view['draw']:.1%}")
    print(f"  Loss: {view['loss']:.1%}")


def main():
    load_dotenv()   # reads FOOTBALL_DATA_API_KEY from the .env file, if there is one
    bundle = load_bundle()
    teams = bundle["teams"]
    print(f"Model trained {bundle['trained_at'][:10]}")

    user_input = sys.argv[1] if len(sys.argv) > 1 else input("Which team? ")
    team = resolve_team(user_input, teams)
    if team is None:
        sys.exit(f"Couldn't find a team matching '{user_input}'.\n"
                 f"Current teams: {', '.join(teams)}")

    api_key = os.environ.get("FOOTBALL_DATA_API_KEY")
    fixture = None
    if api_key:
        try:
            fixture = fetch_next_fixture(team, teams, api_key)
            if fixture is None:
                print(f"No upcoming league fixture found for {team}.")
        except Exception as e:
            print(f"Couldn't fetch fixtures ({e}).")
    else:
        print("(No FOOTBALL_DATA_API_KEY found, so entering the fixture manually.)")
    if fixture is None:
        fixture = ask_for_fixture(team, teams)

    home, away, kickoff = fixture
    odds, source = find_odds(home, away, teams)
    if odds is None:
        tip = "" if os.environ.get("ODDS_API_KEY") else \
            " Add a free ODDS_API_KEY to your .env file to find odds further ahead."
        print(f"No odds published yet for {home} vs {away}, so using Elo only.{tip}")
    probs, used = predict_match(bundle, home, away, odds)
    report(team_view(team, home, away, probs), home, away, kickoff, used, odds, source)


if __name__ == "__main__":
    main()
