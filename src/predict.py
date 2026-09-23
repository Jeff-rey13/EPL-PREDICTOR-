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

from src.config import ALIASES, ELO_NEWCOMER, FEATURES, FIXTURES_URL, MODEL_PATH


def load_bundle():
    if not MODEL_PATH.exists():
        sys.exit("No trained model found. Run this first:  python -m src.train")
    return joblib.load(MODEL_PATH)


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


def predict_match(bundle, home, away):
    """Returns {'H': p, 'D': p, 'A': p} for a single fixture."""
    elo, form = bundle["elo"], bundle["form"]
    h_elo, a_elo = elo.get(home, ELO_NEWCOMER), elo.get(away, ELO_NEWCOMER)
    h_form = form.get(home, (1.0, 1.5, 1.0))   # rough default for an unknown team
    a_form = form.get(away, (1.0, 1.5, 1.0))
    row = pd.DataFrame([[h_elo, a_elo, h_elo - a_elo, *h_form, *a_form]], columns=FEATURES)
    model = bundle["model"]
    return dict(zip(model.classes_, model.predict_proba(row)[0]))


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


def report(view, home, away, kickoff=None):
    when = f" ({kickoff[:10]})" if kickoff else ""
    print(f"\nNext game: {home} vs {away}{when}")
    print(f"  {view['team']} ({view['venue']}) vs {view['opponent']}")
    print(f"  Win:  {view['win']:.1%}")
    print(f"  Draw: {view['draw']:.1%}")
    print(f"  Loss: {view['loss']:.1%}")


def main():
    load_dotenv()   # reads FOOTBALL_DATA_API_KEY from the .env file, if there is one
    bundle = load_bundle()
    teams = bundle["teams"]
    print(f"Model: {bundle['model_name']} (trained {bundle['trained_at'][:10]})")

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
        print("(No API key found, so entering the fixture manually.)")
    if fixture is None:
        fixture = ask_for_fixture(team, teams)

    home, away, kickoff = fixture
    probs = predict_match(bundle, home, away)
    report(team_view(team, home, away, probs), home, away, kickoff)


if __name__ == "__main__":
    main()
