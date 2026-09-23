"""
Web app: pick a Premier League team and see its chances in its next league match.

Run on your computer:
    streamlit run app.py
"""
import json
import os
from datetime import datetime

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from src.config import METRICS_PATH, MODEL_PATH
from src.predict import (fetch_football_data_odds, fetch_odds_api_odds, fetch_upcoming_fixtures,
                         load_bundle, predict_match, team_view)

GITHUB_URL = "https://github.com/Jeff-rey13/EPL-PREDICTOR"

WIN, DRAW, LOSS = "#2F6B45", "#A3A9A5", "#A23B3B"   # pitch green, grey, red

st.set_page_config(page_title="Premier League predictor",
                   page_icon="⚽", layout="centered")

# API keys: from .env on your computer, from the app's "Secrets" once deployed
load_dotenv()
try:
    for key in ["FOOTBALL_DATA_API_KEY", "ODDS_API_KEY"]:
        if key in st.secrets:
            os.environ[key] = st.secrets[key]
except Exception:
    pass    # no secrets file on your computer, which is fine


# ---------------------------------------------------------------- Cached data
# Caching means many visitors share one download, which keeps the free API limits safe.
@st.cache_resource
def get_bundle():
    return load_bundle()


# refresh fixtures every hour
@st.cache_data(ttl=3600, show_spinner=False)
def get_fixtures(teams):
    key = os.environ.get("FOOTBALL_DATA_API_KEY")
    if not key:
        return None
    try:
        return fetch_upcoming_fixtures(list(teams), key)
    except Exception:
        return None


# this week's odds, every 30 minutes
@st.cache_data(ttl=1800, show_spinner=False)
def get_week_odds():
    return fetch_football_data_odds()


# at most 8 calls a day: well inside 500/month
@st.cache_data(ttl=3 * 3600, show_spinner=False)
def get_api_odds(teams):
    key = os.environ.get("ODDS_API_KEY")
    return fetch_odds_api_odds(key, list(teams)) if key else {}


def find_odds(home, away, teams):
    odds = get_week_odds().get((home, away))
    if odds:
        return odds, "football-data.co.uk"
    odds = get_api_odds(teams).get((home, away))
    if odds:
        return odds, "The Odds API"
    return None, None


def bookmaker_probs(odds):
    inv = [1 / o for o in odds]
    return [x / sum(inv) for x in inv]


def format_kickoff(kickoff):
    when = datetime.fromisoformat(kickoff.replace("Z", "+00:00"))
    return when.strftime("%A %d %B %Y, %H:%M UTC")


# ---------------------------------------------------------------- Page parts
def probability_bar(view):
    """One bar split into win / draw / loss: the centrepiece of the page."""
    parts = [(view["win"], WIN), (view["draw"], DRAW), (view["loss"], LOSS)]
    segments = "".join(
        f'<div style="width:{p * 100:.2f}%;background:{colour};display:flex;'
        f'align-items:center;justify-content:center">{f"{p:.0%}" if p >=
                                                      0.1 else ""}</div>'
        for p, colour in parts)
    st.markdown(
        f'<div role="img" aria-label="{view["team"]} win {view["win"]:.0%}, draw '
        f'{view["draw"]:.0%}, {view["opponent"]} win {view["loss"]:.0%}" '
        f'style="display:flex;height:64px;border-radius:8px;overflow:hidden;'
        f'color:white;font-size:1.35rem;font-weight:700;margin:0.5rem 0 1rem">{segments}</div>',
        unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric(f"{view['team']} win", f"{view['win']:.1%}")
    c2.metric("Draw", f"{view['draw']:.1%}")
    c3.metric(f"{view['opponent']} win", f"{view['loss']:.1%}")


def choose_fixture(team, teams, fixtures):
    """Use the real next fixture if we have it; otherwise let the visitor pick."""
    fixture = None
    if fixtures:
        fixture = next((f for f in fixtures if team in (f[0], f[1])), None)
    if fixture:
        return fixture
    if fixtures is None:
        st.caption(
            "Automatic fixtures aren't switched on, so choose the opponent.")
    else:
        st.caption(
            f"No upcoming league match found for {team}, so choose the opponent.")
    c1, c2 = st.columns([2, 1])
    opponent = c1.selectbox("Opponent", [t for t in teams if t != team])
    venue = c2.radio(f"{team} play", ["Home", "Away"], horizontal=True)
    return (team, opponent, None) if venue == "Home" else (opponent, team, None)


def week_table(bundle, teams):
    rows = []
    for (home, away), odds in get_week_odds().items():
        if home in teams and away in teams:
            probs, _ = predict_match(bundle, home, away, odds)
            rows.append({"Home": home, "Away": away, "Home win": f"{probs['H']:.0%}",
                         "Draw": f"{probs['D']:.0%}", "Away win": f"{probs['A']:.0%}"})
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    else:
        st.write("Odds for the next round aren't out yet. They usually appear a few days "
                 "before the matches.")


def accuracy_section():
    with st.expander("How accurate is this?"):
        if not METRICS_PATH.exists():
            st.write("Train the model to see its accuracy.")
            return
        metrics = json.loads(METRICS_PATH.read_text())
        rows = [{"Model": name, "Correct result": f"{m['accuracy']:.1%}",
                 "Log loss (lower is better)": f"{m['log_loss']:.4f}"}
                for name, m in metrics["models"].items()]
        rows.insert(0, {"Model": "Always pick the home team",
                        "Correct result": f"{metrics['baseline_home_win_accuracy']:.1%}",
                        "Log loss (lower is better)": "–"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.write(
            f"Tested on {metrics['test_matches']} Premier League matches the model never saw "
            "while learning. Football is very random, so even bookmakers only pick the right "
            "result about half the time. The model matches their accuracy by combining Elo "
            "ratings (a strength score for each team, updated after every match) with "
            "bookmaker odds. When no odds are published yet, it uses Elo alone.")


# ---------------------------------------------------------------- Page
if not MODEL_PATH.exists():
    st.error(
        "No trained model found. Run  python -m src.train  first, then reload this page.")
    st.stop()

bundle = get_bundle()
teams = tuple(bundle["teams"])

st.title("Premier League predictor")
st.write("Choose a team to see its chances of winning its next league match.")

default = teams.index("Man United") if "Man United" in teams else 0
team = st.selectbox("Team", teams, index=default)

home, away, kickoff = choose_fixture(team, teams, get_fixtures(teams))
odds, source = find_odds(home, away, teams)
probs, used = predict_match(bundle, home, away, odds)
view = team_view(team, home, away, probs)

st.subheader(f"{home} vs {away}")
if kickoff:
    st.caption(format_kickoff(kickoff))
probability_bar(view)

if odds:
    st.caption(f"Based on Elo ratings and bookmaker odds of {odds[0]:.2f} / {odds[1]:.2f} / "
               f"{odds[2]:.2f} (from {source}).")
    with st.expander("Compare with the bookmakers"):
        bm = bookmaker_probs(odds)
        names = [f"{home} win", "Draw", f"{away} win"]
        st.dataframe(pd.DataFrame({
            "Outcome": names,
            "Bookmakers": [f"{p:.1%}" for p in bm],
            "This model": [f"{probs[k]:.1%}" for k in ["H", "D", "A"]],
        }), hide_index=True, width="stretch")
        st.write("Bookmaker odds converted to probabilities, with their profit margin removed. "
                 "The model starts from the odds and adjusts them using each team's Elo rating.")
else:
    st.caption("Bookmakers haven't published odds for this match yet, so this uses Elo ratings "
               "only, which is less accurate. Check again closer to kickoff.")

st.header("This week's matches")
week_table(bundle, teams)

accuracy_section()

st.divider()
st.caption(f"Model last trained on {bundle['trained_at'][:10]}. "
           f"[Code and method on GitHub]({GITHUB_URL}).")
