"""
Backtest: see how the model would have predicted past matches, fairly.

To avoid "knowing the answer", every prediction uses only what was known before kickoff:
  - Elo ratings as they were before the match
  - bookmaker odds from before the match
  - a model trained only on seasons BEFORE the one being tested

Run from the project folder:
    python -m src.backtest "man utd" 2024/25              every Man United match that season
    python -m src.backtest "man utd" "liverpool" 2024/25  just the matches between them

Add --no-odds to use the Elo-only model, which doesn't use bookmaker odds at all:
    python -m src.backtest "man utd" 2024/25 --no-odds
"""
import re
import sys

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import FALLBACK_FEATURES, FEATURES, SEASONS, WARMUP_SEASON
from src.data import load_data
from src.features import build_features
from src.predict import resolve_team

RESULT_NAMES = {"H": "home win", "D": "draw", "A": "away win"}
USAGE = ('Usage:\n  python -m src.backtest "man utd" 2024/25\n'
         '  python -m src.backtest "man utd" "liverpool" 2024/25\n'
         'Add --no-odds to use the Elo-only model (no bookmaker odds at all).')


def parse_season(text):
    """'2024/25', '2024-25', '24/25' or '2425' -> '2425'."""
    digits = re.sub(r"\D", "", text)
    if len(digits) == 4:
        return digits
    if len(digits) == 6:
        return digits[2:4] + digits[4:6]
    if len(digits) == 8:
        return digits[2:4] + digits[6:8]
    return None


def season_label(code):
    return f"20{code[:2]}/{code[2:]}"


def make_model():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))


def train_before(df, season):
    """Train both models only on seasons before the one being tested."""
    past = df[(df["Season"] < season) & (df["Season"] != WARMUP_SEASON)]
    with_odds = past.dropna(subset=FEATURES)
    main_model = make_model().fit(with_odds[FEATURES], with_odds["FTR"])
    fallback = make_model().fit(past.dropna(subset=FALLBACK_FEATURES)[FALLBACK_FEATURES],
                                past.dropna(subset=FALLBACK_FEATURES)["FTR"])
    return main_model, fallback, len(past)


def predict_row(row, main_model, fallback, no_odds=False):
    if not no_odds and row[FEATURES].notna().all():
        model, cols = main_model, FEATURES
    else:
        model, cols = fallback, FALLBACK_FEATURES
    probs = model.predict_proba(row[cols].to_frame().T.astype(float))[0]
    return dict(zip(model.classes_, probs))


def pick(probs):
    return max(probs, key=probs.get)


def main():
    no_odds = "--no-odds" in sys.argv[1:]
    args = [a for a in sys.argv[1:] if a != "--no-odds"]
    if len(args) not in (2, 3):
        sys.exit(USAGE)
    season = parse_season(args[-1])
    testable = [s for s in SEASONS if s > WARMUP_SEASON][1:]   # needs a season to learn from
    if season not in testable:
        sys.exit(f"Season must be one of: {', '.join(season_label(s) for s in testable)}")

    raw = load_data()
    df, _, _ = build_features(raw)      # Elo and odds for each match are pre-kickoff values
    this_season = df[df["Season"] == season]
    teams = sorted(set(this_season["HomeTeam"]) | set(this_season["AwayTeam"]))

    names = [resolve_team(n, teams) for n in args[:-1]]
    for typed, name in zip(args[:-1], names):
        if name not in teams:
            sys.exit(f"'{typed}' wasn't in the Premier League in {season_label(season)}.\n"
                     f"Teams that season: {', '.join(teams)}")

    team = names[0]
    plays = (this_season["HomeTeam"] == team) | (this_season["AwayTeam"] == team)
    if len(names) == 2:
        other = names[1]
        plays &= (this_season["HomeTeam"] == other) | (this_season["AwayTeam"] == other)
    matches = this_season[plays]
    if matches.empty:
        sys.exit("No matches found for that search.")

    main_model, fallback, n_train = train_before(df, season)
    model_name = "Elo only (no bookmaker odds)" if no_odds else "Elo + bookmaker odds"
    print(f"\nModel: {model_name}")
    print(f"Trained on {n_train} matches from before {season_label(season)} "
          f"(it has never seen these results).\n")

    right = 0
    prob_sum = 0.0
    for _, row in matches.iterrows():
        probs = predict_row(row, main_model, fallback, no_odds)
        home, away, result = row["HomeTeam"], row["AwayTeam"], row["FTR"]
        model_pick = pick(probs)
        right += model_pick == result
        prob_sum += probs[result]
        note = "" if no_odds or row[FEATURES].notna().all() else "   (no odds: Elo only)"
        print(f"{row['Date']:%d %b %Y}  {home} {int(row['FTHG'])}-{int(row['FTAG'])} {away}"
              f"  ({RESULT_NAMES[result]})")
        print(f"  home {probs['H']:.0%}  draw {probs['D']:.0%}  away {probs['A']:.0%}"
              f"   pick: {RESULT_NAMES[model_pick]:9s} {'RIGHT' if model_pick == result else 'wrong'}"
              f"{note}\n")

    n = len(matches)
    print("-" * 70)
    print(f"Picked the right result in {right} of {n} matches ({right / n:.0%}).")
    print(f"On average, gave {prob_sum / n:.0%} to what actually happened.")
    print("Higher is better. Picking three outcomes at random would give 33%.")


if __name__ == "__main__":
    main()
