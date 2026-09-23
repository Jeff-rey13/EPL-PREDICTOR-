"""
Trains the prediction models, evaluates them on held-out seasons, and saves them.

Run from the project folder:
    python -m src.train

Two models are saved:
  main      Elo + bookmaker log-odds   (used when odds are available for a fixture)
  fallback  Elo only                   (used when no odds are available yet)
"""
import json
from datetime import datetime, timezone

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import (CURRENT_SEASON, FALLBACK_FEATURES, FEATURES, FOCUS_TEAM, METRICS_PATH,
                        MODEL_PATH, TEST_SEASONS, WARMUP_SEASON)
from src.data import load_data
from src.features import build_features


def make_model():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))


def score(y, probs, classes, team_mask):
    preds = [classes[i] for i in probs.argmax(axis=1)]
    result = {
        "accuracy": round(float(accuracy_score(y, preds)), 4),
        "log_loss": round(float(log_loss(y, probs, labels=classes)), 4),
    }
    if team_mask.sum() > 0:
        team_preds = [p for p, m in zip(preds, team_mask) if m]
        result[f"{FOCUS_TEAM}_accuracy"] = round(float(accuracy_score(y[team_mask],
                                                                      team_preds)), 4)
    return result


def main():
    raw = load_data()
    df, elo, _ = build_features(raw)
    df = df[df["Season"] != WARMUP_SEASON].dropna(subset=FALLBACK_FEATURES)

    # 1. Evaluate: train on older seasons, test on newer ones (never random!)
    train = df[~df["Season"].isin(TEST_SEASONS)]
    test = df[df["Season"].isin(TEST_SEASONS)].dropna(subset=FEATURES)  # needs odds
    team_mask = ((test["HomeTeam"] == FOCUS_TEAM) | (test["AwayTeam"] == FOCUS_TEAM)).values
    y = test["FTR"].values

    main_model = make_model().fit(train.dropna(subset=FEATURES)[FEATURES],
                                  train.dropna(subset=FEATURES)["FTR"])
    fallback_model = make_model().fit(train[FALLBACK_FEATURES], train["FTR"])

    results = {
        "main (Elo + odds)": score(y, main_model.predict_proba(test[FEATURES]),
                                   list(main_model.classes_), team_mask),
        "fallback (Elo only)": score(y, fallback_model.predict_proba(test[FALLBACK_FEATURES]),
                                     list(fallback_model.classes_), team_mask),
        "bookmakers (benchmark)": score(y, test[["Mkt_pA", "Mkt_pD", "Mkt_pH"]].values,
                                        ["A", "D", "H"], team_mask),
    }
    baseline = round(float((test["FTR"] == "H").mean()), 4)

    print(f"\nTrain: {len(train)} matches | Test: {len(test)} matches")
    print(f"Baseline (always predict home win): {baseline:.1%}\n")
    for name, r in results.items():
        print(f"  {name:24s} accuracy={r['accuracy']:.1%}  log loss={r['log_loss']:.4f}")

    # 2. Retrain on ALL matches, so live predictions use the newest data
    final_main = make_model().fit(df.dropna(subset=FEATURES)[FEATURES],
                                  df.dropna(subset=FEATURES)["FTR"])
    final_fallback = make_model().fit(df[FALLBACK_FEATURES], df["FTR"])

    trained_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    this_season = raw[raw["Season"] == CURRENT_SEASON]
    current_teams = sorted(set(this_season["HomeTeam"]) | set(this_season["AwayTeam"]))
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": final_main,
        "features": FEATURES,
        "fallback_model": final_fallback,
        "fallback_features": FALLBACK_FEATURES,
        "elo": elo,
        "teams": current_teams,
        "trained_at": trained_at,
    }, MODEL_PATH)

    METRICS_PATH.write_text(json.dumps({
        "trained_at": trained_at,
        "train_matches": len(train),
        "test_matches": len(test),
        "test_seasons": TEST_SEASONS,
        "baseline_home_win_accuracy": baseline,
        "models": results,
    }, indent=2))
    print(f"\nSaved model to {MODEL_PATH}")
    print(f"Saved metrics to {METRICS_PATH}")


if __name__ == "__main__":
    main()
