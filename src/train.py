"""
Trains the model, evaluates it on held-out seasons, and saves it.

Run from the project folder:
    python -m src.train
"""
import json
from datetime import datetime, timezone

import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import (CURRENT_SEASON, FEATURES, FOCUS_TEAM, METRICS_PATH, MODEL_PATH,
                        TEST_SEASONS, WARMUP_SEASON)
from src.data import load_data
from src.features import build_features


def candidate_models():
    return {
        "logistic_regression": make_pipeline(StandardScaler(),
                                             LogisticRegression(max_iter=1000)),
        "gradient_boosting": GradientBoostingClassifier(n_estimators=150, max_depth=2,
                                                        learning_rate=0.05, random_state=42),
    }


def evaluate(model, X, y, team_mask):
    preds = model.predict(X)
    probs = model.predict_proba(X)
    result = {
        "accuracy": round(float(accuracy_score(y, preds)), 4),
        "log_loss": round(float(log_loss(y, probs, labels=model.classes_)), 4),
    }
    if team_mask.sum() > 0:
        result[f"{FOCUS_TEAM}_accuracy"] = round(float(accuracy_score(y[team_mask],
                                                                      preds[team_mask])), 4)
    return result


def main():
    raw = load_data()
    df, elo, form = build_features(raw)
    df = df[df["Season"] != WARMUP_SEASON].dropna(subset=FEATURES)

    # 1. Evaluate: train on older seasons, test on newer ones (never random!)
    train = df[~df["Season"].isin(TEST_SEASONS)]
    test = df[df["Season"].isin(TEST_SEASONS)]
    team_mask = ((test["HomeTeam"] == FOCUS_TEAM) | (test["AwayTeam"] == FOCUS_TEAM)).values

    baseline = round(float((test["FTR"] == "H").mean()), 4)
    print(f"\nTrain: {len(train)} matches | Test: {len(test)} matches")
    print(f"Baseline (always predict home win): {baseline}")

    results = {}
    for name, model in candidate_models().items():
        model.fit(train[FEATURES], train["FTR"])
        results[name] = evaluate(model, test[FEATURES], test["FTR"], team_mask)
        print(f"\n{name}: {results[name]}")

    # 2. Pick the model with the lowest log loss (best probabilities)
    best_name = min(results, key=lambda n: results[n]["log_loss"])
    print(f"\nBest model: {best_name}")

    # 3. Retrain the winner on ALL matches, so live predictions use the newest data
    final_model = candidate_models()[best_name]
    final_model.fit(df[FEATURES], df["FTR"])

    trained_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    current_teams = sorted(set(raw.loc[raw["Season"] == CURRENT_SEASON, "HomeTeam"]))
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": final_model,
        "model_name": best_name,
        "elo": elo,
        "form": form,
        "teams": current_teams,
        "trained_at": trained_at,
    }, MODEL_PATH)

    metrics = {
        "trained_at": trained_at,
        "train_matches": len(train),
        "test_matches": len(test),
        "test_seasons": TEST_SEASONS,
        "baseline_home_win_accuracy": baseline,
        "models": results,
        "best_model": best_name,
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved metrics to {METRICS_PATH}")


if __name__ == "__main__":
    main()
