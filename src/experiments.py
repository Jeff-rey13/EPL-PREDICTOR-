"""
Experiment runner: tests many versions of the model and records the results.

Run from the project folder:
    python -m src.experiments

Every experiment starts from your current settings in config.py and changes
ONE thing, so you can see exactly what each change does.

Results are saved to:
    results/experiments.csv   every run ever, with a timestamp (your lab notebook)
    results/latest.md         a table from the latest run, ready to paste into README.md

To add an experiment, add a line to the EXPERIMENTS list below.
"""
from datetime import datetime, timezone

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import (ELO_HOME_ADV, ELO_K, FEATURES, FORM_WINDOW, ROOT, TEST_SEASONS,
                        WARMUP_SEASON)
from src.data import load_data
from src.features import build_features

RESULTS_DIR = ROOT / "results"

ELO_FEATURES = ["HomeElo", "AwayElo", "EloDiff"]
FORM_FEATURES = ["H_GF_form", "H_GA_form", "H_Pts_form",
                 "A_GF_form", "A_GA_form", "A_Pts_form"]

# The starting point every experiment is compared against (your config.py settings)
DEFAULTS = {
    "form_window": FORM_WINDOW,
    "elo_k": ELO_K,
    "elo_home_adv": ELO_HOME_ADV,
    "features": FEATURES,
    "model": "logistic_regression",
}

# ---------------------------------------------------------------- Experiments
# Each one changes a single setting from DEFAULTS.
EXPERIMENTS = [
    {"name": "current_setup",       "description": "Your current config.py settings"},

    # Which features matter?
    {"name": "elo_only",            "description": "Elo ratings only, no form",
     "features": ELO_FEATURES},
    {"name": "form_only",           "description": "Form only, no Elo",
     "features": FORM_FEATURES},

    # How many recent games should count as "form"?
    {"name": "form_window_3",
        "description": "Form over last 3 games",  "form_window": 3},
    {"name": "form_window_10",
        "description": "Form over last 10 games", "form_window": 10},

    # How fast should Elo ratings react to results?
    {"name": "elo_k_10",
        "description": "Slower-moving Elo (K=10)", "elo_k": 10},
    {"name": "elo_k_30",
        "description": "Faster-moving Elo (K=30)", "elo_k": 30},

    # How big is home advantage?
    {"name": "home_adv_30",
        "description": "Smaller home advantage (30)", "elo_home_adv": 30},
    {"name": "home_adv_90",
        "description": "Bigger home advantage (90)",  "elo_home_adv": 90},

    # Which model type?
    {"name": "logreg_simpler",      "description": "Logistic regression, stronger regularisation",
     "model": "logistic_regression_c0.1"},
    {"name": "gradient_boosting",   "description": "Gradient boosting instead of logistic regression",
     "model": "gradient_boosting"},
    {"name": "random_forest",       "description": "Random forest instead of logistic regression",
     "model": "random_forest"},
]


def make_model(name):
    models = {
        "logistic_regression": lambda: make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000)),
        "logistic_regression_c0.1": lambda: make_pipeline(
            StandardScaler(), LogisticRegression(C=0.1, max_iter=1000)),
        "gradient_boosting": lambda: GradientBoostingClassifier(
            n_estimators=150, max_depth=2, learning_rate=0.05, random_state=42),
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=300, max_depth=6, min_samples_leaf=20, random_state=42),
    }
    return models[name]()


def run_experiment(settings, featured):
    """Train on older seasons, test on newer ones, return the scores."""
    df = featured[(settings["form_window"], settings["elo_k"],
                   settings["elo_home_adv"])]
    df = df[df["Season"] != WARMUP_SEASON].dropna(
        subset=ELO_FEATURES + FORM_FEATURES)
    train = df[~df["Season"].isin(TEST_SEASONS)]
    test = df[df["Season"].isin(TEST_SEASONS)]

    model = make_model(settings["model"])
    model.fit(train[settings["features"]], train["FTR"])
    probs = model.predict_proba(test[settings["features"]])
    preds = model.predict(test[settings["features"]])
    return {
        "accuracy": accuracy_score(test["FTR"], preds),
        "log_loss": log_loss(test["FTR"], probs, labels=model.classes_),
        "baseline": (test["FTR"] == "H").mean(),
        "test_matches": len(test),
    }


def save_results(table, run_time):
    RESULTS_DIR.mkdir(exist_ok=True)

    # Append to the full history, so you never lose old results
    history = RESULTS_DIR / "experiments.csv"
    table.assign(run_time=run_time).to_csv(history, mode="a", index=False,
                                           header=not history.exists())

    # Markdown table for the README
    lines = ["| Experiment | What changed | Accuracy | Log loss | vs current |",
             "|---|---|---|---|---|"]
    for r in table.itertuples():
        lines.append(f"| {r.name} | {r.description} | {r.accuracy:.1%} | "
                     f"{r.log_loss:.4f} | {r.vs_current:+.4f} |")
    (RESULTS_DIR / "latest.md").write_text("\n".join(lines) + "\n")


def main():
    raw = load_data()
    run_time = datetime.now(timezone.utc).isoformat(timespec="seconds")

    # Building features is the slow part, so build each combination only once
    featured = {}
    rows = []
    for exp in EXPERIMENTS:
        settings = {**DEFAULTS, **exp}
        key = (settings["form_window"], settings["elo_k"],
               settings["elo_home_adv"])
        if key not in featured:
            featured[key], _, _ = build_features(raw, form_window=key[0],
                                                 elo_k=key[1], elo_home_adv=key[2])
        print(f"Running {exp['name']}...")
        scores = run_experiment(settings, featured)
        rows.append(
            {"name": exp["name"], "description": exp["description"], **scores})

    table = pd.DataFrame(rows)
    current = table.loc[table["name"] == "current_setup", "log_loss"].iloc[0]
    table["vs_current"] = table["log_loss"] - current   # negative = better
    table = table.sort_values("log_loss").reset_index(drop=True)

    print(f"\nTest seasons: {', '.join(TEST_SEASONS)} "
          f"({table['test_matches'].iloc[0]} matches)")
    print(
        f"Baseline (always predict home win): {table['baseline'].iloc[0]:.1%}\n")
    print(table[["name", "accuracy", "log_loss", "vs_current"]].to_string(
        index=False, formatters={"accuracy": "{:.1%}".format,
                                 "log_loss": "{:.4f}".format,
                                 "vs_current": "{:+.4f}".format}))
    print("\nSorted best to worst by log loss. Negative 'vs_current' = better than now.")

    save_results(table, run_time)
    print(f"\nSaved: {RESULTS_DIR / 'experiments.csv'}")
    print(f"       {RESULTS_DIR / 'latest.md'}")


if __name__ == "__main__":
    main()
