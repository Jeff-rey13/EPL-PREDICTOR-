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
from src.players import PLAYER_FEATURES, SQUAD_FEATURES, XG_FEATURES, add_player_features

RESULTS_DIR = ROOT / "results"

# Matches played behind closed doors (COVID): home advantage almost disappeared
NO_FANS_START, NO_FANS_END = "2020-06-01", "2021-05-31"

ELO_FEATURES = ["HomeElo", "AwayElo", "EloDiff"]
FORM_FEATURES = ["H_GF_form", "H_GA_form", "H_Pts_form",
                 "A_GF_form", "A_GA_form", "A_Pts_form"]
ODDS_FEATURES = ["Mkt_pH", "Mkt_pA"]      # draw prob = 1 - these two, so not needed
LOGODDS = {p: [f"{p}_lH", f"{p}_lD", f"{p}_lA"] for p in ["Mkt", "Pin", "Cls"]}
MARKET_PROBS = [f"{p}_p{s}" for p in ["Mkt", "Pin", "Cls"] for s in "HDA"]
ORIGINAL_FEATURES = ELO_FEATURES + FORM_FEATURES   # the model before odds were added
ALL_FEATURES = ELO_FEATURES + FORM_FEATURES + PLAYER_FEATURES + MARKET_PROBS

# The starting point every experiment is compared against (your config.py settings)
DEFAULTS = {
    "form_window": FORM_WINDOW,
    "elo_k": ELO_K,
    "elo_home_adv": ELO_HOME_ADV,
    "features": FEATURES,
    "model": "logistic_regression",
    "train_from": None,        # e.g. "2223" = only train on 2022/23 onwards
    "market": "Mkt",           # which odds a "market" benchmark uses: Mkt, Pin or Cls
    "drop_no_fans": False,     # True = leave out COVID no-crowd matches from training
}

# ---------------------------------------------------------------- Experiments
# Each one changes a single setting from DEFAULTS.
EXPERIMENTS = [
    {"name": "current_setup",       "description": "Your current config.py settings"},
    {"name": "original_setup",      "description": "The original model: Elo + form, no odds",
     "features": ORIGINAL_FEATURES},

    # Bookmaker odds: the benchmark to beat, and odds as model features
    {"name": "bookmakers",          "description": "Benchmark: bookmaker odds used directly, no model",
     "model": "market"},
    {"name": "plus_odds",           "description": "Current setup + bookmaker odds",
     "features": ORIGINAL_FEATURES + ODDS_FEATURES},
    {"name": "elo_plus_odds",       "description": "Elo + bookmaker odds, no form",
     "features": ELO_FEATURES + ODDS_FEATURES},
    {"name": "odds_only",           "description": "Bookmaker odds as the only features",
     "features": ODDS_FEATURES},
    {"name": "odds_plus_squad",     "description": "Current setup + odds + starting XI value",
     "features": ORIGINAL_FEATURES + ODDS_FEATURES + SQUAD_FEATURES},

    # Odds in log space (lets the model reproduce the bookmakers, then adjust)
    {"name": "logodds_only",        "description": "Log-odds only (market average)",
     "features": LOGODDS["Mkt"]},
    {"name": "elo_plus_logodds",    "description": "Elo + log-odds (market average)",
     "features": ELO_FEATURES + LOGODDS["Mkt"]},
    {"name": "plus_logodds",        "description": "Current setup + log-odds (market average)",
     "features": ORIGINAL_FEATURES + LOGODDS["Mkt"]},

    # Sharper odds: Pinnacle, and closing odds (set just before kickoff)
    {"name": "bookmakers_pinnacle", "description": "Benchmark: Pinnacle odds used directly",
     "model": "market", "market": "Pin"},
    {"name": "bookmakers_closing",  "description": "Benchmark: closing odds used directly",
     "model": "market", "market": "Cls"},
    {"name": "elo_plus_pin_logodds", "description": "Elo + Pinnacle log-odds",
     "features": ELO_FEATURES + LOGODDS["Pin"]},
    {"name": "elo_plus_cls_logodds", "description": "Elo + closing log-odds",
     "features": ELO_FEATURES + LOGODDS["Cls"]},

    # Leave out the COVID no-crowd matches from training (home advantage vanished)
    {"name": "current_setup_no_covid", "description": "Original setup, no-crowd matches removed",
     "features": ORIGINAL_FEATURES, "drop_no_fans": True},
    {"name": "logodds_only_no_covid", "description": "Log-odds only, no-crowd matches removed",
     "features": LOGODDS["Mkt"], "drop_no_fans": True},
    {"name": "elo_plus_logodds_no_covid", "description": "Elo + log-odds, no-crowd matches removed",
     "features": ELO_FEATURES + LOGODDS["Mkt"], "drop_no_fans": True},
    {"name": "elo_plus_cls_logodds_no_covid", "description": "Elo + closing log-odds, no-crowd removed",
     "features": ELO_FEATURES + LOGODDS["Cls"], "drop_no_fans": True},

    # Same models with much weaker regularisation (C=100 instead of 1)
    {"name": "logodds_only_weakreg", "description": "Log-odds only, weak regularisation",
     "features": LOGODDS["Mkt"], "model": "logistic_regression_c100"},
    {"name": "elo_plus_logodds_weakreg", "description": "Elo + log-odds, weak regularisation",
     "features": ELO_FEATURES + LOGODDS["Mkt"], "model": "logistic_regression_c100"},
    {"name": "elo_plus_pin_logodds_weakreg", "description": "Elo + Pinnacle log-odds, weak reg.",
     "features": ELO_FEATURES + LOGODDS["Pin"], "model": "logistic_regression_c100"},
    {"name": "cls_logodds_only_weakreg", "description": "Closing log-odds only, weak reg.",
     "features": LOGODDS["Cls"], "model": "logistic_regression_c100"},
    {"name": "elo_plus_cls_logodds_weakreg", "description": "Elo + closing log-odds, weak reg.",
     "features": ELO_FEATURES + LOGODDS["Cls"], "model": "logistic_regression_c100"},

    # Which features matter?
    {"name": "elo_only",            "description": "Elo ratings only, no form",
     "features": ELO_FEATURES},
    {"name": "form_only",           "description": "Form only, no Elo",
     "features": FORM_FEATURES},

    # How many recent games should count as "form"?
    {"name": "form_window_3",       "description": "Form over last 3 games",  "form_window": 3},
    {"name": "form_window_10",      "description": "Form over last 10 games", "form_window": 10},

    # How fast should Elo ratings react to results?
    {"name": "elo_k_10",            "description": "Slower-moving Elo (K=10)", "elo_k": 10},
    {"name": "elo_k_30",            "description": "Faster-moving Elo (K=30)", "elo_k": 30},

    # How big is home advantage?
    {"name": "home_adv_30",         "description": "Smaller home advantage (30)", "elo_home_adv": 30},
    {"name": "home_adv_90",         "description": "Bigger home advantage (90)",  "elo_home_adv": 90},

    # Which model type?
    {"name": "logreg_simpler",      "description": "Logistic regression, stronger regularisation",
     "model": "logistic_regression_c0.1"},
    {"name": "gradient_boosting",   "description": "Gradient boosting instead of logistic regression",
     "model": "gradient_boosting"},
    {"name": "random_forest",       "description": "Random forest instead of logistic regression",
     "model": "random_forest"},

    # Player features (from Fantasy Premier League data)
    {"name": "plus_squad_value",    "description": "Add starting XI value (player quality)",
     "features": ORIGINAL_FEATURES + SQUAD_FEATURES},
    {"name": "elo_plus_squad_value", "description": "Elo + starting XI value, no form",
     "features": ELO_FEATURES + SQUAD_FEATURES},
    # xG only exists from 2022/23, so these train on fewer seasons. The first one
    # tells us how much of any change is just from having less training data.
    {"name": "current_setup_recent", "description": "Original setup, trained on 2022/23+ only",
     "features": ORIGINAL_FEATURES, "train_from": "2223"},
    {"name": "plus_xg_form",        "description": "Add xG for/against form (2022/23+ training)",
     "features": ORIGINAL_FEATURES + XG_FEATURES, "train_from": "2223"},
    {"name": "plus_all_player",     "description": "Add XI value + xG form (2022/23+ training)",
     "features": ORIGINAL_FEATURES + PLAYER_FEATURES, "train_from": "2223"},
]


def make_model(name):
    models = {
        "logistic_regression": lambda: make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000)),
        "logistic_regression_c0.1": lambda: make_pipeline(
            StandardScaler(), LogisticRegression(C=0.1, max_iter=1000)),
        # Very weak regularisation: lets the model keep the bookmakers' sharp probabilities
        "logistic_regression_c100": lambda: make_pipeline(
            StandardScaler(), LogisticRegression(C=100, max_iter=5000)),
        "gradient_boosting": lambda: GradientBoostingClassifier(
            n_estimators=150, max_depth=2, learning_rate=0.05, random_state=42),
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=300, max_depth=6, min_samples_leaf=20, random_state=42),
    }
    return models[name]()


def run_experiment(settings, featured):
    """Train on older seasons, test on newer ones, return the scores."""
    df = featured[(settings["form_window"], settings["elo_k"], settings["elo_home_adv"])]
    df = df[df["Season"] != WARMUP_SEASON]
    train = df[~df["Season"].isin(TEST_SEASONS)].dropna(subset=settings["features"])
    if settings["train_from"]:
        train = train[train["Season"] >= settings["train_from"]]
    if settings["drop_no_fans"]:
        train = train[~train["Date"].between(NO_FANS_START, NO_FANS_END)]
    # Every experiment is tested on exactly the same matches: ones with ALL features
    test = df[df["Season"].isin(TEST_SEASONS)].dropna(subset=ALL_FEATURES)

    if settings["model"] == "market":
        # No training: use the bookmakers' probabilities as the prediction
        classes = ["A", "D", "H"]
        m = settings["market"]
        probs = test[[f"{m}_pA", f"{m}_pD", f"{m}_pH"]].values
        preds = [classes[i] for i in probs.argmax(axis=1)]
    else:
        model = make_model(settings["model"])
        model.fit(train[settings["features"]], train["FTR"])
        classes = model.classes_
        probs = model.predict_proba(test[settings["features"]])
        preds = model.predict(test[settings["features"]])
    return {
        "accuracy": accuracy_score(test["FTR"], preds),
        "log_loss": log_loss(test["FTR"], probs, labels=classes),
        "baseline": (test["FTR"] == "H").mean(),
        "train_matches": 0 if settings["model"] == "market" else len(train),
        "test_matches": len(test),
    }


def save_results(table, run_time):
    RESULTS_DIR.mkdir(exist_ok=True)

    # Add to the full history, so you never lose old results
    history = RESULTS_DIR / "experiments.csv"
    new = table.assign(run_time=run_time)
    if history.exists():
        new = pd.concat([pd.read_csv(history), new], ignore_index=True)
    new["train_matches"] = new["train_matches"].astype("Int64")   # 1892, not 1892.0
    new.to_csv(history, index=False)

    # Markdown table for the README
    lines = ["| Experiment | What changed | Train matches | Accuracy | Log loss | vs current |",
             "|---|---|---|---|---|---|"]
    for r in table.itertuples():
        lines.append(f"| {r.name} | {r.description} | {r.train_matches} | {r.accuracy:.1%} | "
                     f"{r.log_loss:.4f} | {r.vs_current:+.4f} |")
    (RESULTS_DIR / "latest.md").write_text("\n".join(lines) + "\n")


def main():
    raw = add_player_features(load_data())
    run_time = datetime.now(timezone.utc).isoformat(timespec="seconds")

    # Building features is the slow part, so build each combination only once
    featured = {}
    rows = []
    for exp in EXPERIMENTS:
        settings = {**DEFAULTS, **exp}
        key = (settings["form_window"], settings["elo_k"], settings["elo_home_adv"])
        if key not in featured:
            featured[key], _, _ = build_features(raw, form_window=key[0],
                                                 elo_k=key[1], elo_home_adv=key[2])
        print(f"Running {exp['name']}...")
        scores = run_experiment(settings, featured)
        rows.append({"name": exp["name"], "description": exp["description"], **scores})

    table = pd.DataFrame(rows)
    current = table.loc[table["name"] == "current_setup", "log_loss"].iloc[0]
    table["vs_current"] = table["log_loss"] - current   # negative = better
    table = table.sort_values("log_loss").reset_index(drop=True)

    print(f"\nTest seasons: {', '.join(TEST_SEASONS)} "
          f"({table['test_matches'].iloc[0]} matches)")
    print(f"Baseline (always predict home win): {table['baseline'].iloc[0]:.1%}\n")
    print(table[["name", "train_matches", "accuracy", "log_loss", "vs_current"]].to_string(
        index=False, formatters={"accuracy": "{:.1%}".format,
                                 "log_loss": "{:.4f}".format,
                                 "vs_current": "{:+.4f}".format}))
    print("\nSorted best to worst by log loss. Negative 'vs_current' = better than now.")

    save_results(table, run_time)
    print(f"\nSaved: {RESULTS_DIR / 'experiments.csv'}")
    print(f"       {RESULTS_DIR / 'latest.md'}")


if __name__ == "__main__":
    main()
