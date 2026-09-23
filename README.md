# ⚽ Premier League Match Predictor

Predicts the win / draw / loss probabilities for any Premier League team's next match,
using Elo ratings and recent form, trained on every Premier League match since 2018/19.

> 🚧 Work in progress. Live app coming in a later phase.

## Results

Evaluated on held-out seasons (2024/25 onward). The model never saw these matches during training.

| Model | Accuracy | Log loss |
|---|---|---|
| Baseline (always predict home win) | _fill in_ | – |
| Logistic regression | _fill in_ | _fill in_ |
| Gradient boosting | _fill in_ | _fill in_ |

Latest numbers are always in [`models/metrics.json`](models/metrics.json).

## How it works

1. **Data:** match results from [football-data.co.uk](https://www.football-data.co.uk);
   upcoming fixtures from the [football-data.org](https://www.football-data.org) API.
2. **Features:**
   - *Elo ratings:* a running strength score for each team, updated after every match.
   - *Recent form:* average points, goals scored and goals conceded over the last 5 games.
3. **Avoiding data leakage:** every feature uses only information available *before*
   kickoff, and the model is tested on later seasons than it was trained on.
4. **Model:** logistic regression and gradient boosting are compared; the one with the
   lowest log loss is retrained on all data and used for predictions.

## Project structure

```
src/
  config.py     settings (seasons, Elo parameters, team name aliases)
  data.py       downloads and caches match results
  features.py   Elo ratings and form features
  train.py      trains, evaluates and saves the model
  predict.py    predicts a team's next match
models/         saved model + evaluation metrics
notebooks/      exploration and learning
```

## Run it yourself

```bash
git clone https://github.com/YOUR_USERNAME/epl-predictor.git
cd epl-predictor
python -m venv .venv
.venv\Scripts\activate          # Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt

python -m src.train             # download data, train and save the model
python -m src.predict "man utd" # predict a team's next match
```

For automatic fixture lookup, copy `.env.example` to `.env` and add a free
football-data.org API key. Without it, the predictor asks you for the opponent.

## Limitations

Football is highly random: even bookmakers are right only about 53–55% of the time.
The model doesn't know about injuries, suspensions or managerial changes.

## Next steps

- [ ] Add bookmaker odds and xG as features
- [ ] Streamlit web app
- [ ] Weekly automatic retraining with GitHub Actions
- [ ] Live track record for the 2026/27 season
