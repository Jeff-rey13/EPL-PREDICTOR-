| Experiment | What changed | Accuracy | Log loss | vs current |
|---|---|---|---|---|
| logreg_simpler | Logistic regression, stronger regularisation | 50.5% | 1.0096 | -0.0003 |
| elo_only | Elo ratings only, no form | 50.1% | 1.0098 | -0.0001 |
| home_adv_90 | Bigger home advantage (90) | 50.4% | 1.0098 | -0.0001 |
| current_setup | Your current config.py settings | 50.2% | 1.0099 | +0.0000 |
| home_adv_30 | Smaller home advantage (30) | 50.1% | 1.0100 | +0.0001 |
| elo_k_30 | Faster-moving Elo (K=30) | 50.6% | 1.0112 | +0.0013 |
| elo_k_10 | Slower-moving Elo (K=10) | 49.1% | 1.0129 | +0.0030 |
| form_window_3 | Form over last 3 games | 50.1% | 1.0130 | +0.0031 |
| random_forest | Random forest instead of logistic regression | 50.4% | 1.0145 | +0.0047 |
| gradient_boosting | Gradient boosting instead of logistic regression | 49.5% | 1.0171 | +0.0072 |
| form_window_10 | Form over last 10 games | 49.1% | 1.0275 | +0.0176 |
| form_only | Form only, no Elo | 47.1% | 1.0513 | +0.0414 |
