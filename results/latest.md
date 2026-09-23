| Experiment | What changed | Train matches | Accuracy | Log loss | vs current |
|---|---|---|---|---|---|
| bookmakers_closing | Benchmark: closing odds used directly | 0 | 52.6% | 0.9889 | -0.0179 |
| elo_plus_cls_logodds | Elo + closing log-odds | 1900 | 52.2% | 0.9908 | -0.0159 |
| elo_plus_cls_logodds_weakreg | Elo + closing log-odds, weak reg. | 1900 | 52.5% | 0.9910 | -0.0157 |
| cls_logodds_only_weakreg | Closing log-odds only, weak reg. | 1900 | 52.7% | 0.9915 | -0.0152 |
| bookmakers | Benchmark: bookmaker odds used directly, no model | 0 | 52.0% | 0.9928 | -0.0139 |
| bookmakers_pinnacle | Benchmark: Pinnacle odds used directly | 0 | 51.8% | 0.9936 | -0.0132 |
| elo_plus_logodds | Elo + log-odds (market average) | 1900 | 51.6% | 0.9947 | -0.0120 |
| elo_plus_logodds_weakreg | Elo + log-odds, weak regularisation | 1900 | 51.4% | 0.9949 | -0.0118 |
| logodds_only | Log-odds only (market average) | 1900 | 52.3% | 0.9953 | -0.0115 |
| plus_logodds | Current setup + log-odds (market average) | 1892 | 51.0% | 0.9953 | -0.0114 |
| elo_plus_pin_logodds | Elo + Pinnacle log-odds | 1900 | 51.8% | 0.9953 | -0.0114 |
| logodds_only_weakreg | Log-odds only, weak regularisation | 1900 | 52.3% | 0.9955 | -0.0113 |
| elo_plus_pin_logodds_weakreg | Elo + Pinnacle log-odds, weak reg. | 1900 | 52.1% | 0.9955 | -0.0112 |
| elo_plus_odds | Elo + bookmaker odds, no form | 1900 | 51.8% | 0.9960 | -0.0107 |
| plus_odds | Current setup + bookmaker odds | 1892 | 51.3% | 0.9966 | -0.0102 |
| odds_only | Bookmaker odds as the only features | 1900 | 52.0% | 0.9968 | -0.0099 |
| odds_plus_squad | Current setup + odds + starting XI value | 1884 | 51.2% | 0.9993 | -0.0074 |
| logreg_simpler | Logistic regression, stronger regularisation | 1892 | 50.8% | 1.0065 | -0.0002 |
| home_adv_90 | Bigger home advantage (90) | 1892 | 50.7% | 1.0067 | -0.0001 |
| current_setup | Your current config.py settings | 1892 | 50.5% | 1.0067 | +0.0000 |
| home_adv_30 | Smaller home advantage (30) | 1892 | 50.5% | 1.0069 | +0.0001 |
| elo_only | Elo ratings only, no form | 1900 | 50.3% | 1.0072 | +0.0005 |
| elo_k_30 | Faster-moving Elo (K=30) | 1892 | 51.0% | 1.0075 | +0.0008 |
| elo_plus_squad_value | Elo + starting XI value, no form | 1884 | 49.9% | 1.0094 | +0.0027 |
| form_window_3 | Form over last 3 games | 1892 | 50.7% | 1.0095 | +0.0028 |
| plus_squad_value | Add starting XI value (player quality) | 1884 | 49.5% | 1.0107 | +0.0040 |
| elo_k_10 | Slower-moving Elo (K=10) | 1892 | 49.2% | 1.0109 | +0.0041 |
| random_forest | Random forest instead of logistic regression | 1892 | 50.5% | 1.0120 | +0.0052 |
| gradient_boosting | Gradient boosting instead of logistic regression | 1892 | 49.6% | 1.0163 | +0.0095 |
| current_setup_recent | Current setup, trained on 2022/23+ only | 758 | 50.8% | 1.0163 | +0.0096 |
| plus_xg_form | Add xG for/against form (2022/23+ training) | 747 | 50.9% | 1.0184 | +0.0116 |
| plus_all_player | Add XI value + xG form (2022/23+ training) | 747 | 50.8% | 1.0190 | +0.0122 |
| form_window_10 | Form over last 10 games | 1892 | 49.5% | 1.0229 | +0.0162 |
| form_only | Form only, no Elo | 1892 | 47.4% | 1.0485 | +0.0418 |
