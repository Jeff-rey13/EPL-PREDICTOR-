"""All settings in one place, so you only change things here."""
from pathlib import Path

# ---------------------------------------------------------------- Paths
ROOT = Path(__file__).resolve().parent.parent      # the project folder
DATA_DIR = ROOT / "data" / "raw"                   # downloaded CSVs (not on GitHub)
MODEL_PATH = ROOT / "models" / "model.joblib"      # trained model
METRICS_PATH = ROOT / "models" / "metrics.json"    # evaluation results

# ---------------------------------------------------------------- Data
SEASONS = ["1819", "1920", "2021", "2122", "2223", "2324", "2425", "2526", "2627"]
CURRENT_SEASON = SEASONS[-1]      # always re-downloaded (new results every week)
WARMUP_SEASON = "1819"            # Elo needs a season to settle; not used for training
TEST_SEASONS = ["2425", "2526", "2627"]
RESULTS_URL = "https://www.football-data.co.uk/mmz4281/{}/E0.csv"
FIXTURES_URL = "https://api.football-data.org/v4/competitions/PL/matches"
ODDS_FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"   # this week's games + odds
ODDS_API_URL = "https://api.the-odds-api.com/v4/sports/soccer_epl/odds"  # games further ahead

# ---------------------------------------------------------------- Model
FOCUS_TEAM = "Man United"         # extra accuracy line printed for this team
FORM_WINDOW = 5                   # number of recent games used for form
ELO_K = 20                        # how fast Elo ratings move
ELO_HOME_ADV = 60                 # Elo points added for playing at home
ELO_START = 1500                  # rating for teams in the first season
ELO_NEWCOMER = 1450               # rating for newly promoted teams

ELO_FEATURES = ["HomeElo", "AwayElo", "EloDiff"]
ODDS_LOG_FEATURES = ["Mkt_lH", "Mkt_lD", "Mkt_lA"]   # log of bookmaker probabilities

# Main model: Elo + bookmaker log-odds (chosen from the experiments, see results/)
FEATURES = ELO_FEATURES + ODDS_LOG_FEATURES
# Fallback model: used when no odds are available yet for a fixture
FALLBACK_FEATURES = ELO_FEATURES

# ---------------------------------------------------------------- Team names
# Nicknames and API names -> the names used in football-data.co.uk files
ALIASES = {
    "man utd": "Man United", "man u": "Man United", "united": "Man United",
    "manchester united": "Man United", "mufc": "Man United",
    "man city": "Man City", "city": "Man City", "manchester city": "Man City",
    "spurs": "Tottenham", "tottenham hotspur": "Tottenham",
    "villa": "Aston Villa", "forest": "Nott'm Forest", "nottingham forest": "Nott'm Forest",
    "wolverhampton": "Wolves", "wolverhampton wanderers": "Wolves",
    "newcastle united": "Newcastle", "west ham united": "West Ham",
    "brighton & hove albion": "Brighton", "brighton and hove albion": "Brighton",
    "leeds united": "Leeds", "palace": "Crystal Palace", "afc bournemouth": "Bournemouth",
    "ipswich town": "Ipswich", "coventry city": "Coventry", "hull city": "Hull",
    "leicester city": "Leicester", "sheffield united": "Sheffield United",
}
