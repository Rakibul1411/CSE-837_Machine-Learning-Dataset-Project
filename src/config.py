"""Central configuration: paths, dataset schema, and shared constants."""
from pathlib import Path

# --- Paths -------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = ROOT_DIR / "data" / "raw" / "bangladesh_crime_statistics_wide.xlsx"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"

OUTPUTS_DIR = ROOT_DIR / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
FIGURES_DIR = OUTPUTS_DIR / "figures"
METRICS_DIR = OUTPUTS_DIR / "metrics"

for _dir in (PROCESSED_DATA_DIR, MODELS_DIR, FIGURES_DIR, METRICS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# --- Dataset schema ------------------------------------------------------
# Individual crime-category columns. `Total Cases` is their exact row-wise
# sum, so they are excluded from the feature set to avoid trivial leakage.
CRIME_CATEGORY_COLUMNS = [
    "Dacoity",
    "Robbery",
    "Murder",
    "Speedy Trial",
    "Riot",
    "Woman & Child Repression",
    "Kidnapping",
    "Police Assault",
    "Burglary",
    "Theft",
    "Other Cases",
    "Arms Act",
    "Explosive Act",
    "Narcotics",
    "Smuggling",
]

TARGET_COLUMN = "Total Cases"

NUMERIC_FEATURES = ["year", "month_number"]
CATEGORICAL_FEATURES = ["unit_name", "unit_type"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# --- Train/test split & reproducibility ----------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2
