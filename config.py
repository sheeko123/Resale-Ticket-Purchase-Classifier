import os
from pathlib import Path

# Project root directory
ROOT_DIR = Path(__file__).parent

# Data directories
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Model directories
MODEL_DIR = ROOT_DIR / "models"
TRAINED_MODELS_DIR = MODEL_DIR / "trained_models"

# Create directories if they don't exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, TRAINED_MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Data file paths
TICKET_DATA_PATH = RAW_DATA_DIR / "ticket_data.csv"

# Model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Analysis parameters
MIN_LISTINGS_PER_EVENT = 5
PRICE_VOLATILITY_THRESHOLD = 0.2 