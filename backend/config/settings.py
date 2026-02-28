import os
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

# DATA PATHS

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
DATA_PATH = os.path.join(PROJECT_ROOT, "data/processed/normalized_patient_features_24h_vitals_clean.csv")

# SIMILARITY SETTINGS
df = pd.read_csv(DATA_PATH)
print(df.columns.tolist())

NON_FEATURE_COLUMNS = ["patient_id", "sepsis_label"]

feature_columns = [
    c for c in df.columns
    if c not in NON_FEATURE_COLUMNS
]

TARGET_COLUMN = "SepsisLabel"

TOP_K_NEIGHBORS = 10

