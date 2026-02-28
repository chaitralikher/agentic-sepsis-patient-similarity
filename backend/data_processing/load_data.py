import pandas as pd
from backend.config.settings import DATA_PATH

def load_feature_dataset():
    """
    Load processed patient feature dataset.
    Must contain aggregated 24h features + SepsisLabel.
    """
    df = pd.read_csv(DATA_PATH)

    if df.empty:
        raise ValueError("Feature dataset is empty")

    return df
