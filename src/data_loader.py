from pathlib import Path
import pandas as pd


DATA_PATH = Path("data/bitext_customer_service.csv")


def load_dataset() -> pd.DataFrame:
    """
    Load the Bitext customer service dataset.
    """
    return pd.read_csv(DATA_PATH)


def inspect_dataset() -> dict:
    """
    Return basic dataset information.
    """
    df = load_dataset()

    return {
        "rows": len(df),
        "columns": list(df.columns),
    }