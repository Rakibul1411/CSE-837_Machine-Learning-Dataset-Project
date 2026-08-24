"""Load and clean the Bangladesh crime statistics dataset."""
import pandas as pd

from src import config


def load_raw_data() -> pd.DataFrame:
    """Read the raw wide-format Excel sheet as-is."""
    return pd.read_excel(config.RAW_DATA_PATH, sheet_name="Wide Format")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only monthly records (drop the yearly 'Annual (Jan-Dec)' rollups,
    which carry no month_number and would double-count the monthly rows).
    """
    df = df.copy()
    monthly = df[df["month"] != "Annual (Jan-Dec)"].reset_index(drop=True)
    monthly["month_number"] = monthly["month_number"].astype(int)
    return monthly


def load_dataset() -> pd.DataFrame:
    """Load and clean the dataset in one call."""
    return clean_data(load_raw_data())


if __name__ == "__main__":
    data = load_dataset()
    print(f"Loaded {len(data)} monthly records, {data['unit_name'].nunique()} units, "
          f"years {data['year'].min()}-{data['year'].max()}")
