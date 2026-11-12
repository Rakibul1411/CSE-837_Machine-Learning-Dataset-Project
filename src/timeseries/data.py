"""Build the national (aggregate) monthly time series used by the ARIMA-family models.

Unlike the tabular models in src/models/, which predict Total Cases from
(year, month, unit_name, unit_type) feature rows, classical time-series models
need a single chronologically ordered series. We use the national total
('unit_name' == "Total") since it's the only series with one row per month
that isn't itself a duplicate breakdown of another row.
"""
import pandas as pd

from src.data_loader import load_dataset


def get_unit_frame(unit_name: str = "Total") -> pd.DataFrame:
    """Monthly totals for a given unit (or 'Total' for national aggregate), indexed by month-start date."""
    df = load_dataset()
    unit_df = df[df["unit_name"] == unit_name].copy()
    if unit_df.empty:
        # Fallback to Total if unit_name is not found
        unit_df = df[df["unit_name"] == "Total"].copy()

    unit_df["date"] = pd.to_datetime(
        unit_df["year"].astype(str) + "-" + unit_df["month_number"].astype(str) + "-01"
    )
    unit_df = unit_df.sort_values("date").set_index("date")
    return unit_df.asfreq("MS")


def get_national_frame() -> pd.DataFrame:
    """Monthly national totals wrapper for backward compatibility."""
    return get_unit_frame("Total")

