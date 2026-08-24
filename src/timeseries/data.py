"""Build the national (aggregate) monthly time series used by the ARIMA-family models.

Unlike the tabular models in src/models/, which predict Total Cases from
(year, month, unit_name, unit_type) feature rows, classical time-series models
need a single chronologically ordered series. We use the national total
('unit_name' == "Total") since it's the only series with one row per month
that isn't itself a duplicate breakdown of another row.
"""
import pandas as pd

from src.data_loader import load_dataset


def get_national_frame() -> pd.DataFrame:
    """Monthly national totals, indexed by month-start date with an explicit frequency."""
    df = load_dataset()
    national = df[df["unit_name"] == "Total"].copy()
    national["date"] = pd.to_datetime(
        national["year"].astype(str) + "-" + national["month_number"].astype(str) + "-01"
    )
    national = national.sort_values("date").set_index("date")
    return national.asfreq("MS")
