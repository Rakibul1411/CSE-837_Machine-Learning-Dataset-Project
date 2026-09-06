"""VAR: Vector Autoregression, jointly modeling Total Cases with related crime categories.

VAR lets Total Cases' forecast draw on lagged values of correlated series
(Theft, Burglary, Robbery) rather than only its own history.
"""
import pandas as pd
from statsmodels.tsa.vector_ar.var_model import VAR

from src import config

VAR_COLUMNS = [config.TARGET_COLUMN, "Theft", "Burglary", "Robbery"]


def fit_forecast(train: pd.DataFrame, horizon: int):
    fitted = VAR(train[VAR_COLUMNS]).fit(maxlags=4)
    lag_order = max(fitted.k_ar, 1)
    forecast_values = fitted.forecast(train[VAR_COLUMNS].to_numpy()[-lag_order:], steps=horizon)
    forecast_df = pd.DataFrame(forecast_values, columns=VAR_COLUMNS)
    return forecast_df[config.TARGET_COLUMN].to_numpy()
