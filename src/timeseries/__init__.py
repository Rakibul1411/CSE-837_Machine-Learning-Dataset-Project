"""Registry for classical time-series forecasting models (ARIMA family + VAR).

These are a different model family from src/models/: they forecast a single
ordered monthly series forward in time rather than predicting from arbitrary
(year, month, unit_name, unit_type) feature rows, so they get their own
registry and pipeline (src/timeseries_pipeline.py) instead of reusing the
tabular sklearn one.

Each module exposes `fit_forecast(train, horizon) -> array-like of length
`horizon``. To add a new one:

  1. Create src/timeseries/<name>_model.py with a `fit_forecast()` function.
  2. Import it below and add it to TIMESERIES_REGISTRY.
"""
from src.timeseries.arima_model import fit_forecast as arima_fit_forecast
from src.timeseries.sarima_model import fit_forecast as sarima_fit_forecast
from src.timeseries.sarimax_model import fit_forecast as sarimax_fit_forecast
from src.timeseries.var_model import fit_forecast as var_fit_forecast

TIMESERIES_REGISTRY = {
    "arima": arima_fit_forecast,
    "sarima": sarima_fit_forecast,
    "sarimax": sarimax_fit_forecast,
    "var": var_fit_forecast,
}
