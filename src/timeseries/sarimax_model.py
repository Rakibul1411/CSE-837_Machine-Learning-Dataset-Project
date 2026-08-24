"""SARIMAX: SARIMA plus exogenous calendar regressors.

The exogenous inputs are Fourier (sin/cos) terms of the calendar month —
deterministic and known in advance for any future date, so they're safe to
use as exogenous features without leaking target information.
"""
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

ORDER = (1, 1, 1)
SEASONAL_ORDER = (1, 1, 1, 12)


def _fourier_terms(index: pd.DatetimeIndex) -> pd.DataFrame:
    month = index.month.values
    return pd.DataFrame(
        {"sin12": np.sin(2 * np.pi * month / 12), "cos12": np.cos(2 * np.pi * month / 12)},
        index=index,
    )


def fit_forecast(train: pd.Series, horizon: int):
    exog_train = _fourier_terms(train.index)
    fitted = SARIMAX(
        train,
        exog=exog_train,
        order=ORDER,
        seasonal_order=SEASONAL_ORDER,
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)

    future_index = pd.date_range(start=train.index[-1], periods=horizon + 1, freq=train.index.freq)[1:]
    exog_future = _fourier_terms(future_index)
    return fitted.forecast(steps=horizon, exog=exog_future)
