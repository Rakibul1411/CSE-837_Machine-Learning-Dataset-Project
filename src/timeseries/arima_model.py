"""ARIMA(p,d,q): univariate forecasting with no seasonal component."""
from statsmodels.tsa.arima.model import ARIMA

ORDER = (2, 2, 2)


def fit_forecast(train, horizon: int):
    fitted = ARIMA(train, order=ORDER).fit()
    return fitted.forecast(steps=horizon)
