"""SARIMA: ARIMA plus a 12-month seasonal component.

Implemented via SARIMAX with no exogenous inputs — statsmodels has no separate
"SARIMA" class, SARIMAX covers both.
"""
from statsmodels.tsa.statespace.sarimax import SARIMAX

ORDER = (0, 1, 1)
SEASONAL_ORDER = (0, 1, 1, 12)


def fit_forecast(train, horizon: int):
    fitted = SARIMAX(
        train,
        order=ORDER,
        seasonal_order=SEASONAL_ORDER,
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)
    return fitted.forecast(steps=horizon)
