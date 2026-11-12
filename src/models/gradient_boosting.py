"""Gradient Boosting Regressor module."""
import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor


def build_model() -> TransformedTargetRegressor:
    return TransformedTargetRegressor(
        regressor=MultiOutputRegressor(
            GradientBoostingRegressor(n_estimators=200, max_depth=5, random_state=42)
        ),
        func=np.log1p,
        inverse_func=np.expm1,
    )
