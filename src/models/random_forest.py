import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor


def build_model() -> TransformedTargetRegressor:
    return TransformedTargetRegressor(
        regressor=RandomForestRegressor(n_estimators=300, max_depth=15, min_samples_split=2, random_state=42),
        func=np.log1p,
        inverse_func=np.expm1,
    )

