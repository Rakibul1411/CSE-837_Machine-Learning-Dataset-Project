"""Feature pipeline and train/test splitting shared by every model."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config


def build_preprocessor() -> ColumnTransformer:
    """Numeric features get scaled; categorical features get one-hot encoded.

    Returned unfitted so each model's pipeline fits it only on training data.
    """
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), config.NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                config.CATEGORICAL_FEATURES,
            ),
        ]
    )


def get_features_and_target(df: pd.DataFrame):
    X = df[config.FEATURE_COLUMNS]
    y = df[config.TARGET_COLUMN]
    return X, y


def split_data(df: pd.DataFrame, test_size: float | None = None):
    X, y = get_features_and_target(df)
    return train_test_split(
        X,
        y,
        test_size=test_size if test_size is not None else config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
    )


def make_pipeline(estimator) -> Pipeline:
    """Wrap any sklearn-compatible estimator with the shared preprocessor."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", estimator),
        ]
    )
