"""Baseline linear regression model."""
from sklearn.linear_model import LinearRegression


def build_model() -> LinearRegression:
    return LinearRegression()
