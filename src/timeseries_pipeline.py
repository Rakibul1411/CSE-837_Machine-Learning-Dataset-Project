"""Train/evaluate/forecast routine for the classical time-series models (src/timeseries/).

Unlike the tabular sklearn models in src/models/ (see train_pipeline.py), these
operate on a single ordered monthly series rather than arbitrary feature rows,
so they use their own pipeline: a chronological holdout instead of a random
train/test split, and a forward forecast instead of row-by-row prediction.
"""
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config
from src.evaluate import compute_metrics, save_metrics
from src.timeseries import TIMESERIES_REGISTRY
from src.timeseries.data import get_national_frame
from src.timeseries.var_model import VAR_COLUMNS


def _run_model(model_name: str, train_frame: pd.DataFrame, horizon: int) -> np.ndarray:
    if model_name not in TIMESERIES_REGISTRY:
        available = ", ".join(sorted(TIMESERIES_REGISTRY))
        raise ValueError(f"Unknown time-series model '{model_name}'. Available: {available}")

    fit_forecast = TIMESERIES_REGISTRY[model_name]
    series_or_frame = (
        train_frame[VAR_COLUMNS] if model_name == "var" else train_frame[config.TARGET_COLUMN]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        forecast = fit_forecast(series_or_frame, horizon)
    return np.asarray(forecast)


def train_and_evaluate(model_name: str, horizon: int = 12, history_months: int = 24) -> dict:
    """Fit on all but the last `horizon` months, forecast that holdout, and score it."""
    national = get_national_frame()
    if len(national) <= horizon:
        raise ValueError(
            f"Only {len(national)} months of national history available; "
            f"can't hold out {horizon} months for testing."
        )

    train_frame, test_frame = national.iloc[:-horizon], national.iloc[-horizon:]
    forecast = _run_model(model_name, train_frame, horizon)
    actual = test_frame[config.TARGET_COLUMN]

    metrics = compute_metrics(actual, forecast)
    save_metrics(metrics, model_name)
    _plot_forecast(
        history=train_frame[config.TARGET_COLUMN],
        actual=actual,
        forecast=pd.Series(forecast, index=test_frame.index),
        model_name=model_name,
        history_months=history_months,
    )

    return {
        "model_name": model_name,
        "train_size": len(train_frame),
        "test_size": len(test_frame),
        "metrics": metrics,
    }


def forecast_future(model_name: str, horizon: int = 6) -> dict:
    """Fit on the full known history and forecast `horizon` months beyond it."""
    national = get_national_frame()
    forecast = _run_model(model_name, national, horizon)

    future_dates = pd.date_range(
        start=national.index[-1], periods=horizon + 1, freq=national.index.freq
    )[1:]
    return {
        "model_name": model_name,
        "forecast": [
            {"date": d.strftime("%Y-%m"), "prediction": max(0.0, float(v))}
            for d, v in zip(future_dates, forecast)
        ],
    }


def forecast_custom_range(model_name: str, start_date: str, end_date: str) -> dict:
    """Fit on national history and generate forecast over a custom start/end date range."""
    national = get_national_frame()
    start_dt = pd.to_datetime(start_date + "-01")
    end_dt = pd.to_datetime(end_date + "-01")

    min_hist_date = national.index[0]
    max_hist_date = national.index[-1]

    # Historical Actuals in range [start_dt, max_hist_date]
    effective_start = max(start_dt, min_hist_date)
    history_subset = national.loc[effective_start:max_hist_date]
    history_points = [
        {"date": d.strftime("%Y-%m"), "value": float(v)}
        for d, v in history_subset[config.TARGET_COLUMN].items()
    ]

    # Out-of-Sample Future Forecast (after max_hist_date up to end_dt)
    if end_dt > max_hist_date:
        horizon = (end_dt.year - max_hist_date.year) * 12 + (end_dt.month - max_hist_date.month)
        forecast_vals = _run_model(model_name, national, horizon)
        future_dates = pd.date_range(start=max_hist_date, periods=horizon + 1, freq=national.index.freq)[1:]
        forecast_points = [
            {"date": d.strftime("%Y-%m"), "prediction": max(0.0, float(v))}
            for d, v in zip(future_dates, forecast_vals)
        ]
    else:
        forecast_points = []

    _plot_custom_range(
        history=history_subset[config.TARGET_COLUMN],
        forecast_points=forecast_points,
        model_name=model_name,
    )

    return {
        "model_name": model_name,
        "start_date": start_date,
        "end_date": end_date,
        "history": history_points,
        "forecast": forecast_points,
    }


def _plot_forecast(
    history: pd.Series, actual: pd.Series, forecast: pd.Series, model_name: str, history_months: int = 24
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    hist_to_plot = history.tail(history_months) if history_months and history_months > 0 else history
    hist_to_plot.plot(ax=ax, label="History", color="#4f7cff")
    actual.plot(ax=ax, label="Actual", color="#1a1a2e", marker="o")
    forecast.plot(ax=ax, label="Forecast", color="#e0752d", linestyle="--", marker="x")
    ax.set_title(f"{model_name}: National Total Cases — Forecast vs Actual")
    ax.set_ylabel(config.TARGET_COLUMN)
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / f"{model_name}_forecast.png", dpi=150)
    plt.close(fig)


def _plot_custom_range(history: pd.Series, forecast_points: list[dict], model_name: str) -> None:
    n_points = len(history) + len(forecast_points)
    fig_width = max(8.0, min(24.0, n_points * 0.15))
    fig, ax = plt.subplots(figsize=(fig_width, 4.5))
    history.plot(ax=ax, label="History", color="#4f7cff", marker="o", markersize=3)

    if forecast_points:
        fc_dates = pd.to_datetime([p["date"] + "-01" for p in forecast_points])
        fc_values = [p["prediction"] for p in forecast_points]
        fc_series = pd.Series(fc_values, index=fc_dates)
        fc_series.plot(ax=ax, label="Future Forecast", color="#e0752d", linestyle="--", marker="x")

    ax.set_title(f"{model_name}: National Total Cases — Custom Range Forecast")
    ax.set_ylabel(config.TARGET_COLUMN)
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / f"{model_name}_forecast_range.png", dpi=150)
    plt.close(fig)
