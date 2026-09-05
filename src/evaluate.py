"""Metrics and diagnostic plots, reused across every model we train."""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src import config


def compute_metrics(y_true, y_pred) -> dict:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def save_metrics(metrics: dict, model_name: str) -> None:
    path = config.METRICS_DIR / f"{model_name}_metrics.json"
    path.write_text(json.dumps(metrics, indent=2))


def _extract_1d(y_true, y_pred):
    if hasattr(y_true, "columns") and config.TARGET_COLUMN in y_true.columns:
        idx = list(y_true.columns).index(config.TARGET_COLUMN)
        yt = y_true[config.TARGET_COLUMN].to_numpy()
        if isinstance(y_pred, np.ndarray) and y_pred.ndim > 1:
            yp = y_pred[:, idx]
        elif hasattr(y_pred, "columns"):
            yp = y_pred[config.TARGET_COLUMN].to_numpy()
        else:
            yp = np.asarray(y_pred)
        return yt, yp
    yt = np.asarray(y_true).ravel()
    yp = np.asarray(y_pred).ravel()
    return yt, yp


def plot_predictions(y_true, y_pred, model_name: str) -> None:
    yt, yp = _extract_1d(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(yt, yp, alpha=0.4, s=15)
    lims = [min(yt.min(), yp.min()), max(yt.max(), yp.max())]
    ax.plot(lims, lims, "r--", linewidth=1, label="Perfect prediction")
    ax.set_xlabel("Actual Total Cases")
    ax.set_ylabel("Predicted Total Cases")
    ax.set_title(f"{model_name}: Actual vs Predicted")
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / f"{model_name}_actual_vs_predicted.png", dpi=150)
    plt.close(fig)


def plot_residuals(y_true, y_pred, model_name: str) -> None:
    yt, yp = _extract_1d(y_true, y_pred)
    residuals = yt - yp
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(yp, residuals, alpha=0.4, s=15)
    ax.axhline(0, color="r", linestyle="--", linewidth=1)
    ax.set_xlabel("Predicted Total Cases")
    ax.set_ylabel("Residual")
    ax.set_title(f"{model_name}: Residual Plot")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / f"{model_name}_residuals.png", dpi=150)
    plt.close(fig)


def evaluate(y_true, y_pred, model_name: str) -> dict:
    """Compute metrics, persist them, and save diagnostic plots."""
    metrics = compute_metrics(y_true, y_pred)
    save_metrics(metrics, model_name)
    plot_predictions(y_true, y_pred, model_name)
    plot_residuals(y_true, y_pred, model_name)
    return metrics
