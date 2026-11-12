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


def _extract_1d(y_true, y_pred, X_test=None):
    if hasattr(y_true, "columns") and config.TARGET_COLUMN in y_true.columns:
        idx = list(y_true.columns).index(config.TARGET_COLUMN)
        yt = y_true[config.TARGET_COLUMN].to_numpy()
        if isinstance(y_pred, np.ndarray) and y_pred.ndim > 1:
            yp = y_pred[:, idx]
        elif hasattr(y_pred, "columns"):
            yp = y_pred[config.TARGET_COLUMN].to_numpy()
        else:
            yp = np.asarray(y_pred)
    else:
        yt = np.asarray(y_true).ravel()
        yp = np.asarray(y_pred).ravel()

    if X_test is not None:
        mask = np.ones(len(yt), dtype=bool)
        if "unit_name" in X_test.columns:
            mask &= (X_test["unit_name"].to_numpy() != "Total")
        if "unit_type" in X_test.columns:
            mask &= (X_test["unit_type"].to_numpy() != "National Total")
        yt = yt[mask]
        yp = yp[mask]

    return yt, yp


def plot_predictions(y_true, y_pred, model_name: str, X_test=None) -> None:
    yt, yp = _extract_1d(y_true, y_pred, X_test=X_test)
    fig, ax = plt.subplots(figsize=(6, 6), facecolor="white")
    ax.set_facecolor("white")
    ax.scatter(yt, yp, alpha=0.5, s=20, color="#1f77b4", label="Predictions")
    lims = [min(yt.min(), yp.min()), max(yt.max(), yp.max())]
    ax.plot(lims, lims, color="#d62728", linestyle="--", linewidth=1.5, label="Perfect Prediction")
    ax.set_xlabel("Actual Total Cases", fontsize=10, fontweight="bold")
    ax.set_ylabel("Predicted Total Cases", fontsize=10, fontweight="bold")
    ax.set_title(f"{model_name}: Actual vs Predicted", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6, color="#cccccc")
    ax.legend(facecolor="white", edgecolor="#cccccc")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / f"{model_name}_actual_vs_predicted.png", dpi=150, facecolor="white")
    plt.close(fig)


def plot_residuals(y_true, y_pred, model_name: str, X_test=None) -> None:
    yt, yp = _extract_1d(y_true, y_pred, X_test=X_test)
    residuals = yt - yp
    fig, ax = plt.subplots(figsize=(6, 4), facecolor="white")
    ax.set_facecolor("white")
    ax.scatter(yp, residuals, alpha=0.5, s=20, color="#1f77b4", label="Residuals")
    ax.axhline(0, color="#d62728", linestyle="--", linewidth=1.5, label="Zero Error")
    ax.set_xlabel("Predicted Total Cases", fontsize=10, fontweight="bold")
    ax.set_ylabel("Residual", fontsize=10, fontweight="bold")
    ax.set_title(f"{model_name}: Residual Plot", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6, color="#cccccc")
    ax.legend(facecolor="white", edgecolor="#cccccc")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / f"{model_name}_residuals.png", dpi=150, facecolor="white")
    plt.close(fig)


def evaluate(y_true, y_pred, model_name: str, X_test=None) -> dict:
    """Compute metrics, persist them, and save diagnostic plots."""
    metrics = compute_metrics(y_true, y_pred)
    save_metrics(metrics, model_name)
    plot_predictions(y_true, y_pred, model_name, X_test=X_test)
    plot_residuals(y_true, y_pred, model_name, X_test=X_test)
    return metrics

