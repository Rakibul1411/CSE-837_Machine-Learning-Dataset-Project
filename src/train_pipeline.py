"""Shared training routine used by both train.py (CLI) and backend/main.py (API)."""
from src import config, utils
from src.data_loader import load_dataset
from src.evaluate import evaluate
from src.models import get_model
from src.preprocessing import make_pipeline, split_data


def train_and_evaluate(model_name: str, test_size: float | None = None) -> dict:
    """Load data, fit `model_name`, evaluate it, and persist model/metrics/plots.

    Returns a summary dict with split sizes and the computed metrics.
    """
    utils.set_seed()

    df = load_dataset()
    X_train, X_test, y_train, y_test = split_data(df, test_size=test_size)

    estimator = get_model(model_name)
    pipeline = make_pipeline(estimator)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    metrics = evaluate(y_test, y_pred, model_name)

    utils.save_model(pipeline, model_name)

    return {
        "model_name": model_name,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "metrics": metrics,
    }
