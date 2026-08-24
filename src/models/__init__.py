"""Model registry.

Each model lives in its own module and registers a zero-arg factory that
returns a fresh, unfitted sklearn estimator. To add a new algorithm:

  1. Create src/models/<name>.py with a `build_model()` function.
  2. Import it below and add it to MODEL_REGISTRY.
  3. Train it via `python train.py --model <name>` or the "Train" page in the UI.
"""
from src.models.linear_regression import build_model as build_linear_regression
from src.models.random_forest import build_model as build_random_forest

MODEL_REGISTRY = {
    "linear_regression": build_linear_regression,
    "random_forest": build_random_forest,
}


def get_model(name: str):
    try:
        factory = MODEL_REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unknown model '{name}'. Available: {available}") from exc
    return factory()
