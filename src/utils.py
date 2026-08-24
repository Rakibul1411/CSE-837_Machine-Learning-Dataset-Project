"""Small shared helpers: model persistence and seeding."""
import random

import joblib
import numpy as np

from src import config


def set_seed(seed: int = config.RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)


def save_model(pipeline, model_name: str) -> None:
    joblib.dump(pipeline, config.MODELS_DIR / f"{model_name}.joblib")


def load_model(model_name: str):
    return joblib.load(config.MODELS_DIR / f"{model_name}.joblib")
