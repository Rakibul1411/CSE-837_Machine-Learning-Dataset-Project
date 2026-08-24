"""Entry point for training and evaluating a model on the crime dataset.

Usage:
    python train.py --model linear_regression
    python train.py --model random_forest --test-size 0.3
"""
import argparse

from src import config
from src.models import MODEL_REGISTRY
from src.train_pipeline import train_and_evaluate


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="linear_regression",
        choices=sorted(MODEL_REGISTRY),
        help="Which registered model to train.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=None,
        help="Fraction of data held out for testing (default: config.TEST_SIZE).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    result = train_and_evaluate(args.model, test_size=args.test_size)

    print(f"Model: {result['model_name']}")
    print(f"Train size: {result['train_size']}  Test size: {result['test_size']}")
    for key, value in result["metrics"].items():
        print(f"  {key.upper():5s}: {value:,.4f}")
    print(f"Saved model -> {config.MODELS_DIR / f'{args.model}.joblib'}")
    print(f"Saved plots -> {config.FIGURES_DIR}")
    print(f"Saved metrics -> {config.METRICS_DIR / f'{args.model}_metrics.json'}")


if __name__ == "__main__":
    main()
