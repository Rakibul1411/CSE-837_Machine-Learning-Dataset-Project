"""CLI for the classical time-series models (ARIMA family + VAR).

Evaluates the model on a held-out tail of the national monthly series, then
forecasts a number of months beyond all known data.

Usage:
    python forecast.py --model sarima
    python forecast.py --model arima --test-horizon 12 --forecast-horizon 6
"""
import argparse

from src.timeseries import TIMESERIES_REGISTRY
from src.timeseries_pipeline import forecast_future, train_and_evaluate


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="sarima",
        choices=sorted(TIMESERIES_REGISTRY),
        help="Which registered time-series model to run.",
    )
    parser.add_argument(
        "--test-horizon",
        type=int,
        default=12,
        help="Months of national history held out for evaluation.",
    )
    parser.add_argument(
        "--history-months",
        type=int,
        default=24,
        help="Months of historical trend to display on graph (0 for full history).",
    )
    parser.add_argument(
        "--forecast-horizon",
        type=int,
        default=6,
        help="Months to forecast beyond all known data.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    result = train_and_evaluate(args.model, horizon=args.test_horizon, history_months=args.history_months)
    print(f"Model: {result['model_name']}")
    print(f"Train months: {result['train_size']}  Test months: {result['test_size']}")
    for key, value in result["metrics"].items():
        print(f"  {key.upper():5s}: {value:,.4f}")

    future = forecast_future(args.model, horizon=args.forecast_horizon)
    print(f"\nForecast for the next {args.forecast_horizon} months:")
    for point in future["forecast"]:
        print(f"  {point['date']}: {point['prediction']:,.0f}")


if __name__ == "__main__":
    main()
