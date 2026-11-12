"""FastAPI service that exposes the trained crime-prediction models to the
Angular frontend: metrics, diagnostic figures, dropdown options, and a
live /predict endpoint.

Run with:
    uvicorn backend.main:app --reload --port 8001
"""
import json

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src import config, utils
from src.data_loader import load_dataset
from src.models import MODEL_REGISTRY
from src.timeseries import TIMESERIES_REGISTRY
from src.train_pipeline import train_and_evaluate
from src.timeseries_pipeline import (
    forecast_custom_range,
    forecast_future,
    train_and_evaluate as train_and_evaluate_timeseries,
)
from backend.schemas import (
    CustomRangeForecastRequest,
    CustomRangeForecastResponse,
    ForecastRequest,
    ForecastResponse,
    ModelInfo,
    Options,
    PredictRequest,
    PredictResponse,
    TimeSeriesTrainRequest,
    TimeSeriesTrainResponse,
    TrainRequest,
    TrainResponse,
)

app = FastAPI(title="Bangladesh Crime Statistics API")

app.add_middleware(
    CORSMiddleware,
    # The Angular dev server's port varies (4200, 4201, or an IDE-assigned
    # port like 54420 when defaults are taken), so allow any localhost port.
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/figures", StaticFiles(directory=config.FIGURES_DIR), name="figures")

_dataset_cache: pd.DataFrame | None = None
_model_cache: dict[str, object] = {}


def get_dataset() -> pd.DataFrame:
    global _dataset_cache
    if _dataset_cache is None:
        _dataset_cache = load_dataset()
    return _dataset_cache


def get_loaded_model(model_name: str):
    if model_name not in MODEL_REGISTRY:
        raise HTTPException(404, f"Unknown model '{model_name}'")
    if model_name not in _model_cache:
        try:
            _model_cache[model_name] = utils.load_model(model_name)
        except FileNotFoundError as exc:
            raise HTTPException(
                404, f"Model '{model_name}' has not been trained yet. Run train.py first."
            ) from exc
    return _model_cache[model_name]


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/models", response_model=list[ModelInfo])
def list_models():
    trained = {p.stem for p in config.MODELS_DIR.glob("*.joblib")}
    return [ModelInfo(name=name, trained=name in trained) for name in sorted(MODEL_REGISTRY)]


@app.get("/api/metrics/{model_name}")
def get_metrics(model_name: str):
    path = config.METRICS_DIR / f"{model_name}_metrics.json"
    if not path.exists():
        raise HTTPException(404, f"No metrics found for '{model_name}'. Train it first.")
    return json.loads(path.read_text())


@app.get("/api/options", response_model=Options)
def get_options():
    df = get_dataset()
    unit_name_to_type = (
        df.drop_duplicates("unit_name").set_index("unit_name")["unit_type"].to_dict()
    )
    return Options(
        unit_names=sorted(df["unit_name"].unique().tolist()),
        unit_types=sorted(df["unit_type"].unique().tolist()),
        unit_name_to_type=unit_name_to_type,
        year_min=int(df["year"].min()),
        year_max=int(df["year"].max()),
    )


@app.post("/api/train", response_model=TrainResponse)
def train_model(request: TrainRequest):
    if request.model_name not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise HTTPException(404, f"Unknown model '{request.model_name}'. Available: {available}")

    result = train_and_evaluate(request.model_name, test_size=request.test_size)
    _model_cache.pop(request.model_name, None)  # force reload of freshly trained weights
    return TrainResponse(**result)


@app.get("/api/timeseries/models", response_model=list[ModelInfo])
def list_timeseries_models():
    # These models refit on demand rather than persisting a .joblib, so
    # "trained" here means "has an evaluated result on disk to show".
    evaluated = {p.stem.replace("_metrics", "") for p in config.METRICS_DIR.glob("*_metrics.json")}
    return [
        ModelInfo(name=name, trained=name in evaluated) for name in sorted(TIMESERIES_REGISTRY)
    ]


@app.post("/api/timeseries/train", response_model=TimeSeriesTrainResponse)
def train_timeseries_model(request: TimeSeriesTrainRequest):
    if request.model_name not in TIMESERIES_REGISTRY:
        available = ", ".join(sorted(TIMESERIES_REGISTRY))
        raise HTTPException(404, f"Unknown time-series model '{request.model_name}'. Available: {available}")

    try:
        result = train_and_evaluate_timeseries(
            request.model_name,
            horizon=request.test_horizon,
            history_months=request.history_months,
            unit_name=request.unit_name,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return TimeSeriesTrainResponse(**result)


@app.post("/api/timeseries/forecast", response_model=ForecastResponse)
def forecast_timeseries(request: ForecastRequest):
    if request.model_name not in TIMESERIES_REGISTRY:
        available = ", ".join(sorted(TIMESERIES_REGISTRY))
        raise HTTPException(404, f"Unknown time-series model '{request.model_name}'. Available: {available}")

    result = forecast_future(request.model_name, horizon=request.horizon, unit_name=request.unit_name)
    return ForecastResponse(**result)


@app.post("/api/timeseries/forecast-range", response_model=CustomRangeForecastResponse)
def forecast_timeseries_custom_range(request: CustomRangeForecastRequest):
    if request.model_name not in TIMESERIES_REGISTRY:
        available = ", ".join(sorted(TIMESERIES_REGISTRY))
        raise HTTPException(404, f"Unknown time-series model '{request.model_name}'. Available: {available}")

    start_date = f"{request.start_year:04d}-{request.start_month:02d}"
    end_date = f"{request.end_year:04d}-{request.end_month:02d}"
    result = forecast_custom_range(
        request.model_name,
        start_date=start_date,
        end_date=end_date,
        unit_name=request.unit_name,
        test_horizon=request.test_horizon,
        include_evaluation=request.include_evaluation,
    )
    return CustomRangeForecastResponse(**result)


@app.post("/api/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    pipeline = get_loaded_model(request.model_name)
    row = pd.DataFrame(
        [
            {
                "year": request.year,
                "month_number": request.month_number,
                "unit_name": request.unit_name,
                "unit_type": request.unit_type,
            }
        ]
    )
    raw_preds = pipeline.predict(row)
    preds_row = raw_preds[0] if raw_preds.ndim > 1 else raw_preds

    predictions: dict[str, float] = {}
    if len(preds_row) == len(config.ALL_TARGET_COLUMNS):
        for col, val in zip(config.ALL_TARGET_COLUMNS, preds_row):
            predictions[col] = max(0.0, float(val))
    elif len(preds_row) == 1:
        predictions[config.TARGET_COLUMN] = max(0.0, float(preds_row[0]))

    if request.target_crime and request.target_crime in predictions:
        main_pred = predictions[request.target_crime]
    else:
        main_pred = predictions.get(config.TARGET_COLUMN, max(0.0, float(preds_row[0])))

    return PredictResponse(
        model_name=request.model_name,
        prediction=main_pred,
        predictions=predictions,
        input=request,
    )
