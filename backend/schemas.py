"""Pydantic request/response models for the API."""
from pydantic import BaseModel, Field

from src import config


class PredictRequest(BaseModel):
    model_name: str = Field(..., description="Registered model key, e.g. 'linear_regression'")
    year: int = Field(..., ge=2015, le=2035)
    month_number: int = Field(..., ge=1, le=12)
    unit_name: str
    unit_type: str


class PredictResponse(BaseModel):
    model_name: str
    prediction: float
    input: PredictRequest


class Metrics(BaseModel):
    mae: float
    rmse: float
    r2: float


class ModelInfo(BaseModel):
    name: str
    trained: bool


class Options(BaseModel):
    unit_names: list[str]
    unit_types: list[str]
    unit_name_to_type: dict[str, str]
    year_min: int
    year_max: int
    target_column: str = config.TARGET_COLUMN


class TrainRequest(BaseModel):
    model_name: str = Field(..., description="Registered model key, e.g. 'random_forest'")
    test_size: float | None = Field(None, gt=0, lt=1)


class TrainResponse(BaseModel):
    model_name: str
    train_size: int
    test_size: int
    metrics: Metrics


class TimeSeriesTrainRequest(BaseModel):
    model_name: str = Field(..., description="Registered time-series key, e.g. 'sarima'")
    test_horizon: int = Field(12, ge=3, le=36, description="Months of history held out for evaluation")


class TimeSeriesTrainResponse(BaseModel):
    model_name: str
    train_size: int
    test_size: int
    metrics: Metrics


class ForecastRequest(BaseModel):
    model_name: str = Field(..., description="Registered time-series key, e.g. 'sarima'")
    horizon: int = Field(6, ge=1, le=24, description="Months to forecast beyond all known data")


class ForecastPoint(BaseModel):
    date: str
    prediction: float


class ForecastResponse(BaseModel):
    model_name: str
    forecast: list[ForecastPoint]
