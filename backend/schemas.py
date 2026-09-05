"""Pydantic request/response models for the API."""
from pydantic import BaseModel, Field

from src import config


class PredictRequest(BaseModel):
    model_name: str = Field(..., description="Registered model key, e.g. 'linear_regression'")
    year: int = Field(..., ge=2015, le=2035)
    month_number: int = Field(..., ge=1, le=12)
    unit_name: str
    unit_type: str
    target_crime: str = Field("All Crimes", description="Selected crime target or 'All Crimes'")


class PredictResponse(BaseModel):
    model_name: str
    prediction: float
    predictions: dict[str, float] = Field(default_factory=dict)
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
    crime_categories: list[str] = Field(default_factory=lambda: config.CRIME_CATEGORY_COLUMNS)
    target_options: list[str] = Field(
        default_factory=lambda: ["All Crimes", config.TARGET_COLUMN] + config.CRIME_CATEGORY_COLUMNS
    )


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
    history_months: int = Field(24, ge=0, le=120, description="Months of history displayed on graph (0 for full history)")


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


class HistoryPoint(BaseModel):
    date: str
    value: float


class ForecastResponse(BaseModel):
    model_name: str
    forecast: list[ForecastPoint]


class CustomRangeForecastRequest(BaseModel):
    model_name: str = Field(..., description="Registered time-series key, e.g. 'sarima'")
    start_year: int = Field(2019, ge=2019, le=2030, description="Start year (minimum 2019, when dataset records begin)")
    start_month: int = Field(1, ge=1, le=12)
    end_year: int = Field(2028, ge=2019, le=2035)
    end_month: int = Field(12, ge=1, le=12)


class CustomRangeForecastResponse(BaseModel):
    model_name: str
    start_date: str
    end_date: str
    history: list[HistoryPoint]
    forecast: list[ForecastPoint]
