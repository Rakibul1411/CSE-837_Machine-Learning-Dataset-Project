# Bangladesh Crime Statistics — ML Project

Predicting monthly crime case counts across Bangladesh's police units
(Metropolitan Police, Police Ranges, Railway Police) using supervised
machine learning, plus classical time-series forecasting of the national
crime trend.

The project has two independent model families, because they solve two
different problems and shouldn't be forced into one abstraction:

- **Tabular regression** (`src/models/`) — predicts `Total Cases` for an
  arbitrary (year, month, unit) combination from those features.
- **Time-series forecasting** (`src/timeseries/`) — forecasts the national
  monthly trend forward in time from its own history (ARIMA, SARIMA,
  SARIMAX, VAR).

## Dataset

`data/raw/bangladesh_crime_statistics_wide.xlsx` — monthly crime records
(2019–2026) per police unit, with counts for 15 crime categories
(Dacoity, Robbery, Murder, Theft, Narcotics, etc.) plus a `Total Cases`
column, which is the exact row-wise sum of those categories.

Because `Total Cases` is a deterministic sum of the other crime columns,
those columns are **not** used as model features (that would be leakage).
Instead, the task is framed as:

> Predict `Total Cases` for a given unit and month, from `year`,
> `month_number`, `unit_name`, and `unit_type` alone.

Annual rollup rows (`month == "Annual (Jan-Dec)"`) are dropped during
cleaning to avoid duplicating the monthly records.

The time-series models instead forecast the **national aggregate**
(`unit_name == "Total"`) series — the only row per month that isn't a
breakdown of another row — using a chronological holdout rather than a
random train/test split, since predicting the past from the future would
be leakage for a time series.

## Project structure

```
ML-Project/
├── data/
│   ├── raw/               # original Excel file (untouched)
│   └── processed/         # (reserved for cached/derived datasets)
├── src/
│   ├── config.py           # paths, schema, constants
│   ├── data_loader.py       # load + clean the raw Excel sheet
│   ├── preprocessing.py     # feature pipeline (scaling + one-hot) & splitting
│   ├── evaluate.py          # metrics + diagnostic plots
│   ├── utils.py             # seeding, model save/load
│   ├── train_pipeline.py     # train_and_evaluate() — shared by CLI and API (tabular)
│   ├── timeseries_pipeline.py # train_and_evaluate() + forecast_future() (time-series)
│   ├── models/                # tabular sklearn regressors
│   │   ├── __init__.py         # MODEL_REGISTRY — one entry per algorithm
│   │   ├── linear_regression.py
│   │   └── random_forest.py
│   └── timeseries/            # classical time-series forecasters
│       ├── __init__.py         # TIMESERIES_REGISTRY
│       ├── data.py              # national monthly series builder
│       ├── arima_model.py
│       ├── sarima_model.py
│       ├── sarimax_model.py
│       └── var_model.py
├── outputs/
│   ├── models/             # trained tabular pipelines (.joblib)
│   ├── figures/            # actual-vs-predicted / residual / forecast plots
│   └── metrics/            # MAE / RMSE / R² per model (.json)
├── backend/                 # FastAPI service exposing models to the UI
│   ├── main.py               # /api routes: models, metrics, options, train, predict,
│   │                          #   timeseries/models, timeseries/train, timeseries/forecast
│   └── schemas.py             # request/response models
├── frontend/                # Angular app (Dashboard, Train, Predict, Forecast pages)
├── train.py                 # CLI entry point (tabular models)
├── forecast.py               # CLI entry point (time-series models)
└── requirements.txt
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

Two algorithms are registered right now: `linear_regression` and
`random_forest` (see [Adding a new algorithm](#adding-a-new-algorithm)
for how the list grows).

```bash
python train.py --model linear_regression
# or
python train.py --model random_forest
```

This loads the dataset, builds a preprocessing + model pipeline, splits
train/test (80/20 by default), fits, evaluates, and writes:

- `outputs/models/linear_regression.joblib`
- `outputs/figures/linear_regression_actual_vs_predicted.png`
- `outputs/figures/linear_regression_residuals.png`
- `outputs/metrics/linear_regression_metrics.json`

### Time-series models

Four are registered: `arima`, `sarima`, `sarimax`, `var`.

```bash
python forecast.py --model sarima
# or, tune the split/horizon:
python forecast.py --model sarima --test-horizon 12 --forecast-horizon 6
```

This holds out the last `--test-horizon` months of the national series,
fits on the rest, scores the holdout (MAE/RMSE/R²), then refits on the
full series and forecasts `--forecast-horizon` months beyond all known
data. Writes `outputs/metrics/sarima_metrics.json` and
`outputs/figures/sarima_forecast.png` (a history/actual/forecast line
chart). No `.joblib` is saved — these models refit in under a couple of
seconds on this dataset's ~90 monthly points, so there's no need to
persist them.

## Web UI

A FastAPI backend serves the registered models to an Angular frontend
with four pages:

- **Train** — pick any registered tabular algorithm from a dropdown,
  click **Train Model**, and the backend trains it on the spot (via
  `POST /api/train`) and returns MAE/RMSE/R² immediately — no need to
  drop back to the terminal.
- **Dashboard** — metrics + diagnostic plots for any already-trained
  tabular model.
- **Predict** — pick a unit/year/month, get a live prediction from a
  tabular model.
- **Forecast** — pick an ARIMA-family/VAR model, **Train & Evaluate** it
  on a chronological holdout (`POST /api/timeseries/train`), then
  **Forecast Future** to get the next N months (`POST /api/timeseries/forecast`).

Start both servers in separate terminals (no need to train anything
up front — do it from the Train page once the UI is open):

```bash
# Terminal 1 — backend (http://localhost:8001)
source venv/bin/activate
uvicorn backend.main:app --reload --port 8001

# Terminal 2 — frontend (http://localhost:4200)
cd frontend
npm install   # first time only
ng serve
```

Open http://localhost:4200. CORS on the backend accepts any
`localhost`/`127.0.0.1` origin/port, so it's fine if `ng serve` picks a
different port because 4200 is already busy (common when another
project is running). If 8001 is taken instead, run
`uvicorn backend.main:app --port <other-port>` and update `apiUrl` in
`frontend/src/environments/environment.ts` to match.

## Adding a new algorithm

The pipeline (`src/preprocessing.py`), training routine
(`src/train_pipeline.py`), and evaluation (`src/evaluate.py`) are all
model-agnostic, so adding e.g. Gradient Boosting is 2 steps:

1. Create `src/models/gradient_boosting.py`:
   ```python
   from sklearn.ensemble import GradientBoostingRegressor

   def build_model():
       return GradientBoostingRegressor(random_state=42)
   ```
2. Register it in `src/models/__init__.py`:
   ```python
   from src.models.gradient_boosting import build_model as build_gradient_boosting
   MODEL_REGISTRY["gradient_boosting"] = build_gradient_boosting
   ```

That's it — it now shows up automatically in the Train/Dashboard/Predict
dropdowns in the UI (restart the backend to pick up the registry change),
and can be trained via `python train.py --model gradient_boosting` or the
Train page's **Train Model** button. No changes to `train.py`,
`preprocessing.py`, `evaluate.py`, or any frontend file are needed.

## Adding a new time-series model

Same pattern, in `src/timeseries/` instead: each module exposes a
`fit_forecast(train, horizon) -> array-like of length horizon`.

1. Create `src/timeseries/holt_winters_model.py`:
   ```python
   from statsmodels.tsa.holtwinters import ExponentialSmoothing

   def fit_forecast(train, horizon: int):
       fitted = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=12).fit()
       return fitted.forecast(horizon)
   ```
2. Register it in `src/timeseries/__init__.py`:
   ```python
   from src.timeseries.holt_winters_model import fit_forecast as holt_winters_fit_forecast
   TIMESERIES_REGISTRY["holt_winters"] = holt_winters_fit_forecast
   ```

It now shows up in the Forecast page's dropdown, and can be run via
`python forecast.py --model holt_winters`. If it needs a DataFrame of
multiple columns instead of a single series (like `var_model.py` does),
add a branch for it in `_run_model()` in `src/timeseries_pipeline.py`.
