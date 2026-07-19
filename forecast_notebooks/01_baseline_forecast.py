# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · Baseline Forecast — Seasonal Naïve, Moving Average & ETS
# MAGIC
# MAGIC **Method level**: Baseline (Level 1 of 5)
# MAGIC
# MAGIC This notebook implements three classical, rule-based forecasting approaches that
# MAGIC require no ML training and serve as the performance floor every other model must beat.
# MAGIC
# MAGIC | Method | Description |
# MAGIC |---|---|
# MAGIC | **Seasonal Naïve** | Forecast = same month one year ago |
# MAGIC | **12-month Moving Average** | Rolling mean of last 12 months |
# MAGIC | **Holt-Winters ETS** | Triple exponential smoothing (trend + multiplicative seasonality) |
# MAGIC
# MAGIC The **best** of the three (lowest RMSE on the Jul–Dec 2023 hold-out) is promoted
# MAGIC as `method_name = "baseline"` in the shared forecast table.

# COMMAND ----------

# MAGIC %md ## 1. Configuration

# COMMAND ----------

dbutils.widgets.text("catalog",     "hive_metastore",   "Catalog")
dbutils.widgets.text("schema",      "demand_forecast",  "Schema")
dbutils.widgets.text("src_table",   "ooh_care_monthly", "Source table")
dbutils.widgets.text("experiment",  "/Shared/ooh_care_demand_forecast", "MLflow experiment")
dbutils.widgets.text("horizon",     "6",  "Forecast horizon (months)")
dbutils.widgets.text("seed",        "42", "Global random seed for reproducibility")

CATALOG    = dbutils.widgets.get("catalog")
SCHEMA     = dbutils.widgets.get("schema")
SRC_TABLE  = f"{CATALOG}.{SCHEMA}.{dbutils.widgets.get('src_table')}"
FCST_TABLE = f"{CATALOG}.{SCHEMA}.ooh_care_forecasts"
EXPERIMENT = dbutils.widgets.get("experiment")
HORIZON    = int(dbutils.widgets.get("horizon"))
SEED       = int(dbutils.widgets.get("seed"))

MODEL_NAME     = "baseline"
TRAIN_END      = "2023-06-01"   # last training month
TEST_START     = "2023-07-01"   # first hold-out month
TEST_END       = "2023-12-01"   # last hold-out month
FUTURE_START   = "2024-01-01"
FUTURE_END     = "2024-06-01"

print(f"Source : {SRC_TABLE}")
print(f"Results: {FCST_TABLE}")
print(f"Model  : {MODEL_NAME}")

# COMMAND ----------

# MAGIC %pip install --quiet statsmodels

# COMMAND ----------

# MAGIC %md ## 2. Reproducibility Seeds
# MAGIC
# MAGIC All randomness is pinned to `SEED` so every run produces identical results.

# COMMAND ----------

import os, random
os.environ["PYTHONHASHSEED"] = str(SEED)  # Python hash randomisation
random.seed(SEED)                          # Python random module

import numpy as np
np.random.seed(SEED)                       # NumPy (used by statsmodels internals)

import pandas as pd
from datetime import datetime, timezone
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import mlflow
import mlflow.pyfunc
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField,
    DateType, StringType, DoubleType, BooleanType, TimestampType
)

# COMMAND ----------

# MAGIC %md ## 2. Load and Prepare Data

# COMMAND ----------

raw = spark.table(SRC_TABLE).toPandas()
raw["period"] = pd.to_datetime(raw["period"])
raw = raw.sort_values(["region", "care_type", "period"]).reset_index(drop=True)

# Split: train ≤ Jun 2023 | test = Jul–Dec 2023
train_df = raw[raw["period"] <= TRAIN_END].copy()
test_df  = raw[(raw["period"] >= TEST_START) & (raw["period"] <= TEST_END)].copy()

print(f"Training months : {train_df['period'].min().date()} → {train_df['period'].max().date()}")
print(f"Test months     : {test_df['period'].min().date()} → {test_df['period'].max().date()}")
print(f"Series          : {raw['region'].nunique()} regions × {raw['care_type'].nunique()} care types")

# COMMAND ----------

# MAGIC %md ## 3. Forecasting Functions

# COMMAND ----------

def seasonal_naive(series: pd.Series, horizon: int) -> np.ndarray:
    """Forecast = same month last year (12-step lag)."""
    values = series.values
    forecasts = np.array([values[-(12 - i % 12)] for i in range(horizon)])
    return forecasts


def moving_average(series: pd.Series, horizon: int, window: int = 12) -> np.ndarray:
    """Forecast = mean of last `window` observations (flat line)."""
    ma = series.iloc[-window:].mean()
    return np.full(horizon, ma)


def holt_winters(series: pd.Series, horizon: int) -> tuple:
    """
    Triple exponential smoothing (additive trend, multiplicative seasonality).
    Returns (point_forecast, lower_80, upper_80).
    """
    model = ExponentialSmoothing(
        series,
        trend="add",
        seasonal="mul",
        seasonal_periods=12,
        initialization_method="estimated",
    ).fit(optimized=True, use_brute=True)

    pred = model.forecast(horizon)

    # Residual std for approximate 80 % prediction interval
    residuals = series.values - model.fittedvalues.values
    sigma = np.std(residuals, ddof=1)
    z80 = 1.282
    lower = pred - z80 * sigma * np.sqrt(np.arange(1, horizon + 1))
    upper = pred + z80 * sigma * np.sqrt(np.arange(1, horizon + 1))
    return pred.values, lower.values, upper.values


# COMMAND ----------

# MAGIC %md ## CV · Time Series Walk-Forward Cross-Validation
# MAGIC
# MAGIC **Expanding-window** scheme: the training window grows with each fold while
# MAGIC the validation window is always the next 6 months — matching the live forecast horizon.
# MAGIC
# MAGIC | Fold | Train period | Validation period |
# MAGIC |---|---|---|
# MAGIC | 1 | Jan 2021 – Dec 2021 | Jan 2022 – Jun 2022 |
# MAGIC | 2 | Jan 2021 – Jun 2022 | Jul 2022 – Dec 2022 |
# MAGIC | 3 | Jan 2021 – Dec 2022 | Jan 2023 – Jun 2023 |
# MAGIC
# MAGIC For each fold and each series the best sub-method (Naïve / MA / Holt-Winters)
# MAGIC is selected independently, exactly as it will be on the final hold-out.

# COMMAND ----------

CV_FOLDS = [
    {"fold": 1, "train_end": "2021-12-01", "val_start": "2022-01-01", "val_end": "2022-06-01"},
    {"fold": 2, "train_end": "2022-06-01", "val_start": "2022-07-01", "val_end": "2022-12-01"},
    {"fold": 3, "train_end": "2022-12-01", "val_start": "2023-01-01", "val_end": "2023-06-01"},
]

cv_records = []

for fold_cfg in CV_FOLDS:
    fold_rmse_list, fold_mae_list, fold_mape_list = [], [], []

    for (region, care_type), grp in raw.groupby(["region", "care_type"]):
        grp = grp.sort_values("period")
        series_tr  = grp[grp["period"] <= fold_cfg["train_end"]]["demand_units"]
        series_val = grp[
            (grp["period"] >= fold_cfg["val_start"]) &
            (grp["period"] <= fold_cfg["val_end"])
        ]["demand_units"]

        if len(series_tr) < 12 or len(series_val) == 0:
            continue

        h = len(series_val)
        actual = series_val.values

        # Select best sub-method using this fold's training data only
        best_sub, _, sn_p, ma_p, hw_p, _, _ = forecast_series(series_tr, series_val)
        best_pred = {"seasonal_naive": sn_p,
                     "moving_average": ma_p,
                     "holt_winters":   hw_p}[best_sub]

        fold_rmse_list.append(np.sqrt(np.mean((best_pred - actual) ** 2)))
        fold_mae_list.append(np.mean(np.abs(best_pred - actual)))
        fold_mape_list.append(np.mean(np.abs((best_pred - actual) / actual)) * 100)

    fold_result = {
        "fold":       fold_cfg["fold"],
        "val_period": f"{fold_cfg['val_start'][:7]} → {fold_cfg['val_end'][:7]}",
        "RMSE":       np.mean(fold_rmse_list),
        "MAE":        np.mean(fold_mae_list),
        "MAPE_%":     np.mean(fold_mape_list),
        "n_series":   len(fold_rmse_list),
    }
    cv_records.append(fold_result)
    print(f"Fold {fold_cfg['fold']} | Val: {fold_result['val_period']} | "
          f"RMSE={fold_result['RMSE']:.2f}  MAE={fold_result['MAE']:.2f}  "
          f"MAPE={fold_result['MAPE_%']:.2f}%")

cv_df = pd.DataFrame(cv_records)
cv_mean_rmse = cv_df["RMSE"].mean()
cv_std_rmse  = cv_df["RMSE"].std()
cv_mean_mape = cv_df["MAPE_%"].mean()
print(f"\n── CV Summary ──────────────────────────────────────────")
print(f"   Mean RMSE : {cv_mean_rmse:.2f} ± {cv_std_rmse:.2f}")
print(f"   Mean MAPE : {cv_mean_mape:.2f}%")

# COMMAND ----------

# MAGIC %md ## 4. Train, Evaluate and Select Best Sub-method

# COMMAND ----------

def rmse(actual, predicted):
    return np.sqrt(np.mean((actual - predicted) ** 2))


def forecast_series(series_train: pd.Series, test_actuals: pd.Series):
    """Run all three baselines on one series; return best by hold-out RMSE."""
    horizon_test = len(test_actuals)

    sn_preds   = seasonal_naive(series_train, horizon_test)
    ma_preds   = moving_average(series_train, horizon_test)
    hw_preds, hw_lo, hw_hi = holt_winters(series_train, horizon_test)

    scores = {
        "seasonal_naive": rmse(test_actuals.values, sn_preds),
        "moving_average": rmse(test_actuals.values, ma_preds),
        "holt_winters":   rmse(test_actuals.values, hw_preds),
    }
    best = min(scores, key=scores.get)
    return best, scores, sn_preds, ma_preds, hw_preds, hw_lo, hw_hi


mlflow.set_experiment(EXPERIMENT)
run_timestamp = datetime.now(timezone.utc)
all_forecasts = []

with mlflow.start_run(run_name=MODEL_NAME) as run:
    mlflow.set_tag("model_name", MODEL_NAME)
    mlflow.log_param("horizon",    HORIZON)
    mlflow.log_param("train_end",  TRAIN_END)
    mlflow.log_param("test_start", TEST_START)
    mlflow.log_param("seed",       SEED)
    mlflow.log_metric("cv_mean_rmse", cv_mean_rmse)
    mlflow.log_metric("cv_std_rmse",  cv_std_rmse)
    mlflow.log_metric("cv_mean_mape", cv_mean_mape)
    mlflow.log_param("cv_n_folds",    len(CV_FOLDS))

    series_rmse = {}

    for (region, care_type), grp in raw.groupby(["region", "care_type"]):
        grp = grp.sort_values("period")
        series_train  = grp[grp["period"] <= TRAIN_END]["demand_units"]
        test_actuals  = grp[(grp["period"] >= TEST_START) & (grp["period"] <= TEST_END)]["demand_units"]

        best_sub, scores, sn_p, ma_p, hw_p, hw_lo, hw_hi = forecast_series(series_train, test_actuals)
        series_rmse[f"{region}|{care_type}"] = scores[best_sub]

        # Build test-set forecast rows using the best sub-method
        test_dates   = grp[(grp["period"] >= TEST_START) & (grp["period"] <= TEST_END)]["period"].values
        if best_sub == "seasonal_naive":
            test_preds = sn_p
            test_lo    = sn_p * 0.90
            test_hi    = sn_p * 1.10
        elif best_sub == "moving_average":
            test_preds = ma_p
            test_lo    = ma_p * 0.90
            test_hi    = ma_p * 1.10
        else:
            test_preds = hw_p
            test_lo    = hw_lo
            test_hi    = hw_hi

        for dt, fv, lo, hi in zip(test_dates, test_preds, test_lo, test_hi):
            all_forecasts.append({
                "period": pd.Timestamp(dt).date(),
                "region": region, "care_type": care_type,
                "model_name": MODEL_NAME,
                "forecast_value": float(fv), "lower_bound": float(lo), "upper_bound": float(hi),
                "is_test_set": True,
                "run_timestamp": run_timestamp,
                "mlflow_run_id": run.info.run_id,
            })

        # 6-month future forecast (Jan–Jun 2024) using full series
        full_series = grp["demand_units"]
        if best_sub == "seasonal_naive":
            fut_preds = seasonal_naive(full_series, HORIZON)
            fut_lo    = fut_preds * 0.90
            fut_hi    = fut_preds * 1.10
        elif best_sub == "moving_average":
            fut_preds = moving_average(full_series, HORIZON)
            fut_lo    = fut_preds * 0.90
            fut_hi    = fut_preds * 1.10
        else:
            fut_preds, fut_lo, fut_hi = holt_winters(full_series, HORIZON)

        future_dates = pd.date_range(FUTURE_START, periods=HORIZON, freq="MS")
        for dt, fv, lo, hi in zip(future_dates, fut_preds, fut_lo, fut_hi):
            all_forecasts.append({
                "period": dt.date(),
                "region": region, "care_type": care_type,
                "model_name": MODEL_NAME,
                "forecast_value": float(fv), "lower_bound": float(lo), "upper_bound": float(hi),
                "is_test_set": False,
                "run_timestamp": run_timestamp,
                "mlflow_run_id": run.info.run_id,
            })

    overall_rmse = float(np.mean(list(series_rmse.values())))
    mlflow.log_metric("mean_rmse_test", overall_rmse)
    print(f"Mean RMSE across all series (hold-out): {overall_rmse:.2f}")

# COMMAND ----------

# MAGIC %md ## 5. Save Forecasts to Delta Table

# COMMAND ----------

forecast_pdf = pd.DataFrame(all_forecasts)
forecast_pdf["run_timestamp"] = pd.to_datetime(forecast_pdf["run_timestamp"])

schema = StructType([
    StructField("period",         DateType(),      False),
    StructField("region",         StringType(),    False),
    StructField("care_type",      StringType(),    False),
    StructField("model_name",     StringType(),    False),
    StructField("forecast_value", DoubleType(),    False),
    StructField("lower_bound",    DoubleType(),    True),
    StructField("upper_bound",    DoubleType(),    True),
    StructField("is_test_set",    BooleanType(),   False),
    StructField("run_timestamp",  TimestampType(), False),
    StructField("mlflow_run_id",  StringType(),    True),
])

sdf = spark.createDataFrame(forecast_pdf, schema=schema)

# Delete any existing rows for this model before appending
spark.sql(f"DELETE FROM {FCST_TABLE} WHERE model_name = '{MODEL_NAME}'")

(
    sdf.write
       .format("delta")
       .mode("append")
       .saveAsTable(FCST_TABLE)
)

print(f"Saved {sdf.count():,} forecast rows → {FCST_TABLE}")

# COMMAND ----------

# MAGIC %md ## 6. Quick Visual Check

# COMMAND ----------

import matplotlib.pyplot as plt

sample_region    = "South"
sample_care_type = "Residential Care"

sample = raw[(raw["region"] == sample_region) & (raw["care_type"] == sample_care_type)].copy()
fcst   = forecast_pdf[
    (forecast_pdf["region"] == sample_region) &
    (forecast_pdf["care_type"] == sample_care_type)
].copy()

fig, ax = plt.subplots(figsize=(13, 4))
ax.plot(sample["period"], sample["demand_units"], label="Actual", color="steelblue", linewidth=2)
ax.axvline(pd.Timestamp(TEST_START), color="grey", linestyle="--", linewidth=1, label="Train/Test split")

test_fcst   = fcst[fcst["is_test_set"] == True].sort_values("period")
future_fcst = fcst[fcst["is_test_set"] == False].sort_values("period")

ax.plot(test_fcst["period"], test_fcst["forecast_value"], "o--", color="darkorange",
        label="Baseline forecast (test)")
ax.fill_between(test_fcst["period"], test_fcst["lower_bound"], test_fcst["upper_bound"],
                alpha=0.2, color="darkorange")

ax.plot(future_fcst["period"], future_fcst["forecast_value"], "o-", color="firebrick",
        label="Baseline forecast (future)")
ax.fill_between(future_fcst["period"], future_fcst["lower_bound"], future_fcst["upper_bound"],
                alpha=0.2, color="firebrick")

ax.set_title(f"Baseline Forecast — {sample_region}: {sample_care_type}", fontsize=13)
ax.set_xlabel("Month"); ax.set_ylabel("Demand (care packages)")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("/tmp/01_baseline_forecast.png", dpi=120)
plt.show()

mlflow.log_artifact("/tmp/01_baseline_forecast.png")

print("✅ Notebook 01 complete.")
