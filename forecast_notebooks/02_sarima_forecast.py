# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · SARIMA Forecast — Seasonal ARIMA
# MAGIC
# MAGIC **Method level**: Statistical (Level 2 of 5)
# MAGIC
# MAGIC SARIMA (Seasonal AutoRegressive Integrated Moving Average) is the classical
# MAGIC workhorse for univariate time-series forecasting with explicit trend and
# MAGIC seasonal components.
# MAGIC
# MAGIC **Model notation**: SARIMA(p,d,q)(P,D,Q)[12]
# MAGIC
# MAGIC * `p,d,q` — non-seasonal AR order, differencing, MA order
# MAGIC * `P,D,Q` — seasonal AR order, differencing, MA order
# MAGIC * `[12]`  — seasonal period = 12 months
# MAGIC
# MAGIC **Strategy**: Use `pmdarima.auto_arima` for automatic order selection per
# MAGIC series, then distribute across all 25 (region × care_type) series using
# MAGIC Spark's `applyInPandas`.

# COMMAND ----------

# MAGIC %md ## 1. Configuration

# COMMAND ----------

dbutils.widgets.text("catalog",    "hive_metastore",   "Catalog")
dbutils.widgets.text("schema",     "demand_forecast",  "Schema")
dbutils.widgets.text("src_table",  "ooh_care_monthly", "Source table")
dbutils.widgets.text("experiment", "/Shared/ooh_care_demand_forecast", "MLflow experiment")
dbutils.widgets.text("horizon",    "6",                "Forecast horizon (months)")

CATALOG    = dbutils.widgets.get("catalog")
SCHEMA     = dbutils.widgets.get("schema")
SRC_TABLE  = f"{CATALOG}.{SCHEMA}.{dbutils.widgets.get('src_table')}"
FCST_TABLE = f"{CATALOG}.{SCHEMA}.ooh_care_forecasts"
EXPERIMENT = dbutils.widgets.get("experiment")
HORIZON    = int(dbutils.widgets.get("horizon"))

MODEL_NAME   = "sarima"
TRAIN_END    = "2023-06-01"
TEST_START   = "2023-07-01"
TEST_END     = "2023-12-01"
FUTURE_START = "2024-01-01"

print(f"Source : {SRC_TABLE}")
print(f"Results: {FCST_TABLE}")
print(f"Model  : {MODEL_NAME}")

# COMMAND ----------

# MAGIC %pip install --quiet pmdarima statsmodels

# COMMAND ----------

import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timezone
import pmdarima as pm
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import mlflow
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField,
    DateType, StringType, DoubleType, BooleanType, TimestampType
)

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# COMMAND ----------

# MAGIC %md ## 2. Load Data

# COMMAND ----------

raw = spark.table(SRC_TABLE).toPandas()
raw["period"] = pd.to_datetime(raw["period"])
raw = raw.sort_values(["region", "care_type", "period"]).reset_index(drop=True)

print(f"Loaded {len(raw):,} rows | "
      f"{raw['region'].nunique()} regions × {raw['care_type'].nunique()} care types × "
      f"{raw['period'].nunique()} months")

# COMMAND ----------

# MAGIC %md ## 3. Define Per-Series SARIMA Forecasting Function
# MAGIC
# MAGIC Each Spark executor receives a group (one time series) and:
# MAGIC 1. Fits `auto_arima` on the training portion
# MAGIC 2. Produces 6-step-ahead hold-out predictions (Jul–Dec 2023)
# MAGIC 3. Retrains on the full series and produces 6-step-ahead future forecasts

# COMMAND ----------

RESULT_SCHEMA = StructType([
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

_TRAIN_END    = TRAIN_END
_TEST_START   = TEST_START
_TEST_END     = TEST_END
_FUTURE_START = FUTURE_START
_HORIZON      = HORIZON
_MODEL_NAME   = MODEL_NAME
_RUN_ID       = None           # filled after MLflow run starts
_RUN_TS       = None

def sarima_one_series(keys, pdf: pd.DataFrame) -> pd.DataFrame:
    """
    Fit SARIMA on one (region, care_type) series.
    Called via applyInPandas — runs on Spark executors.
    """
    import pmdarima as pm
    import warnings
    warnings.filterwarnings("ignore")

    region, care_type = keys
    pdf = pdf.sort_values("period")

    train = pdf[pdf["period"] <= _TRAIN_END]["demand_units"]
    full  = pdf["demand_units"]
    run_ts = datetime.now(timezone.utc)

    def fit_arima(series):
        return pm.auto_arima(
            series,
            start_p=1, start_q=1, max_p=3, max_q=3,
            d=None, D=1,
            m=12,
            seasonal=True,
            information_criterion="aic",
            stepwise=True,
            suppress_warnings=True,
            error_action="ignore",
        )

    rows = []

    # ── Test-set forecast ──────────────────────────────────────────────────
    model_test = fit_arima(train)
    pred_test  = model_test.predict(n_periods=_HORIZON, return_conf_int=True, alpha=0.2)
    fc_vals, ci = pred_test[0], pred_test[1]
    test_dates = pd.date_range(_TEST_START, periods=_HORIZON, freq="MS")
    for dt, fv, lo, hi in zip(test_dates, fc_vals, ci[:, 0], ci[:, 1]):
        rows.append({
            "period": dt.date(), "region": region, "care_type": care_type,
            "model_name": _MODEL_NAME,
            "forecast_value": float(fv),
            "lower_bound": float(lo), "upper_bound": float(hi),
            "is_test_set": True,
            "run_timestamp": run_ts,
            "mlflow_run_id": _RUN_ID or "",
        })

    # ── Future forecast (Jan–Jun 2024) ────────────────────────────────────
    model_full = fit_arima(full)
    pred_fut   = model_full.predict(n_periods=_HORIZON, return_conf_int=True, alpha=0.2)
    fc_fut, ci_fut = pred_fut[0], pred_fut[1]
    future_dates = pd.date_range(_FUTURE_START, periods=_HORIZON, freq="MS")
    for dt, fv, lo, hi in zip(future_dates, fc_fut, ci_fut[:, 0], ci_fut[:, 1]):
        rows.append({
            "period": dt.date(), "region": region, "care_type": care_type,
            "model_name": _MODEL_NAME,
            "forecast_value": float(fv),
            "lower_bound": float(lo), "upper_bound": float(hi),
            "is_test_set": False,
            "run_timestamp": run_ts,
            "mlflow_run_id": _RUN_ID or "",
        })

    return pd.DataFrame(rows)

# COMMAND ----------

# MAGIC %md ## 4. Distributed Execution with `applyInPandas`

# COMMAND ----------

mlflow.set_experiment(EXPERIMENT)

with mlflow.start_run(run_name=MODEL_NAME) as run:
    mlflow.set_tag("model_name", MODEL_NAME)
    mlflow.log_param("method",  "SARIMA (auto_arima)")
    mlflow.log_param("horizon", HORIZON)
    mlflow.log_param("seasonal_period", 12)
    mlflow.log_param("train_end", TRAIN_END)

    # Broadcast the run_id so executors can stamp it on each row
    _RUN_ID = run.info.run_id
    _RUN_TS = datetime.now(timezone.utc)

    sdf_input = (
        spark.createDataFrame(raw[["period", "region", "care_type", "demand_units"]])
             .withColumn("period", F.col("period").cast("date"))
    )

    # Run SARIMA per series in parallel across the cluster
    sdf_forecast = (
        sdf_input
        .groupBy("region", "care_type")
        .applyInPandas(sarima_one_series, schema=RESULT_SCHEMA)
    )

    # Materialise to Delta
    sdf_forecast.cache()
    sdf_forecast.count()   # trigger evaluation

    # Compute hold-out RMSE via Spark SQL (avoid driver collect for large results)
    test_actuals_sdf = (
        sdf_input
        .filter((F.col("period") >= TEST_START) & (F.col("period") <= TEST_END))
        .withColumnRenamed("demand_units", "actual")
    )
    test_fcst_sdf = sdf_forecast.filter(F.col("is_test_set") == True)

    metrics_sdf = (
        test_fcst_sdf
        .join(test_actuals_sdf, on=["period", "region", "care_type"])
        .withColumn("sq_err", F.pow(F.col("forecast_value") - F.col("actual"), 2))
        .agg(
            F.sqrt(F.avg("sq_err")).alias("rmse"),
            F.avg(F.abs((F.col("forecast_value") - F.col("actual")) / F.col("actual"))).alias("mape"),
        )
    )
    metrics = metrics_sdf.first()
    mean_rmse = float(metrics["rmse"])
    mean_mape = float(metrics["mape"]) * 100

    mlflow.log_metric("mean_rmse_test", mean_rmse)
    mlflow.log_metric("mean_mape_pct_test", mean_mape)
    print(f"Hold-out RMSE: {mean_rmse:.2f}  |  MAPE: {mean_mape:.2f}%")

# COMMAND ----------

# MAGIC %md ## 5. Save to Shared Forecast Table

# COMMAND ----------

spark.sql(f"DELETE FROM {FCST_TABLE} WHERE model_name = '{MODEL_NAME}'")

(
    sdf_forecast.write
                .format("delta")
                .mode("append")
                .saveAsTable(FCST_TABLE)
)

print(f"Saved {sdf_forecast.count():,} forecast rows → {FCST_TABLE}")

# COMMAND ----------

# MAGIC %md ## 6. Quick Visual Check

# COMMAND ----------

import matplotlib.pyplot as plt

sample_region    = "South"
sample_care_type = "Residential Care"

actuals = raw[(raw["region"] == sample_region) & (raw["care_type"] == sample_care_type)].copy()
fcst_pdf = sdf_forecast.filter(
    (F.col("region") == sample_region) & (F.col("care_type") == sample_care_type)
).toPandas()
fcst_pdf["period"] = pd.to_datetime(fcst_pdf["period"])

fig, ax = plt.subplots(figsize=(13, 4))
ax.plot(actuals["period"], actuals["demand_units"], label="Actual", color="steelblue", linewidth=2)
ax.axvline(pd.Timestamp(TEST_START), color="grey", linestyle="--", linewidth=1, label="Train/Test split")

test_fcst   = fcst_pdf[fcst_pdf["is_test_set"] == True].sort_values("period")
future_fcst = fcst_pdf[fcst_pdf["is_test_set"] == False].sort_values("period")

ax.plot(test_fcst["period"], test_fcst["forecast_value"], "o--", color="darkorange",
        label="SARIMA forecast (test)")
ax.fill_between(test_fcst["period"], test_fcst["lower_bound"], test_fcst["upper_bound"],
                alpha=0.25, color="darkorange")

ax.plot(future_fcst["period"], future_fcst["forecast_value"], "o-", color="firebrick",
        label="SARIMA forecast (future)")
ax.fill_between(future_fcst["period"], future_fcst["lower_bound"], future_fcst["upper_bound"],
                alpha=0.25, color="firebrick")

ax.set_title(f"SARIMA Forecast — {sample_region}: {sample_care_type}", fontsize=13)
ax.set_xlabel("Month"); ax.set_ylabel("Demand (care packages)")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("/tmp/02_sarima_forecast.png", dpi=120)
plt.show()
mlflow.log_artifact("/tmp/02_sarima_forecast.png")

print("✅ Notebook 02 complete.")
