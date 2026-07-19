# Databricks notebook source
# MAGIC %md
# MAGIC # 06 · Prophet Forecast — Meta (Facebook) Prophet
# MAGIC
# MAGIC **Method level**: Probabilistic Decomposition (Level 6 of 6)
# MAGIC
# MAGIC **Prophet** (Taylor & Letham, 2018) is a procedure for forecasting time series
# MAGIC based on an additive decomposition model:
# MAGIC
# MAGIC ```
# MAGIC y(t) = trend(t) + seasonality(t) + holidays(t) + ε(t)
# MAGIC ```
# MAGIC
# MAGIC ### Why Prophet for OOH care demand
# MAGIC | Strength | Relevance |
# MAGIC |---|---|
# MAGIC | Automatic changepoint detection | Captures COVID impact and recovery |
# MAGIC | Multiplicative seasonality | Scales with the level of demand |
# MAGIC | External regressors | Incorporates 65+ population covariate |
# MAGIC | Built-in uncertainty intervals | Provides 80 % prediction bands out of the box |
# MAGIC | Interpretable components | Trend / seasonality decomposable for stakeholders |
# MAGIC
# MAGIC ### Strategy
# MAGIC One Prophet model is fitted **per series** (25 series total).
# MAGIC Spark's `applyInPandas` distributes the work across the cluster so all
# MAGIC 25 series are trained in parallel.

# COMMAND ----------

# MAGIC %md ## 1. Configuration

# COMMAND ----------

dbutils.widgets.text("catalog",              "hive_metastore",              "Catalog")
dbutils.widgets.text("schema",               "demand_forecast",             "Schema")
dbutils.widgets.text("src_table",            "ooh_care_monthly",            "Source table")
dbutils.widgets.text("experiment",           "/Shared/ooh_care_demand_forecast", "MLflow experiment")
dbutils.widgets.text("horizon",              "6",    "Forecast horizon (months)")
dbutils.widgets.text("changepoint_prior",    "0.1",  "Changepoint prior scale (flexibility of trend)")
dbutils.widgets.text("seasonality_prior",    "10.0", "Seasonality prior scale")
dbutils.widgets.text("uncertainty_samples",  "1000", "MC samples for prediction intervals")
dbutils.widgets.text("seed",                 "42",   "Global random seed for reproducibility")

CATALOG              = dbutils.widgets.get("catalog")
SCHEMA               = dbutils.widgets.get("schema")
SRC_TABLE            = f"{CATALOG}.{SCHEMA}.{dbutils.widgets.get('src_table')}"
FCST_TABLE           = f"{CATALOG}.{SCHEMA}.ooh_care_forecasts"
EXPERIMENT           = dbutils.widgets.get("experiment")
HORIZON              = int(dbutils.widgets.get("horizon"))
CHANGEPOINT_PRIOR    = float(dbutils.widgets.get("changepoint_prior"))
SEASONALITY_PRIOR    = float(dbutils.widgets.get("seasonality_prior"))
UNCERTAINTY_SAMPLES  = int(dbutils.widgets.get("uncertainty_samples"))
SEED                 = int(dbutils.widgets.get("seed"))

MODEL_NAME   = "prophet"
TRAIN_END    = "2023-06-01"
TEST_START   = "2023-07-01"
TEST_END     = "2023-12-01"
FUTURE_START = "2024-01-01"

print(f"Source  : {SRC_TABLE}")
print(f"Results : {FCST_TABLE}")
print(f"Model   : {MODEL_NAME}")
print(f"Changepoint prior: {CHANGEPOINT_PRIOR}  |  Seasonality prior: {SEASONALITY_PRIOR}")

# COMMAND ----------

# MAGIC %md ## 2. Reproducibility Seeds

# COMMAND ----------

import os, random
os.environ["PYTHONHASHSEED"] = str(SEED)  # Python hash randomisation
random.seed(SEED)

import numpy as np
np.random.seed(SEED)   # Prophet's uncertainty sampling uses numpy random

import pandas as pd

print(f"Global seed: {SEED}")

# COMMAND ----------

# MAGIC %pip install --quiet prophet

# COMMAND ----------

# MAGIC %md ## 3. Load Data

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField,
    DateType, StringType, DoubleType, BooleanType, TimestampType
)
import mlflow

raw = spark.table(SRC_TABLE).toPandas()
raw["period"] = pd.to_datetime(raw["period"])
raw = raw.sort_values(["region", "care_type", "period"]).reset_index(drop=True)

# Normalise population covariate (Prophet regressors should be on a similar scale to y)
pop_mean = raw["population_65plus"].mean()
pop_std  = raw["population_65plus"].std()
raw["pop_norm"] = (raw["population_65plus"] - pop_mean) / pop_std

print(f"Loaded {len(raw):,} rows  |  "
      f"{raw['region'].nunique()} regions × {raw['care_type'].nunique()} care types × "
      f"{raw['period'].nunique()} months")

# COMMAND ----------

# MAGIC %md ## 4. Prophet Per-Series Forecast Function
# MAGIC
# MAGIC Prophet expects a DataFrame with two required columns:
# MAGIC * `ds` — datetime (date of each observation)
# MAGIC * `y`  — the target value
# MAGIC
# MAGIC Additional regressors (`pop_norm`) are passed as extra columns.

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

# Broadcast scalars so executors can access them without serialising the full dataset
_TRAIN_END           = TRAIN_END
_TEST_START          = TEST_START
_FUTURE_START        = FUTURE_START
_HORIZON             = HORIZON
_MODEL_NAME          = MODEL_NAME
_CHANGEPOINT_PRIOR   = CHANGEPOINT_PRIOR
_SEASONALITY_PRIOR   = SEASONALITY_PRIOR
_UNCERTAINTY_SAMPLES = UNCERTAINTY_SAMPLES
_SEED                = SEED
_RUN_ID              = None   # populated after MLflow run starts


def prophet_one_series(keys, pdf: pd.DataFrame) -> pd.DataFrame:
    """
    Fit a Prophet model for one (region, care_type) series.
    Invoked by applyInPandas — executes on Spark executor JVMs.
    """
    from prophet import Prophet
    import numpy as np
    import pandas as pd
    from datetime import datetime, timezone
    import warnings, logging

    warnings.filterwarnings("ignore")
    logging.getLogger("prophet").setLevel(logging.ERROR)
    logging.getLogger("cmdstanpy").setLevel(logging.ERROR)

    # Pin numpy seed on this executor for the uncertainty sampling stage
    np.random.seed(_SEED)

    region, care_type = keys
    pdf = pdf.sort_values("period").reset_index(drop=True)

    # Prophet input frame
    pdf_prophet = pdf.rename(columns={"period": "ds", "demand_units": "y"})

    train = pdf_prophet[pdf_prophet["ds"] <= _TRAIN_END].copy()
    full  = pdf_prophet.copy()
    run_ts = datetime.now(timezone.utc)

    def build_and_fit(df_fit):
        """Instantiate and fit a Prophet model on df_fit."""
        m = Prophet(
            seasonality_mode="multiplicative",       # demand scales with level
            changepoint_prior_scale=_CHANGEPOINT_PRIOR,
            seasonality_prior_scale=_SEASONALITY_PRIOR,
            interval_width=0.80,                     # 80 % prediction interval
            uncertainty_samples=_UNCERTAINTY_SAMPLES,
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
        )
        # Population 65+ as an additional linear regressor
        m.add_regressor("pop_norm", standardize=False)
        m.fit(df_fit[["ds", "y", "pop_norm"]])
        return m

    rows = []

    # ── Test-set forecast (trained on training portion only) ──────────────
    model_test = build_and_fit(train)

    # Build future dataframe for test period (Jul–Dec 2023)
    future_test = pd.DataFrame({"ds": pd.date_range(_TEST_START, periods=_HORIZON, freq="MS")})
    # Forward-fill population for test months using last known value
    last_pop = train["pop_norm"].iloc[-1]
    future_test["pop_norm"] = last_pop
    # Merge actual pop_norm if available
    actual_test_pop = pdf_prophet[(pdf_prophet["ds"] >= _TEST_START)][["ds", "pop_norm"]].head(_HORIZON)
    if len(actual_test_pop) > 0:
        future_test = future_test.merge(actual_test_pop.rename(columns={"pop_norm": "pop_actual"}),
                                        on="ds", how="left")
        future_test["pop_norm"] = future_test["pop_actual"].fillna(future_test["pop_norm"])
        future_test = future_test.drop(columns=["pop_actual"])

    pred_test = model_test.predict(future_test)
    for _, r in pred_test.iterrows():
        rows.append({
            "period":         r["ds"].date(),
            "region":         region,
            "care_type":      care_type,
            "model_name":     _MODEL_NAME,
            "forecast_value": float(max(r["yhat"], 0)),   # demand cannot be negative
            "lower_bound":    float(max(r["yhat_lower"], 0)),
            "upper_bound":    float(max(r["yhat_upper"], 0)),
            "is_test_set":    True,
            "run_timestamp":  run_ts,
            "mlflow_run_id":  _RUN_ID or "",
        })

    # ── Future forecast (Jan–Jun 2024) trained on full 2021–2023 series ───
    model_full = build_and_fit(full)

    future_dates = pd.date_range(_FUTURE_START, periods=_HORIZON, freq="MS")
    # Extrapolate population with last-known monthly growth rate
    last_pop_full  = full["pop_norm"].iloc[-1]
    monthly_growth = (full["pop_norm"].iloc[-1] - full["pop_norm"].iloc[-13]) / 12
    future_full = pd.DataFrame({
        "ds":       future_dates,
        "pop_norm": [last_pop_full + monthly_growth * (i + 1) for i in range(_HORIZON)],
    })

    pred_fut = model_full.predict(future_full)
    for _, r in pred_fut.iterrows():
        rows.append({
            "period":         r["ds"].date(),
            "region":         region,
            "care_type":      care_type,
            "model_name":     _MODEL_NAME,
            "forecast_value": float(max(r["yhat"], 0)),
            "lower_bound":    float(max(r["yhat_lower"], 0)),
            "upper_bound":    float(max(r["yhat_upper"], 0)),
            "is_test_set":    False,
            "run_timestamp":  run_ts,
            "mlflow_run_id":  _RUN_ID or "",
        })

    # ── Log trend + seasonality decomposition (driver-side, not here) ─────
    return pd.DataFrame(rows)

# COMMAND ----------

# MAGIC %md ## CV · Prophet Built-in Cross-Validation
# MAGIC
# MAGIC Prophet ships a `cross_validation()` utility that performs **simulated
# MAGIC historical forecasts** automatically — no manual fold management needed.
# MAGIC
# MAGIC | Parameter | Value | Meaning |
# MAGIC |---|---|---|
# MAGIC | `initial` | 365 days (~12 m) | Minimum training window |
# MAGIC | `period` | 182 days (~6 m) | Cutoff advances by this amount per fold |
# MAGIC | `horizon` | 182 days (~6 m) | Forecast horizon evaluated per cutoff |
# MAGIC
# MAGIC This yields **3 cutpoints** within the training period:
# MAGIC * Cutoff ≈ Dec 2021 → forecast Jan–Jun 2022
# MAGIC * Cutoff ≈ Jun 2022 → forecast Jul–Dec 2022
# MAGIC * Cutoff ≈ Dec 2022 → forecast Jan–Jun 2023
# MAGIC
# MAGIC Distributed CV: `applyInPandas` runs Prophet CV on all 25 series in parallel.

# COMMAND ----------

PROPHET_CV_INITIAL = "365 days"    # ~12 months minimum training
PROPHET_CV_PERIOD  = "182 days"    # cutoff advances ~6 months each fold
PROPHET_CV_HORIZON = "182 days"    # evaluate 6 months ahead

CV_RESULT_SCHEMA_PROPHET = StructType([
    StructField("region",    StringType(), False),
    StructField("care_type", StringType(), False),
    StructField("fold",      StringType(), False),
    StructField("cutoff",    StringType(), True),
    StructField("rmse",      DoubleType(), True),
    StructField("mae",       DoubleType(), True),
    StructField("mape_pct",  DoubleType(), True),
])

_CV_INITIAL = PROPHET_CV_INITIAL
_CV_PERIOD  = PROPHET_CV_PERIOD
_CV_HORIZON = PROPHET_CV_HORIZON


def prophet_cv_one_series(keys, pdf: pd.DataFrame) -> pd.DataFrame:
    """
    Run Prophet's built-in cross_validation on one (region, care_type) series.
    Executed on Spark executors via applyInPandas.
    """
    from prophet import Prophet
    from prophet.diagnostics import cross_validation, performance_metrics
    import numpy as np, pandas as pd, warnings, logging
    warnings.filterwarnings("ignore")
    logging.getLogger("prophet").setLevel(logging.ERROR)
    logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
    np.random.seed(_SEED)

    region, care_type = keys
    pdf = pdf.sort_values("period").reset_index(drop=True)
    prophet_input = pdf.rename(columns={"period": "ds", "demand_units": "y"})

    # Only use training portion (up to TRAIN_END) for CV to avoid look-ahead
    prophet_train = prophet_input[prophet_input["ds"] <= _TRAIN_END].copy()
    if len(prophet_train) < 14:
        return pd.DataFrame(columns=["region","care_type","fold","cutoff","rmse","mae","mape_pct"])

    try:
        m = Prophet(
            seasonality_mode="multiplicative",
            changepoint_prior_scale=_CHANGEPOINT_PRIOR,
            seasonality_prior_scale=_SEASONALITY_PRIOR,
            interval_width=0.80,
            uncertainty_samples=200,    # fewer samples for CV speed
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
        )
        m.add_regressor("pop_norm", standardize=False)
        m.fit(prophet_train[["ds", "y", "pop_norm"]])

        df_cv   = cross_validation(
            m, initial=_CV_INITIAL, period=_CV_PERIOD, horizon=_CV_HORIZON,
            parallel=None,   # already inside a Spark executor — no inner parallelism
        )
        df_perf = performance_metrics(df_cv, rolling_window=1.0)

        rows = []
        for fold_i, (cutoff, grp) in enumerate(df_cv.groupby("cutoff"), start=1):
            actual = grp["y"].values
            pred   = grp["yhat"].clip(lower=0).values
            rows.append({
                "region":    region,
                "care_type": care_type,
                "fold":      str(fold_i),
                "cutoff":    str(cutoff.date()),
                "rmse":      float(np.sqrt(np.mean((pred - actual) ** 2))),
                "mae":       float(np.mean(np.abs(pred - actual))),
                "mape_pct":  float(np.mean(np.abs((pred - actual) / np.maximum(actual, 1))) * 100),
            })
        return pd.DataFrame(rows)

    except Exception as e:
        return pd.DataFrame([{
            "region": region, "care_type": care_type,
            "fold": "error", "cutoff": str(e)[:80],
            "rmse": None, "mae": None, "mape_pct": None,
        }])


# Run distributed CV
sdf_cv_input = (
    spark.createDataFrame(
        raw[["period", "region", "care_type", "demand_units", "pop_norm"]]
    )
    .withColumn("period", F.col("period").cast("date"))
)

sdf_prophet_cv = (
    sdf_cv_input
    .groupBy("region", "care_type")
    .applyInPandas(prophet_cv_one_series, schema=CV_RESULT_SCHEMA_PROPHET)
    .filter(F.col("fold") != "error")
)

cv_summary_prophet = (
    sdf_prophet_cv
    .groupBy("fold", "cutoff")
    .agg(
        F.avg("rmse").alias("mean_rmse"),
        F.avg("mae").alias("mean_mae"),
        F.avg("mape_pct").alias("mean_mape_pct"),
    )
    .orderBy("cutoff")
    .toPandas()
)

cv_mean_rmse = float(cv_summary_prophet["mean_rmse"].mean()) if len(cv_summary_prophet) > 0 else float("nan")
cv_std_rmse  = float(cv_summary_prophet["mean_rmse"].std())  if len(cv_summary_prophet) > 1 else 0.0
cv_mean_mape = float(cv_summary_prophet["mean_mape_pct"].mean()) if len(cv_summary_prophet) > 0 else float("nan")

print("Prophet Walk-Forward CV Results (per cutoff, mean across all 25 series):")
print(cv_summary_prophet[["fold", "cutoff", "mean_rmse", "mean_mae", "mean_mape_pct"]].to_string(index=False))
print(f"\n── CV Summary ──────────────────────────────────────────")
print(f"   Mean RMSE : {cv_mean_rmse:.2f} ± {cv_std_rmse:.2f}")
print(f"   Mean MAPE : {cv_mean_mape:.2f}%")

# COMMAND ----------

# MAGIC %md ## 5. Distributed Execution via `applyInPandas`

# COMMAND ----------

mlflow.set_experiment(EXPERIMENT)

with mlflow.start_run(run_name=MODEL_NAME) as run:
    mlflow.set_tag("model_name", MODEL_NAME)
    mlflow.log_params({
        "method":               "Prophet",
        "seasonality_mode":     "multiplicative",
        "changepoint_prior":    CHANGEPOINT_PRIOR,
        "seasonality_prior":    SEASONALITY_PRIOR,
        "uncertainty_samples":  UNCERTAINTY_SAMPLES,
        "regressor":            "pop_norm",
        "horizon":              HORIZON,
        "train_end":            TRAIN_END,
        "seed":                 SEED,
        "cv_initial":           PROPHET_CV_INITIAL,
        "cv_period":            PROPHET_CV_PERIOD,
        "cv_horizon":           PROPHET_CV_HORIZON,
    })
    mlflow.log_metric("cv_mean_rmse", cv_mean_rmse)
    mlflow.log_metric("cv_std_rmse",  cv_std_rmse)
    mlflow.log_metric("cv_mean_mape", cv_mean_mape)

    _RUN_ID = run.info.run_id

    sdf_input = (
        spark.createDataFrame(
            raw[["period", "region", "care_type", "demand_units", "pop_norm"]]
        )
        .withColumn("period", F.col("period").cast("date"))
    )

    # applyInPandas trains one Prophet model per (region, care_type) in parallel
    sdf_forecast = (
        sdf_input
        .groupBy("region", "care_type")
        .applyInPandas(prophet_one_series, schema=RESULT_SCHEMA)
    )

    # Trigger execution and cache
    sdf_forecast.cache()
    sdf_forecast.count()

    # ── Hold-out metrics via Spark ────────────────────────────────────────
    actuals_sdf = (
        sdf_input
        .filter((F.col("period") >= TEST_START) & (F.col("period") <= TEST_END))
        .withColumnRenamed("demand_units", "actual")
        .select("period", "region", "care_type", "actual")
    )
    test_fcst_sdf = sdf_forecast.filter(F.col("is_test_set") == True)

    metrics_sdf = (
        test_fcst_sdf
        .join(actuals_sdf, on=["period", "region", "care_type"])
        .withColumn("sq_err",    F.pow(F.col("forecast_value") - F.col("actual"), 2))
        .withColumn("abs_err",   F.abs(F.col("forecast_value") - F.col("actual")))
        .withColumn("pct_err",   F.abs(F.col("forecast_value") - F.col("actual")) / F.col("actual"))
        .agg(
            F.sqrt(F.avg("sq_err")).alias("rmse"),
            F.avg("abs_err").alias("mae"),
            (F.avg("pct_err") * 100).alias("mape"),
        )
    )
    m = metrics_sdf.first()
    mean_rmse = float(m["rmse"])
    mean_mae  = float(m["mae"])
    mean_mape = float(m["mape"])

    mlflow.log_metric("mean_rmse_test",     mean_rmse)
    mlflow.log_metric("mean_mae_test",      mean_mae)
    mlflow.log_metric("mean_mape_pct_test", mean_mape)
    print(f"Hold-out  RMSE: {mean_rmse:.2f}  |  MAE: {mean_mae:.2f}  |  MAPE: {mean_mape:.2f}%")

# COMMAND ----------

# MAGIC %md ## 6. Save to Shared Forecast Table

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

# MAGIC %md ## 7. Prophet Component Decomposition (Driver)
# MAGIC
# MAGIC Refit one example series on the driver to produce the human-readable
# MAGIC trend + seasonality decomposition chart that is a key deliverable
# MAGIC for OOH care planning teams.

# COMMAND ----------

from prophet import Prophet
from prophet.plot import plot_components
import matplotlib.pyplot as plt
import warnings, logging
warnings.filterwarnings("ignore")
logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)

np.random.seed(SEED)   # re-pin seed for driver-side model

sample_region    = "South"
sample_care_type = "Residential Care"

sample = raw[(raw["region"] == sample_region) & (raw["care_type"] == sample_care_type)].copy()
sample_prophet = sample.rename(columns={"period": "ds", "demand_units": "y"})

m_demo = Prophet(
    seasonality_mode="multiplicative",
    changepoint_prior_scale=CHANGEPOINT_PRIOR,
    seasonality_prior_scale=SEASONALITY_PRIOR,
    interval_width=0.80,
    uncertainty_samples=UNCERTAINTY_SAMPLES,
    yearly_seasonality=True,
    weekly_seasonality=False,
    daily_seasonality=False,
)
m_demo.add_regressor("pop_norm", standardize=False)
m_demo.fit(sample_prophet[["ds", "y", "pop_norm"]])

# Build forecast frame covering actuals + 6 future months
future_demo = m_demo.make_future_dataframe(periods=HORIZON, freq="MS")
# Extrapolate pop_norm for future months
last_pop   = sample_prophet["pop_norm"].iloc[-1]
pop_growth = (sample_prophet["pop_norm"].iloc[-1] - sample_prophet["pop_norm"].iloc[-13]) / 12
future_pops = [
    sample_prophet[sample_prophet["ds"] == d]["pop_norm"].values[0]
    if d in sample_prophet["ds"].values
    else last_pop + pop_growth * (i + 1)
    for i, d in enumerate(future_demo["ds"])
]
future_demo["pop_norm"] = future_pops
forecast_demo = m_demo.predict(future_demo)

# ── Forecast plot ─────────────────────────────────────────────────────────
fig1 = m_demo.plot(forecast_demo, figsize=(13, 4))
fig1.axes[0].set_title(f"Prophet Forecast — {sample_region}: {sample_care_type}", fontsize=13)
fig1.axes[0].axvline(pd.Timestamp(TEST_START), color="grey", linestyle="--", lw=1)
fig1.axes[0].set_xlabel("Month")
fig1.axes[0].set_ylabel("Demand (care packages)")
plt.tight_layout()
plt.savefig("/tmp/06_prophet_forecast.png", dpi=120)
plt.show()

# ── Component decomposition ───────────────────────────────────────────────
fig2 = plot_components(m_demo, forecast_demo, figsize=(13, 8))
fig2.suptitle(f"Prophet Components — {sample_region}: {sample_care_type}", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig("/tmp/06_prophet_components.png", dpi=120)
plt.show()

with mlflow.start_run(run_id=_RUN_ID):
    mlflow.log_artifact("/tmp/06_prophet_forecast.png")
    mlflow.log_artifact("/tmp/06_prophet_components.png")

# COMMAND ----------

# MAGIC %md ## 8. Quick Cross-Plot with Actuals

# COMMAND ----------

fcst_pdf = sdf_forecast.filter(
    (F.col("region") == sample_region) & (F.col("care_type") == sample_care_type)
).toPandas()
fcst_pdf["period"] = pd.to_datetime(fcst_pdf["period"])

fig, ax = plt.subplots(figsize=(13, 4))
ax.plot(sample["period"], sample["demand_units"], label="Actual", color="steelblue", lw=2)
ax.axvline(pd.Timestamp(TEST_START), color="grey", linestyle="--", lw=1, label="Train/Test split")

test_fcst   = fcst_pdf[fcst_pdf["is_test_set"] == True].sort_values("period")
future_fcst = fcst_pdf[fcst_pdf["is_test_set"] == False].sort_values("period")

ax.plot(test_fcst["period"], test_fcst["forecast_value"], "o--", color="darkorange",
        label="Prophet forecast (test)")
ax.fill_between(test_fcst["period"], test_fcst["lower_bound"], test_fcst["upper_bound"],
                alpha=0.25, color="darkorange")

ax.plot(future_fcst["period"], future_fcst["forecast_value"], "o-", color="firebrick",
        label="Prophet forecast (future)")
ax.fill_between(future_fcst["period"], future_fcst["lower_bound"], future_fcst["upper_bound"],
                alpha=0.25, color="firebrick")

ax.set_title(f"Prophet Forecast — {sample_region}: {sample_care_type}", fontsize=13)
ax.set_xlabel("Month"); ax.set_ylabel("Demand (care packages)")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("/tmp/06_prophet_overlay.png", dpi=120)
plt.show()

with mlflow.start_run(run_id=_RUN_ID):
    mlflow.log_artifact("/tmp/06_prophet_overlay.png")

print("✅ Notebook 06 (Prophet) complete.")
