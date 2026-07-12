# Databricks notebook source
# MAGIC %md
# MAGIC # 03 · Simple RNN Forecast
# MAGIC
# MAGIC **Method level**: Deep Learning — Entry (Level 3 of 5)
# MAGIC
# MAGIC Based on **Chapter 15: Processing Sequences Using RNNs and CNNs**
# MAGIC (*Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow*, Géron).
# MAGIC
# MAGIC ### Architecture
# MAGIC ```
# MAGIC Input  →  SimpleRNN(64)  →  SimpleRNN(32)  →  Dense(6)
# MAGIC ```
# MAGIC * **Input window**: last 18 months
# MAGIC * **Output**: 6-step-ahead multi-output forecast (direct strategy from Ch15)
# MAGIC * **Loss**: Huber (robust to outliers)
# MAGIC * **Normalisation**: per-series min–max scaling
# MAGIC
# MAGIC A **global model** is trained on all 25 series simultaneously, with
# MAGIC region and care-type embeddings passed as additional features.

# COMMAND ----------

# MAGIC %md ## 1. Configuration

# COMMAND ----------

dbutils.widgets.text("catalog",    "hive_metastore",   "Catalog")
dbutils.widgets.text("schema",     "demand_forecast",  "Schema")
dbutils.widgets.text("src_table",  "ooh_care_monthly", "Source table")
dbutils.widgets.text("experiment", "/Shared/ooh_care_demand_forecast", "MLflow experiment")
dbutils.widgets.text("horizon",    "6",  "Forecast horizon (months)")
dbutils.widgets.text("window",     "18", "Lookback window (months)")
dbutils.widgets.text("epochs",     "200","Max training epochs")
dbutils.widgets.text("batch_size", "32", "Mini-batch size")

CATALOG    = dbutils.widgets.get("catalog")
SCHEMA     = dbutils.widgets.get("schema")
SRC_TABLE  = f"{CATALOG}.{SCHEMA}.{dbutils.widgets.get('src_table')}"
FCST_TABLE = f"{CATALOG}.{SCHEMA}.ooh_care_forecasts"
EXPERIMENT = dbutils.widgets.get("experiment")
HORIZON    = int(dbutils.widgets.get("horizon"))
WINDOW     = int(dbutils.widgets.get("window"))
EPOCHS     = int(dbutils.widgets.get("epochs"))
BATCH_SIZE = int(dbutils.widgets.get("batch_size"))

MODEL_NAME   = "simple_rnn"
TRAIN_END    = "2023-06-01"
TEST_START   = "2023-07-01"
TEST_END     = "2023-12-01"
FUTURE_START = "2024-01-01"

print(f"Source  : {SRC_TABLE}")
print(f"Results : {FCST_TABLE}")
print(f"Model   : {MODEL_NAME}")
print(f"Window  : {WINDOW}  |  Horizon: {HORIZON}  |  Epochs: {EPOCHS}")

# COMMAND ----------

# MAGIC %pip install --quiet tensorflow

# COMMAND ----------

import numpy as np
import pandas as pd
from datetime import datetime, timezone
import tensorflow as tf
from tensorflow import keras
import mlflow
import mlflow.keras
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField,
    DateType, StringType, DoubleType, BooleanType, TimestampType
)

tf.random.set_seed(42)
np.random.seed(42)

print(f"TensorFlow version: {tf.__version__}")

# COMMAND ----------

# MAGIC %md ## 2. Load and Prepare Data

# COMMAND ----------

raw = spark.table(SRC_TABLE).toPandas()
raw["period"] = pd.to_datetime(raw["period"])
raw = raw.sort_values(["region", "care_type", "period"]).reset_index(drop=True)

# Encode categorical keys as integers for embedding layers
regions    = sorted(raw["region"].unique())
care_types = sorted(raw["care_type"].unique())
region_map    = {r: i for i, r in enumerate(regions)}
care_type_map = {c: i for i, c in enumerate(care_types)}

raw["region_id"]    = raw["region"].map(region_map)
raw["care_type_id"] = raw["care_type"].map(care_type_map)

N_REGIONS    = len(regions)
N_CARE_TYPES = len(care_types)
print(f"Regions: {N_REGIONS}  |  Care types: {N_CARE_TYPES}")

# COMMAND ----------

# MAGIC %md ## 3. Windowed Dataset (Chapter 15 approach)
# MAGIC
# MAGIC We create overlapping windows across all series:
# MAGIC `[t-17, …, t] → [t+1, …, t+6]`

# COMMAND ----------

def make_windowed_dataset(df_train, window=WINDOW, horizon=HORIZON):
    """
    Build (X_seq, X_region, X_care, y) arrays from training data.
    Each row is one window from one series, normalised by that series' stats.
    """
    X_seq, X_region, X_care, y = [], [], [], []
    scalers = {}   # (region, care_type) → (mean, std)

    for (region_id, care_type_id), grp in df_train.groupby(["region_id", "care_type_id"]):
        vals = grp.sort_values("period")["demand_units"].values.astype(float)

        # Per-series z-score normalisation
        mu, sigma = vals.mean(), vals.std(ddof=1)
        sigma = max(sigma, 1e-8)
        scalers[(region_id, care_type_id)] = (mu, sigma)
        vals_norm = (vals - mu) / sigma

        for start in range(len(vals_norm) - window - horizon + 1):
            X_seq.append(vals_norm[start : start + window])
            X_region.append(region_id)
            X_care.append(care_type_id)
            y.append(vals_norm[start + window : start + window + horizon])

    return (
        np.array(X_seq)[..., np.newaxis],   # (N, window, 1)
        np.array(X_region),                  # (N,)
        np.array(X_care),                    # (N,)
        np.array(y),                         # (N, horizon)
        scalers,
    )


train_df = raw[raw["period"] <= TRAIN_END].copy()
X_seq, X_region, X_care, y, scalers = make_windowed_dataset(train_df)

print(f"Windows: {len(X_seq):,}  |  X_seq shape: {X_seq.shape}  |  y shape: {y.shape}")

# COMMAND ----------

# MAGIC %md ## 4. Build Simple RNN Model (Chapter 15)
# MAGIC
# MAGIC Two stacked `SimpleRNN` layers followed by a `Dense` output layer
# MAGIC producing all 6 forecast steps in one shot (direct multi-output).

# COMMAND ----------

EMBED_DIM = 8

# Sequence input
seq_input    = keras.Input(shape=(WINDOW, 1), name="sequence")
region_input = keras.Input(shape=(), dtype="int32", name="region")
care_input   = keras.Input(shape=(), dtype="int32", name="care_type")

# Categorical embeddings (help the model learn series-level patterns)
region_emb    = keras.layers.Embedding(N_REGIONS,    EMBED_DIM, name="region_emb")(region_input)
care_emb      = keras.layers.Embedding(N_CARE_TYPES, EMBED_DIM, name="care_type_emb")(care_input)

# Repeat embeddings across timesteps and concatenate with sequence
region_rep = keras.layers.RepeatVector(WINDOW)(region_emb)
care_rep   = keras.layers.RepeatVector(WINDOW)(care_emb)
merged     = keras.layers.Concatenate(axis=-1)([seq_input, region_rep, care_rep])

# Simple RNN stack (Chapter 15, §"Deep RNNs")
rnn1 = keras.layers.SimpleRNN(64, return_sequences=True, activation="tanh",
                               dropout=0.1, name="rnn_1")(merged)
rnn2 = keras.layers.SimpleRNN(32, return_sequences=False, activation="tanh",
                               dropout=0.1, name="rnn_2")(rnn1)

output = keras.layers.Dense(HORIZON, name="forecast")(rnn2)

model = keras.Model(
    inputs=[seq_input, region_input, care_input],
    outputs=output,
    name="SimpleRNN_OOH_Forecast",
)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=keras.losses.Huber(),
    metrics=["mae"],
)
model.summary()

# COMMAND ----------

# MAGIC %md ## 5. Train

# COMMAND ----------

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=20, restore_best_weights=True, verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=10, min_lr=1e-6, verbose=1
    ),
]

mlflow.set_experiment(EXPERIMENT)

with mlflow.start_run(run_name=MODEL_NAME) as run:
    mlflow.set_tag("model_name", MODEL_NAME)
    mlflow.log_params({
        "architecture": "SimpleRNN",
        "rnn_units":    "64→32",
        "embed_dim":    EMBED_DIM,
        "window":       WINDOW,
        "horizon":      HORIZON,
        "epochs_max":   EPOCHS,
        "batch_size":   BATCH_SIZE,
        "loss":         "Huber",
        "train_end":    TRAIN_END,
    })

    history = model.fit(
        [X_seq, X_region, X_care], y,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.15,
        callbacks=callbacks,
        verbose=1,
    )

    best_val_loss = float(min(history.history["val_loss"]))
    actual_epochs = len(history.history["loss"])
    mlflow.log_metric("best_val_loss", best_val_loss)
    mlflow.log_metric("actual_epochs",  actual_epochs)
    print(f"Training complete: {actual_epochs} epochs | best val_loss={best_val_loss:.4f}")

    # ── Generate forecasts ─────────────────────────────────────────────────
    run_timestamp = datetime.now(timezone.utc)
    all_rows = []

    for (region, care_type), grp in raw.groupby(["region", "care_type"]):
        grp = grp.sort_values("period")
        r_id = region_map[region]
        c_id = care_type_map[care_type]
        mu, sigma = scalers.get((r_id, c_id), (grp["demand_units"].mean(), grp["demand_units"].std()))

        def predict_from_series(series_vals, is_test):
            window_raw  = series_vals[-WINDOW:]
            window_norm = (window_raw - mu) / max(sigma, 1e-8)
            X_s = np.array(window_norm)[np.newaxis, :, np.newaxis]
            X_r = np.array([r_id])
            X_c = np.array([c_id])
            pred_norm = model.predict([X_s, X_r, X_c], verbose=0)[0]
            pred      = pred_norm * sigma + mu
            # Approximate 80 % PI from residual std
            pred_std  = sigma * 0.12
            lo = pred - 1.282 * pred_std
            hi = pred + 1.282 * pred_std
            return pred, lo, hi

        # Test-set forecast (trained on train only)
        train_vals  = grp[grp["period"] <= TRAIN_END]["demand_units"].values.astype(float)
        test_pred, test_lo, test_hi = predict_from_series(train_vals, True)
        test_dates = pd.date_range(TEST_START, periods=HORIZON, freq="MS")
        for dt, fv, lo, hi in zip(test_dates, test_pred, test_lo, test_hi):
            all_rows.append({
                "period": dt.date(), "region": region, "care_type": care_type,
                "model_name": MODEL_NAME,
                "forecast_value": float(fv), "lower_bound": float(lo), "upper_bound": float(hi),
                "is_test_set": True, "run_timestamp": run_timestamp,
                "mlflow_run_id": run.info.run_id,
            })

        # Future forecast (full series)
        full_vals = grp["demand_units"].values.astype(float)
        fut_pred, fut_lo, fut_hi = predict_from_series(full_vals, False)
        future_dates = pd.date_range(FUTURE_START, periods=HORIZON, freq="MS")
        for dt, fv, lo, hi in zip(future_dates, fut_pred, fut_lo, fut_hi):
            all_rows.append({
                "period": dt.date(), "region": region, "care_type": care_type,
                "model_name": MODEL_NAME,
                "forecast_value": float(fv), "lower_bound": float(lo), "upper_bound": float(hi),
                "is_test_set": False, "run_timestamp": run_timestamp,
                "mlflow_run_id": run.info.run_id,
            })

    # Compute hold-out RMSE
    test_rows = [r for r in all_rows if r["is_test_set"]]
    test_pdf  = pd.DataFrame(test_rows)
    actuals   = raw[(raw["period"] >= TEST_START) & (raw["period"] <= TEST_END)][
                    ["period", "region", "care_type", "demand_units"]
                ].copy()
    actuals["period"] = actuals["period"].dt.date
    merged_eval = test_pdf.merge(actuals, on=["period", "region", "care_type"])
    rmse_val = float(np.sqrt(np.mean((merged_eval["forecast_value"] - merged_eval["demand_units"]) ** 2)))
    mape_val = float(np.mean(np.abs(
        (merged_eval["forecast_value"] - merged_eval["demand_units"]) / merged_eval["demand_units"]
    ))) * 100

    mlflow.log_metric("mean_rmse_test", rmse_val)
    mlflow.log_metric("mean_mape_pct_test", mape_val)
    print(f"Hold-out RMSE: {rmse_val:.2f}  |  MAPE: {mape_val:.2f}%")

    mlflow.keras.log_model(model, artifact_path="model")

# COMMAND ----------

# MAGIC %md ## 6. Save to Delta Table

# COMMAND ----------

forecast_pdf = pd.DataFrame(all_rows)
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
spark.sql(f"DELETE FROM {FCST_TABLE} WHERE model_name = '{MODEL_NAME}'")
sdf.write.format("delta").mode("append").saveAsTable(FCST_TABLE)

print(f"Saved {sdf.count():,} forecast rows → {FCST_TABLE}")

# COMMAND ----------

# MAGIC %md ## 7. Training Curves & Forecast Plot

# COMMAND ----------

import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 4))

# Training history
axes[0].plot(history.history["loss"],     label="Train loss")
axes[0].plot(history.history["val_loss"], label="Val loss")
axes[0].set_title("Simple RNN — Training Curves")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Huber Loss")
axes[0].legend(); axes[0].grid(True, alpha=0.3)

# Sample forecast
sample_region = "South"; sample_care = "Residential Care"
actuals_s = raw[(raw["region"] == sample_region) & (raw["care_type"] == sample_care)].copy()
fcst_s    = forecast_pdf[(forecast_pdf["region"] == sample_region) &
                          (forecast_pdf["care_type"] == sample_care)].copy()
fcst_s["period"] = pd.to_datetime(fcst_s["period"])

axes[1].plot(actuals_s["period"], actuals_s["demand_units"], label="Actual", color="steelblue", lw=2)
axes[1].axvline(pd.Timestamp(TEST_START), color="grey", linestyle="--", lw=1)
tf_s = fcst_s[fcst_s["is_test_set"] == True].sort_values("period")
fu_s = fcst_s[fcst_s["is_test_set"] == False].sort_values("period")
axes[1].plot(tf_s["period"], tf_s["forecast_value"], "o--", color="darkorange", label="SimpleRNN (test)")
axes[1].fill_between(tf_s["period"], tf_s["lower_bound"], tf_s["upper_bound"], alpha=0.2, color="darkorange")
axes[1].plot(fu_s["period"], fu_s["forecast_value"], "o-", color="firebrick", label="SimpleRNN (future)")
axes[1].fill_between(fu_s["period"], fu_s["lower_bound"], fu_s["upper_bound"], alpha=0.2, color="firebrick")
axes[1].set_title(f"Simple RNN — {sample_region}: {sample_care}")
axes[1].set_xlabel("Month"); axes[1].set_ylabel("Demand"); axes[1].legend(); axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("/tmp/03_simple_rnn_forecast.png", dpi=120)
plt.show()
mlflow.log_artifact("/tmp/03_simple_rnn_forecast.png")

print("✅ Notebook 03 complete.")
