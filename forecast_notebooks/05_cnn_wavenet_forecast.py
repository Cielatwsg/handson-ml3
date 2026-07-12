# Databricks notebook source
# MAGIC %md
# MAGIC # 05 · CNN + LSTM WaveNet-style Forecast
# MAGIC
# MAGIC **Method level**: Deep Learning — Advanced (Level 5 of 5)
# MAGIC
# MAGIC Based on **Chapter 15: Processing Sequences Using RNNs and CNNs**
# MAGIC (*Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow*, Géron).
# MAGIC Specifically §"Using 1D Convolutional Layers to Process Sequences" and
# MAGIC §"WaveNet".
# MAGIC
# MAGIC ### Motivation
# MAGIC Convolutional layers process sequences in parallel (unlike RNNs) and
# MAGIC **dilated (causal) convolutions** allow the receptive field to grow
# MAGIC exponentially without increasing parameters — a key insight from WaveNet.
# MAGIC Stacking a CNN feature extractor on top of an LSTM combines local pattern
# MAGIC detection with long-range memory.
# MAGIC
# MAGIC ### Architecture
# MAGIC ```
# MAGIC Input (WINDOW, n_features)
# MAGIC   ↓
# MAGIC [Conv1D(64, k=3, dilation=1, causal) → BatchNorm → ReLU]   ─┐
# MAGIC [Conv1D(64, k=3, dilation=2, causal) → BatchNorm → ReLU]    │  WaveNet-style
# MAGIC [Conv1D(64, k=3, dilation=4, causal) → BatchNorm → ReLU]    │  dilated stack
# MAGIC [Conv1D(64, k=3, dilation=8, causal) → BatchNorm → ReLU]   ─┘
# MAGIC   ↓
# MAGIC LSTM(64, return_sequences=False)
# MAGIC   ↓
# MAGIC Dense(32, relu) → Dense(6)
# MAGIC ```
# MAGIC
# MAGIC **Receptive field** = 1 + (k−1)·(1+2+4+8) = 1 + 2·15 = 31 months

# COMMAND ----------

# MAGIC %md ## 1. Configuration

# COMMAND ----------

dbutils.widgets.text("catalog",     "hive_metastore",   "Catalog")
dbutils.widgets.text("schema",      "demand_forecast",  "Schema")
dbutils.widgets.text("src_table",   "ooh_care_monthly", "Source table")
dbutils.widgets.text("experiment",  "/Shared/ooh_care_demand_forecast", "MLflow experiment")
dbutils.widgets.text("horizon",     "6",  "Forecast horizon (months)")
dbutils.widgets.text("window",      "24", "Lookback window (months)")
dbutils.widgets.text("epochs",      "300","Max training epochs")
dbutils.widgets.text("batch_size",  "32", "Mini-batch size")
dbutils.widgets.text("n_filters",   "64", "Conv1D filters per layer")
dbutils.widgets.text("seed",        "42", "Global random seed for reproducibility")

CATALOG    = dbutils.widgets.get("catalog")
SCHEMA     = dbutils.widgets.get("schema")
SRC_TABLE  = f"{CATALOG}.{SCHEMA}.{dbutils.widgets.get('src_table')}"
FCST_TABLE = f"{CATALOG}.{SCHEMA}.ooh_care_forecasts"
EXPERIMENT = dbutils.widgets.get("experiment")
HORIZON    = int(dbutils.widgets.get("horizon"))
WINDOW     = int(dbutils.widgets.get("window"))
EPOCHS     = int(dbutils.widgets.get("epochs"))
BATCH_SIZE = int(dbutils.widgets.get("batch_size"))
N_FILTERS  = int(dbutils.widgets.get("n_filters"))
SEED       = int(dbutils.widgets.get("seed"))

MODEL_NAME   = "cnn_wavenet_lstm"
TRAIN_END    = "2023-06-01"
TEST_START   = "2023-07-01"
TEST_END     = "2023-12-01"
FUTURE_START = "2024-01-01"

DILATIONS = [1, 2, 4, 8]   # WaveNet-style exponential dilation schedule

print(f"Model    : {MODEL_NAME}")
print(f"Dilations: {DILATIONS}  |  Window: {WINDOW}  |  Horizon: {HORIZON}")

# COMMAND ----------

# MAGIC %pip install --quiet tensorflow

# COMMAND ----------

# MAGIC %md ## 2. Reproducibility Seeds

# COMMAND ----------

import os, random
os.environ["PYTHONHASHSEED"]         = str(SEED)
os.environ["TF_DETERMINISTIC_OPS"]   = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
random.seed(SEED)

import numpy as np
np.random.seed(SEED)

import pandas as pd
from datetime import datetime, timezone
import tensorflow as tf
tf.random.set_seed(SEED)

from tensorflow import keras
import mlflow
import mlflow.keras
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField,
    DateType, StringType, DoubleType, BooleanType, TimestampType
)

print(f"TensorFlow: {tf.__version__}")
print(f"Global seed: {SEED}")

# COMMAND ----------

# MAGIC %md ## 2. Load and Prepare Data

# COMMAND ----------

raw = spark.table(SRC_TABLE).toPandas()
raw["period"] = pd.to_datetime(raw["period"])
raw = raw.sort_values(["region", "care_type", "period"]).reset_index(drop=True)

# Cyclical encodings + lagged demand features
raw["month_sin"] = np.sin(2 * np.pi * raw["period"].dt.month / 12)
raw["month_cos"] = np.cos(2 * np.pi * raw["period"].dt.month / 12)

# Year-progress feature
raw["year_progress"] = (raw["period"].dt.month - 1) / 11

# Normalise population
pop_mean = raw["population_65plus"].mean()
pop_std  = raw["population_65plus"].std()
raw["pop_norm"] = (raw["population_65plus"] - pop_mean) / pop_std

regions    = sorted(raw["region"].unique())
care_types = sorted(raw["care_type"].unique())
region_map    = {r: i for i, r in enumerate(regions)}
care_type_map = {c: i for i, c in enumerate(care_types)}
raw["region_id"]    = raw["region"].map(region_map)
raw["care_type_id"] = raw["care_type"].map(care_type_map)

N_REGIONS    = len(regions)
N_CARE_TYPES = len(care_types)

FEATURE_COLS = ["demand_units_norm", "month_sin", "month_cos", "year_progress", "pop_norm"]

# COMMAND ----------

# MAGIC %md ## 3. Windowed Dataset

# COMMAND ----------

def make_windowed_dataset(df, window=WINDOW, horizon=HORIZON):
    X_seq, X_region, X_care, y = [], [], [], []
    scalers = {}

    for (r_id, c_id), grp in df.groupby(["region_id", "care_type_id"]):
        grp = grp.sort_values("period").copy()
        vals = grp["demand_units"].values.astype(float)
        mu, sigma = vals.mean(), vals.std(ddof=1)
        sigma = max(sigma, 1e-8)
        scalers[(r_id, c_id)] = (mu, sigma)

        grp["demand_units_norm"] = (vals - mu) / sigma
        features = grp[FEATURE_COLS].values

        for start in range(len(features) - window - horizon + 1):
            X_seq.append(features[start : start + window])
            X_region.append(r_id)
            X_care.append(c_id)
            target = vals[start + window : start + window + horizon]
            y.append((target - mu) / sigma)

    return (
        np.array(X_seq),
        np.array(X_region),
        np.array(X_care),
        np.array(y),
        scalers,
    )


train_df = raw[raw["period"] <= TRAIN_END].copy()
X_seq, X_region, X_care, y, scalers = make_windowed_dataset(train_df)
N_FEATURES = X_seq.shape[2]

print(f"Windows  : {len(X_seq):,}")
print(f"X_seq    : {X_seq.shape}   (N, window, features)")
print(f"y        : {y.shape}        (N, horizon)")

# COMMAND ----------

# MAGIC %md ## 4. Build WaveNet-style CNN + LSTM Model (Chapter 15)
# MAGIC
# MAGIC **Dilated causal convolutions** (§"WaveNet"):
# MAGIC * `padding="causal"` ensures no information leakage from the future
# MAGIC * `dilation_rate` doubles each layer → receptive field grows exponentially
# MAGIC * `BatchNormalization` stabilises training of deep conv stacks

# COMMAND ----------

EMBED_DIM  = 8
KERNEL_SIZE = 3

seq_input    = keras.Input(shape=(WINDOW, N_FEATURES), name="sequence")
region_input = keras.Input(shape=(),  dtype="int32",   name="region")
care_input   = keras.Input(shape=(),  dtype="int32",   name="care_type")

# Categorical embeddings
region_emb = keras.layers.Embedding(N_REGIONS,    EMBED_DIM)(region_input)
care_emb   = keras.layers.Embedding(N_CARE_TYPES, EMBED_DIM)(care_input)
region_rep = keras.layers.RepeatVector(WINDOW)(region_emb)
care_rep   = keras.layers.RepeatVector(WINDOW)(care_emb)
x = keras.layers.Concatenate(axis=-1)([seq_input, region_rep, care_rep])

# ── WaveNet-style dilated causal Conv1D stack (Chapter 15) ────────────────
# Initial 1×1 projection to filter dimension
x = keras.layers.Conv1D(
    N_FILTERS, kernel_size=1, padding="causal", activation="relu",
    name="input_proj"
)(x)

for i, dilation in enumerate(DILATIONS):
    residual = x
    x = keras.layers.Conv1D(
        N_FILTERS,
        kernel_size=KERNEL_SIZE,
        padding="causal",
        dilation_rate=dilation,
        name=f"dilated_conv_{i}_d{dilation}",
    )(x)
    x = keras.layers.BatchNormalization(name=f"bn_{i}")(x)
    x = keras.layers.Activation("relu")(x)

    # Residual / skip connection (if dims match)
    if residual.shape[-1] == x.shape[-1]:
        x = keras.layers.Add(name=f"skip_{i}")([x, residual])

# ── LSTM reads the CNN feature maps ───────────────────────────────────────
x = keras.layers.LSTM(64, return_sequences=False, name="lstm")(x)
x = keras.layers.Dropout(0.15)(x)

# Output head
x      = keras.layers.Dense(32, activation="relu", name="dense_1")(x)
output = keras.layers.Dense(HORIZON, name="forecast")(x)

model = keras.Model(
    inputs=[seq_input, region_input, care_input],
    outputs=output,
    name="CNN_WaveNet_LSTM_OOH",
)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=5e-4),
    loss=keras.losses.Huber(),
    metrics=["mae"],
)
model.summary()

print(f"\nReceptive field: {1 + (KERNEL_SIZE-1)*sum(DILATIONS)} months")

# COMMAND ----------

# MAGIC %md ## 5. Train

# COMMAND ----------

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=30, restore_best_weights=True, verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=15, min_lr=1e-6, verbose=1
    ),
    keras.callbacks.ModelCheckpoint(
        "/tmp/cnn_wavenet_best.keras", monitor="val_loss", save_best_only=True
    ),
]

mlflow.set_experiment(EXPERIMENT)

with mlflow.start_run(run_name=MODEL_NAME) as run:
    mlflow.set_tag("model_name", MODEL_NAME)
    mlflow.log_params({
        "architecture":     "CNN-WaveNet + LSTM",
        "n_filters":        N_FILTERS,
        "kernel_size":      KERNEL_SIZE,
        "dilations":        str(DILATIONS),
        "lstm_units":       64,
        "embed_dim":        EMBED_DIM,
        "n_features":       N_FEATURES,
        "window":           WINDOW,
        "horizon":          HORIZON,
        "epochs_max":       EPOCHS,
        "batch_size":       BATCH_SIZE,
        "receptive_field":  1 + (KERNEL_SIZE - 1) * sum(DILATIONS),
        "train_end":        TRAIN_END,
        "seed":             SEED,
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
    print(f"Training done: {actual_epochs} epochs | best val_loss={best_val_loss:.4f}")

    # ── Generate forecasts ─────────────────────────────────────────────────
    run_timestamp = datetime.now(timezone.utc)
    all_rows = []

    for (region, care_type), grp in raw.groupby(["region", "care_type"]):
        grp = grp.sort_values("period").copy()
        r_id = region_map[region]
        c_id = care_type_map[care_type]
        mu, sigma = scalers.get((r_id, c_id), (grp["demand_units"].mean(), grp["demand_units"].std()))
        sigma = max(sigma, 1e-8)

        def predict_from_slice(slice_df):
            slice_df = slice_df.copy()
            slice_df["demand_units_norm"] = (slice_df["demand_units"] - mu) / sigma
            feats = slice_df[FEATURE_COLS].values[-WINDOW:]
            X_s = feats[np.newaxis]
            X_r = np.array([r_id])
            X_c = np.array([c_id])
            pred_norm = model.predict([X_s, X_r, X_c], verbose=0)[0]
            pred = pred_norm * sigma + mu
            pred_std = sigma * 0.09
            return pred, pred - 1.282 * pred_std, pred + 1.282 * pred_std

        # Test-set forecast
        train_slice = grp[grp["period"] <= TRAIN_END]
        t_pred, t_lo, t_hi = predict_from_slice(train_slice)
        for dt, fv, lo, hi in zip(
            pd.date_range(TEST_START, periods=HORIZON, freq="MS"),
            t_pred, t_lo, t_hi
        ):
            all_rows.append({
                "period": dt.date(), "region": region, "care_type": care_type,
                "model_name": MODEL_NAME,
                "forecast_value": float(fv), "lower_bound": float(lo), "upper_bound": float(hi),
                "is_test_set": True, "run_timestamp": run_timestamp,
                "mlflow_run_id": run.info.run_id,
            })

        # Future forecast
        f_pred, f_lo, f_hi = predict_from_slice(grp)
        for dt, fv, lo, hi in zip(
            pd.date_range(FUTURE_START, periods=HORIZON, freq="MS"),
            f_pred, f_lo, f_hi
        ):
            all_rows.append({
                "period": dt.date(), "region": region, "care_type": care_type,
                "model_name": MODEL_NAME,
                "forecast_value": float(fv), "lower_bound": float(lo), "upper_bound": float(hi),
                "is_test_set": False, "run_timestamp": run_timestamp,
                "mlflow_run_id": run.info.run_id,
            })

    # Hold-out metrics
    test_pdf   = pd.DataFrame([r for r in all_rows if r["is_test_set"]])
    actuals_df = raw[(raw["period"] >= TEST_START) & (raw["period"] <= TEST_END)][
                     ["period", "region", "care_type", "demand_units"]].copy()
    actuals_df["period"] = actuals_df["period"].dt.date
    eval_df = test_pdf.merge(actuals_df, on=["period", "region", "care_type"])

    rmse_val = float(np.sqrt(np.mean((eval_df["forecast_value"] - eval_df["demand_units"]) ** 2)))
    mape_val = float(np.mean(np.abs(
        (eval_df["forecast_value"] - eval_df["demand_units"]) / eval_df["demand_units"]
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

axes[0].plot(history.history["loss"],     label="Train loss")
axes[0].plot(history.history["val_loss"], label="Val loss")
axes[0].set_title("CNN-WaveNet+LSTM — Training Curves")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Huber Loss")
axes[0].legend(); axes[0].grid(True, alpha=0.3)

sample_region = "South"; sample_care = "Residential Care"
actuals_s = raw[(raw["region"] == sample_region) & (raw["care_type"] == sample_care)].copy()
fcst_s    = forecast_pdf[(forecast_pdf["region"] == sample_region) &
                          (forecast_pdf["care_type"] == sample_care)].copy()
fcst_s["period"] = pd.to_datetime(fcst_s["period"])

axes[1].plot(actuals_s["period"], actuals_s["demand_units"], label="Actual", color="steelblue", lw=2)
axes[1].axvline(pd.Timestamp(TEST_START), color="grey", linestyle="--", lw=1)
tf_s = fcst_s[fcst_s["is_test_set"]].sort_values("period")
fu_s = fcst_s[~fcst_s["is_test_set"]].sort_values("period")
axes[1].plot(tf_s["period"], tf_s["forecast_value"], "o--", color="darkorange", label="CNN-WaveNet (test)")
axes[1].fill_between(tf_s["period"], tf_s["lower_bound"], tf_s["upper_bound"], alpha=0.2, color="darkorange")
axes[1].plot(fu_s["period"], fu_s["forecast_value"], "o-", color="firebrick", label="CNN-WaveNet (future)")
axes[1].fill_between(fu_s["period"], fu_s["lower_bound"], fu_s["upper_bound"], alpha=0.2, color="firebrick")
axes[1].set_title(f"CNN-WaveNet+LSTM — {sample_region}: {sample_care}")
axes[1].set_xlabel("Month"); axes[1].set_ylabel("Demand"); axes[1].legend(); axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("/tmp/05_cnn_wavenet_forecast.png", dpi=120)
plt.show()
mlflow.log_artifact("/tmp/05_cnn_wavenet_forecast.png")

print("✅ Notebook 05 complete.")
