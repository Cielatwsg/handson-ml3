# Databricks notebook source
# MAGIC %md
# MAGIC # 04 · LSTM Forecast
# MAGIC
# MAGIC **Method level**: Deep Learning — Intermediate (Level 4 of 5)
# MAGIC
# MAGIC Based on **Chapter 15: Processing Sequences Using RNNs and CNNs**
# MAGIC (*Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow*, Géron).
# MAGIC
# MAGIC ### Why LSTM over Simple RNN
# MAGIC Simple RNNs suffer from **vanishing gradients** over long sequences.
# MAGIC Long Short-Term Memory (LSTM) units solve this with a gated memory cell
# MAGIC that can retain information across many time steps — critical for annual
# MAGIC seasonality (12-month cycle).
# MAGIC
# MAGIC ### Architecture
# MAGIC ```
# MAGIC Input  →  LSTM(128, return_seq=True)  →  Dropout(0.2)
# MAGIC        →  LSTM(64,  return_seq=True)  →  Dropout(0.2)
# MAGIC        →  LSTM(32,  return_seq=False) →  Dense(16, relu)  →  Dense(6)
# MAGIC ```
# MAGIC * **Multi-step output**: direct 6-step forecasting (one pass)
# MAGIC * **Normalisation**: per-series z-score with inverse transform at inference
# MAGIC * **Month embedding**: adds cyclical month features (sin/cos encoding)

# COMMAND ----------

# MAGIC %md ## 1. Configuration

# COMMAND ----------

dbutils.widgets.text("catalog",    "hive_metastore",   "Catalog")
dbutils.widgets.text("schema",     "demand_forecast",  "Schema")
dbutils.widgets.text("src_table",  "ooh_care_monthly", "Source table")
dbutils.widgets.text("experiment", "/Shared/ooh_care_demand_forecast", "MLflow experiment")
dbutils.widgets.text("horizon",    "6",  "Forecast horizon (months)")
dbutils.widgets.text("window",     "18", "Lookback window (months)")
dbutils.widgets.text("epochs",     "300","Max training epochs")
dbutils.widgets.text("batch_size", "32", "Mini-batch size")
dbutils.widgets.text("seed",       "42", "Global random seed for reproducibility")

CATALOG    = dbutils.widgets.get("catalog")
SCHEMA     = dbutils.widgets.get("schema")
SRC_TABLE  = f"{CATALOG}.{SCHEMA}.{dbutils.widgets.get('src_table')}"
FCST_TABLE = f"{CATALOG}.{SCHEMA}.ooh_care_forecasts"
EXPERIMENT = dbutils.widgets.get("experiment")
HORIZON    = int(dbutils.widgets.get("horizon"))
WINDOW     = int(dbutils.widgets.get("window"))
EPOCHS     = int(dbutils.widgets.get("epochs"))
BATCH_SIZE = int(dbutils.widgets.get("batch_size"))
SEED       = int(dbutils.widgets.get("seed"))

MODEL_NAME   = "lstm"
TRAIN_END    = "2023-06-01"
TEST_START   = "2023-07-01"
TEST_END     = "2023-12-01"
FUTURE_START = "2024-01-01"

print(f"Model  : {MODEL_NAME}")
print(f"Window : {WINDOW}  |  Horizon: {HORIZON}  |  Epochs: {EPOCHS}")

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

# MAGIC %md ## 2. Load and Feature-Engineer Data

# COMMAND ----------

raw = spark.table(SRC_TABLE).toPandas()
raw["period"] = pd.to_datetime(raw["period"])
raw = raw.sort_values(["region", "care_type", "period"]).reset_index(drop=True)

# Cyclical month encoding (sin/cos) prevents ordinal treatment of months
raw["month_sin"] = np.sin(2 * np.pi * raw["period"].dt.month / 12)
raw["month_cos"] = np.cos(2 * np.pi * raw["period"].dt.month / 12)

# Normalise population covariate
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

# FEATURE_COLS: columns to use as multivariate input
FEATURE_COLS = ["demand_units_norm", "month_sin", "month_cos", "pop_norm"]

# COMMAND ----------

# MAGIC %md ## 3. Windowed Dataset with Multivariate Features
# MAGIC
# MAGIC Input per window: `(WINDOW, n_features)`
# MAGIC * demand (normalised)
# MAGIC * month_sin, month_cos
# MAGIC * population covariate (normalised)

# COMMAND ----------

def make_windowed_dataset(df, window=WINDOW, horizon=HORIZON):
    """
    Build multivariate windowed arrays for all series.
    Returns X_seq, X_region, X_care, y arrays and per-series scalers.
    """
    X_seq, X_region, X_care, y = [], [], [], []
    scalers = {}

    for (r_id, c_id), grp in df.groupby(["region_id", "care_type_id"]):
        grp = grp.sort_values("period")
        vals = grp["demand_units"].values.astype(float)

        mu, sigma = vals.mean(), vals.std(ddof=1)
        sigma = max(sigma, 1e-8)
        scalers[(r_id, c_id)] = (mu, sigma)

        demand_norm = (vals - mu) / sigma
        grp = grp.copy()
        grp["demand_units_norm"] = demand_norm

        features = grp[FEATURE_COLS].values   # (T, n_features)

        for start in range(len(features) - window - horizon + 1):
            X_seq.append(features[start : start + window])
            X_region.append(r_id)
            X_care.append(c_id)
            target_raw = vals[start + window : start + window + horizon]
            y.append((target_raw - mu) / sigma)

    return (
        np.array(X_seq),      # (N, WINDOW, n_features)
        np.array(X_region),
        np.array(X_care),
        np.array(y),           # (N, HORIZON)
        scalers,
    )


train_df = raw[raw["period"] <= TRAIN_END].copy()
X_seq, X_region, X_care, y, scalers = make_windowed_dataset(train_df)
N_FEATURES = X_seq.shape[2]

print(f"Windows   : {len(X_seq):,}")
print(f"X_seq     : {X_seq.shape}  (N, window, features)")
print(f"y         : {y.shape}      (N, horizon)")

# COMMAND ----------

# MAGIC %md ## 4. Build Deep LSTM Model (Chapter 15)
# MAGIC
# MAGIC Three stacked LSTM layers with dropout regularisation and a dense output head.
# MAGIC Region and care-type are passed as learned embeddings so the model
# MAGIC can differentiate between series while sharing parameters.

# COMMAND ----------

EMBED_DIM = 8

seq_input    = keras.Input(shape=(WINDOW, N_FEATURES), name="sequence")
region_input = keras.Input(shape=(),  dtype="int32",   name="region")
care_input   = keras.Input(shape=(),  dtype="int32",   name="care_type")

# Embeddings for categorical series identifiers
region_emb = keras.layers.Embedding(N_REGIONS,    EMBED_DIM, name="region_emb")(region_input)
care_emb   = keras.layers.Embedding(N_CARE_TYPES, EMBED_DIM, name="care_type_emb")(care_input)
region_rep = keras.layers.RepeatVector(WINDOW)(region_emb)
care_rep   = keras.layers.RepeatVector(WINDOW)(care_emb)
merged     = keras.layers.Concatenate(axis=-1)([seq_input, region_rep, care_rep])

# Stacked LSTM (Chapter 15, §"LSTMs")
x = keras.layers.LSTM(128, return_sequences=True, name="lstm_1")(merged)
x = keras.layers.Dropout(0.2)(x)
x = keras.layers.LSTM(64,  return_sequences=True, name="lstm_2")(x)
x = keras.layers.Dropout(0.2)(x)
x = keras.layers.LSTM(32,  return_sequences=False, name="lstm_3")(x)

# Output head
x      = keras.layers.Dense(16, activation="relu", name="dense_1")(x)
output = keras.layers.Dense(HORIZON, name="forecast")(x)

model = keras.Model(
    inputs=[seq_input, region_input, care_input],
    outputs=output,
    name="LSTM_OOH_Forecast",
)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=5e-4),
    loss=keras.losses.Huber(),
    metrics=["mae"],
)
model.summary()

# COMMAND ----------

# MAGIC %md ## CV · Time Series Walk-Forward Cross-Validation
# MAGIC
# MAGIC Same **expanding-window** scheme as notebooks 01–03: a **lite LSTM**
# MAGIC (units halved, max 80 epochs, patience=8) is trained from scratch on each
# MAGIC fold — guaranteeing no look-ahead bias.
# MAGIC
# MAGIC | Fold | Train period | Validation period |
# MAGIC |---|---|---|
# MAGIC | 1 | Jan 2021 – Jun 2022 (18 m) | Jul 2022 – Dec 2022 |
# MAGIC | 2 | Jan 2021 – Dec 2022 (24 m) | Jan 2023 – Jun 2023 |

# COMMAND ----------

CV_FOLDS_LSTM = [
    {"fold": 1, "train_end": "2022-06-01", "val_start": "2022-07-01", "val_end": "2022-12-01"},
    {"fold": 2, "train_end": "2022-12-01", "val_start": "2023-01-01", "val_end": "2023-06-01"},
]

def build_lite_lstm(window, n_features, horizon, n_regions, n_care_types, embed_dim,
                    seed=SEED):
    """Lite LSTM for CV — same structure at 50 % unit count."""
    tf.random.set_seed(seed)
    s_in = keras.Input(shape=(window, n_features), name="seq")
    r_in = keras.Input(shape=(), dtype="int32",   name="reg")
    c_in = keras.Input(shape=(), dtype="int32",   name="care")
    r_e  = keras.layers.Embedding(n_regions,    embed_dim)(r_in)
    c_e  = keras.layers.Embedding(n_care_types, embed_dim)(c_in)
    r_r  = keras.layers.RepeatVector(window)(r_e)
    c_r  = keras.layers.RepeatVector(window)(c_e)
    x    = keras.layers.Concatenate(axis=-1)([s_in, r_r, c_r])
    x    = keras.layers.LSTM(64, return_sequences=True)(x);  x = keras.layers.Dropout(0.2)(x)
    x    = keras.layers.LSTM(32, return_sequences=True)(x);  x = keras.layers.Dropout(0.2)(x)
    x    = keras.layers.LSTM(16, return_sequences=False)(x)
    x    = keras.layers.Dense(8, activation="relu")(x)
    out  = keras.layers.Dense(horizon)(x)
    m    = keras.Model(inputs=[s_in, r_in, c_in], outputs=out)
    m.compile(optimizer=keras.optimizers.Adam(5e-4), loss=keras.losses.Huber())
    return m


cv_rmse_list, cv_mae_list, cv_mape_list = [], [], []

for fold_cfg in CV_FOLDS_LSTM:
    fold_train_df = raw[raw["period"] <= fold_cfg["train_end"]].copy()
    Xf, Xrf, Xcf, yf, scf = make_windowed_dataset(fold_train_df, window=WINDOW, horizon=HORIZON)

    if len(Xf) == 0:
        print(f"Fold {fold_cfg['fold']}: insufficient windows — skipped")
        continue

    lite = build_lite_lstm(WINDOW, N_FEATURES, HORIZON, N_REGIONS, N_CARE_TYPES, EMBED_DIM)
    lite.fit(
        [Xf, Xrf, Xcf], yf,
        epochs=80,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        callbacks=[keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True)],
        verbose=0,
    )

    fold_errors = []
    for (region, care_type), grp in raw.groupby(["region", "care_type"]):
        grp = grp.sort_values("period").copy()
        r_id = region_map[region];  c_id = care_type_map[care_type]
        mu, sigma = scf.get((r_id, c_id), (grp["demand_units"].mean(), grp["demand_units"].std()))
        sigma = max(sigma, 1e-8)

        train_sl = grp[grp["period"] <= fold_cfg["train_end"]]
        if len(train_sl) < WINDOW:
            continue
        train_sl = train_sl.copy()
        train_sl["demand_units_norm"] = (train_sl["demand_units"] - mu) / sigma
        feats = train_sl[FEATURE_COLS].values[-WINDOW:]
        pred_norm = lite.predict(
            [feats[np.newaxis], np.array([r_id]), np.array([c_id])], verbose=0
        )[0]
        pred = pred_norm * sigma + mu

        val_mask = (grp["period"] >= fold_cfg["val_start"]) & (grp["period"] <= fold_cfg["val_end"])
        actual   = grp[val_mask]["demand_units"].values
        if len(actual) == 0:
            continue
        n = min(len(pred), len(actual))
        fold_errors.append({
            "rmse": np.sqrt(np.mean((pred[:n] - actual[:n]) ** 2)),
            "mae":  np.mean(np.abs(pred[:n] - actual[:n])),
            "mape": np.mean(np.abs((pred[:n] - actual[:n]) / actual[:n])) * 100,
        })

    if fold_errors:
        f_rmse = np.mean([e["rmse"] for e in fold_errors])
        f_mae  = np.mean([e["mae"]  for e in fold_errors])
        f_mape = np.mean([e["mape"] for e in fold_errors])
        cv_rmse_list.append(f_rmse)
        cv_mae_list.append(f_mae)
        cv_mape_list.append(f_mape)
        print(f"Fold {fold_cfg['fold']} | Val: {fold_cfg['val_start'][:7]}–{fold_cfg['val_end'][:7]} | "
              f"RMSE={f_rmse:.2f}  MAE={f_mae:.2f}  MAPE={f_mape:.2f}%")

    keras.backend.clear_session()

cv_mean_rmse = float(np.mean(cv_rmse_list)) if cv_rmse_list else float("nan")
cv_std_rmse  = float(np.std(cv_rmse_list))  if cv_rmse_list else float("nan")
cv_mean_mape = float(np.mean(cv_mape_list)) if cv_mape_list else float("nan")

print(f"\n── CV Summary ──────────────────────────────────────────")
print(f"   Mean RMSE : {cv_mean_rmse:.2f} ± {cv_std_rmse:.2f}")
print(f"   Mean MAPE : {cv_mean_mape:.2f}%")
print("(Lite model used for CV; full model trained below.)")

# COMMAND ----------

# MAGIC %md ## 5. Train with Callbacks

# COMMAND ----------

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=25, restore_best_weights=True, verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=12, min_lr=1e-6, verbose=1
    ),
    keras.callbacks.ModelCheckpoint(
        "/tmp/lstm_best.keras", monitor="val_loss", save_best_only=True
    ),
]

mlflow.set_experiment(EXPERIMENT)

with mlflow.start_run(run_name=MODEL_NAME) as run:
    mlflow.set_tag("model_name", MODEL_NAME)
    mlflow.log_params({
        "architecture":  "Stacked LSTM",
        "lstm_units":    "128→64→32",
        "embed_dim":     EMBED_DIM,
        "dropout":       0.2,
        "n_features":    N_FEATURES,
        "feature_cols":  str(FEATURE_COLS),
        "window":        WINDOW,
        "horizon":       HORIZON,
        "epochs_max":    EPOCHS,
        "batch_size":    BATCH_SIZE,
        "loss":          "Huber",
        "train_end":     TRAIN_END,
        "seed":          SEED,
        "cv_n_folds":    len(CV_FOLDS_LSTM),
    })
    mlflow.log_metric("cv_mean_rmse", cv_mean_rmse)
    mlflow.log_metric("cv_std_rmse",  cv_std_rmse)
    mlflow.log_metric("cv_mean_mape", cv_mean_mape)

    history = model.fit(
        [X_seq, X_region, X_care], y,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.15,
        callbacks=callbacks,
        verbose=1,
    )

    best_val_loss  = float(min(history.history["val_loss"]))
    actual_epochs  = len(history.history["loss"])
    mlflow.log_metric("best_val_loss", best_val_loss)
    mlflow.log_metric("actual_epochs",  actual_epochs)
    print(f"Training done: {actual_epochs} epochs | best val_loss={best_val_loss:.4f}")

    # ── Generate forecasts ─────────────────────────────────────────────────
    run_timestamp = datetime.now(timezone.utc)
    all_rows = []

    for (region, care_type), grp in raw.groupby(["region", "care_type"]):
        grp = grp.sort_values("period")
        r_id = region_map[region]
        c_id = care_type_map[care_type]
        mu, sigma = scalers.get((r_id, c_id), (grp["demand_units"].mean(), grp["demand_units"].std()))
        sigma = max(sigma, 1e-8)

        def predict_window(series_slice):
            vals  = series_slice["demand_units"].values.astype(float)
            d_norm = (vals[-WINDOW:] - mu) / sigma
            series_slice = series_slice.copy()
            series_slice["demand_units_norm"] = (series_slice["demand_units"] - mu) / sigma
            feats = series_slice[FEATURE_COLS].values[-WINDOW:]  # (WINDOW, n_features)
            feats[:, 0] = d_norm   # override demand col with normalised
            X_s = feats[np.newaxis]
            X_r = np.array([r_id])
            X_c = np.array([c_id])
            pred_norm = model.predict([X_s, X_r, X_c], verbose=0)[0]
            pred = pred_norm * sigma + mu
            pred_std = sigma * 0.10
            return pred, pred - 1.282 * pred_std, pred + 1.282 * pred_std

        # Test-set forecast
        train_slice = grp[grp["period"] <= TRAIN_END].copy()
        t_pred, t_lo, t_hi = predict_window(train_slice)
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
        f_pred, f_lo, f_hi = predict_window(grp)
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
axes[0].set_title("LSTM — Training Curves")
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
axes[1].plot(tf_s["period"], tf_s["forecast_value"], "o--", color="darkorange", label="LSTM (test)")
axes[1].fill_between(tf_s["period"], tf_s["lower_bound"], tf_s["upper_bound"], alpha=0.2, color="darkorange")
axes[1].plot(fu_s["period"], fu_s["forecast_value"], "o-", color="firebrick", label="LSTM (future)")
axes[1].fill_between(fu_s["period"], fu_s["lower_bound"], fu_s["upper_bound"], alpha=0.2, color="firebrick")
axes[1].set_title(f"LSTM — {sample_region}: {sample_care}")
axes[1].set_xlabel("Month"); axes[1].set_ylabel("Demand"); axes[1].legend(); axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("/tmp/04_lstm_forecast.png", dpi=120)
plt.show()
mlflow.log_artifact("/tmp/04_lstm_forecast.png")

print("✅ Notebook 04 complete.")
