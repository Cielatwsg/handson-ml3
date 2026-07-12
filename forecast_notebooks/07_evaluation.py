# Databricks notebook source
# MAGIC %md
# MAGIC # 07 · Evaluation — Which Model Wins?
# MAGIC
# MAGIC This notebook is the **final step** in the Databricks Job/Pipeline.
# MAGIC It runs after all **six** forecast notebooks have completed and written
# MAGIC their results to the shared Delta table `ooh_care_forecasts`.
# MAGIC
# MAGIC ### What this notebook does
# MAGIC 1. Loads actuals (Jul–Dec 2023 hold-out) and all 6 model forecasts
# MAGIC 2. Computes **MAE, RMSE, MAPE, sMAPE, WAPE, Bias** per model
# MAGIC 3. Ranks models overall and **by region / care type** (heatmaps)
# MAGIC 4. Produces publication-ready charts (forecast overlay, monthly RMSE, all-series grid)
# MAGIC 5. Logs everything to MLflow and writes a `model_ranking` Delta table
# MAGIC 6. Prints a clear recommendation of the **winning model**
# MAGIC 7. Registers the winning Keras model (if applicable) in the MLflow Model Registry

# COMMAND ----------

# MAGIC %md ## 1. Configuration

# COMMAND ----------

dbutils.widgets.text("catalog",    "hive_metastore",   "Catalog")
dbutils.widgets.text("schema",     "demand_forecast",  "Schema")
dbutils.widgets.text("src_table",  "ooh_care_monthly", "Source table")
dbutils.widgets.text("experiment", "/Shared/ooh_care_demand_forecast", "MLflow experiment")

CATALOG    = dbutils.widgets.get("catalog")
SCHEMA     = dbutils.widgets.get("schema")
SRC_TABLE  = f"{CATALOG}.{SCHEMA}.{dbutils.widgets.get('src_table')}"
FCST_TABLE = f"{CATALOG}.{SCHEMA}.ooh_care_forecasts"
RANK_TABLE = f"{CATALOG}.{SCHEMA}.model_ranking"
EXPERIMENT = dbutils.widgets.get("experiment")

TEST_START = "2023-07-01"
TEST_END   = "2023-12-01"

MODEL_LABELS = {
    "baseline":          "01 · Baseline (Holt-Winters/Naïve/MA)",
    "sarima":            "02 · SARIMA",
    "simple_rnn":        "03 · Simple RNN",
    "lstm":              "04 · LSTM",
    "cnn_wavenet_lstm":  "05 · CNN-WaveNet + LSTM",
    "prophet":           "06 · Prophet",
}

print(f"Source    : {SRC_TABLE}")
print(f"Forecasts : {FCST_TABLE}")
print(f"Test set  : {TEST_START} → {TEST_END}")

# COMMAND ----------

# MAGIC %pip install --quiet matplotlib seaborn

# COMMAND ----------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from datetime import datetime, timezone
import mlflow
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField,
    StringType, DoubleType, IntegerType, TimestampType
)

sns.set_theme(style="whitegrid", palette="muted")

# COMMAND ----------

# MAGIC %md ## 2. Load Actuals and Forecasts

# COMMAND ----------

actuals_sdf = (
    spark.table(SRC_TABLE)
         .filter((F.col("period") >= TEST_START) & (F.col("period") <= TEST_END))
         .select("period", "region", "care_type", "demand_units")
)

forecasts_sdf = (
    spark.table(FCST_TABLE)
         .filter(F.col("is_test_set") == True)
         .select("period", "region", "care_type", "model_name",
                 "forecast_value", "lower_bound", "upper_bound")
)

# Join forecasts with actuals
eval_sdf = (
    forecasts_sdf
    .join(actuals_sdf, on=["period", "region", "care_type"])
    .withColumn("error",     F.col("forecast_value") - F.col("demand_units"))
    .withColumn("abs_error", F.abs(F.col("error")))
    .withColumn("sq_error",  F.pow(F.col("error"), 2))
    .withColumn("pct_error", F.abs(F.col("error")) / F.col("demand_units"))
    .withColumn("smape_num", F.abs(F.col("error")) /
                             ((F.abs(F.col("forecast_value")) + F.abs(F.col("demand_units"))) / 2))
)

# Convert to pandas for visualisation
eval_pdf = eval_sdf.toPandas()
eval_pdf["period"] = pd.to_datetime(eval_pdf["period"])

# Full actuals for plotting
all_actuals = spark.table(SRC_TABLE).toPandas()
all_actuals["period"] = pd.to_datetime(all_actuals["period"])

# Future forecasts for final chart
future_sdf = (
    spark.table(FCST_TABLE)
         .filter(F.col("is_test_set") == False)
         .toPandas()
)
future_sdf["period"] = pd.to_datetime(future_sdf["period"])

models = sorted(eval_pdf["model_name"].unique())
expected = set(MODEL_LABELS.keys())
missing  = expected - set(models)
if missing:
    print(f"⚠️  Models not yet in forecast table (run their notebooks first): {missing}")
print(f"Models loaded: {models}")
print(f"Evaluation rows: {len(eval_pdf):,}")

# COMMAND ----------

# MAGIC %md ## 3. Compute Metrics

# COMMAND ----------

def compute_metrics(grp: pd.DataFrame) -> pd.Series:
    actual = grp["demand_units"].values
    pred   = grp["forecast_value"].values
    error  = pred - actual

    mae   = np.mean(np.abs(error))
    rmse  = np.sqrt(np.mean(error ** 2))
    mape  = np.mean(np.abs(error) / np.maximum(actual, 1e-8)) * 100
    smape = np.mean(2 * np.abs(error) / (np.abs(pred) + np.abs(actual) + 1e-8)) * 100
    wape  = np.sum(np.abs(error)) / np.sum(np.abs(actual)) * 100
    bias  = np.mean(error)

    return pd.Series({
        "MAE":   mae, "RMSE": rmse, "MAPE_%":  mape,
        "sMAPE_%": smape, "WAPE_%": wape, "Bias": bias,
        "n_obs": len(grp),
    })


# ── Overall metrics ────────────────────────────────────────────────────────
overall_metrics = (
    eval_pdf.groupby("model_name")
            .apply(compute_metrics)
            .reset_index()
            .sort_values("RMSE")
)
overall_metrics["Rank"] = range(1, len(overall_metrics) + 1)
overall_metrics["Model Label"] = overall_metrics["model_name"].map(MODEL_LABELS)

print("\n" + "="*80)
print("OVERALL MODEL RANKINGS (sorted by RMSE, lower = better)")
print("="*80)
print(overall_metrics[["Rank", "Model Label", "MAE", "RMSE", "MAPE_%", "sMAPE_%", "WAPE_%", "Bias"]]
      .to_string(index=False, float_format="{:.2f}".format))

# ── Per–care-type metrics ──────────────────────────────────────────────────
care_metrics = (
    eval_pdf.groupby(["model_name", "care_type"])
            .apply(compute_metrics)
            .reset_index()
)

# ── Per–region metrics ─────────────────────────────────────────────────────
region_metrics = (
    eval_pdf.groupby(["model_name", "region"])
            .apply(compute_metrics)
            .reset_index()
)

# COMMAND ----------

# MAGIC %md ## 4. Log to MLflow

# COMMAND ----------

mlflow.set_experiment(EXPERIMENT)

with mlflow.start_run(run_name="evaluation_comparison") as eval_run:
    mlflow.set_tag("stage", "evaluation")

    for _, row in overall_metrics.iterrows():
        m = row["model_name"]
        mlflow.log_metrics({
            f"{m}_MAE":     row["MAE"],
            f"{m}_RMSE":    row["RMSE"],
            f"{m}_MAPE":    row["MAPE_%"],
            f"{m}_sMAPE":   row["sMAPE_%"],
            f"{m}_WAPE":    row["WAPE_%"],
            f"{m}_Bias":    row["Bias"],
            f"{m}_Rank":    row["Rank"],
        })

    best_model  = overall_metrics.iloc[0]["model_name"]
    best_rmse   = overall_metrics.iloc[0]["RMSE"]
    mlflow.log_param("best_model",     best_model)
    mlflow.log_metric("winner_rmse",   best_rmse)

    eval_run_id = eval_run.info.run_id
    print(f"MLflow eval run: {eval_run_id}")

# COMMAND ----------

# MAGIC %md ## 5. Save Model Ranking to Delta Table

# COMMAND ----------

rank_pdf = overall_metrics[
    ["Rank", "model_name", "Model Label", "MAE", "RMSE", "MAPE_%", "sMAPE_%", "WAPE_%", "Bias"]
].copy()
rank_pdf.columns = [
    "rank", "model_name", "model_label", "mae", "rmse",
    "mape_pct", "smape_pct", "wape_pct", "bias"
]
rank_pdf["evaluated_at"] = datetime.now(timezone.utc)

rank_schema = StructType([
    StructField("rank",        IntegerType(),   False),
    StructField("model_name",  StringType(),    False),
    StructField("model_label", StringType(),    True),
    StructField("mae",         DoubleType(),    True),
    StructField("rmse",        DoubleType(),    True),
    StructField("mape_pct",    DoubleType(),    True),
    StructField("smape_pct",   DoubleType(),    True),
    StructField("wape_pct",    DoubleType(),    True),
    StructField("bias",        DoubleType(),    True),
    StructField("evaluated_at", TimestampType(), True),
])

rank_sdf = spark.createDataFrame(rank_pdf, schema=rank_schema)
(
    rank_sdf.write
            .format("delta")
            .mode("overwrite")
            .option("overwriteSchema", "true")
            .saveAsTable(RANK_TABLE)
)
print(f"Rankings written to {RANK_TABLE}")

# COMMAND ----------

# MAGIC %md ## 6. Visualisations

# COMMAND ----------

COLOURS = {
    "baseline":         "#6c757d",   # grey
    "sarima":           "#fd7e14",   # orange
    "simple_rnn":       "#0d6efd",   # blue
    "lstm":             "#198754",   # green
    "cnn_wavenet_lstm": "#dc3545",   # red
    "prophet":          "#6f42c1",   # purple
}

# ── 6.1 Overall Metrics Bar Chart ─────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for ax, metric in zip(axes, ["RMSE", "MAPE_%", "WAPE_%"]):
    bars = ax.bar(
        [MODEL_LABELS.get(m, m).split("·")[1].strip() for m in overall_metrics["model_name"]],
        overall_metrics[metric],
        color=[COLOURS.get(m, "#888") for m in overall_metrics["model_name"]],
        edgecolor="white", linewidth=0.5,
    )
    ax.bar_label(bars, fmt="{:.1f}", padding=3, fontsize=9)
    ax.set_title(f"{metric} by Model (lower = better)", fontsize=11)
    ax.set_ylabel(metric)
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.4)

plt.suptitle("OOH Care Demand Forecast — Overall Model Comparison (Hold-out: Jul–Dec 2023)",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("/tmp/eval_01_overall_metrics.png", dpi=130, bbox_inches="tight")
plt.show()

# COMMAND ----------

# ── 6.2 RMSE Heatmap by Region × Model ───────────────────────────────────
rmse_pivot = region_metrics.pivot(index="region", columns="model_name", values="RMSE")
rmse_pivot.columns = [MODEL_LABELS.get(c, c).split("·")[1].strip() for c in rmse_pivot.columns]

fig, ax = plt.subplots(figsize=(13, 4))
sns.heatmap(
    rmse_pivot, annot=True, fmt=".0f", cmap="RdYlGn_r",
    linewidths=0.5, ax=ax, cbar_kws={"label": "RMSE"}
)
ax.set_title("RMSE Heatmap — by Region × Model", fontsize=13)
ax.set_xlabel(""); ax.set_ylabel("Region")
plt.tight_layout()
plt.savefig("/tmp/eval_02_region_heatmap.png", dpi=130, bbox_inches="tight")
plt.show()

# COMMAND ----------

# ── 6.3 RMSE Heatmap by Care Type × Model ────────────────────────────────
mape_pivot = care_metrics.pivot(index="care_type", columns="model_name", values="MAPE_%")
mape_pivot.columns = [MODEL_LABELS.get(c, c).split("·")[1].strip() for c in mape_pivot.columns]

fig, ax = plt.subplots(figsize=(13, 5))
sns.heatmap(
    mape_pivot, annot=True, fmt=".1f", cmap="RdYlGn_r",
    linewidths=0.5, ax=ax, cbar_kws={"label": "MAPE %"}
)
ax.set_title("MAPE % Heatmap — by Care Type × Model", fontsize=13)
ax.set_xlabel(""); ax.set_ylabel("Care Type")
plt.tight_layout()
plt.savefig("/tmp/eval_03_care_type_heatmap.png", dpi=130, bbox_inches="tight")
plt.show()

# COMMAND ----------

# ── 6.4 Forecast Overlay — Sample Series ─────────────────────────────────
sample_region    = "South"
sample_care_type = "Residential Care"

actuals_sample = all_actuals[
    (all_actuals["region"] == sample_region) & (all_actuals["care_type"] == sample_care_type)
].sort_values("period")

fig, ax = plt.subplots(figsize=(15, 5))
ax.plot(actuals_sample["period"], actuals_sample["demand_units"],
        color="steelblue", linewidth=2.5, label="Actual", zorder=10)
ax.axvspan(pd.Timestamp(TEST_START), pd.Timestamp(TEST_END),
           alpha=0.06, color="orange", label="Hold-out period")
ax.axvline(pd.Timestamp("2024-01-01"), color="darkgreen", linestyle=":", linewidth=1.2)
ax.text(pd.Timestamp("2024-01-15"), actuals_sample["demand_units"].max() * 1.01,
        "Forecast →", color="darkgreen", fontsize=9)

for model in models:
    colour = COLOURS.get(model, "#888")
    label  = MODEL_LABELS.get(model, model).split("·")[1].strip()

    # Test-set overlay
    test_fcst = eval_pdf[
        (eval_pdf["model_name"] == model) &
        (eval_pdf["region"] == sample_region) &
        (eval_pdf["care_type"] == sample_care_type)
    ].sort_values("period")
    ax.plot(test_fcst["period"], test_fcst["forecast_value"],
            "o--", color=colour, linewidth=1.5, markersize=5, label=f"{label} (test)")

    # Future forecast
    fut = future_sdf[
        (future_sdf["model_name"] == model) &
        (future_sdf["region"] == sample_region) &
        (future_sdf["care_type"] == sample_care_type)
    ].sort_values("period")
    if len(fut) > 0:
        ax.plot(fut["period"], fut["forecast_value"],
                "o-", color=colour, linewidth=2, markersize=6)
        ax.fill_between(fut["period"], fut["lower_bound"], fut["upper_bound"],
                        alpha=0.10, color=colour)

ax.set_title(f"All-Model Forecast Overlay — {sample_region}: {sample_care_type}",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Month"); ax.set_ylabel("Demand (care packages)")
ax.legend(loc="upper left", fontsize=8, ncol=2, framealpha=0.9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("/tmp/eval_04_forecast_overlay.png", dpi=130, bbox_inches="tight")
plt.show()

# COMMAND ----------

# ── 6.5 Future 6-Month Forecast (Jan–Jun 2024) — All Models ──────────────
N_REGIONS_PLOT   = len(regions_list)
N_CARETYPES_PLOT = len(care_types_list)
fig, axes = plt.subplots(N_REGIONS_PLOT, N_CARETYPES_PLOT, figsize=(20, 18), sharey=False)
axes = axes.flatten()

regions_list    = sorted(future_sdf["region"].unique())
care_types_list = sorted(future_sdf["care_type"].unique())
idx = 0

for region in regions_list:
    for care_type in care_types_list:
        ax = axes[idx]
        act_s = actuals_sample = all_actuals[
            (all_actuals["region"] == region) & (all_actuals["care_type"] == care_type)
        ].sort_values("period")
        ax.plot(act_s["period"], act_s["demand_units"], color="steelblue", lw=1.5, alpha=0.7)

        for model in models:
            colour = COLOURS.get(model, "#888")
            fut_s  = future_sdf[
                (future_sdf["model_name"] == model) &
                (future_sdf["region"] == region) &
                (future_sdf["care_type"] == care_type)
            ].sort_values("period")
            if len(fut_s) > 0:
                ax.plot(fut_s["period"], fut_s["forecast_value"],
                        "o-", color=colour, lw=1.5, markersize=3)

        ax.set_title(f"{region}\n{care_type}", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.25)
        idx += 1

# Legend
from matplotlib.lines import Line2D
handles = [Line2D([0], [0], color="steelblue", lw=2, label="Actual")] + [
    Line2D([0], [0], color=COLOURS[m], lw=1.5, marker="o", markersize=4,
           label=MODEL_LABELS[m].split("·")[1].strip())
    for m in models
]
fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=8,
           bbox_to_anchor=(0.5, -0.01), framealpha=0.9)
fig.suptitle("6-Month Future Forecast (Jan–Jun 2024) — All Series × All Models",
             fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("/tmp/eval_05_all_series_future.png", dpi=100, bbox_inches="tight")
plt.show()

# COMMAND ----------

# ── 6.6 Ranked Error Bars Chart ───────────────────────────────────────────
monthly_errors = (
    eval_pdf.groupby(["model_name", "period"])
            .agg(monthly_rmse=("sq_error", lambda x: np.sqrt(x.mean())))
            .reset_index()
)

fig, ax = plt.subplots(figsize=(14, 5))
for model in models:
    m_data = monthly_errors[monthly_errors["model_name"] == model].sort_values("period")
    label  = MODEL_LABELS.get(model, model).split("·")[1].strip()
    colour = COLOURS.get(model, "#888")
    ax.plot(m_data["period"], m_data["monthly_rmse"],
            "o-", color=colour, linewidth=2, markersize=7, label=label)

ax.set_title("Monthly RMSE per Forecast Month (Hold-out Jul–Dec 2023)", fontsize=13)
ax.set_xlabel("Forecast Month"); ax.set_ylabel("RMSE")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("/tmp/eval_06_monthly_rmse.png", dpi=130, bbox_inches="tight")
plt.show()

# COMMAND ----------

# ── Log all charts to MLflow ──────────────────────────────────────────────
chart_files = [
    "/tmp/eval_01_overall_metrics.png",
    "/tmp/eval_02_region_heatmap.png",
    "/tmp/eval_03_care_type_heatmap.png",
    "/tmp/eval_04_forecast_overlay.png",
    "/tmp/eval_05_all_series_future.png",
    "/tmp/eval_06_monthly_rmse.png",
]
with mlflow.start_run(run_id=eval_run_id):
    for f in chart_files:
        mlflow.log_artifact(f)

# COMMAND ----------

# MAGIC %md ## 7. Recommendation

# COMMAND ----------

winner        = overall_metrics.iloc[0]
runner_up     = overall_metrics.iloc[1]
baseline_row  = overall_metrics[overall_metrics["model_name"] == "baseline"].iloc[0]
prophet_row   = overall_metrics[overall_metrics["model_name"] == "prophet"]
prophet_rank  = int(prophet_row["Rank"].values[0]) if len(prophet_row) > 0 else "N/A"

improvement_vs_baseline = (baseline_row["RMSE"] - winner["RMSE"]) / baseline_row["RMSE"] * 100

print("=" * 80)
print("  FINAL RECOMMENDATION  (6 models evaluated)")
print("=" * 80)
print(f"  Winner    : {MODEL_LABELS.get(winner['model_name'], winner['model_name'])}")
print(f"     RMSE   : {winner['RMSE']:.2f}")
print(f"     MAPE   : {winner['MAPE_%']:.2f}%")
print(f"     sMAPE  : {winner['sMAPE_%']:.2f}%")
print(f"     Bias   : {winner['Bias']:+.2f}")
print()
print(f"  2nd place : {MODEL_LABELS.get(runner_up['model_name'], runner_up['model_name'])}")
print(f"     RMSE   : {runner_up['RMSE']:.2f}")
print()
print(f"  Improvement over Baseline (RMSE): {improvement_vs_baseline:.1f}%")
print(f"  Prophet rank: #{prophet_rank} of {len(overall_metrics)}")
print()
print("  FULL RANKING:")
for _, row in overall_metrics.iterrows():
    print(f"    #{int(row['Rank'])}  {MODEL_LABELS.get(row['model_name'], row['model_name']):<45}  "
          f"RMSE={row['RMSE']:.2f}  MAPE={row['MAPE_%']:.1f}%  Bias={row['Bias']:+.0f}")
print("=" * 80)

# COMMAND ----------

# MAGIC %md ## 8. Model Registry — Promote Winner
# MAGIC
# MAGIC Register the winning model in the MLflow Model Registry so it can be
# MAGIC deployed to a Databricks Serving endpoint or batch inference job.

# COMMAND ----------

winning_run_id = (
    spark.table(FCST_TABLE)
         .filter(F.col("model_name") == winner["model_name"])
         .select("mlflow_run_id")
         .first()["mlflow_run_id"]
)

KERAS_MODELS = {"simple_rnn", "lstm", "cnn_wavenet_lstm"}

if winning_run_id and winner["model_name"] in KERAS_MODELS:
    try:
        registered = mlflow.register_model(
            model_uri=f"runs:/{winning_run_id}/model",
            name="ooh_care_demand_forecast_champion",
        )
        print(f"Keras model registered  — version: {registered.version}")
        print(f"Model name: ooh_care_demand_forecast_champion")
    except Exception as e:
        print(f"Model registry skipped (Unity Catalog may need separate setup): {e}")
elif winner["model_name"] == "prophet":
    print("Winner is Prophet — a per-series statistical model (no single Keras artifact).")
    print("To serve, load each series' fitted Prophet object from the MLflow run artifacts.")
    print(f"Winning run ID: {winning_run_id}")
else:
    print(f"Winner '{winner['model_name']}' uses statsmodels — no Keras artifact to register.")
    print("The fitted model objects were logged to the MLflow run as Python artifacts.")

# COMMAND ----------

print("\n✅ Evaluation notebook 07 complete.")
print(f"   View results at: {RANK_TABLE}")
print(f"   MLflow experiment: {EXPERIMENT}")
print(f"   Winner: {MODEL_LABELS.get(winner['model_name'], winner['model_name'])} (RMSE={winner['RMSE']:.2f})")
