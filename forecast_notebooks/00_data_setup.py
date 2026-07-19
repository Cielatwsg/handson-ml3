# Databricks notebook source
# MAGIC %md
# MAGIC # 00 · Data Setup — Out-of-Home Care Demand
# MAGIC
# MAGIC This notebook generates synthetic monthly out-of-home (OOH) care demand data
# MAGIC for 2021–2023 and writes it to a Delta table.
# MAGIC
# MAGIC **Run this notebook once before any of the forecast notebooks.**
# MAGIC
# MAGIC ### Data Schema
# MAGIC | Column | Type | Description |
# MAGIC |---|---|---|
# MAGIC | `period` | date | First day of each month |
# MAGIC | `region` | string | Geographic region |
# MAGIC | `care_type` | string | Type of out-of-home care |
# MAGIC | `demand_units` | double | Number of active care packages (service users) |
# MAGIC | `population_65plus` | double | Regional population aged 65+, used as external regressor |

# COMMAND ----------

# MAGIC %md ## 1. Configuration

# COMMAND ----------

# Widget parameters — override via Databricks Job or manually
dbutils.widgets.text("catalog",   "hive_metastore", "Catalog")
dbutils.widgets.text("schema",    "demand_forecast",  "Schema / Database")
dbutils.widgets.text("table",     "ooh_care_monthly", "Table name")
dbutils.widgets.text("seed",      "42",               "Random seed")

CATALOG    = dbutils.widgets.get("catalog")
SCHEMA     = dbutils.widgets.get("schema")
TABLE      = dbutils.widgets.get("table")
SEED       = int(dbutils.widgets.get("seed"))

FULL_TABLE = f"{CATALOG}.{SCHEMA}.{TABLE}"
print(f"Target table : {FULL_TABLE}")

# COMMAND ----------

# MAGIC %md ## 2. Packages

# COMMAND ----------

# MAGIC %pip install --quiet numpy pandas

# COMMAND ----------

import numpy as np
import pandas as pd
from datetime import date
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField,
    DateType, StringType, DoubleType
)

np.random.seed(SEED)

# COMMAND ----------

# MAGIC %md ## 3. Generate Synthetic Data
# MAGIC
# MAGIC The simulation captures realistic OOH care demand characteristics:
# MAGIC * **Upward trend** — rising ageing population
# MAGIC * **Seasonal pattern** — demand peaks in Jan–Feb (winter) and dips in Jul–Aug
# MAGIC * **Care-type scaling** — residential care > nursing care > supported living > day care > respite
# MAGIC * **Regional variation** — different base volumes per region
# MAGIC * **COVID recovery bump** — suppressed demand in early 2021, recovering through 2022

# COMMAND ----------

REGIONS = {
    "North":   {"base": 4200, "pop65": 310_000},
    "South":   {"base": 5800, "pop65": 420_000},
    "East":    {"base": 3900, "pop65": 280_000},
    "West":    {"base": 4600, "pop65": 350_000},
    "Central": {"base": 5100, "pop65": 380_000},
}

CARE_TYPES = {
    "Residential Care":  {"scale": 1.00, "noise_std": 0.025},
    "Nursing Care":      {"scale": 0.65, "noise_std": 0.030},
    "Supported Living":  {"scale": 0.80, "noise_std": 0.020},
    "Day Care":          {"scale": 0.45, "noise_std": 0.035},
    "Respite Care":      {"scale": 0.30, "noise_std": 0.040},
}

# Monthly seasonality index (January = index 0)
SEASONAL_IDX = np.array([
    1.08, 1.06, 1.02, 0.99, 0.97, 0.95,
    0.93, 0.94, 0.98, 1.02, 1.04, 1.07
])

# COVID suppression factor per month (Jan 2021 – Dec 2023)
N_MONTHS = 36
covid_factor = np.ones(N_MONTHS)
covid_factor[0:6]  = np.linspace(0.82, 0.94, 6)   # H1 2021: suppressed
covid_factor[6:12] = np.linspace(0.94, 1.00, 6)   # H2 2021: recovery
# 2022–2023: fully recovered (stays at 1.0)

# Annual growth rate
ANNUAL_GROWTH = 0.035   # 3.5% year-on-year

periods = pd.date_range("2021-01-01", periods=N_MONTHS, freq="MS")

rows = []
for region, r_cfg in REGIONS.items():
    for care_type, c_cfg in CARE_TYPES.items():
        base = r_cfg["base"] * c_cfg["scale"]

        for t, period in enumerate(periods):
            month_idx = period.month - 1
            year_offset = (period.year - 2021) + (period.month - 1) / 12
            trend = (1 + ANNUAL_GROWTH) ** year_offset
            seasonality = SEASONAL_IDX[month_idx]
            noise = np.random.normal(1.0, c_cfg["noise_std"])

            demand = base * trend * seasonality * covid_factor[t] * noise

            # Population covariate: gentle annual growth
            pop = r_cfg["pop65"] * ((1 + 0.012) ** year_offset)

            rows.append({
                "period":         period.date(),
                "region":         region,
                "care_type":      care_type,
                "demand_units":   round(demand, 2),
                "population_65plus": round(pop, 0),
            })

pdf = pd.DataFrame(rows)
print(f"Generated {len(pdf):,} rows  ({pdf['region'].nunique()} regions × "
      f"{pdf['care_type'].nunique()} care types × {N_MONTHS} months)")
print(pdf.head(10).to_string(index=False))

# COMMAND ----------

# MAGIC %md ## 4. Write to Delta Table

# COMMAND ----------

schema = StructType([
    StructField("period",          DateType(),   False),
    StructField("region",          StringType(), False),
    StructField("care_type",       StringType(), False),
    StructField("demand_units",    DoubleType(), False),
    StructField("population_65plus", DoubleType(), True),
])

sdf = spark.createDataFrame(pdf, schema=schema)

# Create schema if it doesn't exist
spark.sql(f"CREATE DATABASE IF NOT EXISTS {CATALOG}.{SCHEMA}")

(
    sdf.write
       .format("delta")
       .mode("overwrite")
       .option("overwriteSchema", "true")
       .saveAsTable(FULL_TABLE)
)

print(f"Written {sdf.count():,} rows to {FULL_TABLE}")

# COMMAND ----------

# MAGIC %md ## 5. Create Forecast Results Table
# MAGIC
# MAGIC A shared table where all 5 forecast notebooks write their outputs,
# MAGIC making cross-method evaluation easy in the evaluation notebook.

# COMMAND ----------

FORECAST_TABLE = f"{CATALOG}.{SCHEMA}.ooh_care_forecasts"

spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {FORECAST_TABLE} (
        period          DATE,
        region          STRING,
        care_type       STRING,
        model_name      STRING,
        forecast_value  DOUBLE,
        lower_bound     DOUBLE,
        upper_bound     DOUBLE,
        is_test_set     BOOLEAN COMMENT 'TRUE = Jul–Dec 2023 hold-out; FALSE = Jan–Jun 2024 future',
        run_timestamp   TIMESTAMP,
        mlflow_run_id   STRING
    )
    USING DELTA
    PARTITIONED BY (model_name)
    TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true')
""")

print(f"Forecast results table ready: {FORECAST_TABLE}")

# COMMAND ----------

# MAGIC %md ## 6. Quick Validation

# COMMAND ----------

display(
    spark.table(FULL_TABLE)
         .groupBy("region", "care_type")
         .agg(
             F.count("*").alias("n_months"),
             F.round(F.min("demand_units"), 0).alias("min_demand"),
             F.round(F.avg("demand_units"), 0).alias("avg_demand"),
             F.round(F.max("demand_units"), 0).alias("max_demand"),
         )
         .orderBy("region", "care_type")
)

# COMMAND ----------

print("✅ Data setup complete. Run notebooks 01–05 to generate forecasts, then 06 to evaluate.")
