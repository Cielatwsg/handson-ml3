# OOH Care Demand Forecast — Databricks Notebooks

6-month demand projection for **Out-of-Home (OOH) care** using monthly data
(2021–2023) across 5 regions × 5 care types = 25 series.

Inspired by **Chapter 15: Processing Sequences Using RNNs and CNNs**
(*Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow*, Géron).

---

## Notebook Overview

| Notebook | Method | Level | Key library |
|---|---|---|---|
| `00_data_setup.py` | Synthetic data generation | Setup | PySpark / Delta |
| `01_baseline_forecast.py` | Seasonal Naïve + Moving Avg + Holt-Winters ETS | ★☆☆☆☆ | statsmodels |
| `02_sarima_forecast.py` | SARIMA (auto-order selection, distributed) | ★★☆☆☆ | pmdarima |
| `03_simple_rnn_forecast.py` | Simple RNN (Ch15 §"Using a Simple RNN") | ★★★☆☆ | TensorFlow/Keras |
| `04_lstm_forecast.py` | Stacked LSTM + multivariate features (Ch15 §"LSTMs") | ★★★★☆ | TensorFlow/Keras |
| `05_cnn_wavenet_forecast.py` | WaveNet-style dilated Conv1D + LSTM (Ch15 §"WaveNet") | ★★★★★ | TensorFlow/Keras |
| `06_prophet_forecast.py` | Prophet — multiplicative decomposition + changepoints | ★★★☆☆ | prophet |
| `07_evaluation.py` | Cross-model evaluation & ranking (all 6 models) | Final | MLflow / seaborn |

---

## Data Schema

```
ooh_care_monthly (Delta table)
├── period           DATE     — first day of each month (2021-01 to 2023-12)
├── region           STRING   — North | South | East | West | Central
├── care_type        STRING   — Residential Care | Nursing Care | Supported Living
│                              | Day Care | Respite Care
├── demand_units     DOUBLE   — active care packages (service users)
└── population_65plus DOUBLE  — regional 65+ population (covariate)

ooh_care_forecasts (Delta table, partitioned by model_name)
├── period           DATE
├── region           STRING
├── care_type        STRING
├── model_name       STRING   — baseline | sarima | simple_rnn | lstm | cnn_wavenet_lstm
├── forecast_value   DOUBLE
├── lower_bound      DOUBLE   — 80% prediction interval lower
├── upper_bound      DOUBLE   — 80% prediction interval upper
├── is_test_set      BOOLEAN  — TRUE = Jul–Dec 2023 | FALSE = Jan–Jun 2024
├── run_timestamp    TIMESTAMP
└── mlflow_run_id    STRING

model_ranking (Delta table, written by 06_evaluation)
├── rank             INT
├── model_name       STRING
├── model_label      STRING
├── mae              DOUBLE
├── rmse             DOUBLE   — primary ranking metric
├── mape_pct         DOUBLE
├── smape_pct        DOUBLE
├── wape_pct         DOUBLE
├── bias             DOUBLE
└── evaluated_at     TIMESTAMP
```

---

## Train / Test Split

```
Jan 2021 ──────────────────── Jun 2023 │ Jul 2023 ── Dec 2023 │ Jan 2024 ── Jun 2024
         TRAINING (30 months)          │  HOLD-OUT (6 months)  │  FORECAST (6 months)
                                       │  used in evaluation   │  no actuals yet
```

---

## Evaluation Metrics

| Metric | Formula | Interpretation |
|---|---|---|
| **MAE** | mean \|actual − forecast\| | Average absolute error in care packages |
| **RMSE** | √mean(error²) | Primary ranking metric; penalises large errors |
| **MAPE** | mean \|error/actual\| × 100 | Percentage error, interpretable to stakeholders |
| **sMAPE** | mean 2·\|error\|/(actual+forecast) × 100 | Symmetric MAPE, avoids asymmetry |
| **WAPE** | Σ\|error\|/Σactual × 100 | Weighted; robust to small actual values |
| **Bias** | mean(forecast − actual) | Positive = over-forecast, Negative = under-forecast |

---

## How to Run on Databricks

### Option 1 — Manual, one notebook at a time

1. Upload all `.py` files to Databricks Workspace (or push via Repos)
2. Run `00_data_setup` first
3. Run `01` through `06` in any order (they are independent of each other)
4. Run `07_evaluation` last

### Option 2 — Databricks Job (recommended)

Import `pipeline_job_config.json` into Databricks Jobs:

```bash
databricks jobs create --json @pipeline_job_config.json
```

The DAG is:
```
00_data_setup
     │
     ├── 01_baseline    ─┐
     ├── 02_sarima      ─┤
     ├── 03_simple_rnn  ─┼──► 07_evaluation
     ├── 04_lstm        ─┤
     ├── 05_cnn_wavenet ─┤
     └── 06_prophet     ─┘
```

Notebooks 01–06 run **in parallel** after data setup, then the evaluation
runs once all six have finished.

### Parameters (all notebooks accept Databricks Job widgets)

| Parameter | Default | Description |
|---|---|---|
| `catalog` | `hive_metastore` | Unity Catalog name |
| `schema` | `demand_forecast` | Database / schema |
| `experiment` | `/Shared/ooh_care_demand_forecast` | MLflow experiment path |
| `horizon` | `6` | Forecast horizon in months |
| `seed` | `42` | Global random seed — set identically across all notebooks for reproducibility |

---

## Chapter 15 Concepts Used

| Concept | Used in |
|---|---|
| Windowed (sliding) dataset creation | 03, 04, 05 |
| SimpleRNN layer | 03 |
| Deep (stacked) RNNs | 03, 04 |
| LSTM units | 04 |
| 1D Convolutional layers for sequences | 05 |
| WaveNet dilated causal convolutions | 05 |
| Multi-step forecasting (direct strategy) | 03, 04, 05 |
| Multivariate time series | 04, 05 |

## Reproducibility

All notebooks accept a `seed` widget (default `42`). The seed is applied at every level:

| Level | Call |
|---|---|
| Python builtins | `random.seed(SEED)` |
| Python hash | `os.environ["PYTHONHASHSEED"] = str(SEED)` |
| NumPy | `np.random.seed(SEED)` |
| TensorFlow (notebooks 03–05) | `tf.random.set_seed(SEED)` |
| TF GPU ops (notebooks 03–05) | `os.environ["TF_DETERMINISTIC_OPS"] = "1"` |
| cuDNN (notebooks 03–05) | `os.environ["TF_CUDNN_DETERMINISTIC"] = "1"` |
| Prophet (notebook 06) | `np.random.seed(SEED)` before each `fit()` call |

> **Note**: Full bit-for-bit reproducibility in TensorFlow on GPUs is only guaranteed when
> `TF_DETERMINISTIC_OPS=1` is set *before* the TF runtime is initialised (i.e., before
> any `import tensorflow` call). Setting it in the same cell as the import is sufficient
> in Databricks notebook environments.

---

## MLflow Tracking

All five methods log to the same experiment (`/Shared/ooh_care_demand_forecast`).
The evaluation notebook adds a comparison run with aggregate metrics.
Open the MLflow UI in Databricks to compare runs side-by-side.

To register the winning model:
```python
mlflow.register_model("runs:/<run_id>/model", "ooh_care_demand_forecast_champion")
```
