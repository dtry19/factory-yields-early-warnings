# Factory Yield Early-Warning System

An end-to-end manufacturing analytics project that catches yield problems before they reach the customer. Built on Databricks: automated data pipelines, statistical process control (SPC), anomaly detection, a fail-prediction model, and a natural-language AI assistant.

> **Status:** in progress. The datasets and project structure are in place, and the notebooks are being built.

## Business question
Which production runs are likely to fail final test, which sensors and tools drive those failures, and how do marginal runs show up later as customer returns?

## Architecture
```
Factory data ─▶ Bronze (raw) ─▶ Silver (clean) ─▶ Gold (metrics)
                     │ Lakeflow Jobs, data-quality checks + alerts
                     ▼
     SPC + Isolation Forest   XGBoost + SHAP (MLflow)   Genie AI assistant
                     ▼
              Tableau / Databricks dashboards
```

## Data
| File | Rows | Source |
|---|---|---|
| `data/raw/secom.data` | 1,567 runs × 590 sensors | [UCI SECOM](https://archive.ics.uci.edu/dataset/179/secom), real semiconductor fab data |
| `data/raw/secom_labels.data` | 1,567 | UCI SECOM: pass (−1) / fail (1) + timestamp; 104 fails (6.6%) |
| `data/synthetic/production_context.csv` | 1,567 | **Synthetic**: tool, chamber, product family, shift, lot |
| `data/synthetic/customer_returns.csv` | ~1,700 | **Synthetic**: customer returns linked to shipped runs |

See [`data/README.md`](data/README.md) for the full data dictionary. Rebuild the synthetic tables with `python scripts/generate_synthetic.py`.

## Repo layout
```
data/        raw SECOM files + synthetic context tables
scripts/     data generation and utility scripts
notebooks/   Databricks notebooks: bronze → silver → gold → SPC → ML
src/         reusable Python functions (runs locally or on Databricks)
tests/       data-quality checks
genie/       Genie space instructions and sample questions
docs/        screenshots: dashboards, alerts, MLflow runs
```

## Tech stack
Databricks (Delta Lake, Lakeflow Jobs, MLflow, AI/BI Genie) · Python (pandas, scikit-learn, XGBoost, SHAP) · SQL · Tableau

## Data citation
McCann, M. & Johnston, A. (2008). *SECOM* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C54305
