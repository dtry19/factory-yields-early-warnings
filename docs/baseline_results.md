# Baseline results

Time-ordered split: 1175 train runs / 392 test runs (24 test failures, base rate 6.1%).

## Cleaning
- Dropped 24 features with >50% missing values and constant features; 444 of 590 features kept; remaining nulls median-imputed (fit on train).

## Anomaly detection (test runs)
| Method | Runs flagged | Failures caught | Fail rate flagged | Fail rate unflagged |
|---|---|---|---|---|
| SPC (sensors beyond ±3σ) | 11.0% | 4.2% | 2.3% | 6.6% |
| Isolation Forest | 13.0% | 4.2% | 2.0% | 6.7% |
| SPC or Isolation Forest | 16.3% | 4.2% | 1.6% | 7.0% |

## XGBoost fail prediction (test runs)
- ROC-AUC 0.618, PR-AUC 0.092 (random baseline PR-AUC = 0.061)
- At a threshold set on train data (top 20% risk): recall 16.7%, precision 9.1%, 11.2% of test runs flagged
- Top SHAP sensors: feature_059, feature_033, feature_460, feature_000, feature_091

## Synthetic context (tool and customer returns)
Fail rate by tool: ETCH-03 17.0%, ETCH-01 4.8%, ETCH-04 4.0%, ETCH-02 3.4%

Customer returns per 1,000 shipped units by tool: ETCH-03 4.41, ETCH-01 1.99, ETCH-02 1.82, ETCH-04 1.80

## 5-fold stratified cross-validation (all 1,567 runs, 104 failures)
| Model | ROC-AUC | PR-AUC | Recall at top-20% risk | Precision at top-20% | Lift vs. random |
|---|---|---|---|---|---|
| XGBoost | 0.689 | 0.166 | 45.2% | 15.0% | 2.3x |
| Logistic regression (top 40 features) | 0.711 | 0.160 | 45.2% | 15.0% | 2.3x |

Note: the time-ordered test split scores lower than CV, which suggests process drift over time. That's worth investigating and monitoring.
