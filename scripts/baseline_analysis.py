"""Baseline analysis: data cleaning, SPC + Isolation Forest anomaly detection,
and an XGBoost fail-prediction model on SECOM.

Uses a time-ordered split (first 75% of runs = train, last 25% = test) so the
model is evaluated on "future" runs, as it would be in a real fab.

Run from the repo root:  python scripts/baseline_analysis.py
Writes results to docs/baseline_results.md
"""
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score
from xgboost import XGBClassifier

SEED = 42
ROOT = Path(__file__).resolve().parents[1]

# ---------- load ----------
X = pd.read_csv(ROOT / "data/raw/secom.data", sep=" ", header=None)
X.columns = [f"feature_{i:03d}" for i in range(X.shape[1])]
lab = pd.read_csv(ROOT / "data/raw/secom_labels.data", sep=" ", header=None, names=["label", "ts"])
y = (lab["label"] == 1).astype(int).values
ctx = pd.read_csv(ROOT / "data/synthetic/production_context.csv")
rets = pd.read_csv(ROOT / "data/synthetic/customer_returns.csv")

n = len(X)
cut = int(n * 0.75)
tr, te = np.arange(cut), np.arange(cut, n)

# ---------- silver-layer cleaning (fit on train only) ----------
miss = X.iloc[tr].isna().mean()
keep = miss[miss <= 0.5].index
Xk = X[keep]
std = Xk.iloc[tr].std()
keep = std[std > 1e-8].index
Xk = Xk[keep]
med = Xk.iloc[tr].median()
Xc = Xk.fillna(med)
dropped_missing = int((miss > 0.5).sum())
n_features = Xc.shape[1]

# ---------- SPC: count sensors outside +-3 sigma of PASSING train runs ----------
ref = Xc.iloc[tr][y[tr] == 0]
mu, sd = ref.mean(), ref.std().replace(0, np.nan)
z = ((Xc - mu) / sd).abs()
viol = (z > 3).sum(axis=1).values
# Alarm threshold = 90th percentile of violations among passing train runs
spc_thr = np.percentile(viol[tr][y[tr] == 0], 90)
spc_flag = viol > spc_thr

# ---------- Isolation Forest (unsupervised, trained on train runs) ----------
iso = IsolationForest(n_estimators=300, contamination=0.10, random_state=SEED).fit(Xc.iloc[tr])
iso_flag = iso.predict(Xc) == -1

combined = spc_flag | iso_flag

def capture(flag, idx):
    f = flag[idx]
    return {"flagged_pct": f.mean(), "fails_caught_pct": f[y[idx] == 1].mean(),
            "fail_rate_flagged": y[idx][f].mean(), "fail_rate_unflagged": y[idx][~f].mean()}

spc_te, iso_te, comb_te = capture(spc_flag, te), capture(iso_flag, te), capture(combined, te)

# ---------- XGBoost fail prediction ----------
pos_w = (y[tr] == 0).sum() / (y[tr] == 1).sum()
model = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
                      colsample_bytree=0.5, scale_pos_weight=pos_w, eval_metric="aucpr",
                      random_state=SEED)
model.fit(Xc.iloc[tr], y[tr])
p = model.predict_proba(Xc.iloc[te])[:, 1]
# Threshold chosen on TRAIN predictions: lowest threshold giving <=20% flag rate
p_tr = model.predict_proba(Xc.iloc[tr])[:, 1]
thr = np.quantile(p_tr, 0.80)
pred = p >= thr
xgb = {"roc_auc": roc_auc_score(y[te], p), "pr_auc": average_precision_score(y[te], p),
       "base_rate": y[te].mean(), "recall": recall_score(y[te], pred),
       "precision": precision_score(y[te], pred, zero_division=0), "flag_rate": pred.mean()}

expl = shap.TreeExplainer(model)
sv = np.abs(expl.shap_values(Xc.iloc[te])).mean(axis=0)
top = pd.Series(sv, index=Xc.columns).sort_values(ascending=False).head(5)

# ---------- synthetic context findings ----------
ctx["fail"] = y
tool_fail = ctx.groupby("tool_id")["fail"].mean().sort_values(ascending=False)
shipped = ctx[ctx["fail"] == 0]
rr = rets.groupby("run_id").size().reindex(shipped["run_id"], fill_value=0)
ret_per_k = (rr.values / shipped["units_in_run"].values * 1000)
shipped = shipped.assign(ret_per_k=ret_per_k)
tool_ret = shipped.groupby("tool_id")["ret_per_k"].mean().sort_values(ascending=False)

# ---------- report ----------
pct = lambda v: f"{100 * v:.1f}%"
lines = [
    "# Baseline results", "",
    f"Time-ordered split: {len(tr)} train runs / {len(te)} test runs "
    f"({int(y[te].sum())} test failures, base rate {pct(xgb['base_rate'])}).", "",
    "## Cleaning",
    f"- Dropped {dropped_missing} features with >50% missing values and constant features; "
    f"{n_features} of 590 features kept; remaining nulls median-imputed (fit on train).", "",
    "## Anomaly detection (test runs)",
    "| Method | Runs flagged | Failures caught | Fail rate flagged | Fail rate unflagged |",
    "|---|---|---|---|---|",
]
for name, r in [("SPC (sensors beyond ±3σ)", spc_te), ("Isolation Forest", iso_te), ("SPC or Isolation Forest", comb_te)]:
    lines.append(f"| {name} | {pct(r['flagged_pct'])} | {pct(r['fails_caught_pct'])} | "
                 f"{pct(r['fail_rate_flagged'])} | {pct(r['fail_rate_unflagged'])} |")
lines += ["", "## XGBoost fail prediction (test runs)",
          f"- ROC-AUC {xgb['roc_auc']:.3f}, PR-AUC {xgb['pr_auc']:.3f} (random baseline PR-AUC = {xgb['base_rate']:.3f})",
          f"- At a threshold set on train data (top 20% risk): recall {pct(xgb['recall'])}, "
          f"precision {pct(xgb['precision'])}, {pct(xgb['flag_rate'])} of test runs flagged",
          f"- Top SHAP sensors: {', '.join(top.index)}", "",
          "## Synthetic context (tool and customer returns)",
          "Fail rate by tool: " + ", ".join(f"{k} {pct(v)}" for k, v in tool_fail.items()),
          "", "Customer returns per 1,000 shipped units by tool: " + ", ".join(f"{k} {v:.2f}" for k, v in tool_ret.items())]
# ---------- 5-fold stratified CV (more stable than the 24-failure test set) ----------
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

Xcv = X.loc[:, X.isna().mean() <= 0.5]
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
pw_all = (y == 0).sum() / (y == 1).sum()
cv_models = {
    "XGBoost": make_pipeline(SimpleImputer(strategy="median"), VarianceThreshold(1e-8),
                             XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
                                           colsample_bytree=0.5, scale_pos_weight=pw_all, random_state=SEED)),
    "Logistic regression (top 40 features)": make_pipeline(
        SimpleImputer(strategy="median"), VarianceThreshold(1e-8), StandardScaler(),
        SelectKBest(f_classif, k=40), LogisticRegression(class_weight="balanced", C=0.05, max_iter=2000)),
}
lines += ["", "## 5-fold stratified cross-validation (all 1,567 runs, 104 failures)",
          "| Model | ROC-AUC | PR-AUC | Recall at top-20% risk | Precision at top-20% | Lift vs. random |",
          "|---|---|---|---|---|---|"]
for name, m in cv_models.items():
    pc = cross_val_predict(m, Xcv, y, cv=cv, method="predict_proba")[:, 1]
    fl = pc >= np.quantile(pc, 0.80)
    prec = precision_score(y, fl)
    lines.append(f"| {name} | {roc_auc_score(y, pc):.3f} | {average_precision_score(y, pc):.3f} | "
                 f"{pct(recall_score(y, fl))} | {pct(prec)} | {prec / y.mean():.1f}x |")
lines += ["", "Note: the time-ordered test split scores lower than CV, which suggests process drift "
          "over time. That's worth investigating and monitoring."]

out = ROOT / "docs" / "baseline_results.md"
out.parent.mkdir(exist_ok=True)
out.write_text("\n".join(lines) + "\n")
print("\n".join(lines))
