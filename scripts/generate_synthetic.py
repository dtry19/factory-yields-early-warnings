"""Generate synthetic production-context and customer-return tables for SECOM.

SECOM (UCI) gives sensor readings + pass/fail + timestamp per production run, but
no tool, product, or customer information. This script adds clearly-labelled
SYNTHETIC context so the project can show factory -> customer lifecycle analysis.

Deterministic (fixed seed). Run from the repo root:
    python scripts/generate_synthetic.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "synthetic"

TOOLS = ["ETCH-01", "ETCH-02", "ETCH-03", "ETCH-04"]
PRODUCT_FAMILIES = ["DFB laser", "EML laser", "VCSEL array", "Coherent transceiver"]
REGIONS = ["North America", "EMEA", "APAC", "China"]
FAILURE_MODES = ["Low optical power", "Wavelength drift", "Hermeticity leak",
                 "Connector damage", "No fault found"]


def load_labels() -> pd.DataFrame:
    labels = pd.read_csv(RAW / "secom_labels.data", sep=" ", header=None,
                         names=["label", "timestamp"])
    labels["timestamp"] = pd.to_datetime(labels["timestamp"], format="%d/%m/%Y %H:%M:%S")
    labels["run_id"] = [f"RUN-{i:05d}" for i in range(1, len(labels) + 1)]
    labels["fail"] = (labels["label"] == 1).astype(int)
    return labels


def production_context(labels: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    n = len(labels)
    # ETCH-03 is the "problem tool": failed runs are more likely to come from it.
    p_pass = np.array([0.27, 0.27, 0.19, 0.27])
    p_fail = np.array([0.18, 0.18, 0.46, 0.18])
    tool = [rng.choice(TOOLS, p=p_fail if f else p_pass) for f in labels["fail"]]

    hour = labels["timestamp"].dt.hour
    shift = np.select([hour.between(6, 13), hour.between(14, 21)], ["Day", "Swing"], "Night")

    return pd.DataFrame({
        "run_id": labels["run_id"],
        "timestamp": labels["timestamp"],
        "lot_id": [f"LOT-{i // 25 + 1:04d}" for i in range(n)],
        "tool_id": tool,
        "chamber": rng.choice(["A", "B"], size=n),
        "product_family": rng.choice(PRODUCT_FAMILIES, size=n, p=[0.35, 0.25, 0.25, 0.15]),
        "shift": shift,
        "units_in_run": rng.integers(400, 601, size=n),
    })


def customer_returns(ctx: pd.DataFrame, labels: pd.DataFrame,
                     rng: np.random.Generator) -> pd.DataFrame:
    # Only passed runs ship. Return rate is higher for ETCH-03 and for runs
    # within 3 runs of a failure (a "marginal process" signal).
    shipped = ctx[labels["fail"].values == 0].copy()
    fail_idx = np.flatnonzero(labels["fail"].values)
    near_fail = np.array([np.any(np.abs(fail_idx - i) <= 3) for i in shipped.index])
    rate = 0.0015 + 0.0025 * (shipped["tool_id"] == "ETCH-03").values + 0.0015 * near_fail
    counts = rng.poisson(rate * shipped["units_in_run"].values)

    rows = []
    for (_, run), k in zip(shipped.iterrows(), counts):
        for _ in range(k):
            ship = run["timestamp"] + pd.Timedelta(days=int(rng.integers(3, 15)))
            rows.append({
                "run_id": run["run_id"],
                "ship_date": ship.date(),
                "return_date": (ship + pd.Timedelta(days=int(rng.integers(20, 180)))).date(),
                "customer_region": rng.choice(REGIONS, p=[0.35, 0.2, 0.25, 0.2]),
                "failure_mode": rng.choice(FAILURE_MODES, p=[0.3, 0.25, 0.15, 0.1, 0.2]),
            })
    out = pd.DataFrame(rows)
    out.insert(0, "return_id", [f"RMA-{i:05d}" for i in range(1, len(out) + 1)])
    return out


def main() -> None:
    rng = np.random.default_rng(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    labels = load_labels()
    ctx = production_context(labels, rng)
    rets = customer_returns(ctx, labels, rng)
    ctx.to_csv(OUT / "production_context.csv", index=False)
    rets.to_csv(OUT / "customer_returns.csv", index=False)
    print(f"production_context.csv: {len(ctx):,} rows")
    print(f"customer_returns.csv:   {len(rets):,} rows")


if __name__ == "__main__":
    main()
