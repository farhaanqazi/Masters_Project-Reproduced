"""
Cross-dataset validation for the DataLeakageDetector.

Two kinds of checks, all on REAL data (no injected/synthetic leaks):

1. Leak-impact (before/after): for each dataset with a known ORGANIC leak, confirm
   the detector ranks the leaky column(s) at the top, then quantify the leak's effect
   by training models WITH vs WITHOUT the leaky column(s) on identical splits.

2. Precision controls: clean datasets that must NOT trigger the degeneracy hard-flag.

Run:  python -m src.validate_detector
External CSVs must be downloaded into data/external/ (see data/external/README.md).
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

from src.leakage_detector import DataLeakageDetector

SEED = 42
RESULTS_DIR = "results"
MODELS = ["RandomForest", "LogReg", "DecisionTree"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def to_numeric_frame(X):
    """Ordinal-encode object columns and median-fill NaNs so any tabular frame
    becomes a numeric matrix the detector/models can consume. NaN -> -1 for
    categoricals (via cat.codes), median for numerics."""
    X = X.copy()
    for c in X.select_dtypes(include=["object"]).columns:
        X[c] = X[c].astype("category").cat.codes  # NaN -> -1
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median(numeric_only=True)).fillna(0)
    return X


def train_eval(X, y, seed=SEED):
    """Train a small model set and return {model: {acc, auc}} on a held-out split."""
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)

    models = {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1),
        "LogReg": LogisticRegression(max_iter=2000),
        "DecisionTree": DecisionTreeClassifier(random_state=seed),
    }
    out = {}
    for name, model in models.items():
        if name == "LogReg":
            model.fit(Xtr_s, ytr)
            proba = model.predict_proba(Xte_s)[:, 1]
            pred = model.predict(Xte_s)
        else:
            model.fit(Xtr, ytr)
            proba = model.predict_proba(Xte)[:, 1]
            pred = model.predict(Xte)
        out[name] = {
            "acc": round(float(accuracy_score(yte, pred)), 4),
            "auc": round(float(roc_auc_score(yte, proba)), 4),
        }
    return out


def leak_impact_report(name, X, y, leak_cols, leak_kind):
    """Detect the leak, then measure model performance with vs without it."""
    print(f"\n{'='*70}\n  {name}  (leak: {', '.join(leak_cols)} — {leak_kind})\n{'='*70}")
    Xn = to_numeric_frame(X)
    y = pd.Series(y).reset_index(drop=True).astype(int)
    Xn = Xn.reset_index(drop=True)

    # 1. Detection
    det = DataLeakageDetector().fit(Xn, y)
    scores = det.get_scores().reset_index(drop=True)
    print("Top suspicious features:")
    print(scores[["feature", "auc", "ks_stat", "degenerate_flag", "predictiveness_score"]].head(6).to_string(index=False))

    detected = {}
    for c in leak_cols:
        if c in scores["feature"].values:
            row = scores[scores["feature"] == c].iloc[0]
            rank = int(scores.index[scores["feature"] == c][0]) + 1
            detected[c] = {"rank": rank, "degenerate_flag": bool(row["degenerate_flag"]),
                           "auc": round(float(row["auc"]), 4)}
            tag = "HARD degeneracy flag" if row["degenerate_flag"] else f"top-ranked (rank {rank}, AUC {row['auc']:.3f})"
            print(f"  [DETECTED] '{c}' -> {tag}")
        else:
            detected[c] = None
            print(f"  [MISS] '{c}' not found")

    # 2. Impact: with leak vs leak removed
    keep = [c for c in leak_cols if c in Xn.columns]
    before = train_eval(Xn, y)
    after = train_eval(Xn.drop(columns=keep), y)
    print("\nLeak impact (model performance):")
    print(f"  {'Model':<14}{'Acc(leak)':>10}{'Acc(clean)':>11}{'AUC(leak)':>11}{'AUC(clean)':>11}")
    for m in MODELS:
        print(f"  {m:<14}{before[m]['acc']:>10.3f}{after[m]['acc']:>11.3f}"
              f"{before[m]['auc']:>11.3f}{after[m]['auc']:>11.3f}")

    return {"name": name, "leak_cols": leak_cols, "leak_kind": leak_kind,
            "detected": detected, "before": before, "after": after}


def precision_control(name, X, y):
    """Clean dataset: must produce NO degeneracy hard-flags."""
    print(f"\n{'='*70}\n  PRECISION CONTROL: {name}\n{'='*70}")
    Xn = to_numeric_frame(X)
    y = pd.Series(y).reset_index(drop=True).astype(int)
    det = DataLeakageDetector().fit(Xn, y)
    scores = det.get_scores()
    n_deg = int(scores["degenerate_flag"].sum())
    print("Top features:")
    print(scores[["feature", "auc", "degenerate_flag"]].head(5).to_string(index=False))
    verdict = "PASS" if n_deg == 0 else "FAIL"
    print(f"  [{verdict}] degeneracy hard-flags on clean data: {n_deg} (want 0)")
    return {"name": name, "n_degenerate_flags": n_deg, "pass": n_deg == 0}


# ---------------------------------------------------------------------------
# Dataset loaders  (return X, y, leak_cols, leak_kind)
# ---------------------------------------------------------------------------
def load_parkinsons():
    df = pd.read_csv("data/raw/dataset.csv")
    y = df["PC"]
    X = df.drop(columns=[c for c in ["PC", "ID"] if c in df.columns])
    return X, y, ["Duration"], "within-class degeneracy (organic)"


def load_bank():
    df = pd.read_csv("data/external/bank-full.csv", sep=";")
    y = (df["y"] == "yes").astype(int)
    X = df.drop(columns=["y"])
    return X, y, ["duration"], "correlation (provider-documented)"


def load_heart_failure():
    df = pd.read_csv("data/external/heart_failure_clinical_records_dataset.csv")
    y = df["DEATH_EVENT"].astype(int)
    X = df.drop(columns=["DEATH_EVENT"])
    return X, y, ["time"], "correlation (observation window)"


def load_cervical():
    df = pd.read_csv("data/external/risk_factors_cervical_cancer.csv", na_values="?")
    y = df["Biopsy"].astype(int)
    X = df.drop(columns=["Biopsy"])
    return X, y, ["Hinselmann", "Schiller", "Citology"], "target leakage (alternate diagnoses)"


def load_hotel():
    df = pd.read_csv("data/external/hotel_bookings.csv")
    y = df["is_canceled"].astype(int)
    X = df.drop(columns=["is_canceled"])
    return X, y, ["reservation_status", "reservation_status_date"], "target leakage (label restated)"


LEAK_DATASETS = [
    ("Parkinson's Disease", load_parkinsons, "data/raw/dataset.csv"),
    ("Bank Marketing", load_bank, "data/external/bank-full.csv"),
    ("Heart Failure Clinical Records", load_heart_failure, "data/external/heart_failure_clinical_records_dataset.csv"),
    ("Cervical Cancer (Risk Factors)", load_cervical, "data/external/risk_factors_cervical_cancer.csv"),
    ("Hotel Booking Demand", load_hotel, "data/external/hotel_bookings.csv"),
]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main():
    impact_results = []
    for name, loader, path in LEAK_DATASETS:
        if not os.path.exists(path):
            print(f"\n[SKIP] {name}: {path} not found (see data/external/README.md)")
            continue
        try:
            X, y, leak_cols, kind = loader()
            impact_results.append(leak_impact_report(name, X, y, leak_cols, kind))
        except Exception as e:  # keep one dataset's failure from killing the run
            print(f"\n[ERROR] {name}: {e}")

    # Precision controls
    control_results = []
    bc = load_breast_cancer()
    control_results.append(
        precision_control("Breast Cancer Wisconsin",
                           pd.DataFrame(bc.data, columns=bc.feature_names), bc.target)
    )
    pima_path = "data/external/pima-diabetes.csv"
    if os.path.exists(pima_path):
        pima = pd.read_csv(pima_path)
        target = "Outcome" if "Outcome" in pima.columns else pima.columns[-1]
        control_results.append(
            precision_control("Pima Indians Diabetes",
                              pima.drop(columns=[target]), pima[target])
        )

    # Persist + markdown summary
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "leak_impact.json"), "w") as f:
        json.dump({"impact": impact_results, "controls": control_results}, f, indent=2)

    print(f"\n\n{'#'*70}\n  SUMMARY — Random Forest before/after (real organic leaks)\n{'#'*70}")
    print(f"{'Dataset':<34}{'Leak kind':<24}{'Acc':>8}{'->':>4}{'Acc':>7}{'AUC':>8}{'->':>4}{'AUC':>7}")
    md = ["| Dataset | Leak | Acc (with leak) | Acc (removed) | AUC (with leak) | AUC (removed) |",
          "|---|---|---|---|---|---|"]
    for r in impact_results:
        b, a = r["before"]["RandomForest"], r["after"]["RandomForest"]
        print(f"{r['name']:<34}{r['leak_kind'][:23]:<24}"
              f"{b['acc']:>8.3f}{'->':>4}{a['acc']:>7.3f}{b['auc']:>8.3f}{'->':>4}{a['auc']:>7.3f}")
        md.append(f"| {r['name']} | {r['leak_kind']} | {b['acc']:.3f} | {a['acc']:.3f} | {b['auc']:.3f} | {a['auc']:.3f} |")

    with open(os.path.join(RESULTS_DIR, "leak_impact_table.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"\nSaved: {RESULTS_DIR}/leak_impact.json  and  {RESULTS_DIR}/leak_impact_table.md")

    ctrl_ok = all(c["pass"] for c in control_results)
    print(f"\nPrecision controls: {'ALL PASS' if ctrl_ok else 'SOME FAILED'}")


if __name__ == "__main__":
    main()
