# Data Leakage Detector: Implementation Plan

This project began as a faithful reproduction of my Master's thesis on Parkinson's disease classification. During reproduction, the originally reported modest performance (~72–76% in the 2023 thesis) was contrasted against a "too good to be true" 1.0 AUC that appeared once the raw `Duration` column was naively included — exposing a hidden within-class degeneracy leak (`Duration = 0.0` for all Healthy Controls).

That forensic discovery is what birthed the idea for a general-purpose automated **Data Leakage Detector** for tabular datasets. The project's *shape* is therefore the detector, not the thesis reproduction — and a tool that claims to generalize must be proven on **more than one real leak**.

## Goal Description
Build a standalone leakage-detection module that scores features for suspicious label-predictiveness prior to model training, and prove it on **multiple real-world datasets that contain genuine, documented leakage** — no injected or fabricated leaks. For each leaky dataset we demonstrate the full loop: **detect → review → remove → measure the honest before/after impact on model performance.**

The detector is validated as a ranked decision-support tool. The hardest challenge for a leakage detector is avoiding false positives on legitimately strong predictors, so the suite also includes clean datasets as hard-negative precision controls.

## Real-World Datasets (No Injection)

All positive cases use **organic leaks already present in published datasets**. The user downloads each file manually and places it under `data/external/`. Expected leaky columns are listed for verification, but actual before/after numbers are computed by the pipeline — none are assumed.

### Positive cases (contain real leaks)

1. **Parkinson's Disease (primary — already in repo: `data/raw/dataset.csv`)**
   - Target: `PC`. Leaky feature: `Duration`.
   - Leak type: **within-class degeneracy** — `Duration = 0.0` for every Healthy Control (zero variance in class 0). Caught by the detector's hard degeneracy flag. This is the only *hard-flag* case in the suite.

2. **Bank Marketing (download → `data/external/bank-marketing.csv`)**
   - Source: UCI ML Repository (ID 222). File: `bank-additional-full.csv` or `bank-full.csv`.
   - Target: `y` (yes/no → encode 0/1). Leaky feature: `duration` (last-contact call length).
   - Leak type: **extreme univariate correlation**, *documented by the data providers themselves* — the UCI page states `duration` should be discarded for any realistic model (e.g. `duration=0` ⇒ `y='no'`). Strongest "real-world proof": cite the provider's own warning.

3. **Rain in Australia (download → `data/external/weatherAUS.csv`)**
   - Source: Kaggle "Rain in Australia" (`weatherAUS.csv`).
   - Target: `RainTomorrow` (Yes/No). Leaky feature: `RISK_MM` (mm of rain the *next* day — i.e. the answer itself).
   - Leak type: **extreme univariate / target leakage**, documented in the dataset description (drop `RISK_MM`).

4. **Heart Failure Clinical Records (download → `data/external/heart_failure.csv`)**
   - Source: UCI ML Repository (ID 519) / Chicco & Jurman 2020; also on Kaggle.
   - 299 rows, 13 columns. Target: `DEATH_EVENT`. Leaky feature: `time` (follow-up observation window).
   - Leak type: **extreme univariate correlation** — direct analogue of the Parkinson's `Duration` bug (an observation-time column leaking a medical outcome).

5. **Cervical Cancer (Risk Factors) (download → `data/external/cervical_cancer.csv`)**
   - Source: UCI ML Repository (ID 383). Contains `?` missing values requiring cleaning.
   - 858 rows, 36 columns. Target: `Biopsy`. Leaky features: `Hinselmann`, `Schiller`, `Citology`.
   - Leak type: **target leakage** — alternative diagnostic tests for the same outcome, only available at/after diagnosis.

### Negative controls (clean, no leak — precision / false-positive tests)

6. **Breast Cancer Wisconsin (`sklearn.datasets.load_breast_cancer` — no download)**
   - Proves **precision**: legitimately strong clinical markers (e.g. `worst perimeter`, AUC > 0.97) must rank high but **must NOT trigger the degeneracy hard-flag**.

7. **Pima Indians Diabetes (optional download → `data/external/pima-diabetes.csv`)**
   - Edge case: misleading physiological zeros (glucose/BMI = 0 means *missing*, not degenerate). Confirms the detector does not over-flag zero-heavy columns as leaks.

## Proposed Changes

### Core Leakage Detector Module — `src/leakage_detector.py` (exists)
- `DataLeakageDetector`: computes per-feature univariate ROC-AUC, KS-statistic, and a within-class degeneracy hard-flag; returns a ranked decision-support table. Binary classification only (already enforced). No changes expected beyond minor robustness (e.g. coercing/encoding non-numeric columns where needed for the cervical-cancer file).

### Leak-Impact (Before/After) Pipeline — `src/validate_detector.py` (extend)
This is the Option B work: a generic, reusable function that quantifies the *consequence* of each real leak, not just its detection.

- **`leak_impact_report(X, y, leak_cols, dataset_name)`**:
  1. Runs the detector and confirms `leak_cols` appear at the top of the ranking (degeneracy flag where applicable, otherwise top predictiveness).
  2. Trains a fixed, small model set (e.g. Random Forest, Logistic Regression, Decision Tree) **with** the leaky column(s) and **without** them, using identical splits/seed.
  3. Emits a per-dataset before/after table (Accuracy + AUC, with the leaky columns present vs removed).
- **Driver** runs this for the three positive cases and runs the existing detection-only precision check for Breast Cancer.
- **No injection code** — the previous synthetic `INJECTED_*` leaks are removed. Leaks are read from real data only.

### Documentation & Reporting

#### [MODIFY] README.md
- Replace the single Parkinson's before/after with a **cross-dataset leak-impact table** (one row per real leaky dataset), each labelled with its leak type (degeneracy vs correlation/target leak) and its real source.
- State plainly that all leaks are **organic and from published datasets**; none are injected.
- Keep the explicit scope declaration: the tool catches univariate target-correlation and within-class degeneracy only — not temporal, train/test-contamination, or multivariate leakage.
- Add the N-provenance caveat (the original thesis used 58 patients; the reproduction CSV has 325 rows) so comparisons are not read as apples-to-apples.

#### [MODIFY] docs/REPRODUCIBILITY.md, docs/DATA.md
- Document each external dataset's source, version, access procedure, and known leak under `data/external/`.

#### [MODIFY] docs/TRIPOD-AI_checklist.md
- Reflect that the leakage audit is run across multiple real datasets, not just the primary one.

## Verification Plan

### Automated / Programmatic Verification
- **Re-catching Duration:** detector MUST hard-flag `Duration` in the PD dataset via within-class degeneracy.
- **Heart Failure:** detector MUST rank `time` at/near the top by predictiveness; before/after MUST show a measurable drop in Accuracy/AUC when `time` is removed.
- **Cervical Cancer:** detector MUST rank `Hinselmann`/`Schiller`/`Citology` at the top; before/after MUST show a drop when they are removed.
- **Breast Cancer (precision):** NO degeneracy hard-flags on the clean dataset.
- **Honesty constraint:** every before/after number is produced by the committed pipeline from the downloaded files — no hand-entered or assumed metrics.

### Manual Steps (User)
- Download the two external CSVs to `data/external/` per the descriptions above before running `python -m src.validate_detector`.
