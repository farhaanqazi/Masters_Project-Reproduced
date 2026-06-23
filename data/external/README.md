# External Validation Datasets

These datasets are **not redistributed** in this repo (licensing / size). Download each
from its source and save it under `data/external/` with the filename below, then run:

```bash
python -m src.validate_detector
```

All leaks listed are **organic** (present in the published data) — nothing is injected or fabricated.

## Positive cases (contain real leaks)

| Save as | Source | Target | Leaky column(s) | Leak type |
|---|---|---|---|---|
| `bank-marketing.csv` | UCI ID 222 — Bank Marketing (`bank-additional-full.csv`) | `y` | `duration` | correlation — **provider-documented** ("discard for realistic models") |
| `weatherAUS.csv` | Kaggle — Rain in Australia | `RainTomorrow` | `RISK_MM` | target leak — next-day rainfall (the answer) |
| `heart_failure.csv` | UCI ID 519 — Heart Failure Clinical Records | `DEATH_EVENT` | `time` | correlation — follow-up window; analogue of Parkinson's `Duration` |
| `cervical_cancer.csv` | UCI ID 383 — Cervical Cancer (Risk Factors) | `Biopsy` | `Hinselmann`, `Schiller`, `Citology` | target leak — alternate diagnoses (has `?` missing values) |

## Negative controls (clean — precision tests)

| Source | Notes |
|---|---|
| `sklearn.datasets.load_breast_cancer` (no download) | legit strong features must NOT be hard-flagged |
| `pima-diabetes.csv` (optional) | misleading zeros (glucose/BMI = 0 ⇒ missing) must NOT be flagged as degenerate |

## Notes
- The detector requires **binary** targets — all targets above are binary (encode yes/no → 1/0).
- Only `Parkinson's Duration` trips the hard **degeneracy** flag; the others surface as
  top-ranked **suspicious** features (correlation/target leaks) for human review.
