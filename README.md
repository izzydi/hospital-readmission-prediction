# Hospital Readmission Prediction

An applied healthcare analytics project investigating patient-level signals associated with hospital readmission.

## Primary workflow

[`src/hospital_readmission_model.py`](src/hospital_readmission_model.py) is the audited implementation. It detects the readmission target, creates a stratified hold-out split, identifies numeric and categorical predictors from training data, and places imputation/encoding/scaling inside scikit-learn pipelines so every learned transformation remains inside cross-validation folds.

The workflow tunes a balanced Logistic Regression baseline and a balanced Random Forest using training data only, then reports accuracy, balanced accuracy and weighted F1 on the untouched test set. Metrics are written to `outputs/metrics.json`.

## Repository structure

- [`src/hospital_readmission_model.py`](src/hospital_readmission_model.py) — audited modelling pipeline.
- [`tests/test_smoke.py`](tests/test_smoke.py) — lightweight tests for target detection and metric reporting.
- [`.github/workflows/ci.yml`](.github/workflows/ci.yml) — automated Python 3.12 CI.
- [`hospital_readmissions.csv`](hospital_readmissions.csv) — 25,000-row teaching/project derivative described below.
- [`requirements.txt`](requirements.txt) — pinned Python dependencies used by CI.
- [`archive/`](archive/) — historical modelling/exploration notebooks retained for provenance.
- `hospital_1.jfif` through `hospital_5.jpg` — supporting historical presentation assets.

## Dataset provenance

The committed 25,000-row, 17-column table is a reduced/recoded teaching derivative of the **Diabetes 130-US Hospitals for Years 1999–2008** dataset. The authoritative source dataset is maintained by the UCI Machine Learning Repository:

- UCI dataset: https://archive.ics.uci.edu/dataset/296/diabetes+130+us+hospitals+for+years+1999+2008
- DOI: https://doi.org/10.24432/C5230J
- Original creators: John Clore, Krzysztof Cios, Jon DeShazo and Beata Strack
- UCI license: CC BY 4.0

The UCI source contains 101,766 encounters and a substantially larger original feature set. This repository's 25,000-row/17-column CSV uses grouped/reduced fields such as `diag_1`–`diag_3`, `glucose_test`, `A1Ctest`, `diabetes_med` and a binary `readmitted` outcome. The exact historical sampling/recoding recipe that produced this teaching derivative is **not documented in the repository**, so the project does not claim that `hospital_readmissions.csv` can be reconstructed bit-for-bit from the UCI download. The original UCI source should be cited when discussing the underlying clinical dataset, while this committed derivative should be treated as the exact modelling input for this portfolio workflow.

## Validation design

The train/test split happens before model fitting. Imputation, scaling and one-hot encoding are part of each scikit-learn `Pipeline`, so they are refitted independently inside every cross-validation fold during hyperparameter tuning. The held-out test partition is used only after tuning is complete.

## Reproducibility and CI

The Python dependency versions are pinned in [`requirements.txt`](requirements.txt). GitHub Actions creates a clean Python 3.12 environment, installs those exact dependencies, compiles the source and runs the smoke-test suite on every push and pull request.

The historical notebooks remain under `archive/` for provenance, but the smaller `src/` script is the canonical implementation and the target of automated checks.

## Run locally

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m unittest discover -s tests -v
python src/hospital_readmission_model.py
```

## Responsible use

This is a portfolio and educational data-science project. Its outputs are **not clinical guidance** and should not be used for individual patient decisions without appropriate external validation, governance and medical oversight.
