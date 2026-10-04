# Hospital Readmission Prediction

An applied healthcare analytics project investigating patient-level signals associated with hospital readmission.

## Primary workflow

[`src/hospital_readmission_model.py`](src/hospital_readmission_model.py) is the audited implementation. It detects the readmission target, creates a stratified hold-out split, identifies numeric and categorical predictors from training data, and places imputation/encoding/scaling inside scikit-learn pipelines so every transformation is learned correctly within cross-validation folds.

The workflow tunes a balanced logistic-regression baseline and a balanced Random Forest using training data only, then reports accuracy, balanced accuracy and weighted F1 on the untouched test set. Metrics are written to `outputs/metrics.json`.

## Repository structure

- [`src/hospital_readmission_model.py`](src/hospital_readmission_model.py) — audited modelling pipeline.
- [`hospital_readmissions.csv`](hospital_readmissions.csv) — project dataset.
- [`requirements.txt`](requirements.txt) — direct Python dependencies.
- [`archive/`](archive/) — historical modelling/exploration notebooks retained for provenance.
- `hospital_1.jfif` through `hospital_5.jpg` — supporting historical presentation assets.

## Dataset

The dataset includes patient-level variables such as age bracket, time in hospital, procedure and medication counts, prior outpatient/inpatient/emergency visits, medical specialty, diagnoses, glucose/A1C testing, diabetes-medication status and a readmission outcome.

## Run locally

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python src/hospital_readmission_model.py
```

## Historical notebooks

The original notebooks are retained under [`archive/`](archive/) but are no longer presented as the canonical implementation. The cleaned script is intentionally smaller, easier to review and explicit about evaluation boundaries.

## Responsible use

This is a portfolio and educational data-science project. Its outputs are **not clinical guidance** and should not be used for individual patient decisions without appropriate external validation, governance and medical oversight.
