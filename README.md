# Hospital Readmission Prediction

An applied healthcare analytics project investigating which patient characteristics are associated with hospital readmission and how those signals could support more targeted follow-up care.

## Project overview

The analysis focuses on three practical questions:

1. Which primary diagnoses are most common across age groups?
2. Is a diabetes diagnosis associated with readmission patterns?
3. Which patient groups appear to have higher readmission risk and may benefit from additional follow-up?

## Repository structure

- [`hospital_readmissions.csv`](hospital_readmissions.csv) — analysis dataset.
- [`hospital_readmission_modeling.ipynb`](hospital_readmission_modeling.ipynb) — primary modelling notebook.
- [`hospital_readmission_exploration.ipynb`](hospital_readmission_exploration.ipynb) — earlier exploratory analysis retained for transparency.
- `hospital_1.jfif` through `hospital_5.jpg` — supporting presentation assets.
- [`.gitignore`](.gitignore) — excludes local Python/Jupyter artifacts.

## Dataset

The dataset contains patient-level variables such as age bracket, time in hospital, procedure and medication counts, previous outpatient/inpatient/emergency visits, medical specialty, diagnoses, glucose/A1C testing, diabetes medication status and readmission outcome.

## Analytical workflow

The notebooks explore readmission patterns, identify useful predictors and compare higher-risk patient groups. The project emphasizes both model-oriented analysis and translation of findings into operationally useful insights.

## Run locally

1. Clone the repository.
2. Create a Python/Jupyter environment with the packages imported by the notebooks.
3. Keep `hospital_readmissions.csv` in the repository root.
4. Start with `hospital_readmission_modeling.ipynb` and run the notebook sequentially.

## Responsible use

This is a portfolio and educational data-science project. Its outputs are not clinical guidance and should not be used for individual patient decisions without appropriate external validation, governance and medical oversight.
