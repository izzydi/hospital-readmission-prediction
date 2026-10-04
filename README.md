# Reducing Hospital Readmissions with Machine Learning

An applied healthcare analytics project investigating which patient characteristics are associated with hospital readmission and how those signals could help prioritize follow-up care.

## Business problem

The project is framed around three practical questions:

1. Which primary diagnoses are most common across age groups?
2. Is a diabetes diagnosis associated with readmission patterns?
3. Which patient groups appear most suitable for targeted follow-up because of higher readmission risk?

## Repository contents

- [`hospital_readmissions.csv`](hospital_readmissions.csv) — analysis dataset.
- [`readmissions_python_v02.ipynb`](readmissions_python_v02.ipynb) — primary Python notebook.
- [`notebook_git.ipynb`](notebook_git.ipynb) — additional/earlier notebook version.
- `hospital_1.jfif` through `hospital_5.jpg` — supporting presentation images.

## Dataset

The dataset contains patient-level information such as age bracket, time in hospital, procedure counts, medication counts, previous outpatient/inpatient/emergency visits, medical specialty, diagnoses, glucose/A1C testing, diabetes medication status and readmission outcome.

## Analytical workflow

The notebooks explore the data, examine readmission patterns and apply data-science methods to identify useful predictors and higher-risk patient groups. The emphasis is not only on model output but on translating the analysis into operational recommendations for follow-up care.

## Reproducing the analysis

1. Clone the repository.
2. Create a Python/Jupyter environment with the packages imported by the notebooks.
3. Keep `hospital_readmissions.csv` in the repository root or update any data path used by the notebook.
4. Run `readmissions_python_v02.ipynb` sequentially.

## Scope and responsible use

This project is an educational/application-focused data-science analysis. Its outputs should not be used as clinical guidance or for individual patient decisions without appropriate validation, governance and medical oversight.
