# Historical notebooks

The two notebooks in this directory are preserved as historical exploratory artifacts:

- `hospital_readmission_modeling.ipynb`
- `hospital_readmission_exploration.ipynb`

Because these large notebooks cannot be reliably re-executed and audited cell-by-cell through the current repository connection, they are not presented as the canonical implementation.

Use [`../src/hospital_readmission_model.py`](../src/hospital_readmission_model.py) for the audited workflow. It performs preprocessing inside cross-validation pipelines, uses a stratified held-out test set and reports reproducible evaluation metrics.
