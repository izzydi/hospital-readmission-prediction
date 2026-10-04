"""Leakage-safe baseline models for hospital readmission prediction.

The preprocessing pipeline is fitted inside cross-validation folds. The held-out test
set is used once for final evaluation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 1821
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "hospital_readmissions.csv"
OUTPUT_DIR = ROOT / "outputs"


def find_target(columns: list[str]) -> str:
    candidates = {c.lower(): c for c in columns}
    for name in ("readmitted", "readmission", "readmit"):
        if name in candidates:
            return candidates[name]
    raise ValueError(
        "Could not identify the readmission target. Expected a column such as 'readmitted'."
    )


def score(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
    }


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing dataset: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed")].copy()
    target = find_target(df.columns.tolist())

    df = df.dropna(subset=[target])
    x = df.drop(columns=target)
    y = df[target].astype(str)

    if y.nunique() < 2:
        raise ValueError("The readmission target must contain at least two classes.")

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.25,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    numeric_columns = x_train.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = [c for c in x_train.columns if c not in numeric_columns]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ]
    )

    logistic = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "model",
                LogisticRegression(
                    max_iter=3000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    logistic_search = GridSearchCV(
        logistic,
        param_grid={"model__C": [0.1, 1.0, 10.0]},
        scoring="balanced_accuracy",
        cv=5,
        n_jobs=-1,
    )

    forest = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=600,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    forest_search = GridSearchCV(
        forest,
        param_grid={
            "model__max_depth": [None, 8, 16],
            "model__min_samples_leaf": [1, 3, 8],
        },
        scoring="balanced_accuracy",
        cv=5,
        n_jobs=-1,
    )

    results: dict[str, dict[str, object]] = {}
    for name, search in (("logistic_regression", logistic_search), ("random_forest", forest_search)):
        search.fit(x_train, y_train)
        predictions = search.predict(x_test)
        results[name] = {
            "best_cv_balanced_accuracy": float(search.best_score_),
            "best_params": search.best_params_,
            "held_out_test": score(y_test, predictions),
        }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "target": target,
        "rows": int(len(df)),
        "classes": sorted(y.unique().tolist()),
        "evaluation": results,
    }
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
