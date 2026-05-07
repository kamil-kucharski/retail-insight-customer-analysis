import os
import warnings

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
os.environ.setdefault("OMP_NUM_THREADS", "1")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="threadpoolctl")

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = ["Recency", "Frequency", "Monetary", "AverageOrderValue"]


def build_high_value_target(segmented_rfm: pd.DataFrame) -> pd.DataFrame:
    dataset = segmented_rfm.copy()
    dataset["IsHighValue"] = (dataset["Segment"] == "High Value").astype(int)
    return dataset


def train_high_value_models(
    dataset: pd.DataFrame,
    test_size: float = 0.25,
    random_state: int = 42,
) -> dict:
    """Train simple baseline classifiers for the High Value customer label."""
    x = dataset[FEATURE_COLUMNS]
    y = dataset["IsHighValue"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    models = {
        "logistic_regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000, random_state=random_state)),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            random_state=random_state,
            class_weight="balanced",
        ),
    }

    results = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        results[name] = {
            "model": model,
            "classification_report": classification_report(
                y_test,
                predictions,
                output_dict=True,
                zero_division=0,
            ),
            "confusion_matrix": confusion_matrix(y_test, predictions),
        }

    return results
