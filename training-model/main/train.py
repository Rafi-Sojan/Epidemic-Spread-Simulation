from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = ROOT / "results" / "train.csv"
TEST_PATH = ROOT / "results" / "test.csv"
MODEL_DIR = ROOT / "results" / "models"

FEATURE_COLUMNS = [
    "population",
    "initial_infected",
    "infection_rate",
    "recovery_rate",
    "days",
]


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not TRAIN_PATH.exists() or not TEST_PATH.exists():
        raise FileNotFoundError("Split files were not found. Run split.py before train.py.")

    return pd.read_csv(TRAIN_PATH), pd.read_csv(TEST_PATH)


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), FEATURE_COLUMNS),
        ],
        remainder="drop",
    )


def train_classifier(train_data: pd.DataFrame, test_data: pd.DataFrame) -> Pipeline:
    model = Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=250,
                    max_depth=12,
                    random_state=42,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    model.fit(train_data[FEATURE_COLUMNS], train_data["severity"])
    predictions = model.predict(test_data[FEATURE_COLUMNS])

    print("\nSeverity classification")
    print(f"Accuracy: {accuracy_score(test_data['severity'], predictions):.3f}")
    print(classification_report(test_data["severity"], predictions))
    return model


def train_regressor(train_data: pd.DataFrame, test_data: pd.DataFrame) -> Pipeline:
    model = Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=250,
                    max_depth=14,
                    random_state=42,
                ),
            ),
        ]
    )

    model.fit(train_data[FEATURE_COLUMNS], train_data["peak_infected"])
    predictions = model.predict(test_data[FEATURE_COLUMNS])

    print("\nPeak infected regression")
    print(f"MAE: {mean_absolute_error(test_data['peak_infected'], predictions):.2f}")
    print(f"R2: {r2_score(test_data['peak_infected'], predictions):.3f}")
    return model


def main() -> None:
    train_data, test_data = load_data()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    classifier = train_classifier(train_data, test_data)
    regressor = train_regressor(train_data, test_data)

    classifier_path = MODEL_DIR / "severity_classifier.joblib"
    regressor_path = MODEL_DIR / "peak_infected_regressor.joblib"
    joblib.dump(classifier, classifier_path)
    joblib.dump(regressor, regressor_path)

    print(f"\nSaved classifier to {classifier_path}")
    print(f"Saved regressor to {regressor_path}")


if __name__ == "__main__":
    main()
