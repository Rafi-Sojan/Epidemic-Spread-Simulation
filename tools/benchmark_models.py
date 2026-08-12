"""Benchmark the trained epidemic prediction models on the held-out test set."""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


ROOT = Path(__file__).resolve().parents[1]
TEST_PATH = ROOT / "results" / "test.csv"
TRAIN_PATH = ROOT / "results" / "train.csv"
MODEL_DIR = ROOT / "results" / "models"
OUTPUT_DIR = ROOT / "results" / "research"

FEATURE_COLUMNS = [
    "population",
    "initial_infected",
    "infection_rate",
    "recovery_rate",
    "mortality_rate",
    "lockdown_strength",
    "mask_adoption",
    "vaccination_rate",
    "travel_restriction",
    "median_age",
    "elderly_population_ratio",
    "child_population_ratio",
    "temperature_celsius",
    "humidity_percent",
    "rainfall_mm",
    "days",
]


def rmse(actual: pd.Series, predicted) -> float:
    return mean_squared_error(actual, predicted) ** 0.5


def main() -> None:
    if not TEST_PATH.exists() or not TRAIN_PATH.exists():
        raise FileNotFoundError("Run split.py before benchmarking the models.")

    test_data = pd.read_csv(TEST_PATH)
    train_data = pd.read_csv(TRAIN_PATH)
    features = test_data[FEATURE_COLUMNS]
    rows: list[dict[str, object]] = []

    classifier = joblib.load(MODEL_DIR / "severity_classifier.joblib")
    peak_model = joblib.load(MODEL_DIR / "peak_infected_regressor.joblib")
    death_model = joblib.load(MODEL_DIR / "total_deaths_regressor.joblib")

    severity_predictions = classifier.predict(features)
    majority = DummyClassifier(strategy="most_frequent").fit(
        train_data[FEATURE_COLUMNS], train_data["severity"]
    )
    majority_predictions = majority.predict(features)
    accuracy = accuracy_score(test_data["severity"], severity_predictions)
    majority_accuracy = accuracy_score(test_data["severity"], majority_predictions)
    rows.extend(
        [
            {"category": "classification", "metric": "accuracy", "value": accuracy},
            {
                "category": "classification",
                "metric": "balanced_accuracy",
                "value": balanced_accuracy_score(test_data["severity"], severity_predictions),
            },
            {
                "category": "classification",
                "metric": "macro_f1",
                "value": f1_score(test_data["severity"], severity_predictions, average="macro"),
            },
            {
                "category": "classification",
                "metric": "weighted_f1",
                "value": f1_score(test_data["severity"], severity_predictions, average="weighted"),
            },
            {
                "category": "classification",
                "metric": "majority_baseline_accuracy",
                "value": majority_accuracy,
            },
            {
                "category": "classification",
                "metric": "accuracy_lift_percentage_points",
                "value": (accuracy - majority_accuracy) * 100.0,
            },
        ]
    )

    regression_specs = [
        ("peak_infected", peak_model),
        ("total_deaths", death_model),
    ]
    for target, model in regression_specs:
        predictions = model.predict(features)
        baseline = DummyRegressor(strategy="mean").fit(
            train_data[FEATURE_COLUMNS], train_data[target]
        )
        baseline_predictions = baseline.predict(features)
        rows.extend(
            [
                {"category": target, "metric": "mae", "value": mean_absolute_error(test_data[target], predictions)},
                {"category": target, "metric": "rmse", "value": rmse(test_data[target], predictions)},
                {"category": target, "metric": "r2", "value": r2_score(test_data[target], predictions)},
                {
                    "category": target,
                    "metric": "mean_baseline_mae",
                    "value": mean_absolute_error(test_data[target], baseline_predictions),
                },
            ]
        )

    inference_input = features.iloc[[0]]
    start = time.perf_counter()
    for _ in range(100):
        classifier.predict(inference_input)
        peak_model.predict(inference_input)
        death_model.predict(inference_input)
    elapsed_ms = (time.perf_counter() - start) * 1000.0 / 100.0
    rows.append(
        {
            "category": "inference",
            "metric": "mean_single_scenario_latency_ms",
            "value": elapsed_ms,
        }
    )

    model_size_mb = sum(
        path.stat().st_size
        for path in MODEL_DIR.glob("*.joblib")
    ) / (1024 * 1024)
    rows.append(
        {
            "category": "artifacts",
            "metric": "random_forest_models_size_mb",
            "value": model_size_mb,
        }
    )

    output = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_DIR / "benchmark_metrics.csv", index=False)
    (OUTPUT_DIR / "benchmark_metrics.json").write_text(
        json.dumps(rows, indent=2) + "\n",
        encoding="utf-8",
    )

    print("Model benchmark")
    print(output.to_string(index=False))
    print(f"\nSaved benchmark outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
