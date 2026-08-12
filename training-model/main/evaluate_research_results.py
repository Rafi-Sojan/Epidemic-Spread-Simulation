from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


ROOT = Path(__file__).resolve().parents[2]
TEST_PATH = ROOT / "results" / "test.csv"
MODEL_DIR = ROOT / "results" / "models"
RESEARCH_DIR = ROOT / "results" / "research"

CLASSIFIER_PATH = MODEL_DIR / "severity_classifier.joblib"
PEAK_REGRESSOR_PATH = MODEL_DIR / "peak_infected_regressor.joblib"
DEATH_REGRESSOR_PATH = MODEL_DIR / "total_deaths_regressor.joblib"

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

CLASS_LABELS = ["low", "medium", "high"]


def save_confusion_matrix(test_data: pd.DataFrame, predictions: pd.Series) -> None:
    matrix = confusion_matrix(test_data["severity"], predictions, labels=CLASS_LABELS)
    matrix_frame = pd.DataFrame(matrix, index=CLASS_LABELS, columns=CLASS_LABELS)
    matrix_frame.to_csv(RESEARCH_DIR / "severity_confusion_matrix.csv")

    plt.figure(figsize=(7, 5.5))
    sns.heatmap(
        matrix_frame,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        linewidths=0.5,
        linecolor="white",
    )
    plt.title("Severity Classification Confusion Matrix")
    plt.xlabel("Predicted Severity")
    plt.ylabel("Actual Severity")
    plt.tight_layout()
    plt.savefig(RESEARCH_DIR / "severity_confusion_matrix.png", dpi=300)
    plt.close()

def save_classification_results(test_data: pd.DataFrame, predictions: pd.Series) -> None:
    report_dict = classification_report(
        test_data["severity"],
        predictions,
        labels=CLASS_LABELS,
        output_dict=True,
        zero_division=0,
    )
    report_frame = pd.DataFrame(report_dict).transpose()
    report_frame.to_csv(RESEARCH_DIR / "severity_classification_report.csv")

    summary = pd.DataFrame(
        [
            {
                "metric": "accuracy",
                "value": accuracy_score(test_data["severity"], predictions),
            },
            {
                "metric": "test_rows",
                "value": len(test_data),
            },
        ]
    )
    summary.to_csv(RESEARCH_DIR / "severity_summary_metrics.csv", index=False)


def save_regression_results(test_data: pd.DataFrame) -> None:
    peak_model = joblib.load(PEAK_REGRESSOR_PATH)
    death_model = joblib.load(DEATH_REGRESSOR_PATH)
    features = test_data[FEATURE_COLUMNS]

    peak_predictions = peak_model.predict(features)
    death_predictions = death_model.predict(features)

    metrics = pd.DataFrame(
        [
            {
                "target": "peak_infected",
                "mae": mean_absolute_error(test_data["peak_infected"], peak_predictions),
                "rmse": mean_squared_error(
                    test_data["peak_infected"],
                    peak_predictions,
                ) ** 0.5,
                "r2": r2_score(test_data["peak_infected"], peak_predictions),
            },
            {
                "target": "total_deaths",
                "mae": mean_absolute_error(test_data["total_deaths"], death_predictions),
                "rmse": mean_squared_error(
                    test_data["total_deaths"],
                    death_predictions,
                ) ** 0.5,
                "r2": r2_score(test_data["total_deaths"], death_predictions),
            },
        ]
    )
    metrics.to_csv(RESEARCH_DIR / "regression_metrics.csv", index=False)

    prediction_frame = pd.DataFrame(
        {
            "actual_peak_infected": test_data["peak_infected"],
            "predicted_peak_infected": peak_predictions,
            "actual_total_deaths": test_data["total_deaths"],
            "predicted_total_deaths": death_predictions,
        }
    )
    prediction_frame.to_csv(RESEARCH_DIR / "regression_predictions.csv", index=False)

    for actual_column, predicted_column, title, filename in [
        (
            "actual_peak_infected",
            "predicted_peak_infected",
            "Peak Infected: Actual vs Predicted",
            "peak_infected_actual_vs_predicted.png",
        ),
        (
            "actual_total_deaths",
            "predicted_total_deaths",
            "Total Deaths: Actual vs Predicted",
            "total_deaths_actual_vs_predicted.png",
        ),
    ]:
        plt.figure(figsize=(6, 6))
        sns.scatterplot(
            data=prediction_frame,
            x=actual_column,
            y=predicted_column,
            alpha=0.75,
            edgecolor=None,
        )
        axis_min = min(prediction_frame[actual_column].min(), prediction_frame[predicted_column].min())
        axis_max = max(prediction_frame[actual_column].max(), prediction_frame[predicted_column].max())
        plt.plot([axis_min, axis_max], [axis_min, axis_max], color="red", linestyle="--", linewidth=1)
        plt.title(title)
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.tight_layout()
        plt.savefig(RESEARCH_DIR / filename, dpi=300)
        plt.close()


def main() -> None:
    if not TEST_PATH.exists():
        raise FileNotFoundError("results/test.csv was not found. Run split.py first.")
    for model_path in [CLASSIFIER_PATH, PEAK_REGRESSOR_PATH, DEATH_REGRESSOR_PATH]:
        if not model_path.exists():
            raise FileNotFoundError(f"{model_path} was not found. Run train.py first.")

    RESEARCH_DIR.mkdir(parents=True, exist_ok=True)
    test_data = pd.read_csv(TEST_PATH)
    classifier = joblib.load(CLASSIFIER_PATH)
    severity_predictions = classifier.predict(test_data[FEATURE_COLUMNS])

    save_confusion_matrix(test_data, severity_predictions)
    save_classification_results(test_data, severity_predictions)
    save_regression_results(test_data)

    print(f"Saved research results to {RESEARCH_DIR}")
    print(f"Accuracy: {accuracy_score(test_data['severity'], severity_predictions):.3f}")
    print(classification_report(test_data["severity"], severity_predictions, zero_division=0))


if __name__ == "__main__":
    main()
