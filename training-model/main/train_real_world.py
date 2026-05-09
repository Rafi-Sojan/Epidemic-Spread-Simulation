from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "results" / "real_world"
MODEL_DIR = ROOT / "results" / "models"
OWID_URL = "https://covid.ourworldindata.org/data/owid-covid-data.csv"
OWID_GITHUB_URL = (
    "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv"
)
OWID_LOCAL_PATH = DATA_DIR / "owid-covid-data.csv"
METRICS_PATH = DATA_DIR / "real_world_metrics.csv"

FEATURE_COLUMNS = [
    "new_cases_smoothed_per_million",
    "new_deaths_smoothed_per_million",
    "people_vaccinated_per_hundred",
    "people_fully_vaccinated_per_hundred",
    "total_boosters_per_hundred",
    "stringency_index",
    "population_density",
    "median_age",
    "aged_65_older",
    "hospital_beds_per_thousand",
]

TARGET_COLUMNS = [
    "target_new_cases_per_million",
    "target_new_deaths_per_million",
]


def load_owid_data() -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if OWID_LOCAL_PATH.exists():
        return pd.read_csv(OWID_LOCAL_PATH, parse_dates=["date"])

    try:
        data = pd.read_csv(OWID_URL, parse_dates=["date"])
    except Exception:
        data = pd.read_csv(OWID_GITHUB_URL, parse_dates=["date"])

    data.to_csv(OWID_LOCAL_PATH, index=False)
    return data


def prepare_model_data(data: pd.DataFrame) -> pd.DataFrame:
    filtered = data[
        data["continent"].notna()
        & data["population"].notna()
        & data["new_cases_smoothed_per_million"].notna()
        & data["new_deaths_smoothed_per_million"].notna()
    ].copy()

    filtered = filtered.sort_values(["location", "date"])
    filtered["target_new_cases_per_million"] = filtered.groupby("location")[
        "new_cases_smoothed_per_million"
    ].shift(-7)
    filtered["target_new_deaths_per_million"] = filtered.groupby("location")[
        "new_deaths_smoothed_per_million"
    ].shift(-7)

    for column in FEATURE_COLUMNS:
        filtered[column] = filtered.groupby("location")[column].ffill()

    filtered[FEATURE_COLUMNS] = filtered[FEATURE_COLUMNS].fillna(0)
    filtered = filtered.dropna(subset=TARGET_COLUMNS)
    return filtered


def temporal_split(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_date = data["date"].quantile(0.80)
    train_data = data[data["date"] <= split_date].copy()
    test_data = data[data["date"] > split_date].copy()
    return train_data, test_data


def main() -> None:
    data = load_owid_data()
    model_data = prepare_model_data(data)
    train_data, test_data = temporal_split(model_data)

    model = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            (
                "model",
                MultiOutputRegressor(
                    RandomForestRegressor(
                        n_estimators=250,
                        max_depth=18,
                        min_samples_leaf=3,
                        random_state=42,
                        n_jobs=-1,
                    )
                ),
            ),
        ]
    )

    model.fit(train_data[FEATURE_COLUMNS], train_data[TARGET_COLUMNS])
    predictions = model.predict(test_data[FEATURE_COLUMNS])

    metrics = pd.DataFrame(
        [
            {
                "target": "new_cases_per_million_7_days_ahead",
                "model": "random_forest",
                "mae": mean_absolute_error(test_data[TARGET_COLUMNS[0]], predictions[:, 0]),
                "r2": r2_score(test_data[TARGET_COLUMNS[0]], predictions[:, 0]),
                "train_rows": len(train_data),
                "test_rows": len(test_data),
                "locations": model_data["location"].nunique(),
                "test_start": test_data["date"].min().date(),
                "test_end": test_data["date"].max().date(),
            },
            {
                "target": "new_deaths_per_million_7_days_ahead",
                "model": "random_forest",
                "mae": mean_absolute_error(test_data[TARGET_COLUMNS[1]], predictions[:, 1]),
                "r2": r2_score(test_data[TARGET_COLUMNS[1]], predictions[:, 1]),
                "train_rows": len(train_data),
                "test_rows": len(test_data),
                "locations": model_data["location"].nunique(),
                "test_start": test_data["date"].min().date(),
                "test_end": test_data["date"].max().date(),
            },
            {
                "target": "new_cases_per_million_7_days_ahead",
                "model": "persistence_baseline",
                "mae": mean_absolute_error(
                    test_data[TARGET_COLUMNS[0]], test_data["new_cases_smoothed_per_million"]
                ),
                "r2": r2_score(test_data[TARGET_COLUMNS[0]], test_data["new_cases_smoothed_per_million"]),
                "train_rows": len(train_data),
                "test_rows": len(test_data),
                "locations": model_data["location"].nunique(),
                "test_start": test_data["date"].min().date(),
                "test_end": test_data["date"].max().date(),
            },
            {
                "target": "new_deaths_per_million_7_days_ahead",
                "model": "persistence_baseline",
                "mae": mean_absolute_error(
                    test_data[TARGET_COLUMNS[1]], test_data["new_deaths_smoothed_per_million"]
                ),
                "r2": r2_score(test_data[TARGET_COLUMNS[1]], test_data["new_deaths_smoothed_per_million"]),
                "train_rows": len(train_data),
                "test_rows": len(test_data),
                "locations": model_data["location"].nunique(),
                "test_start": test_data["date"].min().date(),
                "test_end": test_data["date"].max().date(),
            },
        ]
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "real_world_covid_forecaster.joblib"
    joblib.dump(
        {
            "model": model,
            "feature_columns": FEATURE_COLUMNS,
            "target_columns": TARGET_COLUMNS,
            "source": OWID_URL,
            "fallback_source": OWID_GITHUB_URL,
        },
        model_path,
    )
    metrics.to_csv(METRICS_PATH, index=False)

    print("Real-world OWID COVID-19 evaluation")
    print(f"Rows: train={len(train_data)}, test={len(test_data)}")
    print(f"Locations: {model_data['location'].nunique()}")
    print(f"Test period: {test_data['date'].min().date()} to {test_data['date'].max().date()}")
    print(metrics[["target", "model", "mae", "r2"]].to_string(index=False))
    print(f"\nSaved model to {model_path}")
    print(f"Saved metrics to {METRICS_PATH}")


if __name__ == "__main__":
    main()
