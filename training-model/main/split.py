from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = ROOT / "results" / "epidemic_dataset.csv"
TRAIN_PATH = ROOT / "results" / "train.csv"
TEST_PATH = ROOT / "results" / "test.csv"


def main() -> None:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"{DATASET_PATH} was not found. Run the C++ simulation generator first."
        )

    data = pd.read_csv(DATASET_PATH)
    train_data, test_data = train_test_split(
        data,
        test_size=0.2,
        random_state=42,
        stratify=data["severity"],
    )

    TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)
    train_data.to_csv(TRAIN_PATH, index=False)
    test_data.to_csv(TEST_PATH, index=False)

    print(f"Saved {len(train_data)} training rows to {TRAIN_PATH}")
    print(f"Saved {len(test_data)} testing rows to {TEST_PATH}")


if __name__ == "__main__":
    main()
