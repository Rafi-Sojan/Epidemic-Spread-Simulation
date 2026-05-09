from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ModuleNotFoundError as error:
    raise ModuleNotFoundError(
        "PyTorch is required for LSTM training. Install it with `pip install -r requirements.txt`."
    ) from error


ROOT = Path(__file__).resolve().parents[2]
DAILY_PATH = ROOT / "results" / "daily_counts.csv"
MODEL_DIR = ROOT / "results" / "models"


class DailyInfectionLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(features)
        return self.output(hidden[-1]).squeeze(-1)


def build_sequences(data: pd.DataFrame, sequence_length: int) -> tuple[np.ndarray, np.ndarray, float]:
    feature_columns = [
        "susceptible",
        "infected",
        "recovered",
        "deceased",
        "vaccinated",
        "new_infections",
        "new_recoveries",
        "new_deaths",
        "new_vaccinations",
    ]
    scale = float(data[["susceptible", "infected", "recovered", "deceased", "vaccinated"]].sum(axis=1).max())
    sequences: list[np.ndarray] = []
    targets: list[float] = []

    for _, scenario in data.sort_values(["scenario_id", "day"]).groupby("scenario_id"):
        values = scenario[feature_columns].to_numpy(dtype=np.float32) / scale
        infected = scenario["infected"].to_numpy(dtype=np.float32) / scale

        if len(values) <= sequence_length:
            continue

        for start in range(0, len(values) - sequence_length):
            end = start + sequence_length
            sequences.append(values[start:end])
            targets.append(infected[end])

    return np.asarray(sequences, dtype=np.float32), np.asarray(targets, dtype=np.float32), scale


def split_by_scenario(data: pd.DataFrame, test_ratio: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    scenario_ids = np.array(sorted(data["scenario_id"].unique()))
    rng = np.random.default_rng(42)
    rng.shuffle(scenario_ids)
    split_index = max(1, int(len(scenario_ids) * (1 - test_ratio)))
    train_ids = set(scenario_ids[:split_index])

    train_data = data[data["scenario_id"].isin(train_ids)].copy()
    test_data = data[~data["scenario_id"].isin(train_ids)].copy()
    return train_data, test_data


def train(args: argparse.Namespace) -> None:
    if not DAILY_PATH.exists():
        raise FileNotFoundError("daily_counts.csv was not found. Run the C++ simulator first.")

    data = pd.read_csv(DAILY_PATH)
    train_data, test_data = split_by_scenario(data)
    x_train, y_train, scale = build_sequences(train_data, args.sequence_length)
    x_test, y_test, _ = build_sequences(test_data, args.sequence_length)

    if len(x_train) == 0 or len(x_test) == 0:
        raise ValueError("Not enough daily rows to build LSTM sequences.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DailyInfectionLSTM(input_size=x_train.shape[-1], hidden_size=args.hidden_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.MSELoss()

    dataset = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model.train()
    for epoch in range(1, args.epochs + 1):
        losses = []
        for features, targets in loader:
            features = features.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            predictions = model(features)
            loss = loss_fn(predictions, targets)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        print(f"Epoch {epoch:03d} | loss={np.mean(losses):.6f}")

    model.eval()
    with torch.no_grad():
        predictions = model(torch.from_numpy(x_test).to(device)).cpu().numpy() * scale

    actual = y_test * scale
    print("\nDaily infected count prediction")
    print(f"MAE: {mean_absolute_error(actual, predictions):.2f}")
    print(f"R2: {r2_score(actual, predictions):.3f}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "daily_infection_lstm.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_size": x_train.shape[-1],
            "hidden_size": args.hidden_size,
            "sequence_length": args.sequence_length,
            "scale": scale,
        },
        model_path,
    )
    joblib.dump({"feature_scale": scale}, MODEL_DIR / "daily_lstm_metadata.joblib")
    print(f"Saved daily LSTM to {model_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an LSTM for daily infected count prediction.")
    parser.add_argument("--sequence-length", type=int, default=14)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
