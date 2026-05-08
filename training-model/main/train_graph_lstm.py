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
        "PyTorch is required for Graph LSTM training. Install it with `pip install -r requirements.txt`."
    ) from error


ROOT = Path(__file__).resolve().parents[2]
GRAPH_PATH = ROOT / "results" / "graph_timeseries.csv"
EDGE_PATH = ROOT / "results" / "graph_edges.csv"
MODEL_DIR = ROOT / "results" / "models"


class GraphLSTM(nn.Module):
    def __init__(
        self,
        node_count: int,
        input_size: int,
        graph_hidden_size: int = 16,
        lstm_hidden_size: int = 64,
    ) -> None:
        super().__init__()
        self.node_count = node_count
        self.graph_projection = nn.Linear(input_size, graph_hidden_size)
        self.lstm = nn.LSTM(
            input_size=node_count * graph_hidden_size,
            hidden_size=lstm_hidden_size,
            batch_first=True,
        )
        self.output = nn.Linear(lstm_hidden_size, node_count)

    def forward(self, features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        propagated = torch.einsum("ij,btjf->btif", adjacency, features)
        graph_features = torch.relu(self.graph_projection(propagated))
        batch_size, time_steps, node_count, hidden_size = graph_features.shape
        sequence = graph_features.reshape(batch_size, time_steps, node_count * hidden_size)
        _, (hidden, _) = self.lstm(sequence)
        return self.output(hidden[-1])


def build_adjacency(edges: pd.DataFrame, node_count: int) -> np.ndarray:
    adjacency = np.eye(node_count, dtype=np.float32)
    first_scenario = edges["scenario_id"].min()

    for _, edge in edges[edges["scenario_id"] == first_scenario].iterrows():
        adjacency[int(edge["source_node"]), int(edge["target_node"])] = float(edge["weight"])

    degree = adjacency.sum(axis=1, keepdims=True)
    return adjacency / np.maximum(degree, 1.0)


def build_sequences(
    data: pd.DataFrame,
    sequence_length: int,
    node_count: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    feature_columns = ["susceptible", "infected", "recovered", "new_infections", "new_recoveries"]
    scale = float(data["population"].max())
    sequences: list[np.ndarray] = []
    targets: list[np.ndarray] = []

    for _, scenario in data.sort_values(["scenario_id", "day", "node_id"]).groupby("scenario_id"):
        days = sorted(scenario["day"].unique())
        frames = []

        for day in days:
            frame = scenario[scenario["day"] == day].sort_values("node_id")
            if len(frame) != node_count:
                continue
            frames.append(frame[feature_columns].to_numpy(dtype=np.float32) / scale)

        if len(frames) <= sequence_length:
            continue

        series = np.asarray(frames, dtype=np.float32)
        for start in range(0, len(series) - sequence_length):
            end = start + sequence_length
            sequences.append(series[start:end])
            targets.append(series[end, :, feature_columns.index("infected")])

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
    if not GRAPH_PATH.exists() or not EDGE_PATH.exists():
        raise FileNotFoundError("Graph CSV files were not found. Run the C++ simulator first.")

    data = pd.read_csv(GRAPH_PATH)
    edges = pd.read_csv(EDGE_PATH)
    node_count = int(data["node_id"].nunique())
    adjacency = build_adjacency(edges, node_count)

    train_data, test_data = split_by_scenario(data)
    x_train, y_train, scale = build_sequences(train_data, args.sequence_length, node_count)
    x_test, y_test, _ = build_sequences(test_data, args.sequence_length, node_count)

    if len(x_train) == 0 or len(x_test) == 0:
        raise ValueError("Not enough graph rows to build Graph LSTM sequences.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GraphLSTM(
        node_count=node_count,
        input_size=x_train.shape[-1],
        graph_hidden_size=args.graph_hidden_size,
        lstm_hidden_size=args.lstm_hidden_size,
    ).to(device)
    adjacency_tensor = torch.from_numpy(adjacency).to(device)
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
            predictions = model(features, adjacency_tensor)
            loss = loss_fn(predictions, targets)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        print(f"Epoch {epoch:03d} | loss={np.mean(losses):.6f}")

    model.eval()
    with torch.no_grad():
        predictions = model(torch.from_numpy(x_test).to(device), adjacency_tensor).cpu().numpy() * scale

    actual = y_test * scale
    print("\nGraph LSTM node-level infected count prediction")
    print(f"Node MAE: {mean_absolute_error(actual.reshape(-1), predictions.reshape(-1)):.2f}")
    print(f"Node R2: {r2_score(actual.reshape(-1), predictions.reshape(-1)):.3f}")
    print(f"Total daily MAE: {mean_absolute_error(actual.sum(axis=1), predictions.sum(axis=1)):.2f}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "graph_lstm.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "node_count": node_count,
            "input_size": x_train.shape[-1],
            "graph_hidden_size": args.graph_hidden_size,
            "lstm_hidden_size": args.lstm_hidden_size,
            "sequence_length": args.sequence_length,
            "scale": scale,
            "adjacency": adjacency,
        },
        model_path,
    )
    joblib.dump({"feature_scale": scale, "adjacency": adjacency}, MODEL_DIR / "graph_lstm_metadata.joblib")
    print(f"Saved Graph LSTM to {model_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a graph-temporal LSTM for node infection prediction.")
    parser.add_argument("--sequence-length", type=int, default=14)
    parser.add_argument("--graph-hidden-size", type=int, default=16)
    parser.add_argument("--lstm-hidden-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
