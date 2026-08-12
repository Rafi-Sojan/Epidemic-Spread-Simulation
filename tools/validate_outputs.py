"""Validate the CSV contract produced by the epidemic simulator."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


SUMMARY_COLUMNS = {
    "scenario_id",
    "population",
    "final_susceptible",
    "final_infected",
    "final_recovered",
    "final_deceased",
    "final_vaccinated",
    "peak_infected",
    "total_deaths",
    "severity",
}
DAILY_STATE_COLUMNS = ["susceptible", "infected", "recovered", "deceased", "vaccinated"]


def validate(results_dir: Path) -> None:
    summary_path = results_dir / "epidemic_dataset.csv"
    daily_path = results_dir / "daily_counts.csv"
    graph_path = results_dir / "graph_timeseries.csv"
    edge_path = results_dir / "graph_edges.csv"
    paths = [summary_path, daily_path, graph_path, edge_path]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing simulator outputs: " + ", ".join(missing))

    summary = pd.read_csv(summary_path)
    daily = pd.read_csv(daily_path)
    graph = pd.read_csv(graph_path)
    edges = pd.read_csv(edge_path)

    if summary.empty:
        raise ValueError("The scenario summary is empty.")
    missing_columns = SUMMARY_COLUMNS.difference(summary.columns)
    if missing_columns:
        raise ValueError(f"Summary is missing columns: {sorted(missing_columns)}")

    population_total = (
        summary["final_susceptible"]
        + summary["final_infected"]
        + summary["final_recovered"]
        + summary["final_deceased"]
        + summary["final_vaccinated"]
    )
    if not population_total.eq(summary["population"]).all():
        raise ValueError("Population conservation failed in the scenario summary.")

    if daily["day"].eq(0).any() and not daily.loc[daily["day"].eq(0), "new_infections"].eq(0).all():
        raise ValueError("Day-zero rows must have zero new infections.")
    if daily[DAILY_STATE_COLUMNS].lt(0).any().any():
        raise ValueError("Daily state counts must not be negative.")
    if graph[DAILY_STATE_COLUMNS].lt(0).any().any():
        raise ValueError("Graph state counts must not be negative.")
    if edges.empty:
        raise ValueError("The graph edge output is empty.")

    print(
        "Validated outputs: "
        f"{len(summary)} scenarios, {len(daily)} daily rows, "
        f"{len(graph)} graph rows, {len(edges)} edges."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory containing simulator CSV outputs.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    validate(parse_args().results_dir)
