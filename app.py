from __future__ import annotations

import math
import subprocess
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
SIMULATOR_EXE = ROOT / "spread-simulation" / "simulation.exe"
SUMMARY_PATH = ROOT / "results" / "epidemic_dataset.csv"
DAILY_PATH = ROOT / "results" / "daily_counts.csv"
GRAPH_PATH = ROOT / "results" / "graph_timeseries.csv"
EDGE_PATH = ROOT / "results" / "graph_edges.csv"
MODEL_DIR = ROOT / "results" / "models"
CLASSIFIER_PATH = MODEL_DIR / "severity_classifier.joblib"
REGRESSOR_PATH = MODEL_DIR / "peak_infected_regressor.joblib"
DAILY_LSTM_PATH = MODEL_DIR / "daily_infection_lstm.pt"
GRAPH_LSTM_PATH = MODEL_DIR / "graph_lstm.pt"

FEATURE_COLUMNS = [
    "population",
    "initial_infected",
    "infection_rate",
    "recovery_rate",
    "days",
]


st.set_page_config(
    page_title="Epidemic Spread Simulation",
    page_icon="",
    layout="wide",
)


@st.cache_data
def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_resource
def load_joblib_model(path: Path):
    return joblib.load(path)


def run_simulator(scenario_count: int) -> tuple[bool, str]:
    if not SIMULATOR_EXE.exists():
        return False, "Simulator executable was not found. Compile simulation.cpp first."

    command = [
        str(SIMULATOR_EXE),
        str(scenario_count),
        str(SUMMARY_PATH),
        str(DAILY_PATH),
        str(GRAPH_PATH),
        str(EDGE_PATH),
    ]
    result = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    load_csv.clear()

    if result.returncode != 0:
        return False, result.stderr or result.stdout
    return True, result.stdout


def predict_from_classic_models(inputs: dict[str, float]) -> tuple[str | None, float | None]:
    feature_frame = pd.DataFrame([inputs], columns=FEATURE_COLUMNS)
    severity = None
    peak_infected = None

    if CLASSIFIER_PATH.exists():
        classifier = load_joblib_model(CLASSIFIER_PATH)
        severity = str(classifier.predict(feature_frame)[0])

    if REGRESSOR_PATH.exists():
        regressor = load_joblib_model(REGRESSOR_PATH)
        peak_infected = float(regressor.predict(feature_frame)[0])

    return severity, peak_infected


def simulate_single_curve(
    population: int,
    initial_infected: int,
    infection_rate: float,
    recovery_rate: float,
    days: int,
) -> pd.DataFrame:
    susceptible = population - initial_infected
    infected = initial_infected
    recovered = 0
    rows = []

    for day in range(days + 1):
        rows.append(
            {
                "day": day,
                "susceptible": round(susceptible),
                "infected": round(infected),
                "recovered": round(recovered),
            }
        )

        new_infections = infection_rate * susceptible * infected / population
        new_recoveries = recovery_rate * infected
        new_infections = min(new_infections, susceptible)
        new_recoveries = min(new_recoveries, infected)

        susceptible -= new_infections
        infected += new_infections - new_recoveries
        recovered += new_recoveries

    return pd.DataFrame(rows)


def graph_snapshot(graph_data: pd.DataFrame, scenario_id: int, day: int) -> pd.DataFrame:
    frame = graph_data[
        (graph_data["scenario_id"] == scenario_id) & (graph_data["day"] == day)
    ].copy()
    if frame.empty:
        return frame

    node_count = frame["node_id"].nunique()
    frame = frame.sort_values("node_id")
    frame["angle"] = frame["node_id"].apply(lambda node: 2 * math.pi * node / node_count)
    frame["x"] = frame["angle"].apply(math.cos)
    frame["y"] = frame["angle"].apply(math.sin)
    frame["infection_share"] = frame["infected"] / frame["population"]
    frame["node"] = frame["node_id"].apply(lambda node: f"Region {node}")
    return frame


st.title("Epidemic Spread Simulation and ML Dashboard")

with st.sidebar:
    st.header("Scenario")
    population = st.slider("Population", 500, 10000, 5000, step=100)
    initial_infected = st.slider("Initial infected", 1, 250, 25)
    infection_rate = st.slider("Infection rate", 0.01, 0.60, 0.22, step=0.01)
    recovery_rate = st.slider("Recovery rate", 0.01, 0.30, 0.08, step=0.01)
    days = st.slider("Days", 30, 240, 120)

    st.divider()
    scenario_count = st.number_input("Simulation rows", min_value=10, max_value=5000, value=250, step=10)
    if st.button("Generate simulation data", type="primary"):
        ok, message = run_simulator(int(scenario_count))
        if ok:
            st.success("Simulation data generated.")
            st.code(message.strip())
        else:
            st.error(message)

inputs = {
    "population": population,
    "initial_infected": initial_infected,
    "infection_rate": infection_rate,
    "recovery_rate": recovery_rate,
    "days": days,
}

severity, peak_infected = predict_from_classic_models(inputs)
curve = simulate_single_curve(population, initial_infected, infection_rate, recovery_rate, days)

metric_columns = st.columns(4)
metric_columns[0].metric("Predicted severity", severity or "model missing")
metric_columns[1].metric(
    "Predicted peak infected",
    f"{peak_infected:,.0f}" if peak_infected is not None else "model missing",
)
metric_columns[2].metric("Curve peak infected", f"{curve['infected'].max():,.0f}")
metric_columns[3].metric("Peak day", int(curve.loc[curve["infected"].idxmax(), "day"]))

tab_overview, tab_data, tab_graph, tab_models = st.tabs(
    ["Overview", "Generated Data", "Graph Spread", "Models"]
)

with tab_overview:
    left, right = st.columns([2, 1])
    with left:
        st.subheader("Daily Infection Curve")
        st.line_chart(curve.set_index("day")[["susceptible", "infected", "recovered"]])
    with right:
        st.subheader("Selected Inputs")
        st.dataframe(pd.DataFrame([inputs]), hide_index=True, use_container_width=True)

with tab_data:
    if SUMMARY_PATH.exists():
        summary = load_csv(SUMMARY_PATH)
        st.subheader("Scenario Summary")
        st.dataframe(summary.head(100), use_container_width=True)

        daily = load_csv(DAILY_PATH) if DAILY_PATH.exists() else pd.DataFrame()
        if not daily.empty:
            selected_scenario = st.selectbox(
                "Daily count scenario",
                sorted(daily["scenario_id"].unique()),
            )
            daily_view = daily[daily["scenario_id"] == selected_scenario].set_index("day")
            st.line_chart(daily_view[["susceptible", "infected", "recovered"]])
    else:
        st.info("Generate simulation data to preview CSV outputs.")

with tab_graph:
    if GRAPH_PATH.exists():
        graph_data = load_csv(GRAPH_PATH)
        scenario_ids = sorted(graph_data["scenario_id"].unique())
        selected_graph_scenario = st.selectbox("Graph scenario", scenario_ids)
        scenario_days = sorted(
            graph_data[graph_data["scenario_id"] == selected_graph_scenario]["day"].unique()
        )
        selected_day = st.slider(
            "Graph day",
            int(min(scenario_days)),
            int(max(scenario_days)),
            int(max(scenario_days) // 2),
        )
        snapshot = graph_snapshot(graph_data, int(selected_graph_scenario), int(selected_day))

        if not snapshot.empty:
            st.scatter_chart(
                snapshot,
                x="x",
                y="y",
                size="infected",
                color="infection_share",
            )
            st.dataframe(
                snapshot[
                    ["node", "population", "susceptible", "infected", "recovered", "infection_share"]
                ],
                hide_index=True,
                use_container_width=True,
            )
    else:
        st.info("Generate graph time-series data to view regional spread.")

with tab_models:
    model_status = pd.DataFrame(
        [
            {
                "model": "Severity classifier",
                "path": str(CLASSIFIER_PATH),
                "available": CLASSIFIER_PATH.exists(),
            },
            {
                "model": "Peak infected regressor",
                "path": str(REGRESSOR_PATH),
                "available": REGRESSOR_PATH.exists(),
            },
            {
                "model": "Daily LSTM",
                "path": str(DAILY_LSTM_PATH),
                "available": DAILY_LSTM_PATH.exists(),
            },
            {
                "model": "Graph LSTM",
                "path": str(GRAPH_LSTM_PATH),
                "available": GRAPH_LSTM_PATH.exists(),
            },
        ]
    )
    st.subheader("Model Files")
    st.dataframe(model_status, hide_index=True, use_container_width=True)
    st.caption(
        "The dashboard uses trained Random Forest models when available. "
        "Daily and graph LSTM files appear here after PyTorch training."
    )
