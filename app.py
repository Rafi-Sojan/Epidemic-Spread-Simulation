from __future__ import annotations

import math
import subprocess
from pathlib import Path

import joblib
import networkx as nx
import pandas as pd
import plotly.graph_objects as go
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
DEATH_REGRESSOR_PATH = MODEL_DIR / "total_deaths_regressor.joblib"
DAILY_LSTM_PATH = MODEL_DIR / "daily_infection_lstm.pt"
GRAPH_LSTM_PATH = MODEL_DIR / "graph_lstm.pt"
REAL_WORLD_MODEL_PATH = MODEL_DIR / "real_world_covid_forecaster.joblib"
REAL_WORLD_METRICS_PATH = ROOT / "results" / "real_world" / "real_world_metrics.csv"

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


def predict_from_classic_models(inputs: dict[str, float]) -> tuple[str | None, float | None, float | None]:
    feature_frame = pd.DataFrame([inputs], columns=FEATURE_COLUMNS)
    severity = None
    peak_infected = None
    total_deaths = None

    if CLASSIFIER_PATH.exists():
        classifier = load_joblib_model(CLASSIFIER_PATH)
        severity = str(classifier.predict(feature_frame)[0])

    if REGRESSOR_PATH.exists():
        regressor = load_joblib_model(REGRESSOR_PATH)
        peak_infected = float(regressor.predict(feature_frame)[0])

    if DEATH_REGRESSOR_PATH.exists():
        death_regressor = load_joblib_model(DEATH_REGRESSOR_PATH)
        total_deaths = float(death_regressor.predict(feature_frame)[0])

    return severity, peak_infected, total_deaths


def simulate_single_curve(
    population: int,
    initial_infected: int,
    infection_rate: float,
    recovery_rate: float,
    mortality_rate: float,
    median_age: float,
    elderly_population_ratio: float,
    child_population_ratio: float,
    temperature_celsius: float,
    humidity_percent: float,
    rainfall_mm: float,
    days: int,
    lockdown_strength: float,
    mask_adoption: float,
    vaccination_rate: float,
    travel_restriction: float,
) -> pd.DataFrame:
    susceptible = population - initial_infected
    infected = initial_infected
    recovered = 0
    deceased = 0
    vaccinated = 0
    rows = []

    for day in range(days + 1):
        policy_effect = (
            1.0
            - 0.55 * lockdown_strength
            - 0.35 * mask_adoption
            - 0.20 * travel_restriction
        )
        temperature_factor = 1.0 + max(0.0, 22.0 - temperature_celsius) * 0.012
        humidity_factor = 1.0 + max(0.0, 45.0 - humidity_percent) * 0.006
        rainfall_factor = 1.0 - min(0.18, rainfall_mm * 0.004)
        child_contact_factor = 1.0 + child_population_ratio * 0.25
        age_mortality_factor = (
            1.0
            + max(0.0, median_age - 35.0) * 0.018
            + elderly_population_ratio * 1.8
        )
        effective_infection_rate = max(
            0.0,
            infection_rate
            * policy_effect
            * temperature_factor
            * humidity_factor
            * rainfall_factor
            * child_contact_factor,
        )
        effective_mortality_rate = mortality_rate * age_mortality_factor

        rows.append(
            {
                "day": day,
                "susceptible": round(susceptible),
                "infected": round(infected),
                "recovered": round(recovered),
                "deceased": round(deceased),
                "vaccinated": round(vaccinated),
                "effective_infection_rate": effective_infection_rate,
                "effective_mortality_rate": effective_mortality_rate,
            }
        )

        new_vaccinations = min(susceptible, population * vaccination_rate)
        susceptible -= new_vaccinations
        vaccinated += new_vaccinations

        new_infections = effective_infection_rate * susceptible * infected / population
        new_infections = min(new_infections, susceptible)

        susceptible -= new_infections
        infected_after_spread = infected + new_infections
        new_deaths = min(effective_mortality_rate * infected_after_spread, infected_after_spread)
        infected_after_deaths = infected_after_spread - new_deaths
        new_recoveries = min(recovery_rate * infected_after_deaths, infected_after_deaths)

        infected = infected_after_deaths - new_recoveries
        recovered += new_recoveries
        deceased += new_deaths

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
    frame["death_share"] = frame["deceased"] / frame["population"]
    frame["node"] = frame["node_id"].apply(lambda node: f"Region {node}")
    return frame


def build_graph_figure(snapshot: pd.DataFrame, edges: pd.DataFrame, scenario_id: int) -> go.Figure:
    graph = nx.Graph()
    for _, row in snapshot.iterrows():
        graph.add_node(
            int(row["node_id"]),
            infected=int(row["infected"]),
            population=int(row["population"]),
            infection_share=float(row["infection_share"]),
            deceased=int(row["deceased"]),
        )

    scenario_edges = edges[edges["scenario_id"] == scenario_id]
    for _, row in scenario_edges.iterrows():
        graph.add_edge(
            int(row["source_node"]),
            int(row["target_node"]),
            weight=float(row["weight"]),
        )

    positions = nx.circular_layout(graph)

    edge_x = []
    edge_y = []
    for source, target in graph.edges():
        x0, y0 = positions[source]
        x1, y1 = positions[target]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    ordered_nodes = sorted(graph.nodes())
    node_x = [positions[node][0] for node in ordered_nodes]
    node_y = [positions[node][1] for node in ordered_nodes]
    infected = [graph.nodes[node]["infected"] for node in ordered_nodes]
    infection_share = [graph.nodes[node]["infection_share"] for node in ordered_nodes]
    labels = [
        f"Region {node}<br>Infected: {graph.nodes[node]['infected']:,}"
        f"<br>Deceased: {graph.nodes[node]['deceased']:,}"
        f"<br>Population: {graph.nodes[node]['population']:,}"
        for node in ordered_nodes
    ]

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line={"width": 2, "color": "#9aa4b2"},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    figure.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=[str(node) for node in ordered_nodes],
            textposition="middle center",
            hovertext=labels,
            hoverinfo="text",
            marker={
                "size": [max(22, min(70, value / 4)) for value in infected],
                "color": infection_share,
                "colorscale": "YlOrRd",
                "showscale": True,
                "colorbar": {"title": "Infected share"},
                "line": {"width": 2, "color": "#1f2937"},
            },
            textfont={"color": "#111827", "size": 13},
            showlegend=False,
        )
    )
    figure.update_layout(
        height=520,
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        xaxis={"visible": False},
        yaxis={"visible": False},
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return figure


st.title("Epidemic Spread Simulation and ML Dashboard")

with st.sidebar:
    st.header("Scenario")
    population = st.slider("Population", 500, 10000, 5000, step=100)
    initial_infected = st.slider("Initial infected", 1, 250, 25)
    infection_rate = st.slider("Infection rate", 0.01, 0.60, 0.22, step=0.01)
    recovery_rate = st.slider("Recovery rate", 0.01, 0.30, 0.08, step=0.01)
    mortality_rate = st.slider("Mortality rate", 0.001, 0.05, 0.008, step=0.001)
    days = st.slider("Days", 30, 240, 120)

    with st.expander("Demographics and climate", expanded=False):
        median_age = st.slider("Median age", 18.0, 55.0, 32.0, step=0.5)
        elderly_population_ratio = st.slider("Elderly population ratio", 0.03, 0.24, 0.09, step=0.01)
        child_population_ratio = st.slider("Child population ratio", 0.12, 0.34, 0.22, step=0.01)
        temperature_celsius = st.slider("Temperature (C)", 5.0, 40.0, 28.0, step=0.5)
        humidity_percent = st.slider("Humidity (%)", 25.0, 95.0, 65.0, step=1.0)
        rainfall_mm = st.slider("Rainfall (mm/day)", 0.0, 45.0, 4.0, step=0.5)

    with st.expander("Policy controls", expanded=True):
        lockdown_strength = st.slider("Lockdown strength", 0.0, 1.0, 0.20, step=0.05)
        mask_adoption = st.slider("Mask adoption", 0.0, 1.0, 0.35, step=0.05)
        vaccination_rate = st.slider("Daily vaccination rate", 0.0, 0.02, 0.002, step=0.001)
        travel_restriction = st.slider("Travel restriction", 0.0, 1.0, 0.15, step=0.05)

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
    "mortality_rate": mortality_rate,
    "lockdown_strength": lockdown_strength,
    "mask_adoption": mask_adoption,
    "vaccination_rate": vaccination_rate,
    "travel_restriction": travel_restriction,
    "median_age": median_age,
    "elderly_population_ratio": elderly_population_ratio,
    "child_population_ratio": child_population_ratio,
    "temperature_celsius": temperature_celsius,
    "humidity_percent": humidity_percent,
    "rainfall_mm": rainfall_mm,
    "days": days,
}

severity, peak_infected, total_deaths = predict_from_classic_models(inputs)
curve = simulate_single_curve(
    population,
    initial_infected,
    infection_rate,
    recovery_rate,
    mortality_rate,
    median_age,
    elderly_population_ratio,
    child_population_ratio,
    temperature_celsius,
    humidity_percent,
    rainfall_mm,
    days,
    lockdown_strength,
    mask_adoption,
    vaccination_rate,
    travel_restriction,
)

metric_columns = st.columns(5)
metric_columns[0].metric("Predicted severity", severity or "model missing")
metric_columns[1].metric(
    "Predicted peak infected",
    f"{peak_infected:,.0f}" if peak_infected is not None else "model missing",
)
metric_columns[2].metric(
    "Predicted total deaths",
    f"{total_deaths:,.0f}" if total_deaths is not None else "model missing",
)
metric_columns[3].metric("Curve peak infected", f"{curve['infected'].max():,.0f}")
metric_columns[4].metric("Curve total deaths", f"{curve['deceased'].max():,.0f}")

tab_overview, tab_data, tab_graph, tab_models = st.tabs(
    ["Overview", "Generated Data", "Graph Spread", "Models"]
)

with tab_overview:
    left, right = st.columns([2, 1])
    with left:
        st.subheader("Daily Infection Curve With Policies")
        st.line_chart(
            curve.set_index("day")[["susceptible", "infected", "recovered", "deceased", "vaccinated"]]
        )
    with right:
        st.subheader("Selected Inputs")
        st.dataframe(pd.DataFrame([inputs]), hide_index=True, use_container_width=True)
        st.subheader("Policy Effect")
        policy_frame = pd.DataFrame(
            [
                {
                    "lockdown_strength": lockdown_strength,
                    "mask_adoption": mask_adoption,
                    "daily_vaccination_rate": vaccination_rate,
                    "travel_restriction": travel_restriction,
                    "average_effective_infection_rate": curve["effective_infection_rate"].mean(),
                    "average_effective_mortality_rate": curve["effective_mortality_rate"].mean(),
                }
            ]
        )
        st.dataframe(policy_frame, hide_index=True, use_container_width=True)

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
            st.line_chart(daily_view[["susceptible", "infected", "recovered", "deceased", "vaccinated"]])
    else:
        st.info("Generate simulation data to preview CSV outputs.")

with tab_graph:
    if GRAPH_PATH.exists() and EDGE_PATH.exists():
        graph_data = load_csv(GRAPH_PATH)
        edges = load_csv(EDGE_PATH)
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
            figure = build_graph_figure(snapshot, edges, int(selected_graph_scenario))
            st.plotly_chart(figure, use_container_width=True)
            st.dataframe(
                snapshot[
                    [
                        "node",
                        "population",
                        "susceptible",
                        "infected",
                        "recovered",
                        "deceased",
                        "vaccinated",
                        "infection_share",
                        "death_share",
                    ]
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
                "model": "Total deaths regressor",
                "path": str(DEATH_REGRESSOR_PATH),
                "available": DEATH_REGRESSOR_PATH.exists(),
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
