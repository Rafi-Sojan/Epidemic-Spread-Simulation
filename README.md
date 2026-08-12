# Epidemic Spread Simulation and Machine Learning

An end-to-end, simulation-driven epidemic analysis system. The project uses a
probabilistic C++ SIRD model to generate controlled epidemic scenarios and
Python-based Random Forest models to predict outbreak severity, peak infected
count, and total deaths.

The system is intended for education, experimentation, and research prototyping.
Its reported metrics are measured on synthetic simulation data and must not be
interpreted as validated real-world forecasting accuracy.

## Overview

The project combines four components:

1. A stochastic SIRD epidemic simulator written in C++.
2. A CSV data pipeline for scenario, daily, and regional graph outputs.
3. Random Forest classification and regression models written in Python.
4. A Streamlit dashboard for interactive scenario exploration.

The main workflow is:

```text
Scenario parameters
        |
        v
Probabilistic SIRD simulation
        |
        v
CSV dataset generation
        |
        v
Train/test split and model training
        |
        v
Evaluation and Streamlit visualization
```

## Features

- SIRD compartments: susceptible, infected, recovered, and deceased.
- Vaccination as an additional population transition.
- Policy controls for lockdown, masks, vaccination, and travel restriction.
- Demographic factors including median age, elderly ratio, and child ratio.
- Climate factors including temperature, humidity, and rainfall.
- Probabilistic infections, recoveries, deaths, and regional spread.
- Severity classification into low, medium, and high risk.
- Peak infected count and total deaths regression.
- Eight-node regional graph simulation.
- Interactive Streamlit visualization.
- Reproducible runs through an explicit random seed.
- Automated CSV contract validation.

## Repository Structure

```text
Epidemic-Spread-Simulation/
├── app.py
├── README.md
├── requirements.txt
├── requirements-optional.txt
├── .gitignore
│
├── spread-simulation/
│   ├── simulation.exe                  # Generated build artifact
│   └── simulation-generation-source-code/
│       └── simulation.cpp              # Probabilistic SIRD simulator
│
├── training-model/
│   └── main/
│       ├── split.py                    # Dataset splitting
│       ├── train.py                    # Random Forest training
│       ├── evaluate_research_results.py
│       ├── train_daily_lstm.py         # Optional extension
│       └── train_graph_lstm.py         # Optional extension
│
├── tools/
│   ├── run_pipeline.ps1               # Complete Windows workflow
│   └── validate_outputs.py             # Output contract checks
│
└── results/                            # Generated and Git-ignored
    ├── epidemic_dataset.csv
    ├── daily_counts.csv
    ├── graph_timeseries.csv
    ├── graph_edges.csv
    ├── train.csv
    ├── test.csv
    ├── models/
    └── research/
```

Report documents, figures, presentations, virtual environments, and generated
results are excluded from the public GitHub source workflow.

## Requirements

### Core requirements

- C++ compiler with C++11 support, such as MinGW g++.
- Python 3.10 or newer.
- Python packages listed in `requirements.txt`.

Install the core Python dependencies with:

```powershell
python -m pip install -r requirements.txt
```

### Optional neural-network requirements

The daily LSTM and Graph LSTM scripts require PyTorch. Install it separately
only when those extensions are needed:

```powershell
python -m pip install -r requirements-optional.txt
```

The final reported project results use Random Forest models and do not require
PyTorch.

## Quick Start

The recommended workflow compiles the simulator, generates seeded data,
validates the outputs, trains the models, and produces evaluation results:

```powershell
.\tools\run_pipeline.ps1 -ScenarioCount 1000 -Seed 42
```

The pipeline creates:

- `results\epidemic_dataset.csv`
- `results\daily_counts.csv`
- `results\graph_timeseries.csv`
- `results\graph_edges.csv`
- `results\train.csv`
- `results\test.csv`
- trained model files under `results\models\`
- evaluation outputs under `results\research\`

## Manual Workflow

### 1. Compile the simulator

```powershell
g++ -std=c++11 -O2 -Wall -Wextra -pedantic `
  "spread-simulation\simulation-generation-source-code\simulation.cpp" `
  -o "spread-simulation\simulation.exe"
```

### 2. Generate data

```powershell
.\spread-simulation\simulation.exe 1000 `
  results\epidemic_dataset.csv `
  results\daily_counts.csv `
  results\graph_timeseries.csv `
  results\graph_edges.csv `
  42
```

The final argument is the random seed. Supplying the same seed and parameters
produces identical simulator outputs. Use `--help` to display the command-line
syntax.

### 3. Validate the generated data

```powershell
python tools\validate_outputs.py
```

### 4. Split and train

```powershell
python training-model\main\split.py
python training-model\main\train.py
```

### 5. Generate evaluation outputs

```powershell
python training-model\main\evaluate_research_results.py
```

## Machine-Learning Models

The final system trains three Random Forest models:

| Model | Task | Output |
|---|---|---|
| Random Forest Classifier | Classification | Low, medium, or high severity |
| Random Forest Regressor | Regression | Peak infected count |
| Random Forest Regressor | Regression | Total deaths |

The models use disease, policy, demographic, climate, and simulation-duration
features. The complete feature schema and model hyperparameters are recorded in
`results\models\model_metadata.json` after training.

The LSTM and Graph LSTM scripts are experimental extensions for temporal and
regional forecasting. They are not used by the current dashboard prediction
cards or the reported Random Forest metrics.

## Simulation Model

The simulator maintains the population identity:

```text
N = S(t) + I(t) + R(t) + D(t) + V(t)
```

where `S` is susceptible, `I` is infected, `R` is recovered, `D` is deceased,
and `V` is vaccinated.

The effective transmission rate combines disease, policy, demographic, and
climate effects:

```text
beta_eff = beta
            x policy_factor
            x temperature_factor
            x humidity_factor
            x rainfall_factor
            x child_contact_factor
```

New infections, deaths, and recoveries are sampled using binomial distributions.
The severity score is calculated as:

```text
Risk_Score = Infected_Share + 2.5 x Death_Share
```

The thresholds are:

```text
Risk_Score < 0.20           Low
0.20 <= Risk_Score < 0.50  Medium
Risk_Score >= 0.50         High
```

## Generated Data

### Scenario summary

`epidemic_dataset.csv` stores one row per scenario, including input parameters,
final compartment counts, peak infected count, total deaths, and severity.

### Daily time series

`daily_counts.csv` stores one row per scenario per day with compartment counts
and daily transitions:

```text
scenario_id, day, susceptible, infected, recovered, deceased, vaccinated,
new_infections, new_recoveries, new_deaths, new_vaccinations
```

### Regional graph data

`graph_timeseries.csv` stores one row per scenario, region, and day.
`graph_edges.csv` stores the connections between the eight simulated regions.
The regional graph is a separate stochastic simulation used for visualization;
its totals are not required to match the global daily time series exactly.

## Streamlit Dashboard

Start the dashboard with:

```powershell
streamlit run app.py
```

The dashboard provides controls for:

- population and initial infected count,
- infection, recovery, and mortality rates,
- simulation duration,
- lockdown, masks, vaccination, and travel restriction,
- median age and population ratios,
- temperature, humidity, and rainfall.

It displays predicted severity, peak infected count, total deaths, expected
daily SIRD curves, policy effects, generated CSV data, and regional graph spread.

## Results

The current evaluation uses 1000 synthetic scenarios split into 800 training
rows and 200 testing rows.

| Task | Result |
|---|---:|
| Severity classification accuracy | 84.5% |
| Peak infected prediction R² | 0.618 |
| Total deaths prediction R² | 0.752 |
| Peak infected MAE | 124.00 |
| Total deaths MAE | 86.16 |

The classifier performs strongly for the low and high classes. The medium class
requires improvement because it is a transitional risk range with fewer and
more overlapping samples.

## Validation

The project has been checked using:

- C++ compilation with warnings enabled.
- Seed reproducibility checks.
- Population-conservation checks.
- Non-negative state validation.
- Day-zero transition validation.
- Python syntax validation.
- Dataset splitting and Random Forest evaluation.
- PowerShell workflow syntax validation.

## Scope and Limitations

- The primary dataset is synthetic and generated from model assumptions.
- Results do not represent validated real-world epidemiological accuracy.
- Reporting delays, testing availability, mobility, healthcare capacity, and
  changing variants are not modeled.
- The graph simulation is intended for regional visualization and experimentation.
- LSTM and Graph LSTM extensions require additional data and computation.
- The medium severity class has weaker recall than the other classes.

## Future Work

Potential extensions include:

- real-world dataset integration and temporal validation,
- uncertainty intervals and Monte Carlo confidence bands,
- hospital-capacity and mobility features,
- calibrated probabilities for severity classes,
- improved class balancing for medium-severity scenarios,
- automated CI testing and containerized deployment,
- trained temporal and graph-temporal neural models.
