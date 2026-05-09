# Epidemic Spread Simulation and Machine Learning

This project generates synthetic epidemic-spread scenarios with a C++ SIRD-style
simulation, then trains machine-learning models in Python to predict outbreak
severity, peak infection count, daily infection counts, and graph/node-level
infection dynamics.

## Project Structure

```text
spread-simulation/
  simulation-generation-source-code/
    simulation.cpp
training-model/
  main/
    split.py
    train.py
    train_daily_lstm.py
    train_graph_lstm.py
results/
  epidemic_dataset.csv
  daily_counts.csv
  graph_timeseries.csv
  graph_edges.csv
  train.csv
  test.csv
  models/
```

## Build and Generate Data

Compile the C++ simulator:

```powershell
g++ -std=c++11 -O2 -Wall -Wextra -pedantic `
  "spread-simulation\simulation-generation-source-code\simulation.cpp" `
  -o "spread-simulation\simulation.exe"
```

Generate 1,000 simulation rows:

```powershell
.\spread-simulation\simulation.exe 1000 results\epidemic_dataset.csv
```

The simulator also writes daily and graph outputs by default:

```text
results\daily_counts.csv
results\graph_timeseries.csv
results\graph_edges.csv
```

You can pass all output paths explicitly:

```powershell
.\spread-simulation\simulation.exe 1000 `
  results\epidemic_dataset.csv `
  results\daily_counts.csv `
  results\graph_timeseries.csv `
  results\graph_edges.csv
```

The simulator writes these main inputs and labels:

- `population`
- `initial_infected`
- `infection_rate`
- `recovery_rate`
- `mortality_rate`
- `lockdown_strength`
- `mask_adoption`
- `vaccination_rate`
- `travel_restriction`
- `days`
- `peak_infected`
- `total_infected`
- `total_deaths`
- `severity`

The daily file stores one row per scenario per day:

- `scenario_id`
- `day`
- `susceptible`
- `infected`
- `recovered`
- `deceased`
- `vaccinated`
- `new_infections`
- `new_recoveries`
- `new_deaths`
- `new_vaccinations`

The graph time-series file stores one row per scenario, node, and day. The
current simulator uses 8 connected regions as graph nodes.

## Train Models

Install Python dependencies:

```powershell
pip install -r requirements.txt
```

Split the generated dataset:

```powershell
python training-model\main\split.py
```

Train the models:

```powershell
python training-model\main\train.py
```

The training script saves:

- `results\models\severity_classifier.joblib`
- `results\models\peak_infected_regressor.joblib`
- `results\models\total_deaths_regressor.joblib`

Generate research-paper evaluation outputs:

```powershell
python training-model\main\evaluate_research_results.py
```

This saves confusion matrix figures, classification metrics, regression metrics,
and actual-vs-predicted plots under `results\research\`.

Train the aggregate daily infected-count LSTM:

```powershell
python training-model\main\train_daily_lstm.py
```

Train the graph-temporal LSTM:

```powershell
python training-model\main\train_graph_lstm.py
```

The neural scripts require PyTorch from `requirements.txt`.

Test a model on the real-world Our World in Data COVID-19 dataset:

```powershell
python training-model\main\train_real_world.py
```

This trains on earlier country-level history and evaluates 7-day-ahead
predictions on later dates for cases and deaths per million.

## Streamlit Dashboard

Run the interactive dashboard:

```powershell
streamlit run app.py
```

The dashboard lets you adjust scenario inputs, apply policy controls, view
predicted severity and peak infection count, plot daily infection curves,
inspect generated CSV data, and visualize node-level graph spread with
NetworkX and Plotly.

Policy controls include:

- lockdown strength
- mask adoption
- daily vaccination rate
- travel restriction

## Current Model Tasks

- Classification: predict `severity` as `low`, `medium`, or `high`.
- Regression: predict `peak_infected`.
- Regression: predict `total_deaths`.
- Daily LSTM: predict the next aggregate daily `infected` count.
- Graph LSTM: predict the next day infected count for each graph node, then
  sum node predictions for total daily infections.
- Real-world forecaster: predict 7-day-ahead COVID cases and deaths per million
  using the Our World in Data dataset.

## Epidemic Model

The simulator now uses an SIRD-style compartment model:

```text
Susceptible -> Infected -> Recovered
                     \-> Deceased
```

Vaccination moves people out of the susceptible group, while policy controls
reduce the effective transmission rate.
