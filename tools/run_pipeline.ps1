param(
    [int]$ScenarioCount = 1000,
    [uint32]$Seed = 42
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Source = Join-Path $Root "spread-simulation\simulation-generation-source-code\simulation.cpp"
$Executable = Join-Path $Root "spread-simulation\simulation.exe"
$Python = if (Test-Path (Join-Path $Root "env\Scripts\python.exe")) {
    Join-Path $Root "env\Scripts\python.exe"
} else {
    "python"
}

Push-Location $Root
try {
    if ($ScenarioCount -le 0) {
        throw "ScenarioCount must be positive."
    }

    Write-Host "Compiling simulator..."
    & g++ -std=c++11 -O2 -Wall -Wextra -pedantic $Source -o $Executable

    Write-Host "Generating $ScenarioCount scenarios with seed $Seed..."
    & $Executable $ScenarioCount `
        "results\epidemic_dataset.csv" `
        "results\daily_counts.csv" `
        "results\graph_timeseries.csv" `
        "results\graph_edges.csv" `
        $Seed

    Write-Host "Validating simulator outputs..."
    & $Python "tools\validate_outputs.py"

    Write-Host "Splitting and training models..."
    & $Python "training-model\main\split.py"
    & $Python "training-model\main\train.py"
    & $Python "training-model\main\evaluate_research_results.py"
    & $Python "tools\benchmark_models.py"

    Write-Host "Pipeline completed successfully."
}
finally {
    Pop-Location
}
