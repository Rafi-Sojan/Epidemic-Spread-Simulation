#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#endif

struct SimulationConfig {
    int population;
    int initialInfected;
    double infectionRate;
    double recoveryRate;
    int days;
};

struct SimulationResult {
    int scenarioId;
    SimulationConfig config;
    int finalSusceptible;
    int finalInfected;
    int finalRecovered;
    int peakInfected;
    int peakDay;
    int totalInfected;
    std::string severity;
};

struct DailyRecord {
    int scenarioId;
    int day;
    int susceptible;
    int infected;
    int recovered;
    int newInfections;
    int newRecoveries;
};

struct NodeDailyRecord {
    int scenarioId;
    int nodeId;
    int day;
    int population;
    int susceptible;
    int infected;
    int recovered;
    int newInfections;
    int newRecoveries;
};

struct EdgeRecord {
    int scenarioId;
    int sourceNode;
    int targetNode;
    double weight;
};

struct ScenarioBundle {
    SimulationResult summary;
    std::vector<DailyRecord> dailyRecords;
    std::vector<NodeDailyRecord> nodeDailyRecords;
    std::vector<EdgeRecord> edgeRecords;
};

std::string classifySeverity(double infectedShare) {
    if (infectedShare < 0.20) {
        return "low";
    }
    if (infectedShare < 0.50) {
        return "medium";
    }
    return "high";
}

double clampProbability(double value) {
    if (value < 0.0) {
        return 0.0;
    }
    if (value > 1.0) {
        return 1.0;
    }
    return value;
}

void createDirectoryIfMissing(const std::string& directory) {
#ifdef _WIN32
    _mkdir(directory.c_str());
#else
    mkdir(directory.c_str(), 0755);
#endif
}

ScenarioBundle runSimulation(int scenarioId, const SimulationConfig& config, std::mt19937& rng) {
    int susceptible = config.population - config.initialInfected;
    int infected = config.initialInfected;
    int recovered = 0;
    int peakInfected = infected;
    int peakDay = 0;
    std::vector<DailyRecord> dailyRecords;

    dailyRecords.push_back({scenarioId, 0, susceptible, infected, recovered, infected, 0});

    for (int day = 1; day <= config.days; ++day) {
        const double contactProbability =
            config.infectionRate * static_cast<double>(infected) / config.population;
        const double boundedContactProbability = clampProbability(contactProbability);
        const double boundedRecoveryProbability = clampProbability(config.recoveryRate);

        std::binomial_distribution<int> newInfectionsDist(susceptible, boundedContactProbability);
        std::binomial_distribution<int> newRecoveriesDist(infected, boundedRecoveryProbability);

        const int newInfections = std::min(newInfectionsDist(rng), susceptible);
        const int newRecoveries = std::min(newRecoveriesDist(rng), infected);

        susceptible -= newInfections;
        infected += newInfections - newRecoveries;
        recovered += newRecoveries;

        dailyRecords.push_back({
            scenarioId,
            day,
            susceptible,
            infected,
            recovered,
            newInfections,
            newRecoveries
        });

        if (infected > peakInfected) {
            peakInfected = infected;
            peakDay = day;
        }

        if (infected == 0) {
            break;
        }
    }

    const int totalInfected = recovered + infected;
    const double infectedShare = static_cast<double>(totalInfected) / config.population;

    const SimulationResult summary {
        scenarioId,
        config,
        susceptible,
        infected,
        recovered,
        peakInfected,
        peakDay,
        totalInfected,
        classifySeverity(infectedShare)
    };

    const int nodeCount = 8;
    std::vector<int> nodePopulations(nodeCount, config.population / nodeCount);
    nodePopulations.back() += config.population % nodeCount;

    std::vector<int> nodeSusceptible(nodeCount);
    std::vector<int> nodeInfected(nodeCount, 0);
    std::vector<int> nodeRecovered(nodeCount, 0);
    std::vector<NodeDailyRecord> nodeDailyRecords;
    std::vector<EdgeRecord> edgeRecords;

    for (int node = 0; node < nodeCount; ++node) {
        const int target = (node + 1) % nodeCount;
        edgeRecords.push_back({scenarioId, node, target, 1.0});
        edgeRecords.push_back({scenarioId, target, node, 1.0});
    }

    for (int remaining = config.initialInfected; remaining > 0; --remaining) {
        std::uniform_int_distribution<int> nodeDist(0, nodeCount - 1);
        ++nodeInfected[nodeDist(rng)];
    }

    for (int node = 0; node < nodeCount; ++node) {
        nodeInfected[node] = std::min(nodeInfected[node], nodePopulations[node]);
        nodeSusceptible[node] = nodePopulations[node] - nodeInfected[node];
        nodeDailyRecords.push_back({
            scenarioId,
            node,
            0,
            nodePopulations[node],
            nodeSusceptible[node],
            nodeInfected[node],
            nodeRecovered[node],
            nodeInfected[node],
            0
        });
    }

    for (int day = 1; day <= config.days; ++day) {
        std::vector<int> newInfections(nodeCount, 0);
        std::vector<int> newRecoveries(nodeCount, 0);

        for (int node = 0; node < nodeCount; ++node) {
            const int left = (node + nodeCount - 1) % nodeCount;
            const int right = (node + 1) % nodeCount;
            const double localPressure =
                static_cast<double>(nodeInfected[node]) / nodePopulations[node];
            const double neighborPressure =
                0.5 * (
                    static_cast<double>(nodeInfected[left]) / nodePopulations[left] +
                    static_cast<double>(nodeInfected[right]) / nodePopulations[right]
                );
            const double contactProbability =
                config.infectionRate * (0.75 * localPressure + 0.25 * neighborPressure);

            std::binomial_distribution<int> infectionDist(
                nodeSusceptible[node],
                clampProbability(contactProbability)
            );
            std::binomial_distribution<int> recoveryDist(
                nodeInfected[node],
                clampProbability(config.recoveryRate)
            );

            newInfections[node] = std::min(infectionDist(rng), nodeSusceptible[node]);
            newRecoveries[node] = std::min(recoveryDist(rng), nodeInfected[node]);
        }

        int activeInfections = 0;
        for (int node = 0; node < nodeCount; ++node) {
            nodeSusceptible[node] -= newInfections[node];
            nodeInfected[node] += newInfections[node] - newRecoveries[node];
            nodeRecovered[node] += newRecoveries[node];
            activeInfections += nodeInfected[node];

            nodeDailyRecords.push_back({
                scenarioId,
                node,
                day,
                nodePopulations[node],
                nodeSusceptible[node],
                nodeInfected[node],
                nodeRecovered[node],
                newInfections[node],
                newRecoveries[node]
            });
        }

        if (activeInfections == 0) {
            break;
        }
    }

    return {summary, dailyRecords, nodeDailyRecords, edgeRecords};
}

void writeDataset(const std::vector<SimulationResult>& results, const std::string& outputPath) {
    const std::size_t slashPosition = outputPath.find_last_of("/\\");
    if (slashPosition != std::string::npos) {
        createDirectoryIfMissing(outputPath.substr(0, slashPosition));
    }

    std::ofstream file(outputPath);
    if (!file) {
        throw std::runtime_error("Unable to open output file: " + outputPath);
    }

    file << "scenario_id,population,initial_infected,infection_rate,recovery_rate,days,"
         << "final_susceptible,final_infected,final_recovered,peak_infected,"
         << "peak_day,total_infected,severity\n";

    file << std::fixed << std::setprecision(4);
    for (const auto& result : results) {
        file << result.scenarioId << ','
             << result.config.population << ','
             << result.config.initialInfected << ','
             << result.config.infectionRate << ','
             << result.config.recoveryRate << ','
             << result.config.days << ','
             << result.finalSusceptible << ','
             << result.finalInfected << ','
             << result.finalRecovered << ','
             << result.peakInfected << ','
             << result.peakDay << ','
             << result.totalInfected << ','
             << result.severity << '\n';
    }
}

void writeDailyCounts(const std::vector<DailyRecord>& records, const std::string& outputPath) {
    const std::size_t slashPosition = outputPath.find_last_of("/\\");
    if (slashPosition != std::string::npos) {
        createDirectoryIfMissing(outputPath.substr(0, slashPosition));
    }

    std::ofstream file(outputPath);
    if (!file) {
        throw std::runtime_error("Unable to open output file: " + outputPath);
    }

    file << "scenario_id,day,susceptible,infected,recovered,new_infections,new_recoveries\n";
    for (const auto& record : records) {
        file << record.scenarioId << ','
             << record.day << ','
             << record.susceptible << ','
             << record.infected << ','
             << record.recovered << ','
             << record.newInfections << ','
             << record.newRecoveries << '\n';
    }
}

void writeNodeTimeSeries(const std::vector<NodeDailyRecord>& records, const std::string& outputPath) {
    const std::size_t slashPosition = outputPath.find_last_of("/\\");
    if (slashPosition != std::string::npos) {
        createDirectoryIfMissing(outputPath.substr(0, slashPosition));
    }

    std::ofstream file(outputPath);
    if (!file) {
        throw std::runtime_error("Unable to open output file: " + outputPath);
    }

    file << "scenario_id,node_id,day,population,susceptible,infected,recovered,"
         << "new_infections,new_recoveries\n";
    for (const auto& record : records) {
        file << record.scenarioId << ','
             << record.nodeId << ','
             << record.day << ','
             << record.population << ','
             << record.susceptible << ','
             << record.infected << ','
             << record.recovered << ','
             << record.newInfections << ','
             << record.newRecoveries << '\n';
    }
}

void writeEdges(const std::vector<EdgeRecord>& records, const std::string& outputPath) {
    const std::size_t slashPosition = outputPath.find_last_of("/\\");
    if (slashPosition != std::string::npos) {
        createDirectoryIfMissing(outputPath.substr(0, slashPosition));
    }

    std::ofstream file(outputPath);
    if (!file) {
        throw std::runtime_error("Unable to open output file: " + outputPath);
    }

    file << std::fixed << std::setprecision(4);
    file << "scenario_id,source_node,target_node,weight\n";
    for (const auto& record : records) {
        file << record.scenarioId << ','
             << record.sourceNode << ','
             << record.targetNode << ','
             << record.weight << '\n';
    }
}

int main(int argc, char* argv[]) {
    const int scenarioCount = argc > 1 ? std::stoi(argv[1]) : 1000;
    const std::string outputPath =
        argc > 2 ? argv[2] : "results/epidemic_dataset.csv";
    const std::string dailyOutputPath =
        argc > 3 ? argv[3] : "results/daily_counts.csv";
    const std::string graphOutputPath =
        argc > 4 ? argv[4] : "results/graph_timeseries.csv";
    const std::string edgeOutputPath =
        argc > 5 ? argv[5] : "results/graph_edges.csv";

    std::random_device device;
    std::mt19937 rng(device());

    std::uniform_int_distribution<int> populationDist(500, 10000);
    std::uniform_int_distribution<int> daysDist(60, 180);
    std::uniform_real_distribution<double> infectionRateDist(0.05, 0.45);
    std::uniform_real_distribution<double> recoveryRateDist(0.02, 0.20);

    std::vector<SimulationResult> results;
    std::vector<DailyRecord> dailyRecords;
    std::vector<NodeDailyRecord> nodeDailyRecords;
    std::vector<EdgeRecord> edgeRecords;
    results.reserve(static_cast<std::size_t>(scenarioCount));

    for (int i = 0; i < scenarioCount; ++i) {
        const int population = populationDist(rng);
        const int maxInitialInfected = std::max(2, static_cast<int>(std::sqrt(population)));
        std::uniform_int_distribution<int> initialInfectedDist(1, maxInitialInfected);

        const SimulationConfig config {
            population,
            initialInfectedDist(rng),
            infectionRateDist(rng),
            recoveryRateDist(rng),
            daysDist(rng)
        };

        const ScenarioBundle bundle = runSimulation(i, config, rng);
        results.push_back(bundle.summary);
        dailyRecords.insert(dailyRecords.end(), bundle.dailyRecords.begin(), bundle.dailyRecords.end());
        nodeDailyRecords.insert(
            nodeDailyRecords.end(),
            bundle.nodeDailyRecords.begin(),
            bundle.nodeDailyRecords.end()
        );
        edgeRecords.insert(edgeRecords.end(), bundle.edgeRecords.begin(), bundle.edgeRecords.end());
    }

    writeDataset(results, outputPath);
    writeDailyCounts(dailyRecords, dailyOutputPath);
    writeNodeTimeSeries(nodeDailyRecords, graphOutputPath);
    writeEdges(edgeRecords, edgeOutputPath);

    std::cout << "Generated " << results.size() << " simulations at " << outputPath << '\n';
    std::cout << "Generated daily counts at " << dailyOutputPath << '\n';
    std::cout << "Generated graph time series at " << graphOutputPath << '\n';
    std::cout << "Generated graph edges at " << edgeOutputPath << '\n';
    return 0;
}
