#include <algorithm>
#include <cerrno>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
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
    double mortalityRate;
    double lockdownStrength;
    double maskAdoption;
    double vaccinationRate;
    double travelRestriction;
    double medianAge;
    double elderlyPopulationRatio;
    double childPopulationRatio;
    double temperatureCelsius;
    double humidityPercent;
    double rainfallMm;
    int days;
};

struct SimulationResult {
    int scenarioId;
    SimulationConfig config;
    int finalSusceptible;
    int finalInfected;
    int finalRecovered;
    int finalDeceased;
    int finalVaccinated;
    int peakInfected;
    int peakDay;
    int totalInfected;
    int totalDeaths;
    std::string severity;
};

struct DailyRecord {
    int scenarioId;
    int day;
    int susceptible;
    int infected;
    int recovered;
    int deceased;
    int vaccinated;
    int newInfections;
    int newRecoveries;
    int newDeaths;
    int newVaccinations;
};

struct NodeDailyRecord {
    int scenarioId;
    int nodeId;
    int day;
    int population;
    int susceptible;
    int infected;
    int recovered;
    int deceased;
    int vaccinated;
    int newInfections;
    int newRecoveries;
    int newDeaths;
    int newVaccinations;
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

std::string classifySeverity(double infectedShare, double deathShare) {
    const double riskScore = infectedShare + 2.5 * deathShare;
    if (riskScore < 0.20) {
        return "low";
    }
    if (riskScore < 0.50) {
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
    if (directory.empty()) {
        return;
    }

    std::string current;
    for (std::size_t index = 0; index <= directory.size(); ++index) {
        const bool atEnd = index == directory.size();
        const char character = atEnd ? '/' : directory[index];
        if (character != '/' && character != '\\') {
            current += character;
            continue;
        }

        if (!current.empty() && !(current.size() == 2 && current[1] == ':')) {
#ifdef _WIN32
            if (_mkdir(current.c_str()) != 0 && errno != EEXIST) {
                throw std::runtime_error("Unable to create directory: " + current);
            }
#else
            if (mkdir(current.c_str(), 0755) != 0 && errno != EEXIST) {
                throw std::runtime_error("Unable to create directory: " + current);
            }
#endif
        }

        if (!atEnd) {
            current += character;
        }
    }
}

double effectiveInfectionRate(const SimulationConfig& config) {
    const double policyEffect =
        1.0
        - 0.55 * config.lockdownStrength
        - 0.35 * config.maskAdoption
        - 0.20 * config.travelRestriction;
    const double temperatureFactor = 1.0 + std::max(0.0, 22.0 - config.temperatureCelsius) * 0.012;
    const double humidityFactor = 1.0 + std::max(0.0, 45.0 - config.humidityPercent) * 0.006;
    const double rainfallFactor = 1.0 - std::min(0.18, config.rainfallMm * 0.004);
    const double childContactFactor = 1.0 + config.childPopulationRatio * 0.25;
    return config.infectionRate
        * std::max(0.0, policyEffect)
        * temperatureFactor
        * humidityFactor
        * rainfallFactor
        * childContactFactor;
}

double effectiveMortalityRate(const SimulationConfig& config) {
    const double ageFactor =
        1.0
        + std::max(0.0, config.medianAge - 35.0) * 0.018
        + config.elderlyPopulationRatio * 1.8;
    return config.mortalityRate * ageFactor;
}

ScenarioBundle runSimulation(int scenarioId, const SimulationConfig& config, std::mt19937& rng) {
    int susceptible = config.population - config.initialInfected;
    int infected = config.initialInfected;
    int recovered = 0;
    int deceased = 0;
    int vaccinated = 0;
    int peakInfected = infected;
    int peakDay = 0;
    std::vector<DailyRecord> dailyRecords;

    dailyRecords.push_back({scenarioId, 0, susceptible, infected, recovered, deceased, vaccinated, 0, 0, 0, 0});

    for (int day = 1; day <= config.days; ++day) {
        const int newVaccinations = std::min(
            susceptible,
            static_cast<int>(std::round(config.population * config.vaccinationRate))
        );
        susceptible -= newVaccinations;
        vaccinated += newVaccinations;

        const double contactProbability =
            effectiveInfectionRate(config) * static_cast<double>(infected) / config.population;
        const double boundedContactProbability = clampProbability(contactProbability);
        const double boundedRecoveryProbability = clampProbability(config.recoveryRate);
        const double boundedMortalityProbability = clampProbability(effectiveMortalityRate(config));

        std::binomial_distribution<int> newInfectionsDist(susceptible, boundedContactProbability);
        const int newInfections = std::min(newInfectionsDist(rng), susceptible);
        int infectedAfterSpread = infected + newInfections;

        std::binomial_distribution<int> newDeathsDist(infectedAfterSpread, boundedMortalityProbability);
        const int newDeaths = std::min(newDeathsDist(rng), infectedAfterSpread);
        infectedAfterSpread -= newDeaths;

        std::binomial_distribution<int> newRecoveriesDist(infectedAfterSpread, boundedRecoveryProbability);
        const int newRecoveries = std::min(newRecoveriesDist(rng), infectedAfterSpread);

        susceptible -= newInfections;
        infected = infectedAfterSpread - newRecoveries;
        recovered += newRecoveries;
        deceased += newDeaths;

        dailyRecords.push_back({
            scenarioId,
            day,
            susceptible,
            infected,
            recovered,
            deceased,
            vaccinated,
            newInfections,
            newRecoveries,
            newDeaths,
            newVaccinations
        });

        if (infected > peakInfected) {
            peakInfected = infected;
            peakDay = day;
        }

        if (infected == 0) {
            break;
        }
    }

    const int totalInfected = recovered + infected + deceased;
    const double infectedShare = static_cast<double>(totalInfected) / config.population;
    const double deathShare = static_cast<double>(deceased) / config.population;

    const SimulationResult summary {
        scenarioId,
        config,
        susceptible,
        infected,
        recovered,
        deceased,
        vaccinated,
        peakInfected,
        peakDay,
        totalInfected,
        deceased,
        classifySeverity(infectedShare, deathShare)
    };

    const int nodeCount = 8;
    std::vector<int> nodePopulations(nodeCount, config.population / nodeCount);
    nodePopulations.back() += config.population % nodeCount;

    std::vector<int> nodeSusceptible(nodeCount);
    std::vector<int> nodeInfected(nodeCount, 0);
    std::vector<int> nodeRecovered(nodeCount, 0);
    std::vector<int> nodeDeceased(nodeCount, 0);
    std::vector<int> nodeVaccinated(nodeCount, 0);
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
            nodeDeceased[node],
            nodeVaccinated[node],
            0,
            0,
            0,
            0
        });
    }

    for (int day = 1; day <= config.days; ++day) {
        std::vector<int> newInfections(nodeCount, 0);
        std::vector<int> newRecoveries(nodeCount, 0);
        std::vector<int> newDeaths(nodeCount, 0);
        std::vector<int> newVaccinations(nodeCount, 0);

        for (int node = 0; node < nodeCount; ++node) {
            newVaccinations[node] = std::min(
                nodeSusceptible[node],
                static_cast<int>(std::round(nodePopulations[node] * config.vaccinationRate))
            );
            nodeSusceptible[node] -= newVaccinations[node];
            nodeVaccinated[node] += newVaccinations[node];

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
                effectiveInfectionRate(config) * (0.75 * localPressure + 0.25 * neighborPressure);

            std::binomial_distribution<int> infectionDist(
                nodeSusceptible[node],
                clampProbability(contactProbability)
            );

            newInfections[node] = std::min(infectionDist(rng), nodeSusceptible[node]);
            int infectedAfterSpread = nodeInfected[node] + newInfections[node];

            std::binomial_distribution<int> deathDist(
                infectedAfterSpread,
                clampProbability(effectiveMortalityRate(config))
            );
            newDeaths[node] = std::min(deathDist(rng), infectedAfterSpread);
            infectedAfterSpread -= newDeaths[node];

            std::binomial_distribution<int> recoveryDist(
                infectedAfterSpread,
                clampProbability(config.recoveryRate)
            );
            newRecoveries[node] = std::min(recoveryDist(rng), infectedAfterSpread);
        }

        int activeInfections = 0;
        for (int node = 0; node < nodeCount; ++node) {
            nodeSusceptible[node] -= newInfections[node];
            nodeInfected[node] += newInfections[node] - newRecoveries[node] - newDeaths[node];
            nodeRecovered[node] += newRecoveries[node];
            nodeDeceased[node] += newDeaths[node];
            activeInfections += nodeInfected[node];

            nodeDailyRecords.push_back({
                scenarioId,
                node,
                day,
                nodePopulations[node],
                nodeSusceptible[node],
                nodeInfected[node],
                nodeRecovered[node],
                nodeDeceased[node],
                nodeVaccinated[node],
                newInfections[node],
                newRecoveries[node],
                newDeaths[node],
                newVaccinations[node]
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

    file << "scenario_id,population,initial_infected,infection_rate,recovery_rate,mortality_rate,"
         << "lockdown_strength,mask_adoption,vaccination_rate,travel_restriction,"
         << "median_age,elderly_population_ratio,child_population_ratio,"
         << "temperature_celsius,humidity_percent,rainfall_mm,days,"
         << "final_susceptible,final_infected,final_recovered,final_deceased,final_vaccinated,"
         << "peak_infected,peak_day,total_infected,total_deaths,severity\n";

    file << std::fixed << std::setprecision(4);
    for (const auto& result : results) {
        file << result.scenarioId << ','
             << result.config.population << ','
             << result.config.initialInfected << ','
             << result.config.infectionRate << ','
             << result.config.recoveryRate << ','
             << result.config.mortalityRate << ','
             << result.config.lockdownStrength << ','
             << result.config.maskAdoption << ','
             << result.config.vaccinationRate << ','
             << result.config.travelRestriction << ','
             << result.config.medianAge << ','
             << result.config.elderlyPopulationRatio << ','
             << result.config.childPopulationRatio << ','
             << result.config.temperatureCelsius << ','
             << result.config.humidityPercent << ','
             << result.config.rainfallMm << ','
             << result.config.days << ','
             << result.finalSusceptible << ','
             << result.finalInfected << ','
             << result.finalRecovered << ','
             << result.finalDeceased << ','
             << result.finalVaccinated << ','
             << result.peakInfected << ','
             << result.peakDay << ','
             << result.totalInfected << ','
             << result.totalDeaths << ','
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

    file << "scenario_id,day,susceptible,infected,recovered,deceased,vaccinated,"
         << "new_infections,new_recoveries,new_deaths,new_vaccinations\n";
    for (const auto& record : records) {
        file << record.scenarioId << ','
             << record.day << ','
             << record.susceptible << ','
             << record.infected << ','
             << record.recovered << ','
             << record.deceased << ','
             << record.vaccinated << ','
             << record.newInfections << ','
             << record.newRecoveries << ','
             << record.newDeaths << ','
             << record.newVaccinations << '\n';
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

    file << "scenario_id,node_id,day,population,susceptible,infected,recovered,deceased,vaccinated,"
         << "new_infections,new_recoveries,new_deaths,new_vaccinations\n";
    for (const auto& record : records) {
        file << record.scenarioId << ','
             << record.nodeId << ','
             << record.day << ','
             << record.population << ','
             << record.susceptible << ','
             << record.infected << ','
             << record.recovered << ','
             << record.deceased << ','
             << record.vaccinated << ','
             << record.newInfections << ','
             << record.newRecoveries << ','
             << record.newDeaths << ','
             << record.newVaccinations << '\n';
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

int parseScenarioCount(int argc, char* argv[]) {
    if (argc <= 1) {
        return 1000;
    }

    try {
        std::size_t consumed = 0;
        const std::string argument(argv[1]);
        const int value = std::stoi(argument, &consumed);
        if (consumed != argument.size() || value <= 0) {
            throw std::invalid_argument("scenario count must be positive");
        }
        return value;
    } catch (const std::exception&) {
        throw std::runtime_error("Scenario count must be a positive integer.");
    }
}

unsigned int parseSeed(int argc, char* argv[]) {
    if (argc <= 6) {
        return std::random_device{}();
    }

    try {
        std::size_t consumed = 0;
        const std::string argument(argv[6]);
        const unsigned long value = std::stoul(argument, &consumed);
        if (consumed != argument.size() || value > std::numeric_limits<unsigned int>::max()) {
            throw std::invalid_argument("seed must be an unsigned integer");
        }
        return static_cast<unsigned int>(value);
    } catch (const std::exception& error) {
        throw std::runtime_error("Seed must be an unsigned integer.");
    }
}

void printUsage(const char* executable) {
    std::cout
        << "Usage: " << executable << " [scenario_count] [summary_csv] [daily_csv] "
        << "[graph_csv] [edges_csv] [seed]\n"
        << "Defaults: 1000 scenarios, results/*.csv, random seed.\n";
}

int runApplication(int argc, char* argv[]) {
    if (argc > 1 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
        printUsage(argv[0]);
        return 0;
    }

    const int scenarioCount = parseScenarioCount(argc, argv);
    const unsigned int seed = parseSeed(argc, argv);
    const std::string outputPath =
        argc > 2 ? argv[2] : "results/epidemic_dataset.csv";
    const std::string dailyOutputPath =
        argc > 3 ? argv[3] : "results/daily_counts.csv";
    const std::string graphOutputPath =
        argc > 4 ? argv[4] : "results/graph_timeseries.csv";
    const std::string edgeOutputPath =
        argc > 5 ? argv[5] : "results/graph_edges.csv";

    std::mt19937 rng(seed);

    std::uniform_int_distribution<int> populationDist(500, 10000);
    std::uniform_int_distribution<int> daysDist(60, 180);
    std::uniform_real_distribution<double> infectionRateDist(0.05, 0.45);
    std::uniform_real_distribution<double> recoveryRateDist(0.02, 0.20);
    std::uniform_real_distribution<double> mortalityRateDist(0.001, 0.025);
    std::uniform_real_distribution<double> lockdownStrengthDist(0.0, 0.8);
    std::uniform_real_distribution<double> maskAdoptionDist(0.0, 0.9);
    std::uniform_real_distribution<double> vaccinationRateDist(0.0, 0.01);
    std::uniform_real_distribution<double> travelRestrictionDist(0.0, 0.8);
    std::uniform_real_distribution<double> medianAgeDist(18.0, 55.0);
    std::uniform_real_distribution<double> elderlyPopulationRatioDist(0.03, 0.24);
    std::uniform_real_distribution<double> childPopulationRatioDist(0.12, 0.34);
    std::uniform_real_distribution<double> temperatureDist(5.0, 40.0);
    std::uniform_real_distribution<double> humidityDist(25.0, 95.0);
    std::uniform_real_distribution<double> rainfallDist(0.0, 45.0);

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
            mortalityRateDist(rng),
            lockdownStrengthDist(rng),
            maskAdoptionDist(rng),
            vaccinationRateDist(rng),
            travelRestrictionDist(rng),
            medianAgeDist(rng),
            elderlyPopulationRatioDist(rng),
            childPopulationRatioDist(rng),
            temperatureDist(rng),
            humidityDist(rng),
            rainfallDist(rng),
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

    std::cout << "Seed: " << seed << '\n';
    std::cout << "Generated " << results.size() << " simulations at " << outputPath << '\n';
    std::cout << "Generated daily counts at " << dailyOutputPath << '\n';
    std::cout << "Generated graph time series at " << graphOutputPath << '\n';
    std::cout << "Generated graph edges at " << edgeOutputPath << '\n';
    return 0;
}

int main(int argc, char* argv[]) {
    try {
        return runApplication(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
