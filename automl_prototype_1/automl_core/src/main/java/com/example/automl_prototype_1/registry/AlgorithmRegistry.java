package com.example.automl_prototype_1.registry;

import com.example.automl_prototype_1.algorithm.*;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;

/**
 * Central registry and SINGLE source of truth for all optimization algorithms.
 * Acts as a Factory to ensure fresh instances are created for each request.
 *
 * ALL algorithm name→instance resolution MUST go through this class.
 * Do NOT add switch/if-else algorithm mapping logic in other services.
 */
public class AlgorithmRegistry {
    private static final AlgorithmRegistry instance = new AlgorithmRegistry();

    public static AlgorithmRegistry getInstance() {
        return instance;
    }

    private AlgorithmRegistry() {
    }

    /**
     * Retrieves a new instance of an algorithm by its name/ID.
     * Uses case-insensitive matching for robustness.
     *
     * @param name The name/ID of the algorithm (e.g. "GA", "IGPSO", "Manual
     *             Selection").
     * @return A new OptimizationAlgorithm instance, or null if not found.
     */
    public OptimizationAlgorithm getAlgorithm(String name) {
        if (name == null)
            return null;

        // Case-insensitive matching to prevent silent null returns from case mismatches
        switch (name.toUpperCase()) {
            case "GA":
            case "GENETIC ALGORITHM":
                return new GeneticAlgorithm();
            case "IGPSO":
                return new IGPSO();
            case "WWO":
            case "WATER WAVE OPTIMIZATION":
                return new WaterWaveOptimization();
            case "BPSO":
            case "BINARY PSO":
                return new BinaryParticleSwarmOptimization();
            case "ASO":
            case "ATOM SEARCH OPTIMIZATION":
                return new AtomSearchOptimization();
            case "PLS":
            case "NNP":
            case "PARALLEL LOCAL SEARCH":
                return new ParallelLocalSearchAlgorithm();
            case "SA":
            case "SIMULATED ANNEALING":
                return new SimulatedAnnealing();
            case "GWO":
            case "GREY WOLF OPTIMIZER":
                return new GreyWolfOptimizer();
            case "DE":
            case "DIFFERENTIAL EVOLUTION":
                return new DifferentialEvolution();
            case "ABLATION GA":
                return new AblationAwareGeneticAlgorithm(50, 100);

            // Non-optimization pass-through algorithms
            case "MANUAL SELECTION":
                return new ManualSelectionAlgorithm();
            case "NO OPTIMIZATION":
            case "NO OPTIMIZATION (FULL FEATURE SET)":
                return new NoOptimizationAlgorithm();

            default:
                return null;
        }
    }

    /**
     * Creates a new algorithm instance with randomized hyperparameters for tuning.
     * Population size is randomized between 20-50, iterations between 50-150.
     *
     * @param name The algorithm name/ID.
     * @return A new algorithm with random hyperparameters, or null if not found.
     */
    public OptimizationAlgorithm createRandomized(String name) {
        if (name == null)
            return null;

        var rand = ThreadLocalRandom.current();
        int popSize = rand.nextInt(20, 51);
        int maxIter = rand.nextInt(50, 151);

        return switch (name.toUpperCase()) {
            case "GA", "GENETIC ALGORITHM" -> new GeneticAlgorithm(popSize, maxIter);
            case "BPSO", "BINARY PSO" -> new BinaryParticleSwarmOptimization(popSize, maxIter);
            case "WWO", "WATER WAVE OPTIMIZATION" -> new WaterWaveOptimization(popSize, maxIter);
            case "ASO", "ATOM SEARCH OPTIMIZATION" -> new AtomSearchOptimization(popSize, maxIter);
            case "IGPSO" -> new IGPSO(popSize, maxIter);
            case "SA", "SIMULATED ANNEALING" -> {
                double initialTemp = rand.nextDouble(0.8, 1.2);
                double coolingRate = rand.nextDouble(0.99, 0.999);
                yield new SimulatedAnnealing(maxIter * 5, initialTemp, coolingRate);
            }
            case "GWO", "GREY WOLF OPTIMIZER" -> new GreyWolfOptimizer(popSize, maxIter);
            case "DE", "DIFFERENTIAL EVOLUTION" -> new DifferentialEvolution(popSize, maxIter);
            case "PLS", "NNP", "PARALLEL LOCAL SEARCH" -> new ParallelLocalSearchAlgorithm(maxIter);
            default -> null;
        };
    }

    /**
     * Returns a list of all supported algorithm names (short codes).
     *
     * @return Unmodifiable list of names.
     */
    public List<String> getAlgorithmNames() {
        List<String> codes = new ArrayList<>();
        codes.add("GA");
        codes.add("IGPSO");
        codes.add("WWO");
        codes.add("BPSO");
        codes.add("ASO");
        codes.add("PLS");
        codes.add("SA");
        codes.add("GWO");
        codes.add("DE");
        return Collections.unmodifiableList(codes);
    }
}
