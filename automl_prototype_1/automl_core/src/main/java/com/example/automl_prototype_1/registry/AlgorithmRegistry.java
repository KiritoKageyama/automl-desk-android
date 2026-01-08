package com.example.automl_prototype_1.registry;

import com.example.automl_prototype_1.algorithm.*;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
// import java.util.function.Supplier; // Removed unused import

/**
 * Central registry and factory for managing available optimization algorithms.
 * Acts as a Factory to ensure fresh instances are created for each request.
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
     * 
     * @param name The name/ID of the algorithm (e.g. "GA", "IGPSO").
     * @return A new OptimizationAlgorithm instance, or null if not found.
     */
    public OptimizationAlgorithm getAlgorithm(String name) {
        if (name == null)
            return null;

        // Factory Logic - Returns FRESH instances to avoid state sharing
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
                return new AblationAwareGeneticAlgorithm(50, 100); // Default test params
            default:
                return null;
        }
    }

    /**
     * Returns a list of all supported algorithm names.
     * 
     * @return List of names.
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
