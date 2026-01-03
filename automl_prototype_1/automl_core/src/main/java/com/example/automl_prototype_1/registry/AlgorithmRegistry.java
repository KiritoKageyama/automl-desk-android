package com.example.automl_prototype_1.registry;

import com.example.automl_prototype_1.algorithm.OptimizationAlgorithm;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Central registry for managing available optimization algorithms.
 * Allows dynamic registration and discovery of algorithms.
 */
public class AlgorithmRegistry {
    private static final AlgorithmRegistry instance = new AlgorithmRegistry();

    public static AlgorithmRegistry getInstance() {
        return instance;
    }

    private final Map<String, OptimizationAlgorithm> algorithms = new ConcurrentHashMap<>();

    private AlgorithmRegistry() {
    }

    /**
     * Registers a new optimization algorithm.
     * 
     * @param algorithm The algorithm to register.
     */
    public void register(OptimizationAlgorithm algorithm) {
        if (algorithm != null) {
            algorithms.put(algorithm.getAlgorithmName(), algorithm);
            System.out.println("Registered algorithm: " + algorithm.getAlgorithmName());
        }
    }

    /**
     * Retrieves an algorithm by its name.
     * 
     * @param name The name of the algorithm.
     * @return The algorithm instance, or null if not found.
     */
    public OptimizationAlgorithm getAlgorithm(String name) {
        return algorithms.get(name);
    }

    /**
     * Returns a list of all registered algorithms.
     * 
     * @return Unmodifiable list of algorithms.
     */
    public List<OptimizationAlgorithm> getAllAlgorithms() {
        return Collections.unmodifiableList(new ArrayList<>(algorithms.values()));
    }

    /**
     * Returns a list of all registered algorithm names.
     * 
     * @return Unmodifiable list of names.
     */
    public List<String> getAlgorithmNames() {
        return Collections.unmodifiableList(new ArrayList<>(algorithms.keySet()));
    }
}
