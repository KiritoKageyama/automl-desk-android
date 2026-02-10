package com.example.automl_prototype_1.algorithm;

import com.example.automl_prototype_1.model.Dataset;
import com.example.automl_prototype_1.model.ExecutionResult;
import com.example.automl_prototype_1.service.IClassifier;

import java.util.Map; // For configuration later

// Interface for all optimization algorithms
public interface OptimizationAlgorithm {

    String getAlgorithmName();

    /**
     * Executes the algorithm on the given dataset.
     * 
     * @param dataset       The input dataset.
     * @param configuration Algorithm-specific configuration (optional for now).
     * @return An ExecutionResult object containing performance metrics.
     */
    ExecutionResult execute(Dataset dataset, Map<String, Object> configuration);

    /**
     * Set a manual override for the maximum number of iterations/generations.
     * 
     * @param maxIterations The new maximum.
     */
    default void setMaxIterations(int maxIterations) {
    }

    /**
     * Executes the algorithm with specific trial context for progress tracking.
     * Default implementation delegates to execute() for backward compatibility.
     */
    default ExecutionResult optimize(Dataset dataset, IClassifier classifier, int trialNum, int totalTrials) {
        return optimize(dataset, classifier, new java.util.HashMap<>(), trialNum, totalTrials);
    }

    /**
     * Executes the algorithm with specific trial context AND configuration.
     * This is the master method that implementations like
     * AbstractFeatureSelectionAlgorithm override.
     */
    default ExecutionResult optimize(Dataset dataset, IClassifier classifier, Map<String, Object> configuration,
            int trialNum, int totalTrials) {
        if (configuration == null)
            configuration = new java.util.HashMap<>();
        configuration.put("classifier", classifier);
        return execute(dataset, configuration);
    }
}