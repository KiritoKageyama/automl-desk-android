package com.example.automl_prototype_1.service;

import com.example.automl_prototype_1.algorithm.*;
import com.example.automl_prototype_1.model.AlgorithmRecommendation;
import com.example.automl_prototype_1.model.Dataset;
import com.example.automl_prototype_1.model.ExecutionResult;
import com.example.automl_prototype_1.classifier.*;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Service responsible for loading data, providing recommendations, and
 * executing algorithms.
 * This version is backwards-compatible with the original UI controllers.
 */
public class ExecutionService {

    private final RecommendationService recommendationService;

    /**
     * Constructor initializes the new services.
     */
    public ExecutionService() {
        this.recommendationService = new RecommendationService();
        LogService.info("CORE", "ExecutionService Initialized.");
    }

    /**
     * Executes the selected optimization algorithms SYNCHRONOUSLY.
     * This method is kept for compatibility with existing UI controllers.
     * It now requires a pre-selected classifier to be passed in the configuration.
     *
     * @param dataset                The loaded Dataset object.
     * @param selectedAlgorithmNames List of names of algorithms to run.
     * @param configuration          Must contain an IClassifier instance under the
     *                               key "classifier".
     * @return A List containing ExecutionResult objects.
     */
    public List<ExecutionResult> executeAlgorithms(Dataset dataset,
            List<String> selectedAlgorithmNames,
            Map<String, Object> configuration) throws Exception {
        if (dataset == null) {
            throw new IllegalArgumentException("Dataset cannot be null for execution.");
        }
        if (selectedAlgorithmNames == null || selectedAlgorithmNames.isEmpty()) {
            return new ArrayList<>();
        }

        // Initialize Logging
        LogService.getInstance().clear();
        LogService.info("CORE", "Starting execution pipeline.");

        // Check Hardware Acceleration
        boolean gpuAvailable = HardwareAcceleratorService.isGpuAvailable();
        if (gpuAvailable) {
            String gpuName = HardwareAcceleratorService.getGpuName();
            LogService.info("CORE", "Hardware Acceleration Detected: " + gpuName);
        } else {
            LogService.info("CORE", "No GPU detected. Using CPU for all tasks.");
        }

        // Apply Scaling
        if (AppStateService.getInstance().isStandardizationEnabled()) {
            LogService.info("CORE", "Applying Standardization (Z-Score) to Dataset...");
            dataset.standardize();
        }

        if (configuration == null || !(configuration.get("classifier") instanceof IClassifier)) {
            LogService.info("CORE", "Warning: No classifier provided. Defaulting to KnnClassifier.");
            configuration = new HashMap<>();
            configuration.put("classifier", new KnnClassifier());
        }
        IClassifier classifier = (IClassifier) configuration.get("classifier");

        LogService.info("CORE", "Starting PARALLEL execution of " + selectedAlgorithmNames.size()
                + " algorithms with classifier: " + classifier.getName());

        // Create a Thread Pool for parallel execution of algorithms
        java.util.concurrent.ExecutorService executor = java.util.concurrent.Executors.newCachedThreadPool();
        List<java.util.concurrent.Future<ExecutionResult>> futures = new ArrayList<>();
        // final IClassifier finalClassifier = classifier; // Unused
        final Map<String, Object> finalConfig = configuration;

        for (String algoName : selectedAlgorithmNames) {
            futures.add(executor.submit(() -> {
                long threadId = Thread.currentThread().getId();
                String resourceType = (finalConfig
                        .get("classifier") instanceof com.example.automl_prototype_1.classifier.GpuKnnClassifier ||
                        finalConfig.get(
                                "classifier") instanceof com.example.automl_prototype_1.classifier.GpuLogisticRegression)
                                        ? "GPU"
                                        : "CPU";
                LogService.info("CORE", "[Thread " + threadId + "] Running " + algoName + " on " + resourceType);
                OptimizationAlgorithm algorithm = getAlgorithmImplementation(algoName);
                if (algorithm != null) {
                    try {
                        return algorithm.execute(dataset, finalConfig);
                    } catch (Exception e) {
                        LogService.error("CORE", "Error executing " + algoName + ": " + e.getMessage());
                        return new ExecutionResult(algoName + " (Failed)", 0, 0, 0, 0L);
                    }
                } else {
                    return new ExecutionResult(algoName + " (Not Found)", 0, 0, 0, 0L);
                }
            }));
        }

        List<ExecutionResult> allResults = new ArrayList<>();
        for (java.util.concurrent.Future<ExecutionResult> future : futures) {
            try {
                allResults.add(future.get()); // Wait for completion
            } catch (Exception e) {
                LogService.error("CORE", "Error retrieving future result: " + e.getMessage());
            }
        }

        executor.shutdown();
        return allResults;
    }

    /**
     * Returns a ranked list of algorithm recommendations for a given dataset.
     * 
     * @param dataset The dataset to analyze.
     * @return A sorted list of AlgorithmRecommendation objects.
     */
    public List<AlgorithmRecommendation> getRecommendations(Dataset dataset) {
        return recommendationService.getRecommendations(dataset);
    }

    /**
     * A high-level method that gets recommendations and then runs the top-ranked
     * one.
     * Simulates the full, automated AutoML workflow.
     *
     * @param dataset The dataset to analyze and run on.
     * @return The ExecutionResult from the best recommended algorithm.
     */
    public ExecutionResult runTopRecommendation(Dataset dataset) {
        List<AlgorithmRecommendation> recommendations = getRecommendations(dataset);
        if (recommendations.isEmpty()) {
            LogService.error("CORE", "No recommendations could be generated for this dataset.");
            return new ExecutionResult("No Recommendation", 0, 0, 0, 0L);
        }
        AlgorithmRecommendation topChoice = recommendations.get(0);
        LogService.info("CORE", "Top Recommendation: " + topChoice.optimizer().getAlgorithmName() + " with "
                + topChoice.classifier().getName());
        LogService.info("CORE", "Reason: " + topChoice.reason());

        return runExperiment(topChoice.optimizer(), topChoice.classifier(), dataset);
    }

    /**
     * Runs a specific combination of an optimization algorithm and a classifier.
     * 
     * @param algorithm  The optimization algorithm instance.
     * @param classifier The base classifier instance.
     * @param dataset    The dataset for the experiment.
     * @return An ExecutionResult.
     */
    public ExecutionResult runExperiment(OptimizationAlgorithm algorithm, IClassifier classifier, Dataset dataset) {
        Map<String, Object> configuration = new HashMap<>();
        configuration.put("classifier", classifier);
        return algorithm.execute(dataset, configuration);
    }

    private OptimizationAlgorithm getAlgorithmImplementation(String name) {
        if (name == null)
            return null;
        // FIX: Handle aliases for GA and SA
        if ("Genetic Algorithm".equalsIgnoreCase(name) || "GA".equalsIgnoreCase(name))
            return new GeneticAlgorithm();
        if ("IGPSO".equalsIgnoreCase(name))
            return new IGPSO();
        if ("WWO".equalsIgnoreCase(name))
            return new WaterWaveOptimization();
        if ("BPSO".equalsIgnoreCase(name))
            return new BinaryParticleSwarmOptimization();
        if ("ASO".equalsIgnoreCase(name))
            return new AtomSearchOptimization();
        if ("PLS".equalsIgnoreCase(name) || "NNP".equalsIgnoreCase(name)) // Changed NNP to PLS and kept NNP as alias
            return new ParallelLocalSearchAlgorithm();
        if ("simulated annealing".equalsIgnoreCase(name) || "SA".equalsIgnoreCase(name))
            return new SimulatedAnnealing();
        if ("GWO".equalsIgnoreCase(name) || "Grey Wolf Optimizer".equalsIgnoreCase(name))
            return new GreyWolfOptimizer();
        if ("DE".equalsIgnoreCase(name) || "Differential Evolution".equalsIgnoreCase(name))
            return new DifferentialEvolution();

        LogService.error("ExecutionService", "No implementation found for algorithm name '" + name + "'");
        return null;
    }
}