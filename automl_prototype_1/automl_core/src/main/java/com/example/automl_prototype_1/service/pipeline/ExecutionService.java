package com.example.automl_prototype_1.service.pipeline;

import com.example.automl_prototype_1.service.LogService;
import com.example.automl_prototype_1.service.AppStateService;
import com.example.automl_prototype_1.service.IClassifier;

import com.example.automl_prototype_1.algorithm.AblationAwareGeneticAlgorithm;
import com.example.automl_prototype_1.algorithm.AtomSearchOptimization;
import com.example.automl_prototype_1.algorithm.BinaryParticleSwarmOptimization;
import com.example.automl_prototype_1.algorithm.DifferentialEvolution;
import com.example.automl_prototype_1.algorithm.GeneticAlgorithm;
import com.example.automl_prototype_1.algorithm.GreyWolfOptimizer;
import com.example.automl_prototype_1.algorithm.IGPSO;
import com.example.automl_prototype_1.algorithm.OptimizationAlgorithm;
import com.example.automl_prototype_1.algorithm.NoOptimizationAlgorithm;
import com.example.automl_prototype_1.algorithm.ParallelLocalSearchAlgorithm;
import com.example.automl_prototype_1.algorithm.SimulatedAnnealing;
import com.example.automl_prototype_1.algorithm.WaterWaveOptimization;
import com.example.automl_prototype_1.model.AlgorithmRecommendation;
import com.example.automl_prototype_1.model.Dataset;
import com.example.automl_prototype_1.model.ExecutionResult;
import com.example.automl_prototype_1.classifier.cpu.KnnClassifier;
// GpuKnnClassifier and GpuLogisticRegression are used FQCN in code, so imports removed.
// Services in same package (com.example.automl_prototype_1.service) do not need imports.

import com.example.automl_prototype_1.gpu.PythonExecutionService;
import com.example.automl_prototype_1.gpu.HardwareAcceleratorService;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Service responsible for loading data, providing recommendations, and
 * executing algorithms.
 * This version is backwards-compatible with the original UI controllers.
 */
public class ExecutionService { // Forced IDE Re-index

    private final RecommendationService recommendationService;

    private static ExecutionService instance;

    /**
     * Constructor initializes the new services.
     */
    public ExecutionService() {
        this.recommendationService = new RecommendationService();
        LogService.info("CORE", "ExecutionService Initialized.");
    }

    public static synchronized ExecutionService getInstance() {
        if (instance == null) {
            instance = new ExecutionService();
        }
        return instance;
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
    private java.util.concurrent.ExecutorService executor;

    public void stop() {
        LogService.info("CORE", "STOP Requested. Terminating processes...");
        if (executor != null && !executor.isShutdown()) {
            executor.shutdownNow();
        }
        PythonExecutionService.getInstance().terminateAll();
        LogService.error("CORE", "=== EXECUTION FORCEFULLY STOPPED BY USER ===");
    }

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

        // --- Phase 3: Pipeline Control (Scaling) ---
        String scaling = AppStateService.getInstance().getScalingStrategy();
        LogService.info("CORE", "Scaling Strategy: " + scaling + " (delegated to algorithm execution)");

        if (configuration == null) {
            configuration = new HashMap<>();
        }
        configuration.put("scaling", scaling);
        configuration.put("imputation", AppStateService.getInstance().getPreprocessingConfig().isImputeMissingValues());

        // --- Phase 3: Feature Selection Bypass ---
        boolean fsEnabled = AppStateService.getInstance().isFeatureSelectionEnabled();
        List<String> finalAlgorithms = new ArrayList<>(selectedAlgorithmNames);

        if (!fsEnabled) {
            LogService.info("CORE", "Feature Selection DISABLED. Forcing 'No Optimization' pass-through.");
            finalAlgorithms.clear();
            finalAlgorithms.add("No Optimization (Full Feature Set)");
        } else if (AppStateService.getInstance().isManualFeatureSelectionEnabled()) {
            LogService.info("CORE", "Manual Feature Selection ENABLED. Overriding algorithms.");
            finalAlgorithms.clear();
            finalAlgorithms.add("Manual Selection");
        }

        if (configuration == null || !(configuration.get("classifier") instanceof IClassifier)) {
            LogService.info("CORE", "Warning: No classifier provided. Defaulting to KnnClassifier.");
            configuration = new HashMap<>();
            configuration.put("classifier", new KnnClassifier());
        }
        IClassifier classifier = (IClassifier) configuration.get("classifier");

        // --- Phase 3: Image Classification Support ---
        if (dataset.getDataType() == Dataset.DataType.IMAGE
                && classifier instanceof com.example.automl_prototype_1.classifier.CnnClassifier) {
            ((com.example.automl_prototype_1.classifier.CnnClassifier) classifier)
                    .setImagePaths(dataset.getFilePaths());
            LogService.info("CORE", "Injected " + dataset.getFilePaths().size() + " image paths into CnnClassifier.");
        }

        // --- Phase 7: Gen AI Text Support ---
        if (dataset.getDataType() == Dataset.DataType.TEXT
                && classifier instanceof com.example.automl_prototype_1.classifier.BertClassifier) {
            ((com.example.automl_prototype_1.classifier.BertClassifier) classifier)
                    .setFilePaths(dataset.getFilePaths());
            LogService.info("CORE", "Injected " + (dataset.getFilePaths() != null ? dataset.getFilePaths().size() : 0)
                    + " text paths into BertClassifier.");
        }

        LogService.info("CORE", "Starting PARALLEL execution of " + finalAlgorithms.size()
                + " algorithms with classifier: " + classifier.getName());

        // Create a Thread Pool for parallel execution of algorithms
        this.executor = java.util.concurrent.Executors.newCachedThreadPool();
        List<java.util.concurrent.Future<ExecutionResult>> futures = new ArrayList<>();
        // final IClassifier finalClassifier = classifier; // Unused
        final Map<String, Object> finalConfig = configuration;

        for (String algoName : finalAlgorithms) {
            futures.add(executor.submit(() -> {
                if (Thread.currentThread().isInterrupted())
                    return null;
                // long threadId = Thread.currentThread().getId();
                long threadId = Thread.currentThread().getId();
                String resourceType = (finalConfig
                        .get("classifier") instanceof com.example.automl_prototype_1.classifier.gpu.GpuKnnClassifier ||
                        finalConfig.get(
                                "classifier") instanceof com.example.automl_prototype_1.classifier.gpu.GpuLogisticRegression)
                                        ? "GPU"
                                        : "CPU";
                LogService.info("CORE", "[Thread " + threadId + "] Running " + algoName + " on " + resourceType);
                OptimizationAlgorithm algorithm = getAlgorithmImplementation(algoName);
                if (algorithm != null) {
                    try {
                        ExecutionResult result = algorithm.execute(dataset, finalConfig);
                        // Self-Correction Loop: log success
                        com.example.automl_prototype_1.service.agentic.SelfCorrectionLoopService.getInstance()
                                .onExecutionComplete(algoName, true, null, result.getAccuracy());
                        return result;
                    } catch (Exception e) {
                        LogService.error("CORE", "Error executing " + algoName + ": " + e.getMessage());
                        // Self-Correction Loop: log failure
                        com.example.automl_prototype_1.service.agentic.SelfCorrectionLoopService.getInstance()
                                .onExecutionComplete(algoName, false, e.getClass().getSimpleName(), 0.0);
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
                // Global safety timeout: 3 hours per algorithm (generous, but prevents eternal
                // dev hangs)
                ExecutionResult res = future.get(3, java.util.concurrent.TimeUnit.HOURS);
                if (res != null)
                    allResults.add(res);
            } catch (java.util.concurrent.CancellationException ce) {
                LogService.info("CORE", "Task cancelled.");
            } catch (java.util.concurrent.TimeoutException te) {
                LogService.error("CORE", "Algorithm execution timed out (3 hours). cancelling...");
                future.cancel(true);
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
        if ("Ablation GA".equalsIgnoreCase(name))
            return new AblationAwareGeneticAlgorithm(50, 100);
        if (name.contains("No Optimization") || name.contains("Full Feature Set"))
            return new NoOptimizationAlgorithm();
        if ("Manual Selection".equalsIgnoreCase(name))
            return new com.example.automl_prototype_1.algorithm.ManualSelectionAlgorithm();

        LogService.error("ExecutionService", "No implementation found for algorithm name '" + name + "'");
        return null;
    }

    /**
     * Retrains the best model on the FULL dataset using the selected features and
     * global settings.
     * This step is crucial for generating a deployable model (ONNX).
     */
    public void refitBestModel(ExecutionResult result, IClassifier classifier, Dataset dataset) {
        if (result.getFeatureIndices() == null) {
            LogService.warn("ExecutionService", "Cannot refit model: No feature indices found in result.");
            return;
        }

        LogService.info("CORE", "Refitting Final Model on Full Dataset (100% data) for Export...");

        // 1. Prepare Global Config (Enable Scaling/Imputation for Final Model)
        com.example.automl_prototype_1.model.PreprocessingConfig config = AppStateService.getInstance()
                .getPreprocessingConfig();
        // Force scaling/imputation to be active as per global settings
        // Dashboard disabled them for CV, but here we want them.

        // 2. Run Preprocessing (Clean, Impute, Scale)
        com.example.automl_prototype_1.service.pipeline.PreprocessorService preprocessor = new com.example.automl_prototype_1.service.pipeline.PreprocessorService();
        String targetCol = dataset.getTargetColumnName();
        Dataset cleanData = preprocessor.runPreprocessingPipeline(dataset, targetCol, config);

        // 3. Filter Features
        double[][] X = cleanData.getFeatures();
        int[] y = cleanData.getLabelVector(); // Assuming labels are preserved
        if (y == null) {
            LogService.error("CORE", "Cannot refit: No labels found in processed dataset.");
            return;
        }

        int[] indices = result.getFeatureIndices();
        // Validate indices
        int maxIndex = 0;
        for (int idx : indices)
            maxIndex = Math.max(maxIndex, idx);

        if (maxIndex >= X[0].length) {
            LogService.error("CORE", "Feature Indices out of bounds. Max: " + maxIndex + ", Cols: " + X[0].length);
            // Verify if preprocessing removed columns (e.g. ID removal)
            // Indices come from Algorithm which used Preprocessed folds?
            // AbstractFeatureSelectionAlgorithm used Folds created from RAW data.
            // Wait. createStratifiedFolds creates folds from provided features.
            // Optimize calls 'dataset.getFeatureMatrix(TARGET)'.
            // Does 'runPreprocessingPipeline' modify column counts (e.g. remove ID)?
            // YES. 'autoRemoveIdColumns'.
            // So if Validation used logic that skipped ID columns...
            // AbstractFeatureSelectionAlgorithm receives 'dataset'.
            // In 'executeAlgorithms', dataset is PASSED IN.
            // In DashboardController, 'finalDataset' is passed to 'executeAlgorithms'.
            // 'finalDataset' is the result of 'runPreprocessingPipeline(safeConfig)'.
            // 'safeConfig' has 'autoRemoveIdColumns(true)' (inherited from config).
            // So 'finalDataset' HAS NO IDs.
            // 'indices' are based on 'finalDataset'.
            // HERE in 'refitBestModel', we act on 'dataset' (the ORIGINAL raw one passed
            // from Dashboard?).
            // NO. `refitBestModel` receives `dataset`.
            // If Dashboard passes the SAME `finalDataset` (already preprocessed safely),
            // then we shouldn't run preprocessor again on it?
            // OR Dashboard passes `rawDataset`?

            // If Dashboard passes `finalDataset` (No IDs, Raw Values, NaN preserved),
            // Then we just need to Apply Impute/Scale.
            // We should NOT run 'autoRemoveIdColumns' again if they are already gone.
            // BUT `runPreprocessingPipeline` might fail if IDs are missing? No.

            // Safer Check: If `DashboardController` calls this with the `finalDataset` used
            // for CV,
            // Then we are good on column alignment.
            // But we need to Apply Impute/Scale.

            // Modification: `runPreprocessingPipeline` applies ALL steps in config.
            // If we pass `finalDataset` (which has ID removed), and Config says
            // `removeID=true`,
            // Preprocessor checks heuristic. If no ID found, it does nothing. Safe.

            // The main issue: Impute/Scale.
            // `cleanData` will be Imputed & Scaled.
        }

        double[][] selectedX = new double[X.length][indices.length];

        for (int i = 0; i < X.length; i++) {
            for (int k = 0; k < indices.length; k++) {
                int colIdx = indices[k];
                if (colIdx < X[0].length) {
                    selectedX[i][k] = X[i][colIdx];
                } else {
                    selectedX[i][k] = 0.0; // Fallback
                }
            }
        }

        // 4b. Inject Image Paths if applicable (Phase 6)
        if (dataset.getDataType() == Dataset.DataType.IMAGE
                && classifier instanceof com.example.automl_prototype_1.classifier.CnnClassifier) {
            ((com.example.automl_prototype_1.classifier.CnnClassifier) classifier)
                    .setImagePaths(dataset.getFilePaths());
            LogService.info("CORE",
                    "Refit: Injected " + dataset.getFilePaths().size() + " image paths into CnnClassifier.");
        }

        // 4. Train Classifier
        LogService.info("CORE", "Training " + classifier.getName() + " on " + X.length + " samples...");
        classifier.train(selectedX, y);

        // Save reference for Export
        result.setTrainedClassifier(classifier);

        // 5. Extract ONNX Path
        if (classifier instanceof com.example.automl_prototype_1.classifier.gpu.GpuRandomForestClassifier) {
            String path = ((com.example.automl_prototype_1.classifier.gpu.GpuRandomForestClassifier) classifier)
                    .getOnnxModelPath();
            if (path != null) {
                result.setOnnxModelPath(path);
                LogService.info("CORE", "ONNX Model Exported: " + path);
            }
        } else {
            // Future support for other GPU classifiers
            // if (classifier instanceof
            // com.example.automl_prototype_1.classifier.gpu.GpuSvmClassifier) ...
        }
    }
}
