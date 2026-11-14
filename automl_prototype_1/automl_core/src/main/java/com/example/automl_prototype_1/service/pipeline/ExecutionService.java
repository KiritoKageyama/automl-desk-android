package com.example.automl_prototype_1.service.pipeline;

import com.example.automl_prototype_1.service.LogService;
import com.example.automl_prototype_1.service.AppStateService;
import com.example.automl_prototype_1.service.CrashReportService;
import com.example.automl_prototype_1.service.IClassifier;

import com.example.automl_prototype_1.algorithm.OptimizationAlgorithm;
import com.example.automl_prototype_1.model.AlgorithmRecommendation;
import com.example.automl_prototype_1.model.Dataset;
import com.example.automl_prototype_1.model.ExecutionResult;
import com.example.automl_prototype_1.registry.AlgorithmRegistry;
import com.example.automl_prototype_1.classifier.ml.KnnClassifier;
import com.example.automl_prototype_1.service.progress.ProgressTrackingService;

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

    private final java.util.concurrent.ExecutorService executor;

    public ExecutionService() {
        this.recommendationService = new RecommendationService();
        // Persistent ThreadPool with dynamic sizing
        int cores = Runtime.getRuntime().availableProcessors();
        this.executor = java.util.concurrent.Executors.newFixedThreadPool(Math.max(2, cores));
        LogService.info("CORE", "ExecutionService Initialized with " + Math.max(2, cores) + " threads.");
    }

    public static synchronized ExecutionService getInstance() {
        if (instance == null) {
            instance = new ExecutionService();
        }
        return instance;
    }

    /**
     * Stops the current execution. The thread pool is kept alive for reuse;
     * only Python processes are forcefully terminated.
     */
    public void stop() {
        LogService.info("CORE", "STOP Requested. Clearing tasks...");
        if (executor != null && !executor.isShutdown()) {
            LogService.warn("CORE",
                    "Stop requested. Active threads may continue unless they check interruption.");
        }
        PythonExecutionService.getInstance().terminateAll();
        LogService.error("CORE", "=== EXECUTION FORCEFULLY STOPPED BY USER (Python Terminated) ===");
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
                && classifier instanceof com.example.automl_prototype_1.classifier.cnn.CnnClassifier) {
            ((com.example.automl_prototype_1.classifier.cnn.CnnClassifier) classifier)
                    .setImagePaths(dataset.getFilePaths());
            LogService.info("CORE", "Injected " + dataset.getFilePaths().size() + " image paths into CnnClassifier.");
        }

        // --- Phase 7: Gen AI Text Support ---
        if (dataset.getDataType() == Dataset.DataType.TEXT
                && classifier instanceof com.example.automl_prototype_1.classifier.llm.BertClassifier) {
            ((com.example.automl_prototype_1.classifier.llm.BertClassifier) classifier)
                    .setFilePaths(dataset.getFilePaths());
            LogService.info("CORE", "Injected " + (dataset.getFilePaths() != null ? dataset.getFilePaths().size() : 0)
                    + " text paths into BertClassifier.");
        }

        LogService.info("CORE", "Starting PARALLEL execution of " + finalAlgorithms.size()
                + " algorithms with classifier: " + classifier.getName());

        // Use Persistent Executor
        List<java.util.concurrent.Future<ExecutionResult>> futures = new ArrayList<>();
        final Map<String, Object> finalConfig = configuration;

        for (String algoName : finalAlgorithms) {
            futures.add(executor.submit(() -> {
                if (Thread.currentThread().isInterrupted())
                    return null;
                long threadId = Thread.currentThread().getId();
                String resourceType = (finalConfig
                        .get("classifier") instanceof com.example.automl_prototype_1.classifier.ml.GpuKnnClassifier ||
                        finalConfig.get(
                                "classifier") instanceof com.example.automl_prototype_1.classifier.ml.GpuLogisticRegression)
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
                        CrashReportService.getInstance().report("ExecutionService:" + algoName, e);
                        // Self-Correction Loop: log failure
                        com.example.automl_prototype_1.service.agentic.SelfCorrectionLoopService.getInstance()
                                .onExecutionComplete(algoName, false, e.getClass().getSimpleName(), 0.0);
                        return ExecutionResult.failed(algoName, e.getMessage());
                    }
                } else {
                    return ExecutionResult.failed(algoName, "Algorithm implementation not found");
                }
            }));
        }

        List<ExecutionResult> allResults = new ArrayList<>();
        for (java.util.concurrent.Future<ExecutionResult> future : futures) {
            try {
                // Global safety timeout: 3 hours per algorithm
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

        // Do NOT shutdown the persistent executor
        // executor.shutdown();
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
        OptimizationAlgorithm algo = com.example.automl_prototype_1.registry.AlgorithmRegistry.getInstance()
                .getAlgorithm(name);
        if (algo == null && name != null) {
            LogService.error("ExecutionService", "No implementation found for algorithm name '" + name + "'");
        }
        return algo;
    }

    // -----------------------------------------------------------------------
    // Sequential Pipeline — used by DashboardController's UI task
    // -----------------------------------------------------------------------

    /**
     * Callback interface for reporting progress from the sequential pipeline.
     * The controller implements this to update UI (e.g., JavaFX
     * Task.updateMessage).
     */
    @FunctionalInterface
    public interface ProgressCallback {
        void update(String message);
    }

    /**
     * Executes the full sequential pipeline: preprocess → GPU warmup → run
     * algorithms → refit.
     * Matches the DashboardController's execution flow but lives in core so it can
     * be reused.
     *
     * @param dataset     The preprocessed dataset (IDs already removed, scaling
     *                    deferred to CV).
     * @param algorithms  Ordered list of algorithm names to run.
     * @param classifier  The classifier to use.
     * @param isTuning    Whether to run hyperparameter tuning.
     * @param cancelCheck Returns true when the caller wants to abort.
     * @param progress    Optional callback for UI status messages.
     * @return Ordered list of results (one per algorithm).
     */
    public List<ExecutionResult> runSequentialPipeline(
            Dataset dataset,
            List<String> algorithms,
            IClassifier classifier,
            boolean isTuning,
            java.util.function.BooleanSupplier cancelCheck,
            ProgressCallback progress) {

        if (progress == null)
            progress = msg -> {
            };

        // 1. GPU warmup (if needed)
        if (classifier.getName().contains("GPU") || classifier.getName().contains("Python")) {
            progress.update("Starting GPU Server...");
            PythonExecutionService.getInstance().startGpuServer();
            if (!PythonExecutionService.getInstance().isServerRunning()) {
                throw new RuntimeException("GPU Server failed to start. Check logs.");
            }
        }

        // 2. Calculate tuning trials dynamically
        int tuningTrials = calculateTrials(dataset);

        // 3. Sequential execution with progress tracking
        HyperparameterTuningService tuningService = new HyperparameterTuningService();
        ProgressTrackingService progressService = ProgressTrackingService.getInstance();
        progressService.reset(algorithms);

        List<ExecutionResult> results = new ArrayList<>();
        int total = algorithms.size();

        // Phase 10: Knowledge Graph live glow — activate pipeline stages as they run
        com.example.automl_prototype_1.service.agentic.KnowledgeBaseService kb = com.example.automl_prototype_1.service.agentic.KnowledgeBaseService
                .getInstance();
        com.example.automl_prototype_1.service.agentic.TrainingWatchdogService watchdog = com.example.automl_prototype_1.service.agentic.TrainingWatchdogService
                .getInstance();

        // Activate Sensor+Preprocessing nodes at start of pipeline
        kb.activateNode("Sensor Layer");
        kb.activateNode("Preprocessing");
        progress.update("Preprocessing dataset...");

        for (int i = 0; i < total; i++) {
            if (cancelCheck.getAsBoolean())
                break;

            String algoName = algorithms.get(i);
            OptimizationAlgorithm algorithm = AlgorithmRegistry.getInstance().getAlgorithm(algoName);

            // Phase 9.2: Start watchdog for this algorithm
            watchdog.beginWatch(algoName);

            // Phase 10: Activate Feature Selection node
            kb.activateNode("Feature Selection");

            // Apply manual iteration override
            Integer iterOverride = AppStateService.getInstance().getManualIterationsOverride();
            if (algorithm != null && iterOverride != null && iterOverride > 0) {
                algorithm.setMaxIterations(iterOverride);
                LogService.info("CORE", "Manual iterations for " + algoName + ": " + iterOverride);
            }

            if (isTuning) {
                progress.update("Tuning " + algoName + " (" + (i + 1) + "/" + total + ")");
                kb.activateNode("Model Training");
                watchdog.heartbeat();
                ExecutionResult best = tuningService.tune(algoName, tuningTrials, dataset, classifier);
                if (best != null)
                    results.add(best);
            } else {
                progress.update("Running " + algoName + " (" + (i + 1) + "/" + total + ")");
                kb.activateNode("Model Training");
                if (algorithm != null) {
                    watchdog.heartbeat(); // signal progress before long run
                    results.add(algorithm.execute(dataset, Map.of("classifier", classifier)));
                }
            }

            // Phase 9.2: Job done — stop watchdog timer
            watchdog.endWatch();
        }

        // 4. Refit best model for export
        if (!results.isEmpty() && !cancelCheck.getAsBoolean()) {
            progress.update("Refitting Best Model for Export...");
            kb.activateNode("ONNX Export");
            kb.activateNode("Action Layer");
            kb.activateNode("Meta-Cognition");
            ExecutionResult best = results.stream()
                    .max(java.util.Comparator.comparingDouble(ExecutionResult::getAccuracy))
                    .orElse(results.get(0));
            try {
                refitBestModel(best, classifier, dataset);
            } catch (Exception e) {
                LogService.error("CORE", "Refit Failed: " + e.getMessage());
            }
        }

        return results;
    }

    /**
     * Calculates dynamic tuning trial count based on dataset size.
     */
    private int calculateTrials(Dataset dataset) {
        Integer manual = AppStateService.getInstance().getManualTrialsOverride();
        if (manual != null && manual > 0)
            return manual;

        long rows = dataset.getRecordCount();
        if (rows < 1000)
            return 3;
        if (rows < 10000)
            return 6;
        return 9;
    }

    /**
     * Retrains the best model on the FULL dataset using the selected features.
     * Crucial for generating a deployable model (ONNX).
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

        // 2. Run Preprocessing (Clean, Impute, Scale)
        PreprocessorService preprocessor = new PreprocessorService();
        String targetCol = dataset.getTargetColumnName();
        Dataset cleanData = preprocessor.runPreprocessingPipeline(dataset, targetCol, config);

        // 3. Filter Features
        double[][] X = cleanData.getFeatures();
        int[] y = cleanData.getLabelVector();
        if (y == null) {
            LogService.error("CORE", "Cannot refit: No labels found in processed dataset.");
            return;
        }

        int[] indices = result.getFeatureIndices();
        int maxIndex = 0;
        for (int idx : indices)
            maxIndex = Math.max(maxIndex, idx);

        if (maxIndex >= X[0].length) {
            LogService.error("CORE", "Feature Indices out of bounds. Max: " + maxIndex + ", Cols: " + X[0].length);
            // Indices may be misaligned if preprocessing changed column count.
            // Proceeding with bounds-safe fallback below.
        }

        double[][] selectedX = new double[X.length][indices.length];
        for (int i = 0; i < X.length; i++) {
            for (int k = 0; k < indices.length; k++) {
                int colIdx = indices[k];
                selectedX[i][k] = (colIdx < X[0].length) ? X[i][colIdx] : 0.0;
            }
        }

        // Inject Image Paths if applicable
        if (dataset.getDataType() == Dataset.DataType.IMAGE
                && classifier instanceof com.example.automl_prototype_1.classifier.cnn.CnnClassifier) {
            ((com.example.automl_prototype_1.classifier.cnn.CnnClassifier) classifier)
                    .setImagePaths(dataset.getFilePaths());
            LogService.info("CORE",
                    "Refit: Injected " + dataset.getFilePaths().size() + " image paths into CnnClassifier.");
        }

        // 4. Train Classifier on full data
        LogService.info("CORE", "Training " + classifier.getName() + " on " + X.length + " samples...");
        classifier.train(selectedX, y);
        result.setTrainedClassifier(classifier);

        // 5. Extract / produce ONNX path for any supported hybrid/Python classifier
        String path = resolveOnnxModelPath(classifier);
        if (path == null) {
            path = tryExportOnnx(classifier);
        }
        if (path != null) {
            result.setOnnxModelPath(path);
            try {
                OnnxInferenceService.getInstance().loadModel(path, path);
            } catch (Exception e) {
                LogService.warn("CORE", "ONNX preload failed (path kept for export): " + e.getMessage());
            }
            LogService.info("CORE", "ONNX Model Exported: " + path);
        } else {
            LogService.warn("CORE", "No ONNX artifact available for classifier: " + classifier.getName());
        }
    }

    /**
     * Reads onnx path from hybrid GPU/Python classifiers after training.
     */
    private String resolveOnnxModelPath(IClassifier classifier) {
        if (classifier instanceof com.example.automl_prototype_1.classifier.ml.GpuRandomForestClassifier) {
            return ((com.example.automl_prototype_1.classifier.ml.GpuRandomForestClassifier) classifier)
                    .getOnnxModelPath();
        }
        if (classifier instanceof com.example.automl_prototype_1.classifier.ml.GpuPythonXGBoostClassifier) {
            return ((com.example.automl_prototype_1.classifier.ml.GpuPythonXGBoostClassifier) classifier)
                    .getOnnxModelPath();
        }
        if (classifier instanceof com.example.automl_prototype_1.classifier.ml.GpuPythonLightGbmClassifier) {
            return ((com.example.automl_prototype_1.classifier.ml.GpuPythonLightGbmClassifier) classifier)
                    .getOnnxModelPath();
        }
        if (classifier instanceof com.example.automl_prototype_1.classifier.ml.GpuPythonSvmClassifier) {
            return ((com.example.automl_prototype_1.classifier.ml.GpuPythonSvmClassifier) classifier)
                    .getOnnxModelPath();
        }
        if (classifier instanceof com.example.automl_prototype_1.classifier.ml.GpuPythonCatBoostClassifier) {
            return ((com.example.automl_prototype_1.classifier.ml.GpuPythonCatBoostClassifier) classifier)
                    .getOnnxModelPath();
        }
        return null;
    }

    /**
     * Fallback: ask Python bridge / OnnxExportService to materialize an .onnx file.
     */
    private String tryExportOnnx(IClassifier classifier) {
        try {
            java.io.File dest = java.io.File.createTempFile("automl_export_", ".onnx");
            dest.deleteOnExit();
            boolean ok = new com.example.automl_prototype_1.service.agentic.inference.OnnxExportService()
                    .exportModel(classifier, dest.getAbsolutePath());
            if (ok && dest.exists() && dest.length() > 0
                    && dest.getName().endsWith(".onnx")
                    && !dest.getAbsolutePath().contains(".fallback.ser")) {
                return dest.getAbsolutePath();
            }
            // Export may have written a .fallback.ser next to destination
            java.io.File ser = new java.io.File(dest.getAbsolutePath() + ".fallback.ser");
            if (ser.exists()) {
                LogService.info("CORE", "ONNX unsupported; Java serialization fallback at: " + ser.getAbsolutePath());
            }
        } catch (Exception e) {
            LogService.warn("CORE", "ONNX export fallback failed: " + e.getMessage());
        }
        return null;
    }
}
