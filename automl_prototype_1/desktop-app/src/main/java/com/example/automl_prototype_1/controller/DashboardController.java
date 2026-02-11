package com.example.automl_prototype_1.controller;

import javafx.scene.control.Alert;
import com.example.automl_prototype_1.algorithm.*;
import com.example.automl_prototype_1.algorithm.AtomSearchOptimization;
import com.example.automl_prototype_1.service.pipeline.HyperparameterTuningService;
import com.example.automl_prototype_1.dataprovider.CsvDataProvider;
import com.example.automl_prototype_1.model.Dataset;
import com.example.automl_prototype_1.model.ExecutionResult;
import com.example.automl_prototype_1.service.*;
import com.example.automl_prototype_1.service.pipeline.ExecutionService;
import com.example.automl_prototype_1.service.pipeline.PreprocessorService;
import com.example.automl_prototype_1.service.progress.ProgressTrackingService;
import com.example.automl_prototype_1.gpu.HardwareAcceleratorService;
import com.example.automl_prototype_1.gpu.PythonExecutionService;
import javafx.application.Platform;
import javafx.concurrent.Task;
import javafx.fxml.FXML;
import javafx.fxml.FXMLLoader;

//import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.Parent;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.layout.StackPane;
import javafx.scene.layout.VBox;

import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.net.URL;
import java.util.ArrayList;
//import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class DashboardController {

    @FXML
    private Button uploadBtn, algoBtn, runBtn, resultsBtn, settingsBtn, simulatorBtn, rlLabBtn, visualizerBtn,
            creatorLabBtn, perfHistoryBtn, exitBtn;
    @FXML
    private StackPane mainContent;
    @FXML
    private Label datasetLabel;
    @FXML
    private Label resourceStatusLabel; // Phase 6 UI Indicator
    @FXML
    private Label systemStatsLabel; // Live system stats
    @FXML
    private StackPane chatContainer;
    @FXML
    private Button chatToggleBtn;

    // --- Services ---

    private final AppStateService appStateService; // Final for DI
    private CsvDataProvider csvDataProvider;
    private PreprocessorService preprocessorService;
    private HyperparameterTuningService tuningService;
    private Object currentViewController;

    // --- Constructor Injection ---
    public DashboardController(AppStateService appStateService) {
        this.appStateService = appStateService;
    }

    @FXML
    private void initialize() {
        // Instantiate other services (could also be injected, but keeping simple for
        // now)

        // this.appStateService = AppStateService.getInstance(); // REMOVED
        this.csvDataProvider = new CsvDataProvider();
        this.preprocessorService = new PreprocessorService();
        this.tuningService = new HyperparameterTuningService();

        // Phase 6 Step 6: UI Resource Mode
        checkResourceMode();

        // Live system stats polling (RAM/CPU/GPU) in header
        startSystemStatsPolling();

        // Set initial view
        loadView("upload_view.fxml");

        // Set button actions
        uploadBtn.setOnAction(event -> loadView("upload_view.fxml"));
        algoBtn.setOnAction(event -> loadView("algo_select_view.fxml"));
        runBtn.setOnAction(event -> handleRunModel());
        resultsBtn.setOnAction(event -> loadView("results_view.fxml"));
        settingsBtn.setOnAction(event -> loadView("settings_view.fxml"));
        simulatorBtn.setOnAction(event -> loadView("pipeline_simulator.fxml"));
        if (rlLabBtn != null)
            rlLabBtn.setOnAction(event -> loadView("rl_lab_view.fxml"));
        if (visualizerBtn != null)
            visualizerBtn.setOnAction(event -> loadView("rl_visualizer_view.fxml"));
        if (creatorLabBtn != null)
            creatorLabBtn.setOnAction(event -> loadView("creator_lab_view.fxml"));
        if (perfHistoryBtn != null)
            perfHistoryBtn.setOnAction(event -> loadView("performance_history_view.fxml"));
        exitBtn.setOnAction(event -> onExitClicked());

        // Chat Integration
        if (chatContainer != null && chatToggleBtn != null) {
            loadChatView();
            chatToggleBtn.setOnAction(e -> {
                boolean isVisible = chatContainer.isVisible();
                chatContainer.setVisible(!isVisible);
                chatContainer.setManaged(!isVisible); // Ensure it doesn't take space when hidden? Actually it's in
                                                      // overlay so managed doesn't matter much for layout but good
                                                      // practice.
                // Actually in an overlay stackpane, managed=false or true mainly affects if it
                // can be picked?
                // Let's just toggle Visibility.
            });
        }
    }

    private void loadChatView() {
        try {
            FXMLLoader loader = new FXMLLoader(
                    getClass().getResource("/com/example/automl_prototype_1/chat_view.fxml"));
            Parent chatView = loader.load();
            ChatViewController controller = loader.getController();

            // Set close action
            controller.setOnCloseAction(() -> chatContainer.setVisible(false));

            chatContainer.getChildren().setAll(chatView);
        } catch (IOException e) {
            LogService.error("Dashboard", "Failed to load Chat View: " + e.getMessage());
        }
    }

    private void checkResourceMode() {
        // Run async check to avoid blocking UI init
        new Thread(() -> {
            boolean gpu = HardwareAcceleratorService.isGpuAvailable();
            String gpuName = HardwareAcceleratorService.getGpuName();

            Platform.runLater(() -> {
                if (resourceStatusLabel != null) {
                    if (gpu) {
                        resourceStatusLabel.setText("GPU ACTIVE: " + gpuName);
                        // Green accent for success
                        resourceStatusLabel.setStyle(
                                "-fx-font-size: 11px; -fx-padding: 5 10 5 10; -fx-background-radius: 4; -fx-background-color: #dcfce7; -fx-text-fill: #166534; -fx-font-weight: bold;");
                    } else {
                        resourceStatusLabel.setText("CPU FALLBACK MODE");
                        // Orange accent for warning
                        resourceStatusLabel.setStyle(
                                "-fx-font-size: 11px; -fx-padding: 5 10 5 10; -fx-background-radius: 4; -fx-background-color: #ffedd5; -fx-text-fill: #9a3412; -fx-font-weight: bold;");
                    }
                }
            });
        }).start();
    }

    /** Starts a background timer that refreshes live system stats every 5s. */
    private void startSystemStatsPolling() {
        java.util.concurrent.ScheduledExecutorService statsScheduler = java.util.concurrent.Executors
                .newSingleThreadScheduledExecutor(r -> {
                    Thread t = new Thread(r, "SystemStatsPoller");
                    t.setDaemon(true);
                    return t;
                });
        com.example.automl_prototype_1.service.agentic.SystemMonitorService sysMonitor = new com.example.automl_prototype_1.service.agentic.SystemMonitorService();

        statsScheduler.scheduleAtFixedRate(() -> {
            String stats = sysMonitor.getSystemContextString();
            Platform.runLater(() -> {
                if (systemStatsLabel != null) {
                    systemStatsLabel.setText(stats);
                }
            });
        }, 0, 5, java.util.concurrent.TimeUnit.SECONDS);
    }

    // In DashboardController.java

    private Task<List<ExecutionResult>> executionTask;
    private boolean isRunning = false;

    private void handleRunModel() {
        // Toggle Logic for STOP
        if (isRunning) {
            LogService.info("CORE", "Stop requested by user.");
            // UI Feedback IMMEDIATELY
            runBtn.setDisable(true); // Disable to prevent double clicks
            runBtn.setText("Stopping...");

            if (executionTask != null && executionTask.isRunning()) {
                executionTask.cancel();
            }

            // Run heavy termination in background to avoid UI freeze
            new Thread(() -> {
                // Force Kill Mechanism (Deep Kill)
                PythonExecutionService.getInstance().terminateAll();

                // UI Cleanup on JavaFX Thread
                Platform.runLater(() -> {
                    isRunning = false;
                    runBtn.setText("Run Experiment");
                    runBtn.setStyle(""); // Reset style
                    runBtn.setDisable(false); // Ensure enabled

                    // Log Interruption
                    LogService.error("CORE", "=== EXECUTION STOPPED BY USER ===");

                    // Critical Log Analysis
                    String analysisReport = LogService.getInstance().analyzeCriticalErrors();
                    LogService.info("CORE", analysisReport);

                    // Show Analysis to User
                    showAlert(Alert.AlertType.WARNING, "Execution Stopped",
                            "Process terminated by user.\n\n" + analysisReport);
                });
            }).start();
            return;
        }

        // ===================================================================
        // 1. GATHER AND VALIDATE STATE
        // ===================================================================
        File datasetFile = appStateService.getSelectedDatasetFile();
        List<String> algorithms = appStateService.getSelectedAlgorithmNames();
        IClassifier classifier = appStateService.getSelectedClassifier();
        boolean isTuning = appStateService.isTuningEnabled();
        String targetColumn = appStateService.getTargetColumnName();

        // Validate that all necessary selections have been made
        if (datasetFile == null) {
            showError("Please select a dataset file first.");
            loadView("upload_view.fxml");
            return;
        }
        if (algorithms == null || algorithms.isEmpty()) {
            showError("Please select at least one algorithm.");
            loadView("algo_select_view.fxml");
            return;
        }
        if (classifier == null) {
            showError("Please select a base ML model.");
            loadView("algo_select_view.fxml");
            return;
        }
        if (targetColumn == null) {
            showError("Please select a target column from the Upload tab.");
            loadView("upload_view.fxml");
            return;
        }

        // Perform a quick "fail-fast" validation on the target column before starting
        // the long process
        try {
            Dataset tempDataset = csvDataProvider.loadDataset(new FileReader(datasetFile));
            if (!preprocessorService.isTargetColumnSuitableForClassification(tempDataset, targetColumn)) {
                showAlert(Alert.AlertType.WARNING, "Invalid Target Column",
                        "The selected target column '" + targetColumn
                                + "' appears unsuitable for classification (e.g., too many unique values). Please choose a categorical target.");
                return;
            }
        } catch (IOException e) {
            showError("Failed to load data for validation: " + e.getMessage());
            return;
        }

        // ===================================================================
        // 2. SETUP THE UI AND BACKEND FOR THE EXPERIMENT RUN
        // ===================================================================

        // Phase 4: Manual Feature Selection Dialog
        if (appStateService.isManualFeatureSelectionEnabled()) {
            try {
                FXMLLoader fxmlLoader = new FXMLLoader(
                        getClass().getResource("/com/example/automl_prototype_1/feature_selection_dialog.fxml"));
                Parent parent = fxmlLoader.load();
                FeatureSelectionDialogController dialogController = fxmlLoader.getController();

                javafx.stage.Stage stage = new javafx.stage.Stage();
                stage.initModality(javafx.stage.Modality.APPLICATION_MODAL);
                stage.setTitle("Select Features"); // Keep title simple
                stage.setScene(new javafx.scene.Scene(parent));
                dialogController.setDialogStage(stage);

                stage.showAndWait();

                if (!dialogController.isConfirmed()) {
                    LogService.info("Dashboard", "Manual Feature Selection Cancelled. Aborting run.");
                    return;
                }
            } catch (IOException e) {
                LogService.error("Dashboard", "Failed to open Feature Selection Dialog: " + e.getMessage());
                showError("Failed to open Feature Selection Dialog.");
                return;
            }
        }

        ProgressTrackingService.getInstance().reset(algorithms);

        // Update UI to Running State
        isRunning = true;
        runBtn.setText("STOP");
        runBtn.setStyle("-fx-background-color: #ef4444; -fx-text-fill: white;"); // Tailwind red-500

        // Load the progress view and get a reference to its controller
        ProgressViewController progressController = null;
        try {
            String fullPath = "/com/example/automl_prototype_1/progress_view.fxml";
            FXMLLoader loader = new FXMLLoader(getClass().getResource(fullPath));
            loader.setControllerFactory(param -> new ProgressViewController()); // Default constructor
            Parent progressView = loader.load();
            progressController = loader.getController(); // Get the controller instance
            currentViewController = progressController; // Store for later cleanup
            mainContent.getChildren().setAll(progressView);
        } catch (IOException e) {
            e.printStackTrace();
            LogService.error("Dashboard", "Failed to load progress view: " + e.getMessage());
            showError("Failed to load progress view.");
            isRunning = false;
            runBtn.setText("Run Experiment");
            runBtn.setStyle("");
            return;
        }

        // ===================================================================
        // 3. DEFINE THE LONG-RUNNING WORK IN A BACKGROUND TASK
        // ===================================================================
        executionTask = new Task<>() {
            @Override
            protected List<ExecutionResult> call() throws Exception {
                updateMessage("Loading and Preprocessing Data...");
                Dataset rawDataset;
                try (FileReader reader = new FileReader(datasetFile)) {
                    rawDataset = csvDataProvider.loadDataset(reader);
                }
                // Auto-apply scaling: Dropdown takes precedence, Checkbox is fallback default
                // Retrieve full config from State
                com.example.automl_prototype_1.model.PreprocessingConfig config = appStateService
                        .getPreprocessingConfig();

                // Fallback check if config is somehow null (safety)
                if (config == null) {
                    config = new com.example.automl_prototype_1.model.PreprocessingConfig();
                    String effectiveScaling = appStateService.getScalingStrategy();
                    if (effectiveScaling == null) {
                        effectiveScaling = "Standardization"; // Default to Standardization only if null
                    }
                    config.setScalingStrategy(effectiveScaling);
                }

                // --- CRITICAL FIX: PREVENT DATA LEAKAGE ---
                // We must NOT apply scaling here globally. Scaling must be done inside the
                // Algorithm's Cross-Validation loop.
                // We create a "Safe Config" that disables scaling for this pre-processing step,
                // while preserving other steps like ID removal.
                com.example.automl_prototype_1.model.PreprocessingConfig safeConfig = new com.example.automl_prototype_1.model.PreprocessingConfig();
                safeConfig.setAutoRemoveIdColumns(config.isAutoRemoveIdColumns());
                // Imputation must also be done per-fold in CV loop.
                safeConfig.setImputeMissingValues(false); // FORCE FALSE
                safeConfig.setRemoveOutliers(config.isRemoveOutliers()); // Outlier removal logic might still be risky
                                                                         // globally but accepted for now
                safeConfig.setScalingStrategy("None"); // FORCE NONE

                LogService.info("Dashboard", "Applying Preprocessing (Scaling Disabled to prevent Leakage)...");
                Dataset finalDataset = preprocessorService.runPreprocessingPipeline(rawDataset, targetColumn,
                        safeConfig);

                // Phase 7: Metric & Data Flow Audit
                // Optimization: Release rawDataset immediately to free RAM
                rawDataset = null;
                System.gc(); // Suggest GC (aggressive optimization for 16GB limit)

                // --- EXPLICIT GPU STARTUP ---
                // Pre-warm the GPU server if needed. This avoids race conditions and provides
                // UI feedback.
                if (classifier.getName().contains("GPU") || classifier.getName().contains("Python")) {
                    updateMessage("Starting GPU Server...");
                    PythonExecutionService.getInstance().startGpuServer();
                    // Quick check to ensure it's up
                    if (!PythonExecutionService.getInstance().isServerRunning()) {
                        throw new RuntimeException("GPU Server failed to start. Check logs.");
                    }
                }

                // --- THIS IS THE CORRECT DYNAMIC TRIAL LOGIC ---
                int tuningTrials;
                // MANUAL OVERRIDE: Trials
                if (appStateService.getManualTrialsOverride() != null
                        && appStateService.getManualTrialsOverride() > 0) {
                    tuningTrials = appStateService.getManualTrialsOverride();
                    LogService.info("Dashboard", "Using Manual Tuning Trials: " + tuningTrials);
                } else {
                    if (finalDataset.getRecordCount() < 1000) {
                        tuningTrials = 3; // Fewer trials for small datasets like WDBC
                    } else if (finalDataset.getRecordCount() < 10000) {
                        tuningTrials = 6; // Standard trials for medium datasets
                    } else {
                        tuningTrials = 9; // More trials for large datasets
                    }
                }
                // -------------------------------------------------

                List<ExecutionResult> finalResults = new ArrayList<>();
                int totalAlgos = algorithms.size();
                ProgressTrackingService progressService = ProgressTrackingService.getInstance();
                progressService.reset(algorithms); // Reset and prime the service with the new queue

                for (int i = 0; i < totalAlgos; i++) {
                    if (isCancelled())
                        break; // Check for cancellation

                    String algoName = algorithms.get(i);
                    OptimizationAlgorithm algorithm = getAlgorithmImplementation(algoName); // Declare before use

                    // MANUAL OVERRIDE: Iterations (If supported)
                    if (appStateService.getManualIterationsOverride() != null
                            && appStateService.getManualIterationsOverride() > 0) {
                        algorithm.setMaxIterations(appStateService.getManualIterationsOverride());
                        LogService.info("Dashboard", "Using Manual Iterations for " + algoName + ": "
                                + appStateService.getManualIterationsOverride());
                    }

                    if (isTuning) {
                        LogService.info("Dashboard", "Tuning enabled for " + algoName);
                        updateMessage("Tuning " + algoName + " (" + (i + 1) + "/" + totalAlgos + ")");
                        // Use the CORRECT 'tuningTrials' variable here
                        ExecutionResult bestTunedResult = tuningService.tune(algoName, tuningTrials, finalDataset,
                                classifier);
                        if (bestTunedResult != null)
                            finalResults.add(bestTunedResult);
                    } else {
                        updateMessage("Running " + algoName + " (" + (i + 1) + "/" + totalAlgos + ")");
                        // Reuse previously declared variable
                        if (algorithm != null) {
                            finalResults.add(algorithm.execute(finalDataset, Map.of("classifier", classifier)));
                        }
                    }
                }

                // --- Phase 8: Final Refit & Model Export ---
                if (!finalResults.isEmpty() && !isCancelled()) {
                    updateMessage("Refitting Best Model for Export...");
                    // Find best result based on "Optimization Metric" - defaulting to Accuracy for
                    // now
                    ExecutionResult best = finalResults.stream()
                            .max(java.util.Comparator.comparingDouble(ExecutionResult::getAccuracy))
                            .orElse(finalResults.get(0));

                    try {
                        ExecutionService.getInstance().refitBestModel(
                                best,
                                classifier, finalDataset);
                    } catch (Exception e) {
                        LogService.error("Dashboard", "Refit Failed: " + e.getMessage());
                    }
                }

                return finalResults;
            }
        };

        // ===================================================================
        // 4. BIND THE UI TO THE TASK AND DEFINE COMPLETION BEHAVIOR
        // ===================================================================

        // This is the critical link that makes the UI update in real-time
        if (progressController != null)

        {
            progressController.bindToTask(executionTask);
        }

        // Define what happens when the task finishes successfully
        executionTask.setOnSucceeded(event -> Platform.runLater(() -> {
            appStateService.setExecutionResults(executionTask.getValue());
            isRunning = false;
            runBtn.setText("Run Experiment");
            runBtn.setStyle("");
            runBtn.setDisable(false);
            loadView("results_view.fxml"); // Automatically cleans up progress view and shows results
        }));

        // Define what happens if the task fails with an exception
        executionTask.setOnFailed(event -> Platform.runLater(() -> {
            isRunning = false;
            runBtn.setText("Run Experiment");
            runBtn.setStyle("");
            runBtn.setDisable(false);

            Throwable e = executionTask.getException();
            LogService.error("Dashboard", "Execution task failed: " + e.getMessage());
            e.printStackTrace(); // Keep logging to console

            // Show a persistent Alert dialog so the user can see the error
            showAlert(Alert.AlertType.ERROR, "Execution Failed",
                    "An error occurred during execution:\n" + e.getMessage());

            // Do NOT redirect to upload_view.fxml. Stay on the current view (Progress View)
            // or let the user decide.
            // If we want to return to Algo Select, we can, but let's let the user read the
            // error first.
        }));

        executionTask.setOnCancelled(event -> Platform.runLater(() -> {
            isRunning = false;
            runBtn.setText("Run Experiment");
            runBtn.setStyle("");
            runBtn.setDisable(false);
            LogService.info("CORE", "Task cancelled confirmation received in UI.");
        }));

        // ===================================================================
        // 5. START THE EXPERIMENT
        // ===================================================================
        new Thread(executionTask).start();
    }

    private OptimizationAlgorithm getAlgorithmImplementation(String name) {
        if (name == null)
            return null;

        // --- THIS IS THE FIX ---
        // The switch now uses the short names from the UI checkboxes.
        return switch (name) {
            case "Genetic Algorithm", "GA" -> new GeneticAlgorithm();
            case "IGPSO" -> new IGPSO();
            case "WWO" -> new WaterWaveOptimization();
            case "BPSO" -> new BinaryParticleSwarmOptimization();
            case "ASO" -> new AtomSearchOptimization();
            case "PLS", "Parallel Local Search" -> new ParallelLocalSearchAlgorithm();
            case "Simulated Annealing", "SA" -> new SimulatedAnnealing();
            case "GWO", "Grey Wolf Optimizer" -> new GreyWolfOptimizer();
            case "DE", "Differential Evolution" -> new DifferentialEvolution();
            default -> null;
        };
    }

    // --- UI Helper Methods ---
    private void loadView(String fxmlFile) {
        // Step 1: Clean up the previous controller if it's "Cleanable"
        if (currentViewController instanceof Cleanable) {
            ((Cleanable) currentViewController).cleanup();
        }

        runBtn.setDisable(false);
        try {
            String fullPath = "/com/example/automl_prototype_1/" + fxmlFile;
            URL fxmlUrl = getClass().getResource(fullPath);
            if (fxmlUrl == null) {
                showError("Cannot find FXML resource: " + fullPath);
                return;
            }

            // THIS IS THE FIX: Create an FXMLLoader instance
            FXMLLoader loader = new FXMLLoader(fxmlUrl);

            // --- Dependency Injection for Sub-Controllers ---
            loader.setControllerFactory(param -> {
                if (param == ResultsViewController.class) {
                    ResultsViewController controller = new ResultsViewController();
                    controller.setOnBackAction(() -> loadView("algo_select_view.fxml"));
                    return controller;
                }

                try {
                    return param.getConstructor().newInstance();
                } catch (Exception e) {
                    throw new RuntimeException(e);
                }
            });

            Parent viewRoot = loader.load();

            // Step 2: Store the new controller for the next cleanup cycle
            currentViewController = loader.getController();

            mainContent.getChildren().setAll(viewRoot);

            // Update dataset label whenever view changes
            updateDatasetLabel();
        } catch (IOException e) {
            // This is a good place for more robust logging later
            LogService.error("Dashboard", "Error loading view " + fxmlFile + ": " + e.getMessage());
            e.printStackTrace();

            // Show detailed stack trace to user for debugging
            java.io.StringWriter sw = new java.io.StringWriter();
            java.io.PrintWriter pw = new java.io.PrintWriter(sw);
            e.printStackTrace(pw);
            String exceptionText = sw.toString();

            Alert alert = new Alert(Alert.AlertType.ERROR);
            alert.setTitle("View Load Error");
            alert.setHeaderText("Failed to load: " + fxmlFile);
            alert.setContentText(e.getMessage());

            // Expandable Exception Content
            javafx.scene.control.TextArea textArea = new javafx.scene.control.TextArea(exceptionText);
            textArea.setEditable(false);
            textArea.setWrapText(true);
            textArea.setMaxWidth(Double.MAX_VALUE);
            textArea.setMaxHeight(Double.MAX_VALUE);
            javafx.scene.layout.GridPane.setVgrow(textArea, javafx.scene.layout.Priority.ALWAYS);
            javafx.scene.layout.GridPane.setHgrow(textArea, javafx.scene.layout.Priority.ALWAYS);

            javafx.scene.layout.GridPane expContent = new javafx.scene.layout.GridPane();
            expContent.setMaxWidth(Double.MAX_VALUE);
            expContent.add(new Label("Full Stack Trace:"), 0, 0);
            expContent.add(textArea, 0, 1);

            alert.getDialogPane().setExpandableContent(expContent);
            alert.showAndWait();
        }
    }

    private void showError(String message) {
        LogService.error("Dashboard", "UI Error: " + message);
        Label errorLabel = new Label("Error: " + message);
        errorLabel.setStyle("-fx-text-fill: red; -fx-font-weight: bold;");
        VBox errorBox = new VBox(errorLabel);
        errorBox.setAlignment(Pos.CENTER);
        mainContent.getChildren().setAll(errorBox);
    }

    @FXML
    private void onExitClicked() {
        Platform.exit();
        System.exit(0);
    }

    private void showAlert(Alert.AlertType alertType, String title, String content) {
        Alert alert = new Alert(alertType);
        alert.setTitle(title);
        alert.setHeaderText(null); // No header text
        alert.setContentText(content);
        alert.showAndWait();
    }

    private void updateDatasetLabel() {
        if (datasetLabel != null) {
            File file = appStateService.getSelectedDatasetFile();
            if (file != null) {
                datasetLabel.setText("Dataset: " + file.getName());
                datasetLabel.setStyle("-fx-text-fill: -color-accent-500; -fx-font-size: 12px;");
            } else {
                datasetLabel.setText("No dataset selected");
                datasetLabel.setStyle("-fx-text-fill: -color-text-secondary; -fx-font-size: 12px; -fx-opacity: 0.8;");
            }
        }
    }
}
