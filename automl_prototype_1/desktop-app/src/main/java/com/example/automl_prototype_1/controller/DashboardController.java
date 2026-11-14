package com.example.automl_prototype_1.controller;

import javafx.scene.control.Alert;
import com.example.automl_prototype_1.algorithm.*;
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
import javafx.scene.layout.Region;
import javafx.scene.layout.StackPane;
import javafx.scene.layout.VBox;
import com.example.automl_prototype_1.ui.UiMotion;

import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.net.URL;
import java.util.List;

public class DashboardController {

    @FXML
    private Button homeBtn, uploadBtn, algoSelectBtn, mlBtn, dlBtn, cnnBtn, llmBtn, genAiBtn, runBtn, resultsBtn, settingsBtn, simulatorBtn,
            rlLabBtn, visualizerBtn,
            dataVizBtn,
            creatorLabBtn, architectLabBtn, modelHubBtn, agentDebugBtn, perfHistoryBtn, exitBtn, optimizeOsBtn,
            popOutBtn, helpBtn;
    @FXML
    private StackPane mainContent;
    @FXML
    private Label datasetLabel;
    @FXML
    private Label resourceStatusLabel; // Phase 6 UI Indicator
    @FXML
    private Label systemStatsLabel; // Live system stats
    @FXML
    private Label remoteStatusLabel;
    @FXML
    private Label viewTitleLabel;
    @FXML
    private StackPane chatContainer;
    @FXML
    private Button chatToggleBtn;
    @FXML
    private StackPane shellRoot;
    @FXML
    private Region orbAmber;
    @FXML
    private Region orbTeal;
    @FXML
    private VBox sidebarPane;

    // --- Services ---

    private final AppStateService appStateService; // Final for DI
    private CsvDataProvider csvDataProvider;
    private PreprocessorService preprocessorService;
    private Object currentViewController;

    private Parent currentViewRoot;
    private javafx.stage.Stage poppedOutStage;
    private String currentActiveViewFxml = "";
    private boolean routeSmokeTestRunning;
    private final List<String> routeLoadFailures = new java.util.ArrayList<>();

    private final java.util.Map<String, String> viewTitles = java.util.Map.ofEntries(
            java.util.Map.entry("overview_view.fxml", "Overview"),
            java.util.Map.entry("upload_view.fxml", "Data & Upload"),
            java.util.Map.entry("algo_select_view.fxml", "Algorithm Selection"),
            java.util.Map.entry("ml_workspace.fxml", "Machine Learning"),
            java.util.Map.entry("dl_workspace.fxml", "Deep Learning"),
            java.util.Map.entry("cnn_workspace.fxml", "Vision / CNN"),
            java.util.Map.entry("llm_workspace.fxml", "LLM / SLM"),
            java.util.Map.entry("genai_workspace.fxml", "Generative AI"),
            java.util.Map.entry("pipeline_simulator.fxml", "Pipeline Designer"),
            java.util.Map.entry("rl_lab_view.fxml", "Reinforcement Learning Lab"),
            java.util.Map.entry("rl_visualizer_view.fxml", "RL Visualizer"),
            java.util.Map.entry("data_visualization_view.fxml", "Data Visualizer"),
            java.util.Map.entry("neural_forge_view.fxml", "Neural Forge"),
            java.util.Map.entry("creator_lab_view.fxml", "Creator Lab"),
            java.util.Map.entry("model_hub_view.fxml", "Model Hub"),
            java.util.Map.entry("agent_debug_view.fxml", "Agent Debug"),
            java.util.Map.entry("performance_history_view.fxml", "Performance History"),
            java.util.Map.entry("results_view.fxml", "Results & Exports"),
            java.util.Map.entry("settings_view.fxml", "Settings"),
            java.util.Map.entry("progress_view.fxml", "Experiment Progress"));

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

        // Phase 6 Step 6: UI Resource Mode
        checkResourceMode();

        // Live system stats polling (RAM/CPU/GPU) in header
        startSystemStatsPolling();

        // Set initial view
        loadView("overview_view.fxml");

        // Set button actions
        homeBtn.setOnAction(event -> loadView("overview_view.fxml"));
        uploadBtn.setOnAction(event -> loadView("upload_view.fxml"));
        if (algoSelectBtn != null)
            algoSelectBtn.setOnAction(event -> loadView("algo_select_view.fxml"));
        if (mlBtn != null)
            mlBtn.setOnAction(event -> loadView("ml_workspace.fxml"));
        if (dlBtn != null)
            dlBtn.setOnAction(event -> loadView("dl_workspace.fxml"));
        if (cnnBtn != null)
            cnnBtn.setOnAction(event -> loadView("cnn_workspace.fxml"));
        if (llmBtn != null)
            llmBtn.setOnAction(event -> loadView("llm_workspace.fxml"));
        if (genAiBtn != null)
            genAiBtn.setOnAction(event -> loadView("genai_workspace.fxml"));
        runBtn.setOnAction(event -> handleRunModel());
        resultsBtn.setOnAction(event -> loadView("results_view.fxml"));
        settingsBtn.setOnAction(event -> loadView("settings_view.fxml"));
        simulatorBtn.setOnAction(event -> loadView("pipeline_simulator.fxml"));
        if (rlLabBtn != null)
            rlLabBtn.setOnAction(event -> loadView("rl_lab_view.fxml"));
        if (dataVizBtn != null)
            dataVizBtn.setOnAction(evt -> loadView("data_visualization_view.fxml"));
        if (visualizerBtn != null)
            visualizerBtn.setOnAction(event -> loadView("rl_visualizer_view.fxml"));
        if (creatorLabBtn != null)
            creatorLabBtn.setOnAction(event -> loadView("neural_forge_view.fxml"));
        if (architectLabBtn != null)
            architectLabBtn.setOnAction(event -> loadView("creator_lab_view.fxml"));
        if (modelHubBtn != null)
            modelHubBtn.setOnAction(event -> loadView("model_hub_view.fxml"));
        if (agentDebugBtn != null)
            agentDebugBtn.setOnAction(event -> loadView("agent_debug_view.fxml"));
        if (perfHistoryBtn != null)
            perfHistoryBtn.setOnAction(event -> loadView("performance_history_view.fxml"));
        exitBtn.setOnAction(event -> onExitClicked());

        if (optimizeOsBtn != null) {
            optimizeOsBtn.setOnAction(event -> showOptimizerDialog());
        }

        if (popOutBtn != null) {
            popOutBtn.setOnAction(event -> handlePopOutWindow());
        }
        if (helpBtn != null) {
            helpBtn.setOnAction(event -> showHelpDialog());
        }

        updateRemoteStatus();

        // Chat Integration
        if (chatContainer != null && chatToggleBtn != null) {
            loadChatView();
            chatToggleBtn.setOnAction(e -> {
                boolean isVisible = chatContainer.isVisible();
                chatContainer.setVisible(!isVisible);
                chatContainer.setManaged(!isVisible); // Ensure it doesn't take space when hidden
            });
        }

        // Initial setup for visible modules based on AppState
        Platform.runLater(this::refreshNavigationModules);
        Platform.runLater(this::startShellAtmosphere);
    }

    private void startShellAtmosphere() {
        if (orbAmber != null && shellRoot != null) {
            StackPane.setAlignment(orbAmber, javafx.geometry.Pos.TOP_RIGHT);
            orbAmber.setTranslateX(40);
            orbAmber.setTranslateY(-40);
            UiMotion.pulseOpacity(orbAmber, 0.35, 0.7, 4200);
            UiMotion.drift(orbAmber, -36, 28, 9000);
        }
        if (orbTeal != null && shellRoot != null) {
            StackPane.setAlignment(orbTeal, javafx.geometry.Pos.BOTTOM_LEFT);
            orbTeal.setTranslateX(-20);
            orbTeal.setTranslateY(40);
            UiMotion.pulseOpacity(orbTeal, 0.25, 0.55, 5600);
            UiMotion.drift(orbTeal, 30, -24, 11000);
        }
        if (sidebarPane != null) {
            sidebarPane.setOpacity(0);
            sidebarPane.setTranslateX(-18);
            javafx.animation.FadeTransition fade = new javafx.animation.FadeTransition(
                    javafx.util.Duration.millis(420), sidebarPane);
            fade.setFromValue(0);
            fade.setToValue(1);
            javafx.animation.TranslateTransition slide = new javafx.animation.TranslateTransition(
                    javafx.util.Duration.millis(420), sidebarPane);
            slide.setFromX(-18);
            slide.setToX(0);
            slide.setInterpolator(javafx.animation.Interpolator.EASE_OUT);
            new javafx.animation.ParallelTransition(fade, slide).play();
        }
    }

    private void loadChatView() {
        try {
            FXMLLoader loader = new FXMLLoader(
                    getClass().getResource("/com/example/automl_prototype_1/chat_view.fxml"));
            Parent chatView = loader.load();
            ChatViewController controller = loader.getController();

            // Set close action
            controller.setOnCloseAction(() -> {
                chatContainer.setVisible(false);
                chatContainer.setManaged(false);
            });

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
            cleanupCurrentView();
            currentViewController = progressController; // Store for later cleanup
            currentViewRoot = progressView;
            currentActiveViewFxml = "progress_view.fxml";
            mainContent.getChildren().setAll(progressView);
            updateViewChrome("progress_view.fxml");
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
                // Retrieve preprocessing config from global state
                com.example.automl_prototype_1.model.PreprocessingConfig config = appStateService
                        .getPreprocessingConfig();
                if (config == null) {
                    config = new com.example.automl_prototype_1.model.PreprocessingConfig();
                    String effectiveScaling = appStateService.getScalingStrategy();
                    if (effectiveScaling == null)
                        effectiveScaling = "Standardization";
                    config.setScalingStrategy(effectiveScaling);
                }

                // Create safe config: disable scaling/imputation for preprocessing
                // (they are applied per-fold inside the algorithm's CV loop to prevent data
                // leakage)
                com.example.automl_prototype_1.model.PreprocessingConfig safeConfig = new com.example.automl_prototype_1.model.PreprocessingConfig();
                safeConfig.setAutoRemoveIdColumns(config.isAutoRemoveIdColumns());
                safeConfig.setImputeMissingValues(false);
                safeConfig.setRemoveOutliers(config.isRemoveOutliers());
                safeConfig.setScalingStrategy("None");

                LogService.info("Dashboard", "Applying Preprocessing (Scaling Disabled to prevent Leakage)...");
                Dataset finalDataset = preprocessorService.runPreprocessingPipeline(rawDataset, targetColumn,
                        safeConfig);
                rawDataset = null;
                System.gc();

                // Delegate execution pipeline to ExecutionService
                return ExecutionService.getInstance().runSequentialPipeline(
                        finalDataset,
                        algorithms,
                        classifier,
                        isTuning,
                        this::isCancelled,
                        this::updateMessage);
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
            List<ExecutionResult> results = executionTask.getValue();
            appStateService.setExecutionResults(results);
            isRunning = false;
            runBtn.setText("Run Experiment");
            runBtn.setStyle("");
            runBtn.setDisable(false);
            loadView("results_view.fxml");

            // --- Phase 5: AgenticAdvisor post-run callback ---
            if (results != null && !results.isEmpty()) {
                new Thread(() -> {
                    try {
                        com.example.automl_prototype_1.service.agentic.AgenticAdvisor advisor = new com.example.automl_prototype_1.service.agentic.AgenticAdvisor();
                        String advice = advisor.quickAdvice(results);
                        Platform.runLater(() -> {
                            if (advice != null && !advice.isBlank()) {
                                showAlert(Alert.AlertType.INFORMATION, "AutoML Advisor", advice);
                            }
                        });
                    } catch (Exception ex) {
                        LogService.warn("Dashboard", "AgenticAdvisor unavailable: " + ex.getMessage());
                    }
                }, "AgenticAdvisor").start();
            }
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

    // --- UI Helper Methods ---
    private void loadView(String fxmlFile) {
        try {
            if (poppedOutStage != null && poppedOutStage.isShowing()) {
                poppedOutStage.close();
            }

            String fullPath = "/com/example/automl_prototype_1/" + fxmlFile;
            URL fxmlUrl = getClass().getResource(fullPath);
            if (fxmlUrl == null) {
                showError("Cannot find FXML resource: " + fullPath);
                return;
            }

            FXMLLoader loader = new FXMLLoader(fxmlUrl);

            // --- Dependency Injection for Sub-Controllers ---
            loader.setControllerFactory(param -> {
                if (param == OverviewViewController.class) {
                    return new OverviewViewController(this::loadView, this::handleRunModel);
                }
                if (param == PipelineSimulatorController.class) {
                    return new PipelineSimulatorController(this::handleRunModel);
                }
                if (param == ResultsViewController.class) {
                    ResultsViewController controller = new ResultsViewController();
                    controller.setOnBackAction(() -> loadView("overview_view.fxml"));
                    return controller;
                }
                if (param == SettingsViewController.class) {
                    SettingsViewController controller = new SettingsViewController();
                    controller.setOnVisibilityChanged(() -> Platform.runLater(this::refreshNavigationModules));
                    return controller;
                }

                try {
                    return param.getConstructor().newInstance();
                } catch (Exception e) {
                    throw new RuntimeException(e);
                }
            });

            Parent viewRoot = loader.load();
            Object nextController = loader.getController();

            // Stop timers, listeners, and background work owned by the outgoing view.
            // Stateful views are recreated on navigation so stale UI state cannot survive.
            cleanupCurrentView();
            currentViewController = nextController;
            currentViewRoot = viewRoot;
            if (routeSmokeTestRunning) {
                mainContent.getChildren().setAll(viewRoot);
            } else {
                UiMotion.swapContent(mainContent, viewRoot);
            }
            currentActiveViewFxml = fxmlFile;

            updateViewChrome(fxmlFile);
            updateDatasetLabel();
            LogService.info("UI", "Loaded view: " + fxmlFile);
        } catch (Exception e) {
            LogService.error("Dashboard", "Error loading view " + fxmlFile + ": " + e.getMessage());
            e.printStackTrace();
            if (routeSmokeTestRunning) {
                routeLoadFailures.add(fxmlFile + ": " + e.getMessage());
                return;
            }

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

    private void cleanupCurrentView() {
        if (currentViewController instanceof Cleanable cleanable) {
            try {
                cleanable.cleanup();
            } catch (Exception e) {
                LogService.warn("Dashboard", "View cleanup failed: " + e.getMessage());
            }
        }
    }

    private void updateViewChrome(String fxmlFile) {
        if (viewTitleLabel != null) {
            viewTitleLabel.setText(viewTitles.getOrDefault(fxmlFile, "AutoML Studio"));
        }

        java.util.Map<String, Button> routeButtons = java.util.Map.ofEntries(
                java.util.Map.entry("overview_view.fxml", homeBtn),
                java.util.Map.entry("upload_view.fxml", uploadBtn),
                java.util.Map.entry("data_visualization_view.fxml", dataVizBtn),
                java.util.Map.entry("algo_select_view.fxml", algoSelectBtn),
                java.util.Map.entry("pipeline_simulator.fxml", simulatorBtn),
                java.util.Map.entry("ml_workspace.fxml", mlBtn),
                java.util.Map.entry("dl_workspace.fxml", dlBtn),
                java.util.Map.entry("cnn_workspace.fxml", cnnBtn),
                java.util.Map.entry("llm_workspace.fxml", llmBtn),
                java.util.Map.entry("genai_workspace.fxml", genAiBtn),
                java.util.Map.entry("rl_lab_view.fxml", rlLabBtn),
                java.util.Map.entry("rl_visualizer_view.fxml", visualizerBtn),
                java.util.Map.entry("neural_forge_view.fxml", creatorLabBtn),
                java.util.Map.entry("creator_lab_view.fxml", architectLabBtn),
                java.util.Map.entry("model_hub_view.fxml", modelHubBtn),
                java.util.Map.entry("agent_debug_view.fxml", agentDebugBtn),
                java.util.Map.entry("performance_history_view.fxml", perfHistoryBtn),
                java.util.Map.entry("results_view.fxml", resultsBtn),
                java.util.Map.entry("settings_view.fxml", settingsBtn));

        routeButtons.values().forEach(button -> button.getStyleClass().remove("active"));
        Button active = routeButtons.get(fxmlFile);
        if (active != null && !active.getStyleClass().contains("active")) {
            active.getStyleClass().add("active");
        }
    }

    /**
     * Development/runtime diagnostic: instantiate every dashboard route in the same
     * way a user opens it. Enabled with -Dautoml.ui.smoke=true.
     */
    public void runViewSmokeTest() {
        List<String> routes = List.of(
                "upload_view.fxml",
                "data_visualization_view.fxml",
                "algo_select_view.fxml",
                "pipeline_simulator.fxml",
                "ml_workspace.fxml",
                "dl_workspace.fxml",
                "cnn_workspace.fxml",
                "llm_workspace.fxml",
                "genai_workspace.fxml",
                "rl_lab_view.fxml",
                "rl_visualizer_view.fxml",
                "neural_forge_view.fxml",
                "creator_lab_view.fxml",
                "model_hub_view.fxml",
                "agent_debug_view.fxml",
                "performance_history_view.fxml",
                "results_view.fxml",
                "settings_view.fxml",
                "overview_view.fxml");
        routeLoadFailures.clear();
        routeSmokeTestRunning = true;
        try {
            routes.forEach(this::loadView);
        } finally {
            routeSmokeTestRunning = false;
        }
        if (routeLoadFailures.isEmpty()) {
            LogService.info("UI_SMOKE", "All " + routes.size() + " dashboard routes loaded successfully.");
        } else {
            LogService.error("UI_SMOKE", routeLoadFailures.size() + " route(s) failed: "
                    + String.join(" | ", routeLoadFailures));
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

    private void handlePopOutWindow() {
        if (currentActiveViewFxml == null || currentActiveViewFxml.isEmpty()) {
            return;
        }

        Parent activeView = currentViewRoot;
        if (activeView == null)
            return;

        // If it's already popped out (not in our mainContent), ignore
        if (!mainContent.getChildren().contains(activeView)) {
            showAlert(Alert.AlertType.WARNING, "Already Popped Out", "This window is already popped out!");
            return;
        }

        // Remove from main dashboard and show placeholder
        mainContent.getChildren().clear();
        Label placeholder = new Label("This view corresponds to an external window.");
        placeholder.setStyle("-fx-text-fill: #aaa; -fx-font-size: 16px;");
        mainContent.getChildren().add(placeholder);

        // Create new Stage
        javafx.stage.Stage popOutStage = new javafx.stage.Stage();
        poppedOutStage = popOutStage;
        popOutStage.setTitle("AutoML Studio - Popped Out Window");

        // We wrap it in a new border pane with our dark theme style
        StackPane rootPane = new StackPane(activeView);
        rootPane.setStyle("-fx-background-color: #1e1e1e;");

        javafx.scene.Scene scene = new javafx.scene.Scene(rootPane, 1000, 700);

        // Inherit atelier stylesheets
        java.net.URL atelierTheme = getClass().getResource("/styles/atelier_theme.css");
        java.net.URL darkBase = getClass().getResource("/styles/dark_theme.css");
        if (darkBase != null) {
            scene.getStylesheets().add(darkBase.toExternalForm());
        }
        if (atelierTheme != null) {
            scene.getStylesheets().add(atelierTheme.toExternalForm());
        }

        popOutStage.setScene(scene);

        // When external window closes, dock it back in the cache/main view if it is
        // still the active tab
        String fxmlRef = currentActiveViewFxml;
        popOutStage.setOnCloseRequest(e -> {
            Platform.runLater(() -> {
                LogService.info("UI", "Restoring popped out window: " + fxmlRef);
                // If the user hasn't switched tabs, immediately dock it back into view
                if (currentActiveViewFxml.equals(fxmlRef)) {
                    mainContent.getChildren().setAll(activeView);
                }
                poppedOutStage = null;
            });
        });

        popOutStage.show();
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

    private void loadAlgoSelectView(String tabName) {
        loadView("algo_select_view.fxml");
        if (currentViewController instanceof AlgoSelectViewController) {
            ((AlgoSelectViewController) currentViewController).selectTab(tabName);
        }
    }

    public void refreshNavigationModules() {
        if (mlBtn != null) {
            mlBtn.setManaged(appStateService.isShowMl());
            mlBtn.setVisible(appStateService.isShowMl());
        }
        if (dlBtn != null) {
            dlBtn.setManaged(appStateService.isShowDl());
            dlBtn.setVisible(appStateService.isShowDl());
        }
        if (cnnBtn != null) {
            cnnBtn.setManaged(appStateService.isShowCnn());
            cnnBtn.setVisible(appStateService.isShowCnn());
        }
        if (llmBtn != null) {
            llmBtn.setManaged(appStateService.isShowLlm());
            llmBtn.setVisible(appStateService.isShowLlm());
        }
        if (genAiBtn != null) {
            genAiBtn.setManaged(appStateService.isShowGenAi());
            genAiBtn.setVisible(appStateService.isShowGenAi());
        }
        if (rlLabBtn != null) {
            rlLabBtn.setManaged(appStateService.isShowRl());
            rlLabBtn.setVisible(appStateService.isShowRl());
        }
    }

    private void updateRemoteStatus() {
        if (remoteStatusLabel == null) {
            return;
        }
        RemoteStatusServer server = RemoteStatusServer.getInstance();
        if (server.isRunning()) {
            String auth = server.isAuthEnabled() ? " · token on" : "";
            remoteStatusLabel.setText("Remote online · :" + server.getPort() + auth);
            remoteStatusLabel.setStyle("-fx-text-fill: #3fb950;");
        } else {
            remoteStatusLabel.setText("Remote offline");
            remoteStatusLabel.setStyle("-fx-text-fill: #f85149;");
        }
    }

    private void showHelpDialog() {
        RemoteStatusServer remote = RemoteStatusServer.getInstance();
        String tokenHint = remote.isAuthEnabled()
                ? "\nAndroid token (X-AutoML-Token): " + remote.getAuthToken()
                        + "\n(also in ~/.automl_remote_token)"
                : "\n(Remote auth disabled via AUTOML_REMOTE_AUTH=false)";
        showAlert(Alert.AlertType.INFORMATION, "AutoML Studio — Quick Start",
                "1. Open Data & Upload and select a CSV dataset.\n"
                        + "2. Choose the target column.\n"
                        + "3. Open Algorithms or an AI domain workspace.\n"
                        + "4. Configure the model and click Run Experiment.\n"
                        + "5. Track execution in Progress, then export from Results.\n\n"
                        + "The Android companion monitors this desktop at:\n"
                        + "http://<this-PC-LAN-IP>:" + remote.getPort() + "/api/status"
                        + tokenHint);
    }

    private void showOptimizerDialog() {
        com.example.automl_prototype_1.service.agentic.SystemOptimizerService optimizer = new com.example.automl_prototype_1.service.agentic.SystemOptimizerService();

        // Show a loading dialog first
        Alert loadingDialog = new Alert(Alert.AlertType.INFORMATION);
        loadingDialog.setTitle("Agentic OS Optimizer");
        loadingDialog.setHeaderText("Scanning Windows Processes...");
        loadingDialog.setContentText("The AI is determining safe processes to kill. Please wait...");
        loadingDialog.show();

        Task<List<com.example.automl_prototype_1.model.SystemProcess>> scanTask = new Task<>() {
            @Override
            protected List<com.example.automl_prototype_1.model.SystemProcess> call() throws Exception {
                // Fetch > 150MB processes
                List<com.example.automl_prototype_1.model.SystemProcess> procs = optimizer
                        .getHeavyBackgroundProcesses(150);
                // Get AI Analysis for each
                for (com.example.automl_prototype_1.model.SystemProcess p : procs) {
                    optimizer.analyzeProcessImpact(p);
                }
                return procs;
            }
        };

        scanTask.setOnSucceeded(e -> {
            loadingDialog.close();
            List<com.example.automl_prototype_1.model.SystemProcess> results = scanTask.getValue();

            if (results.isEmpty()) {
                showAlert(Alert.AlertType.INFORMATION, "Optimizer",
                        "No heavy non-essential processes found. Your OS is optimized!");
                return;
            }

            // Build custom Dialog UI
            javafx.scene.control.Dialog<List<String>> dialog = new javafx.scene.control.Dialog<>();
            dialog.setTitle("OS Optimizer - AI Analysis");
            dialog.setHeaderText("Select the processes you allow the AI to forcefully kill:");

            dialog.getDialogPane().getButtonTypes().addAll(
                    javafx.scene.control.ButtonType.OK, javafx.scene.control.ButtonType.CANCEL);

            VBox vbox = new VBox(10);
            vbox.setStyle("-fx-padding: 10; -fx-background-color: #2b2b2b;");

            // We use checkboxes parallel to the processes
            List<javafx.scene.control.CheckBox> checkBoxes = new java.util.ArrayList<>();

            for (com.example.automl_prototype_1.model.SystemProcess proc : results) {
                javafx.scene.control.CheckBox cb = new javafx.scene.control.CheckBox();
                cb.setStyle("-fx-text-fill: white; -fx-font-weight: bold;");
                cb.setText(proc.getName() + " (" + proc.getMemoryUsageMb() + " MB)");

                Label aiLabel = new Label("🤖 AI Impact: " + proc.getAiAnalysis());
                aiLabel.setStyle("-fx-text-fill: #aaa; -fx-font-size: 11px; -fx-padding: 0 0 5 25;");
                aiLabel.setWrapText(true);
                aiLabel.setMaxWidth(400);

                vbox.getChildren().addAll(cb, aiLabel);

                // Tie checkbox to process
                cb.selectedProperty().addListener((obs, oldV, newV) -> proc.setSelectedForKill(newV));
            }

            javafx.scene.control.ScrollPane scroll = new javafx.scene.control.ScrollPane(vbox);
            scroll.setFitToWidth(true);
            scroll.setPrefViewportHeight(300);
            scroll.setStyle("-fx-background:#1e1e1e; -fx-control-inner-background:#1e1e1e;");

            dialog.getDialogPane().setContent(scroll);

            dialog.setResultConverter(dialogButton -> {
                if (dialogButton == javafx.scene.control.ButtonType.OK) {
                    List<String> pidsToKill = new java.util.ArrayList<>();
                    for (com.example.automl_prototype_1.model.SystemProcess p : results) {
                        if (p.isSelectedForKill())
                            pidsToKill.add(p.getPid());
                    }
                    return pidsToKill;
                }
                return null;
            });

            java.util.Optional<List<String>> result = dialog.showAndWait();
            result.ifPresent(pids -> {
                if (!pids.isEmpty()) {
                    boolean success = optimizer.killProcesses(pids);
                    if (success) {
                        showAlert(Alert.AlertType.INFORMATION, "Optimizer",
                                "Successfully killed selected processes. RAM freed!");
                    } else {
                        showAlert(Alert.AlertType.ERROR, "Optimizer",
                                "Failed to kill some processes. You may need Administrator rights.");
                    }
                }
            });
        });

        scanTask.setOnFailed(e -> {
            loadingDialog.close();
            showAlert(Alert.AlertType.ERROR, "Optimizer Failed",
                    "Failed to scan OS processes: " + scanTask.getException().getMessage());
        });

        new Thread(scanTask).start();
    }
}
