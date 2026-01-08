package com.example.automl_prototype_1.controller;

import javafx.scene.control.Alert;
import com.example.automl_prototype_1.algorithm.*;
import com.example.automl_prototype_1.dataprovider.CsvDataProvider;
import com.example.automl_prototype_1.model.Dataset;
import com.example.automl_prototype_1.model.ExecutionResult;
import com.example.automl_prototype_1.service.*;
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
    private Button uploadBtn, algoBtn, runBtn, resultsBtn, settingsBtn, simulatorBtn, exitBtn;
    @FXML
    private StackPane mainContent;
    @FXML
    private Label datasetLabel;

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

        // Set initial view
        loadView("upload_view.fxml");

        // Set button actions
        uploadBtn.setOnAction(event -> loadView("upload_view.fxml"));
        algoBtn.setOnAction(event -> loadView("algo_select_view.fxml"));
        runBtn.setOnAction(event -> handleRunModel());
        resultsBtn.setOnAction(event -> loadView("results_view.fxml"));
        settingsBtn.setOnAction(event -> loadView("settings_view.fxml"));
        simulatorBtn.setOnAction(event -> loadView("pipeline_simulator.fxml"));
        exitBtn.setOnAction(event -> onExitClicked());
    }

    // In DashboardController.java

    private void handleRunModel() {
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
        ProgressTrackingService.getInstance().reset(algorithms);
        runBtn.setDisable(true);

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
            showError("Failed to load progress view.");
            runBtn.setDisable(false); // Re-enable the button on UI failure
            return;
        }

        // ===================================================================
        // 3. DEFINE THE LONG-RUNNING WORK IN A BACKGROUND TASK
        // ===================================================================
        Task<List<ExecutionResult>> executionTask = new Task<>() {
            @Override
            protected List<ExecutionResult> call() throws Exception {
                updateMessage("Loading and Preprocessing Data...");
                Dataset rawDataset;
                try (FileReader reader = new FileReader(datasetFile)) {
                    rawDataset = csvDataProvider.loadDataset(reader);
                }
                // Auto-apply scaling for distance-based algorithms if not explicitly set
                // Auto-apply scaling: Dropdown takes precedence, Checkbox is fallback default
                String effectiveScaling = appStateService.getScalingStrategy();
                if (effectiveScaling == null || effectiveScaling.equalsIgnoreCase("None")) {
                    if (appStateService.isStandardizationEnabled()) {
                        effectiveScaling = "Standardization";
                    } else {
                        effectiveScaling = "None";
                    }
                }

                System.out.println("Scaling Strategy: " + effectiveScaling);

                Dataset finalDataset = preprocessorService.runPreprocessingPipeline(rawDataset, targetColumn,
                        effectiveScaling);

                // --- THIS IS THE CORRECT DYNAMIC TRIAL LOGIC ---
                int tuningTrials;
                if (finalDataset.getRecordCount() < 1000) {
                    tuningTrials = 3; // Fewer trials for small datasets like WDBC
                } else if (finalDataset.getRecordCount() < 10000) {
                    tuningTrials = 6; // Standard trials for medium datasets
                } else {
                    tuningTrials = 9; // More trials for large datasets
                }
                // -------------------------------------------------

                List<ExecutionResult> finalResults = new ArrayList<>();
                int totalAlgos = algorithms.size();
                ProgressTrackingService progressService = ProgressTrackingService.getInstance();
                progressService.reset(algorithms); // Reset and prime the service with the new queue

                for (int i = 0; i < totalAlgos; i++) {
                    String algoName = algorithms.get(i);

                    if (isTuning) {
                        updateMessage("Tuning " + algoName + " (" + (i + 1) + "/" + totalAlgos + ")");
                        // Use the CORRECT 'tuningTrials' variable here
                        ExecutionResult bestTunedResult = tuningService.tune(algoName, tuningTrials, finalDataset,
                                classifier);
                        if (bestTunedResult != null)
                            finalResults.add(bestTunedResult);
                    } else {
                        updateMessage("Running " + algoName + " (" + (i + 1) + "/" + totalAlgos + ")");
                        OptimizationAlgorithm algorithm = getAlgorithmImplementation(algoName);
                        if (algorithm != null) {
                            finalResults.add(algorithm.execute(finalDataset, Map.of("classifier", classifier)));
                        }
                    }
                }
                return finalResults;
            }
        };

        // ===================================================================
        // 4. BIND THE UI TO THE TASK AND DEFINE COMPLETION BEHAVIOR
        // ===================================================================

        // This is the critical link that makes the UI update in real-time
        if (progressController != null) {
            progressController.bindToTask(executionTask);
        }

        // Define what happens when the task finishes successfully
        executionTask.setOnSucceeded(event -> Platform.runLater(() -> {
            appStateService.setExecutionResults(executionTask.getValue());
            runBtn.setDisable(false);
            loadView("results_view.fxml"); // Automatically cleans up progress view and shows results
        }));

        // Define what happens if the task fails with an exception
        executionTask.setOnFailed(event -> Platform.runLater(() -> {
            Throwable e = executionTask.getException();
            e.printStackTrace(); // Keep logging to console

            // Show a persistent Alert dialog so the user can see the error
            showAlert(Alert.AlertType.ERROR, "Execution Failed",
                    "An error occurred during execution:\n" + e.getMessage());

            runBtn.setDisable(false);
            // Do NOT redirect to upload_view.fxml. Stay on the current view (Progress View)
            // or let the user decide.
            // If we want to return to Algo Select, we can, but let's let the user read the
            // error first.
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
            e.printStackTrace();
            showError("Error loading view: " + fxmlFile);
        }
    }

    private void showError(String message) {
        System.err.println("UI Error: " + message);
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