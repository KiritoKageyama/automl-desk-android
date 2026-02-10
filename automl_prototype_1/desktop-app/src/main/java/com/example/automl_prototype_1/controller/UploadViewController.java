package com.example.automl_prototype_1.controller;

import com.example.automl_prototype_1.model.Dataset;
import com.example.automl_prototype_1.service.AppStateService;
import com.example.automl_prototype_1.dataprovider.CsvDataProvider;
import com.example.automl_prototype_1.model.AttributeSummary;
import com.example.automl_prototype_1.service.RecommendationService;
import com.example.automl_prototype_1.experiment.SyntheticDataGenerator;
import javafx.collections.FXCollections;
import javafx.fxml.FXML;
import javafx.scene.control.Alert;
import javafx.scene.control.Button;
import javafx.scene.control.ChoiceBox;
import javafx.scene.control.Label;
import javafx.scene.control.TextArea;
import javafx.stage.FileChooser;
import javafx.stage.Stage;

import java.io.File;
import java.io.FileReader;
import java.util.List;

public class UploadViewController {
    @FXML
    private TextArea analysisTextArea;
    @FXML
    private Button browseButton;
    @FXML
    private Label selectedFileLabel;
    @FXML
    private ChoiceBox<String> targetColumnChoiceBox;
    @FXML
    private Button analyzeButton;

    private final AppStateService appStateService = AppStateService.getInstance();
    private final RecommendationService recommendationService = new RecommendationService();

    public UploadViewController() {
        // No-arg constructor for FXML
    }

    @FXML
    private void initialize() {
        browseButton.setOnAction(event -> handleBrowse());

        // Listener for when a new target column is selected
        targetColumnChoiceBox.getSelectionModel().selectedItemProperty().addListener(
                (obs, oldVal, newVal) -> {
                    if (newVal != null) {
                        appStateService.setTargetColumnName(newVal);
                    }
                });

        // Initialize view based on current state
        updateViewFromState();
        analyzeButton.setOnAction(event -> handleAnalyze());
        updateViewFromState(); // Call this to set initial button state
    }

    private void handleAnalyze() {
        Dataset tempDataset;
        try (FileReader reader = new FileReader(appStateService.getSelectedDatasetFile())) {
            tempDataset = new CsvDataProvider().loadDataset(reader);
        } catch (Exception e) {
            analysisTextArea.setText("Error: Could not load and analyze the dataset.\n\n" + e.getMessage());
            return;
        }

        List<AttributeSummary> summaries = recommendationService.analyzeAttributes(tempDataset);

        StringBuilder report = new StringBuilder();
        report.append(String.format("%-25s | %-11s | %-8s | %-8s | %-10s | %-10s | %-10s | %s\n",
                "Attribute", "Data Type", "Unique", "Missing", "Mean", "Std Dev", "Min", "Max"));
        report.append("-".repeat(110)).append("\n");

        for (AttributeSummary summary : summaries) {
            report.append(String.format("%-25s | %-11s | %-8d | %-8d | %-10s | %-10s | %-10s | %s\n",
                    summary.attributeName(), summary.dataType(), summary.uniqueValues(), summary.missingValues(),
                    summary.mean(), summary.stdDev(), summary.min(), summary.max()));
        }

        report.append("\n\n--- Insights & Suggestions ---\n");
        report.append(
                "\n* Target Column: Look for a 'Categorical' attribute with a low number of unique values (e.g., 2 for binary classification).\n");

        for (AttributeSummary summary : summaries) {
            if (summary.missingValues() > 0) {
                report.append("* Warning: The column '").append(summary.attributeName())
                        .append("' has missing values.\n");
            }
            if (summary.dataType().equals("Numeric") && summary.uniqueValues() <= 2) {
                report.append("* Info: The numeric column '").append(summary.attributeName())
                        .append("' might be a binary categorical feature.\n");
            }
        }

        analysisTextArea.setText(report.toString());
        analysisTextArea.setText(report.toString());
    }

    @FXML
    private void handleAdvancedSettings() {
        // Create a custom dialog for preprocessing settings
        javafx.scene.control.Dialog<com.example.automl_prototype_1.model.PreprocessingConfig> dialog = new javafx.scene.control.Dialog<>();
        dialog.setTitle("Advanced Preprocessing Settings");
        dialog.setHeaderText("Configure Data Cleaning & Scaling");

        // Set the button types
        javafx.scene.control.ButtonType saveButtonType = new javafx.scene.control.ButtonType("Save",
                javafx.scene.control.ButtonBar.ButtonData.OK_DONE);
        dialog.getDialogPane().getButtonTypes().addAll(saveButtonType, javafx.scene.control.ButtonType.CANCEL);

        // Create the layout
        javafx.scene.layout.GridPane grid = new javafx.scene.layout.GridPane();
        grid.setHgap(10);
        grid.setVgap(10);
        grid.setPadding(new javafx.geometry.Insets(20, 150, 10, 10));

        // Get current config
        com.example.automl_prototype_1.model.PreprocessingConfig currentConfig = appStateService
                .getPreprocessingConfig();

        // Add controls
        javafx.scene.control.CheckBox removeIds = new javafx.scene.control.CheckBox("Auto-remove ID Columns");
        removeIds.setSelected(currentConfig.isAutoRemoveIdColumns());

        javafx.scene.control.CheckBox impute = new javafx.scene.control.CheckBox("Impute Missing Values (Mean/Mode)");
        impute.setSelected(currentConfig.isImputeMissingValues());

        javafx.scene.control.CheckBox outliers = new javafx.scene.control.CheckBox("Remove Outliers (IQR Method)");
        outliers.setSelected(currentConfig.isRemoveOutliers());

        javafx.scene.control.ChoiceBox<String> scaling = new javafx.scene.control.ChoiceBox<>();
        scaling.getItems().addAll("Standardization", "Normalization", "None");
        scaling.setValue(currentConfig.getScalingStrategy());

        grid.add(new javafx.scene.control.Label("Scaling Strategy:"), 0, 0);
        grid.add(scaling, 1, 0);
        grid.add(removeIds, 0, 1);
        grid.add(impute, 0, 2);
        grid.add(outliers, 0, 3);

        dialog.getDialogPane().setContent(grid);

        // Convert the result to a config object when the login button is clicked.
        dialog.setResultConverter(dialogButton -> {
            if (dialogButton == saveButtonType) {
                return new com.example.automl_prototype_1.model.PreprocessingConfig(
                        removeIds.isSelected(),
                        impute.isSelected(),
                        outliers.isSelected(),
                        scaling.getValue());
            }
            return null;
        });

        java.util.Optional<com.example.automl_prototype_1.model.PreprocessingConfig> result = dialog.showAndWait();

        result.ifPresent(newConfig -> {
            appStateService.setPreprocessingConfig(newConfig);
            analysisTextArea.setText(analysisTextArea.getText() + "\n\n[Settings Updated]: " + newConfig.toString());
        });
    }

    private void handleBrowse() {
        FileChooser fileChooser = new FileChooser();
        fileChooser.setTitle("Select Dataset File");
        fileChooser.getExtensionFilters()
                .add(new FileChooser.ExtensionFilter("Dataset Files", "*.csv", "*.data", "*.txt"));
        Stage stage = (Stage) browseButton.getScene().getWindow();
        File file = fileChooser.showOpenDialog(stage);

        if (file != null) {
            appStateService.setSelectedDatasetFile(file);
            // After selecting a file, we MUST load it to get the headers
            try (FileReader reader = new FileReader(file)) {
                Dataset tempDataset = new CsvDataProvider().loadDataset(reader);
                appStateService.setDatasetHeaders(tempDataset.getHeaders()); // Store headers in state
            } catch (Exception e) {
                AppStateService.getInstance().setDatasetHeaders(null);
                com.example.automl_prototype_1.service.LogService.error("UploadView",
                        "Failed to load dataset: " + e.getMessage());
                showError("Failed to load dataset: " + e.getMessage());
            }
        }
        updateViewFromState();
    }

    // A helper to sync the UI with the AppStateService
    private void updateViewFromState() {
        File currentFile = appStateService.getSelectedDatasetFile();
        List<String> headers = appStateService.getDatasetHeaders();

        if (currentFile != null) {
            if (headers != null && !headers.isEmpty()) {
                selectedFileLabel.setText("Selected: " + currentFile.getName());
                selectedFileLabel.setStyle(""); // Reset style
                analyzeButton.setDisable(false);

                targetColumnChoiceBox.setItems(FXCollections.observableArrayList(headers));
                targetColumnChoiceBox.setDisable(false);

                // Set selected item if it's already in the list
                String currentTarget = appStateService.getTargetColumnName();
                if (currentTarget != null && headers.contains(currentTarget)) {
                    targetColumnChoiceBox.setValue(currentTarget);
                } else {
                    // Default to the last column
                    targetColumnChoiceBox.setValue(headers.get(headers.size() - 1));
                }
            } else {
                selectedFileLabel.setText("Invalid File: " + currentFile.getName());
                selectedFileLabel.setStyle("-fx-text-fill: red;");
                analyzeButton.setDisable(true);
                targetColumnChoiceBox.getItems().clear();
                targetColumnChoiceBox.setDisable(true);
            }
        } else {
            selectedFileLabel.setText("No file selected.");
            selectedFileLabel.setStyle(""); // Reset style
            analyzeButton.setDisable(true);
            targetColumnChoiceBox.getItems().clear();
            targetColumnChoiceBox.setDisable(true);
        }
    }

    @FXML
    private void handleGenerateSynthetic() {
        try {
            // Generate synthetic dataset: 500 samples, 20 features, 10 informative
            Dataset syntheticDataset = SyntheticDataGenerator.generate(500, 20, 10, false);

            // Store in app state
            appStateService.setCurrentDataset(syntheticDataset);
            appStateService.setDatasetHeaders(syntheticDataset.getHeaders());
            appStateService.setTargetColumnName("Class");

            // Update UI
            selectedFileLabel.setText("Synthetic Dataset (500 x 20)");
            selectedFileLabel.setStyle("-fx-text-fill: #4CAF50;");

            targetColumnChoiceBox.setItems(FXCollections.observableArrayList(syntheticDataset.getHeaders()));
            targetColumnChoiceBox.setValue("Class");
            targetColumnChoiceBox.setDisable(false);
            analyzeButton.setDisable(false);

            analysisTextArea.setText("✅ Synthetic Dataset Generated Successfully!\n\n" +
                    "Samples: 500\n" +
                    "Features: 20 (10 informative, 10 noise)\n" +
                    "Target: Class (binary: 0 or 1)\n\n" +
                    "This dataset is ideal for testing the pipeline quickly.\n" +
                    "You can proceed to Algorithm Selection.");

            com.example.automl_prototype_1.service.LogService.info("UploadView",
                    "Generated synthetic dataset: 500 samples, 20 features");

        } catch (Exception e) {
            showError("Failed to generate synthetic data: " + e.getMessage());
            com.example.automl_prototype_1.service.LogService.error("UploadView",
                    "Synthetic generation failed: " + e.getMessage());
        }
    }

    private void showError(String message) {
        Alert alert = new Alert(Alert.AlertType.ERROR);
        alert.setTitle("Error");
        alert.setHeaderText(null);
        alert.setContentText(message);
        alert.showAndWait();
    }
}