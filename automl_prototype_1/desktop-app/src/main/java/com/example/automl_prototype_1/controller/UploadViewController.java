package com.example.automl_prototype_1.controller;

import com.example.automl_prototype_1.model.Dataset;
import com.example.automl_prototype_1.service.AppStateService;
import com.example.automl_prototype_1.dataprovider.CsvDataProvider;
import com.example.automl_prototype_1.model.AttributeSummary;
import com.example.automl_prototype_1.service.RecommendationService;
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
                e.printStackTrace();
                appStateService.setDatasetHeaders(null);
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

    private void showError(String message) {
        Alert alert = new Alert(Alert.AlertType.ERROR);
        alert.setTitle("Error");
        alert.setHeaderText(null);
        alert.setContentText(message);
        alert.showAndWait();
    }
}