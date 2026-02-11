package com.example.automl_prototype_1.controller;

import com.example.automl_prototype_1.model.ExecutionResult;
import com.example.automl_prototype_1.service.AppStateService;
import com.example.automl_prototype_1.service.LogService;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.fxml.FXML;
import javafx.fxml.Initializable;
import javafx.scene.chart.LineChart;
import javafx.scene.chart.NumberAxis;
import javafx.scene.chart.XYChart;
import javafx.scene.control.Alert;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.control.TableCell;
import javafx.scene.control.TableColumn;
import javafx.scene.control.TableView;
import javafx.scene.control.cell.PropertyValueFactory;
import javafx.stage.FileChooser;
import javafx.stage.Stage;
import javafx.scene.layout.GridPane;
import javafx.scene.chart.BarChart;
import javafx.geometry.Pos;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.net.URL;
import java.text.DecimalFormat;
import java.util.List;
import java.util.Locale;
import java.util.ResourceBundle;
import java.util.Comparator;

public class ResultsViewController implements Initializable {

    // --- FXML Injected Fields ---
    @FXML
    private TableView<ExecutionResult> resultsTable;
    @FXML
    private TableColumn<ExecutionResult, String> algoCol;
    @FXML
    private TableColumn<ExecutionResult, Double> accuracyCol;
    @FXML
    private TableColumn<ExecutionResult, Double> precisionCol;
    @FXML
    private TableColumn<ExecutionResult, Double> recallCol;
    @FXML
    private TableColumn<ExecutionResult, Double> f1Col;
    @FXML
    private TableColumn<ExecutionResult, Double> originalFeaturesCol;
    @FXML
    private TableColumn<ExecutionResult, Double> selectedFeaturesCol;
    @FXML
    private TableColumn<ExecutionResult, Long> timeCol;
    @FXML
    private Button exportOnnxButton;
    @FXML
    private Button exportButton;
    @FXML
    private Button saveButton;
    @FXML
    private Button viewReportButton;
    @FXML
    private LineChart<Number, Number> resultsChart;
    @FXML
    private LineChart<Number, Number> lossChart;
    @FXML
    private NumberAxis chartXAxis;
    @FXML
    private NumberAxis chartYAxis;
    @FXML
    private Label bestAccuracyLabel;
    @FXML
    private Label bestAlgoLabel;
    @FXML
    private Label avgFeaturesLabel;
    @FXML
    private Label originalFeaturesLabel;
    @FXML
    private Label totalTimeLabel;

    @FXML
    private GridPane confusionMatrixGrid;
    @FXML
    private BarChart<String, Number> performanceChart;

    // --- Services and Helpers ---
    private final AppStateService appStateService = AppStateService.getInstance();
    private final ObjectMapper objectMapper = new ObjectMapper().enable(SerializationFeature.INDENT_OUTPUT);
    private final DecimalFormat formatter = new DecimalFormat("#.####");

    public ResultsViewController() {
        // No-arg constructor
    }

    @Override
    public void initialize(URL url, ResourceBundle resourceBundle) {
        setupTableColumns();
        saveButton.setOnAction(e -> handleSaveJson());
        exportButton.setOnAction(e -> handleExport());
        configureChart();

        // Add listener for table selection
        resultsTable.getSelectionModel().selectedItemProperty().addListener((obs, oldSelection, newSelection) -> {
            if (newSelection != null) {
                populateConfusionMatrix(newSelection);
                // Optionally highlight the selected algo in charts
            }
        });

        loadAndDisplayResults();

        // --- ONNX Export Visibility ---
        List<ExecutionResult> results = appStateService.getExecutionResults();
        boolean hasOnnx = results != null && results.stream().anyMatch(r -> r.getOnnxModelPath() != null);
        if (exportOnnxButton != null) {
            exportOnnxButton.setVisible(hasOnnx);
            exportOnnxButton.setManaged(hasOnnx);
        }
    }

    private void setupTableColumns() {
        algoCol.setCellValueFactory(new PropertyValueFactory<>("algorithmName"));
        accuracyCol.setCellValueFactory(new PropertyValueFactory<>("accuracy"));
        precisionCol.setCellValueFactory(new PropertyValueFactory<>("precision"));
        recallCol.setCellValueFactory(new PropertyValueFactory<>("recall"));
        f1Col.setCellValueFactory(new PropertyValueFactory<>("f1Score"));
        originalFeaturesCol.setCellValueFactory(new PropertyValueFactory<>("originalFeatures"));
        selectedFeaturesCol.setCellValueFactory(new PropertyValueFactory<>("selectedFeatures"));
        timeCol.setCellValueFactory(new PropertyValueFactory<>("executionTimeMs"));

        // Assign cell factories for formatting
        accuracyCol.setCellFactory(tc -> createDoubleFormattingCell());
        precisionCol.setCellFactory(tc -> createDoubleFormattingCell());
        recallCol.setCellFactory(tc -> createDoubleFormattingCell());
        f1Col.setCellFactory(tc -> createDoubleFormattingCell());

        // Set numeric columns to right alignment
        accuracyCol.setStyle("-fx-alignment: CENTER-RIGHT;");
        precisionCol.setStyle("-fx-alignment: CENTER-RIGHT;");
        recallCol.setStyle("-fx-alignment: CENTER-RIGHT;");
        f1Col.setStyle("-fx-alignment: CENTER-RIGHT;");
        originalFeaturesCol.setStyle("-fx-alignment: CENTER-RIGHT;");
        selectedFeaturesCol.setStyle("-fx-alignment: CENTER-RIGHT;");
        timeCol.setStyle("-fx-alignment: CENTER-RIGHT;");
    }

    private TableCell<ExecutionResult, Double> createDoubleFormattingCell() {
        return new TableCell<ExecutionResult, Double>() {
            @Override
            protected void updateItem(Double item, boolean empty) {
                super.updateItem(item, empty);
                if (empty || item == null || item.isNaN()) {
                    setText(null);
                } else {
                    setText(formatter.format(item));
                }
            }
        };
    }

    @FXML
    private javafx.scene.control.CheckBox showAllCheckBox;

    private void configureChart() {
        resultsChart.setTitle("Convergence Plot (Fitness vs Iteration)");
        chartXAxis.setLabel("Iteration");
        chartYAxis.setLabel("Fitness");
        resultsChart.setAnimated(false);
        resultsChart.getData().clear();

        performanceChart.setTitle("Algorithm Performance Comparison");
        performanceChart.getYAxis().setLabel("Score (0.0 - 1.0)");
        performanceChart.setAnimated(false);

        lossChart.setTitle("Loss over Time (1 - Fitness)");
        lossChart.getXAxis().setLabel("Iteration");
        lossChart.getYAxis().setLabel("Loss");
        lossChart.setAnimated(false);
        lossChart.getData().clear();

        // Add listener for Show All checkbox
        if (showAllCheckBox != null) {
            showAllCheckBox.selectedProperty().addListener((obs, oldVal, newVal) -> {
                loadAndDisplayResults(); // Reload chart based on new setting
            });
            showAllCheckBox.setSelected(true); // Default to showing all
        }
    }

    private void loadAndDisplayResults() {
        List<ExecutionResult> results = appStateService.getExecutionResults();

        // Don't clear table here if just refreshing chart, but for simplicity we reload
        // all
        if (resultsTable.getItems().isEmpty() && results != null) {
            resultsTable.setItems(FXCollections.observableArrayList(results));
        }

        resultsChart.getData().clear();
        lossChart.getData().clear();
        performanceChart.getData().clear();
        confusionMatrixGrid.getChildren().clear();

        if (results != null && !results.isEmpty()) {
            ObservableList<ExecutionResult> observableResults = resultsTable.getItems();

            // Logic for Chart Filtering
            boolean showAll = showAllCheckBox != null && showAllCheckBox.isSelected();
            ExecutionResult selected = resultsTable.getSelectionModel().getSelectedItem();

            if (showAll || selected == null) {
                populateChart(observableResults);
            } else {
                populateChart(FXCollections.observableArrayList(selected));
            }

            populatePerformanceChart(observableResults);

            // Only update stats/selection if this is the initial load (table empty or
            // selection empty)
            if (selected == null) {
                ExecutionResult bestResult = results.stream()
                        .max(Comparator.comparingDouble(ExecutionResult::getAccuracy))
                        .orElse(null);

                if (bestResult != null) {
                    bestAccuracyLabel.setText(String.format("%.2f%%", bestResult.getAccuracy() * 100));
                    bestAlgoLabel.setText("Algorithm: " + bestResult.getAlgorithmName());
                    originalFeaturesLabel.setText("Original: " + bestResult.getOriginalFeatures());
                    resultsTable.getSelectionModel().select(bestResult);
                }
            } else {
                // Restore selection stats
                populateConfusionMatrix(selected);
            }

            double avgSelected = results.stream()
                    .filter(r -> r.getSelectedFeatures() > 0)
                    .mapToInt(ExecutionResult::getSelectedFeatures)
                    .average().orElse(0.0);

            long totalTimeMs = results.stream().mapToLong(ExecutionResult::getExecutionTimeMs).sum();

            avgFeaturesLabel.setText(String.format("%.1f", avgSelected));
            totalTimeLabel.setText(String.format("%.2f", totalTimeMs / 1000.0));
            resultsTable.setPlaceholder(null);
        } else {
            bestAccuracyLabel.setText("N/A");
            resultsTable.setPlaceholder(new Label("No results to display. Run a model first."));
        }
    }

    private void populateChart(ObservableList<ExecutionResult> results) {
        for (ExecutionResult result : results) {
            List<Double> history = result.getFitnessHistory();
            // List<Double> avgHistory = result.getAvgFitnessHistory();
            // List<Double> diversityHistory = result.getPopulationDiversityHistory();

            if (history != null && !history.isEmpty()) {
                XYChart.Series<Number, Number> fitnessSeries = new XYChart.Series<>();
                fitnessSeries.setName(result.getAlgorithmName()); // Simplified name

                XYChart.Series<Number, Number> lossSeries = new XYChart.Series<>();
                lossSeries.setName(result.getAlgorithmName());

                for (int i = 0; i < history.size(); i++) {
                    double fitness = history.get(i);
                    fitnessSeries.getData().add(new XYChart.Data<>(i + 1, fitness));
                    lossSeries.getData().add(new XYChart.Data<>(i + 1, 1.0 - fitness));
                }

                resultsChart.getData().add(fitnessSeries);
                lossChart.getData().add(lossSeries);
            }
        }
    }

    private void populatePerformanceChart(ObservableList<ExecutionResult> results) {
        XYChart.Series<String, Number> accuracySeries = new XYChart.Series<>();
        accuracySeries.setName("Accuracy");

        XYChart.Series<String, Number> precisionSeries = new XYChart.Series<>();
        precisionSeries.setName("Precision");

        XYChart.Series<String, Number> recallSeries = new XYChart.Series<>();
        recallSeries.setName("Recall");

        XYChart.Series<String, Number> f1Series = new XYChart.Series<>();
        f1Series.setName("F1-Score");

        for (ExecutionResult result : results) {
            String algoName = result.getAlgorithmName();
            accuracySeries.getData().add(new XYChart.Data<>(algoName, result.getAccuracy()));
            precisionSeries.getData().add(new XYChart.Data<>(algoName, result.getPrecision()));
            recallSeries.getData().add(new XYChart.Data<>(algoName, result.getRecall()));
            f1Series.getData().add(new XYChart.Data<>(algoName, result.getF1Score()));
        }

        performanceChart.getData()
                .addAll(java.util.Arrays.asList(accuracySeries, precisionSeries, recallSeries, f1Series));
    }

    private void populateConfusionMatrix(ExecutionResult result) {
        confusionMatrixGrid.getChildren().clear();
        int[][] matrix = result.getConfusionMatrix();

        if (matrix == null || matrix.length == 0) {
            confusionMatrixGrid.add(new Label("No Confusion Matrix available"), 0, 0);
            return;
        }

        int numClasses = matrix.length;

        // Add Headers
        for (int i = 0; i < numClasses; i++) {
            Label predHeader = new Label("Pred " + i);
            predHeader.setStyle("-fx-font-weight: bold;");
            confusionMatrixGrid.add(predHeader, i + 1, 0);

            Label trueHeader = new Label("True " + i);
            trueHeader.setStyle("-fx-font-weight: bold;");
            confusionMatrixGrid.add(trueHeader, 0, i + 1);
        }

        // Find max value for color scaling
        int maxVal = 0;
        for (int[] row : matrix) {
            for (int val : row) {
                if (val > maxVal)
                    maxVal = val;
            }
        }

        // Populate Cells
        for (int i = 0; i < numClasses; i++) {
            for (int j = 0; j < numClasses; j++) {
                int value = matrix[i][j];
                Label cell = new Label(String.valueOf(value));
                cell.setMinSize(40, 40);
                cell.setAlignment(Pos.CENTER); // Use Pos.CENTER

                // Color scaling (Green for diagonal, Red for errors)
                String color;
                double intensity = (double) value / (maxVal > 0 ? maxVal : 1);
                if (i == j) {
                    // Diagonal (Correct) - Shades of Green
                    int green = (int) (150 + (105 * intensity));
                    color = String.format("rgb(0, %d, 0)", green);
                    cell.setStyle("-fx-background-color: " + color + "; -fx-text-fill: white; -fx-border-color: #ddd;");
                } else {
                    // Off-diagonal (Error) - Shades of Red
                    if (value > 0) {
                        int red = (int) (150 + (105 * intensity));
                        color = String.format("rgb(%d, 0, 0)", red);
                        cell.setStyle(
                                "-fx-background-color: " + color + "; -fx-text-fill: white; -fx-border-color: #ddd;");
                    } else {
                        cell.setStyle("-fx-background-color: white; -fx-text-fill: black; -fx-border-color: #ddd;");
                    }
                }

                confusionMatrixGrid.add(cell, j + 1, i + 1);
            }
        }
    }

    @FXML
    private void handleViewReport() {
        String report = com.example.automl_prototype_1.service.LogService.getInstance().generateReport();

        Stage reportStage = new Stage();
        reportStage.setTitle("Execution Detail Report");

        javafx.scene.control.TextArea textArea = new javafx.scene.control.TextArea(report);
        textArea.setEditable(false);
        textArea.setWrapText(true);
        textArea.setStyle("-fx-font-family: 'Consolas', 'Monospaced'; -fx-font-size: 12px;");

        javafx.scene.layout.StackPane root = new javafx.scene.layout.StackPane(textArea);
        javafx.scene.Scene scene = new javafx.scene.Scene(root, 800, 600);

        reportStage.setScene(scene);
        reportStage.show();
    }

    @FXML
    private void handleSaveJson() {
        ObservableList<ExecutionResult> results = resultsTable.getItems();
        if (results == null || results.isEmpty()) {
            showAlert(Alert.AlertType.WARNING, "Save Error", "No results available to save.");
            return;
        }

        FileChooser fileChooser = new FileChooser();
        fileChooser.setTitle("Save Results as JSON");
        fileChooser.getExtensionFilters().add(new FileChooser.ExtensionFilter("JSON files (*.json)", "*.json"));
        fileChooser.setInitialFileName("automl_results_" + System.currentTimeMillis() + ".json");

        File file = fileChooser.showSaveDialog(getStage());
        if (file != null) {
            File finalFile = file.getName().toLowerCase().endsWith(".json") ? file
                    : new File(file.getParentFile(), file.getName() + ".json");

            // Async Save Task
            javafx.concurrent.Task<Void> saveTask = new javafx.concurrent.Task<>() {
                @Override
                protected Void call() throws Exception {
                    objectMapper.writeValue(finalFile, results);
                    return null;
                }
            };

            saveTask.setOnSucceeded(e -> showAlert(Alert.AlertType.INFORMATION, "Save Successful",
                    "Results saved to:\n" + finalFile.getAbsolutePath()));

            saveTask.setOnFailed(e -> {
                Throwable ex = saveTask.getException();
                LogService.error("ResultsView", "Error during JSON serialization: " + ex.getMessage());
                showAlert(Alert.AlertType.ERROR, "Save Failed", "Error during JSON serialization:\n" + ex.getMessage());
            });

            new Thread(saveTask).start();
        }
    }

    @FXML
    private void handleExportOnnx() {
        ExecutionResult selected = resultsTable.getSelectionModel().getSelectedItem();
        if (selected == null) {
            showAlert(Alert.AlertType.WARNING, "No Selection", "Please select a result to export.");
            return;
        }

        com.example.automl_prototype_1.service.IClassifier classifier = selected.getTrainedClassifier();
        if (classifier == null) {
            showAlert(Alert.AlertType.WARNING, "No Model",
                    "The selected result does not have a trained model in memory.\nDid you run 'Refit' or is this a loaded result?");
            return;
        }

        FileChooser fileChooser = new FileChooser();
        fileChooser.setTitle("Export Model");
        // Smart filter: suggest .onnx if python/DL, else .ser
        String name = classifier.getName();
        boolean isPython = name.contains("Python") || name.contains("Deep") || name.contains("XGBoost")
                || name.contains("MLP") || name.contains("BERT") || name.contains("CNN");

        if (isPython) {
            fileChooser.getExtensionFilters().add(new FileChooser.ExtensionFilter("ONNX Model (*.onnx)", "*.onnx"));
            fileChooser.setInitialFileName("model_" + System.currentTimeMillis() + ".onnx");
        } else {
            fileChooser.getExtensionFilters().add(new FileChooser.ExtensionFilter("Java Model (*.ser)", "*.ser"));
            fileChooser.setInitialFileName("model_" + System.currentTimeMillis() + ".ser");
        }

        File destFile = fileChooser.showSaveDialog(getStage());
        if (destFile != null) {
            com.example.automl_prototype_1.service.agentic.inference.OnnxExportService exportService = new com.example.automl_prototype_1.service.agentic.inference.OnnxExportService();

            boolean success = exportService.exportModel(classifier, destFile.getAbsolutePath());

            if (success) {
                showAlert(Alert.AlertType.INFORMATION, "Export Successful",
                        "Model saved to:\n" + destFile.getAbsolutePath());
            } else {
                showAlert(Alert.AlertType.ERROR, "Export Failed", "Check logs for details.");
            }
        }
    }

    @FXML
    private void handleExport() {
        ObservableList<ExecutionResult> results = resultsTable.getItems();
        if (results == null || results.isEmpty()) {
            showAlert(Alert.AlertType.WARNING, "Export Error", "No results available to export.");
            return;
        }

        FileChooser fileChooser = new FileChooser();
        fileChooser.setTitle("Export Results as CSV");
        fileChooser.getExtensionFilters().add(new FileChooser.ExtensionFilter("CSV files (*.csv)", "*.csv"));
        fileChooser.setInitialFileName("automl_export_" + System.currentTimeMillis() + ".csv");

        File file = fileChooser.showSaveDialog(getStage());
        if (file != null) {
            if (!file.getName().toLowerCase().endsWith(".csv")) {
                file = new File(file.getParentFile(), file.getName() + ".csv");
            }
            try (FileWriter writer = new FileWriter(file)) {
                // Write Header
                writer.append(
                        "Algorithm,Accuracy,Precision,Recall,F1-Score,Original Features,Selected Features,Execution Time (ms)\n");

                // Write Data
                for (ExecutionResult result : results) {
                    writer.append(escapeCsv(result.getAlgorithmName())).append(",");
                    writer.append(formatCsvDouble(result.getAccuracy())).append(",");
                    writer.append(formatCsvDouble(result.getPrecision())).append(",");
                    writer.append(formatCsvDouble(result.getRecall())).append(",");
                    writer.append(formatCsvDouble(result.getF1Score())).append(",");
                    writer.append(String.valueOf(result.getOriginalFeatures())).append(",");
                    writer.append(String.valueOf(result.getSelectedFeatures())).append(",");
                    writer.append(String.valueOf(result.getExecutionTimeMs()));
                    writer.append("\n");
                }
                writer.flush();
                showAlert(Alert.AlertType.INFORMATION, "Export Successful",
                        "Results exported to:\n" + file.getAbsolutePath());

            } catch (IOException e) {
                LogService.error("ResultsView", "Could not export results to CSV: " + e.getMessage());
                showAlert(Alert.AlertType.ERROR, "Export Failed",
                        "Could not export results to CSV file:\n" + e.getMessage());
            }
        }
    }

    private Runnable onBackAction;

    public void setOnBackAction(Runnable onBackAction) {
        this.onBackAction = onBackAction;
    }

    @FXML
    private void handleBack() {
        if (onBackAction != null) {
            onBackAction.run();
        } else {
            com.example.automl_prototype_1.service.LogService.warn("ResultsView",
                    "Back button clicked, but no action defined.");
        }
    }

    private String escapeCsv(String value) {
        if (value == null)
            return "";
        String escaped = value.replace("\"", "\"\"");
        if (escaped.contains(",") || escaped.contains("\"") || escaped.contains("\n") || escaped.contains("\r")) {
            return "\"" + escaped + "\"";
        }
        return escaped;
    }

    private String formatCsvDouble(double value) {
        return Double.isNaN(value) ? "" : String.format(Locale.US, "%.4f", value);
    }

    private Stage getStage() {
        return (Stage) resultsTable.getScene().getWindow();
    }

    private void showAlert(Alert.AlertType type, String title, String content) {
        Alert alert = new Alert(type);
        alert.setTitle(title);
        alert.setHeaderText(null);
        alert.setContentText(content);
        alert.showAndWait();
    }
}