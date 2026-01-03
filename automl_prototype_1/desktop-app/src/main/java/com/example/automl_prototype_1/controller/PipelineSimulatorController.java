package com.example.automl_prototype_1.controller;

import com.example.automl_prototype_1.model.*;
import com.example.automl_prototype_1.service.AppStateService;
import com.example.automl_prototype_1.service.SimulationService;
import javafx.beans.property.SimpleStringProperty;
import javafx.collections.FXCollections;
import javafx.fxml.FXML;
import javafx.scene.control.*;

import java.io.File;
import java.util.List;
import java.util.Map;

public class PipelineSimulatorController {

    @FXML
    private Label datasetNameLabel;
    @FXML
    private ComboBox<String> scalingCombo;
    @FXML
    private TableView<Map<String, String>> rawTable;
    @FXML
    private TableView<Map<String, String>> transformedTable;
    @FXML
    private TextArea transformationExplanation;

    @FXML
    private ComboBox<String> algoCombo;
    @FXML
    private Label algoFormula;
    @FXML
    private Label algoDescription;
    @FXML
    private Label algoOutput;

    @FXML
    private ComboBox<String> classifierCombo;
    @FXML
    private Label classifierFunction;
    @FXML
    private Label classifierLoss;
    @FXML
    private Label classifierOutput;

    @FXML
    private Label tuningStrategy;
    @FXML
    private Label tuningFormula;
    @FXML
    private TextArea tuningOutput;
    @FXML
    private TextArea estimationOutput;

    private final SimulationService simulationService = SimulationService.getInstance();
    private final AppStateService appStateService = AppStateService.getInstance();

    @FXML
    public void initialize() {
        // Init Combos
        scalingCombo.setItems(FXCollections.observableArrayList("None", "Standardization", "Normalization"));
        scalingCombo.getSelectionModel().select("Standardization");

        algoCombo.setItems(FXCollections.observableArrayList(
                "Binary PSO",
                "Genetic Algorithm",
                "Simulated Annealing",
                "Atom Search Optimization",
                "Water Wave Optimization",
                "Parallel Local Search"));
        algoCombo.getSelectionModel().select(0);

        // Updated with NEW Python/GPU Classifiers
        classifierCombo.setItems(FXCollections.observableArrayList(
                "K-Nearest Neighbors",
                "Support Vector Machine",
                "Logistic Regression",
                "Random Forest",
                "XGBoost",
                "GPU-KNN",
                "GPU-Logistic Regression",
                "Random Forest (GPU Accelerated)",
                "XGBoost (Python/GPU Bridge)",
                "SVM (Python/GPU/RBF Bridge)"));
        classifierCombo.getSelectionModel().select(0);

        // Set Dataset Label
        File datasetFile = appStateService.getSelectedDatasetFile();
        if (datasetFile != null) {
            datasetNameLabel.setText(datasetFile.getName());
        } else {
            datasetNameLabel.setText("No Dataset Loaded");
        }

        // Populate initial logic views to not show "..."
        handleShowAlgoLogic();
        handleShowClassifierLogic();
        handleShowTuningLogic();
    }

    @FXML
    private void handleSimulateScaling() {
        File datasetFile = appStateService.getSelectedDatasetFile();
        if (datasetFile == null) {
            transformationExplanation.setText("Please load a dataset in the Dashboard first.");
            return;
        }

        try {
            com.example.automl_prototype_1.dataprovider.CsvDataProvider provider = new com.example.automl_prototype_1.dataprovider.CsvDataProvider();
            java.io.FileReader reader = new java.io.FileReader(datasetFile);
            Dataset dataset = provider.loadDataset(reader);
            reader.close();

            String strategy = scalingCombo.getValue();
            TransformationPreview preview = simulationService.simulatePreprocessing(dataset, strategy);

            // Populate Explanation
            transformationExplanation.setText(preview.getExplanation());

            // Populate Tables
            // 1. Raw (Just first 10 rows)
            populateTable(rawTable, dataset.getHeaders(),
                    dataset.getRecords().subList(0, Math.min(10, dataset.getRecords().size())));

            // 2. Transformed
            populateTable(transformedTable, dataset.getHeaders(), preview.getPreviewRows());

        } catch (Exception e) {
            transformationExplanation.setText("Error loading/processing dataset: " + e.getMessage());
            e.printStackTrace();
        }
    }

    private void populateTable(TableView<Map<String, String>> table, List<String> headers,
            List<Map<String, String>> rows) {
        table.getColumns().clear();
        table.getItems().clear();

        for (String header : headers) {
            TableColumn<Map<String, String>, String> col = new TableColumn<>(header);
            col.setCellValueFactory(data -> new SimpleStringProperty(data.getValue().get(header)));
            table.getColumns().add(col);
        }

        table.setItems(FXCollections.observableArrayList(rows));
    }

    @FXML
    private void handleShowAlgoLogic() {
        String algo = algoCombo.getValue();
        if (algo == null)
            return;
        AlgoSimulationData data = simulationService.getAlgorithmLogic(algo);

        algoFormula.setText(data.getCoreFormula());
        algoDescription.setText(data.getProcessDescription());
        algoOutput.setText(data.getSampleOutput());

        // --- FEATURE SELECTION VISUALIZATION ---
        // Randomly hide columns in the preview table to simulate "Feature Selection"
        if (!rawTable.getColumns().isEmpty()) {
            java.util.Random rnd = new java.util.Random();
            int visibleCount = 0;
            for (TableColumn<?, ?> col : rawTable.getColumns()) {
                if (col.getText().equals("Class")) {
                    col.setVisible(true); // Always show target
                    continue;
                }

                // Hide 50% of columns to simulate selection
                boolean keep = rnd.nextBoolean();
                col.setVisible(keep);
                if (keep)
                    visibleCount++;

                // Also sync Transformed table
                for (TableColumn<?, ?> tCol : transformedTable.getColumns()) {
                    if (tCol.getText().equals(col.getText())) {
                        tCol.setVisible(keep);
                    }
                }
            }
            algoDescription.setText(data.getProcessDescription() + "\n\n(Visualizing Feature Selection: " + visibleCount
                    + " features retained in table above.)");
        }
    }

    @FXML
    private void handleShowClassifierLogic() {
        String clf = classifierCombo.getValue();
        if (clf == null)
            return;

        List<Map<String, String>> currentData = transformedTable.getItems();
        ClassifierSimulationData data = simulationService.getClassifierLogic(clf, currentData);

        classifierFunction.setText(data.getDecisionFunction());
        classifierLoss.setText(data.getLossFormula());
        classifierOutput.setText(data.getSampleOutput());
    }

    @FXML
    private void handleShowTuningLogic() {
        TuningSimulationData data = simulationService.getTuningLogic();
        tuningStrategy.setText(data.getStrategyName());
        tuningFormula.setText(data.getParameterDistribution());
        tuningOutput.setText(data.getSampleOutput());
    }

    @FXML
    private void handleCalculateEstimation() {
        String algo = algoCombo.getValue();
        String clf = classifierCombo.getValue();

        // Mock data sizing if dataset not loaded, or use real
        int rows = 1000;
        int cols = 20;
        File ds = appStateService.getSelectedDatasetFile();
        if (ds != null) {
            // Heuristic: file size / 100 bytes/row approx
            long size = ds.length();
            rows = (int) (size / 100);
            // We can't know columns easily without parsing but we assume 20
        }

        boolean hasGpu = com.example.automl_prototype_1.service.HardwareAcceleratorService.isGpuAvailable();

        String est = simulationService.getTimeEstimation(algo, clf, rows, cols, hasGpu);
        estimationOutput.setText(est);
    }
}
