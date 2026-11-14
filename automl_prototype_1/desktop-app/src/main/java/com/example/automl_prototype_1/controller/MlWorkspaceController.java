package com.example.automl_prototype_1.controller;

import com.example.automl_prototype_1.dataprovider.CsvDataProvider;
import com.example.automl_prototype_1.model.Dataset;
import com.example.automl_prototype_1.model.Domain;
import com.example.automl_prototype_1.service.AppStateService;
import com.example.automl_prototype_1.service.LogService;
import com.example.automl_prototype_1.service.agentic.AgenticAdvisor;
import com.example.automl_prototype_1.service.agentic.EmpiricalBenchmarker;
import com.example.automl_prototype_1.gpu.HardwareAcceleratorService;
import com.example.automl_prototype_1.dataprovider.SyntheticDataGenerator;
import javafx.application.Platform;
import javafx.collections.FXCollections;
import javafx.fxml.FXML;
import javafx.scene.control.*;
import javafx.stage.FileChooser;
import java.io.File;
import java.io.FileReader;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

public class MlWorkspaceController implements Cleanable {

    @FXML
    private Button browseButton;
    @FXML
    private Label selectedFileLabel;
    @FXML
    private ChoiceBox<String> targetColumnChoiceBox;
    @FXML
    private Button analyzeButton;
    @FXML
    private TextArea analysisTextArea;

    @FXML
    private CheckBox checkGA, checkIGPSO, checkWWO, checkBPSO, checkASO, checkPLS, checkSA, checkGWO, checkDE;
    @FXML
    private CheckBox selectAllAlgorithms;
    @FXML
    private ChoiceBox<String> baseModelChoiceBox;

    @FXML
    private TextField kFoldsTextField;
    @FXML
    private CheckBox enableTuningCheckBox;
    @FXML
    private CheckBox chkUseGpu, chkUseWsl, chkBalancedOpt, enableLoggingCheckBox;
    @FXML
    private ComboBox<String> cmbGpuSelector;

    @FXML
    private Button recommendButton;
    @FXML
    private TextArea adviceTextArea;
    @FXML
    private Button quickScanBtn;
    @FXML
    private TextArea quickScanResultArea;

    private final AppStateService appState = AppStateService.getInstance();
    private final CsvDataProvider csvDataProvider = new CsvDataProvider();
    private Dataset currentAnalysisDataset;

    @FXML
    private void initialize() {
        appState.setCurrentDomain(Domain.ML);

        // 1. Upload Section
        browseButton.setOnAction(event -> handleBrowseBtn());
        targetColumnChoiceBox.getSelectionModel().selectedItemProperty().addListener((obs, oldVal, newVal) -> {
            if (newVal != null) {
                appState.setTargetColumnName(newVal);
                analyzeButton.setDisable(false);
            }
        });
        analyzeButton.setOnAction(event -> handleAnalyzeBtn());

        // Restore file state if it exists
        File existingFile = appState.getSelectedDatasetFile();
        if (existingFile != null) {
            selectedFileLabel.setText(existingFile.getName());
            loadHeadersToChoiceBox(existingFile, appState.getTargetColumnName());
        }

        // 2. Algorithm Section
        setupAlgorithms();

        // 3. Settings
        setupExecutionSettings();

        // 4. Advisor bindings
        recommendButton.setOnAction(e -> handleAdvice());
        quickScanBtn.setOnAction(e -> handleQuickScan());
    }

    private void handleBrowseBtn() {
        FileChooser fileChooser = new FileChooser();
        fileChooser.setTitle("Open Dataset File");
        File defaultDir = new File(System.getProperty("user.dir"), "datasets");
        if (defaultDir.exists())
            fileChooser.setInitialDirectory(defaultDir);

        fileChooser.getExtensionFilters().add(new FileChooser.ExtensionFilter("CSV Files", "*.csv"));
        File file = fileChooser.showOpenDialog(browseButton.getScene().getWindow());

        if (file != null) {
            selectedFileLabel.setText(file.getName());
            appState.setSelectedDatasetFile(file);
            appState.setCurrentDataset(null); // Clear any synthetic data
            loadHeadersToChoiceBox(file, null);
        }
    }

    @FXML
    private void handleGenerateSynthetic() {
        try {
            Dataset synthData = SyntheticDataGenerator.generate(500, 20, 10, false);

            appState.setSelectedDatasetFile(null);
            appState.setCurrentDataset(synthData);
            selectedFileLabel.setText("Synthetic Data (500 rows, 20 features)");

            List<String> headers = synthData.getHeaders();
            targetColumnChoiceBox.setItems(FXCollections.observableArrayList(headers));
            targetColumnChoiceBox.setDisable(false);
            targetColumnChoiceBox.getSelectionModel().select("Target");
        } catch (Exception e) {
            LogService.error("MlWorkspace", "Synthetic generation failed: " + e.getMessage());
        }
    }

    private void loadHeadersToChoiceBox(File file, String preselect) {
        try (FileReader reader = new FileReader(file)) {
            currentAnalysisDataset = csvDataProvider.loadDataset(reader);
            List<String> headers = currentAnalysisDataset.getHeaders();
            appState.setDatasetHeaders(headers);

            targetColumnChoiceBox.setItems(FXCollections.observableArrayList(headers));
            targetColumnChoiceBox.setDisable(false);

            if (preselect != null && headers.contains(preselect)) {
                targetColumnChoiceBox.getSelectionModel().select(preselect);
            } else if (!headers.isEmpty()) {
                targetColumnChoiceBox.getSelectionModel().select(headers.size() - 1);
            }
        } catch (Exception e) {
            analysisTextArea.setText("Failed to load headers: " + e.getMessage());
        }
    }

    private void handleAnalyzeBtn() {
        if (currentAnalysisDataset == null)
            return;
        String target = targetColumnChoiceBox.getValue();
        if (target == null)
            return;

        analysisTextArea.setText("Analyzing...");
        new Thread(() -> {
            try (AgenticAdvisor advisor = new AgenticAdvisor()) {
                String fileName = appState.getSelectedDatasetFile() != null
                        ? appState.getSelectedDatasetFile().getName()
                        : "Synthetic Dataset";
                String report = advisor.generateStrategicReport(currentAnalysisDataset, fileName,
                        appState.getCurrentDomain());
                Platform.runLater(() -> analysisTextArea.setText(report));
            } catch (Exception e) {
                Platform.runLater(() -> analysisTextArea.setText("Error analyzing dataset: " + e.getMessage()));
            }
        }).start();
    }

    @FXML
    private void handleAdvancedSettings() {
        com.example.automl_prototype_1.model.PreprocessingConfig current = appState.getPreprocessingConfig();
        if (current == null) {
            current = new com.example.automl_prototype_1.model.PreprocessingConfig();
        }

        Dialog<com.example.automl_prototype_1.model.PreprocessingConfig> dialog = new Dialog<>();
        dialog.setTitle("Advanced Preprocessing");
        dialog.setHeaderText("Configure the preprocessing applied to this ML workspace");

        ButtonType save = new ButtonType("Save", ButtonBar.ButtonData.OK_DONE);
        dialog.getDialogPane().getButtonTypes().addAll(save, ButtonType.CANCEL);

        CheckBox removeIds = new CheckBox("Auto-remove ID columns");
        removeIds.setSelected(current.isAutoRemoveIdColumns());
        CheckBox impute = new CheckBox("Impute missing values");
        impute.setSelected(current.isImputeMissingValues());
        CheckBox outliers = new CheckBox("Remove outliers (IQR)");
        outliers.setSelected(current.isRemoveOutliers());
        ChoiceBox<String> scaling = new ChoiceBox<>(
                FXCollections.observableArrayList("Standardization", "Normalization", "None"));
        scaling.setValue(current.getScalingStrategy());

        javafx.scene.layout.GridPane form = new javafx.scene.layout.GridPane();
        form.setHgap(12);
        form.setVgap(12);
        form.setPadding(new javafx.geometry.Insets(16));
        form.add(new Label("Scaling"), 0, 0);
        form.add(scaling, 1, 0);
        form.add(removeIds, 0, 1, 2, 1);
        form.add(impute, 0, 2, 2, 1);
        form.add(outliers, 0, 3, 2, 1);
        dialog.getDialogPane().setContent(form);

        dialog.setResultConverter(button -> button == save
                ? new com.example.automl_prototype_1.model.PreprocessingConfig(
                        removeIds.isSelected(),
                        impute.isSelected(),
                        outliers.isSelected(),
                        scaling.getValue())
                : null);
        dialog.showAndWait().ifPresent(config -> {
            appState.setPreprocessingConfig(config);
            appState.setScalingStrategy(config.getScalingStrategy());
            LogService.info("MlWorkspace", "Preprocessing settings updated");
        });
    }

    private void setupAlgorithms() {
        List<String> savedAlgos = appState.getSelectedAlgorithmNames();
        if (savedAlgos != null) {
            checkGA.setSelected(savedAlgos.contains("GA"));
            checkIGPSO.setSelected(savedAlgos.contains("IGPSO"));
            checkWWO.setSelected(savedAlgos.contains("WWO"));
            // ... apply others
        }

        javafx.beans.value.ChangeListener<Boolean> algoUpdater = (obs, oldV, newV) -> updateSelectedAlgorithms();
        checkGA.selectedProperty().addListener(algoUpdater);
        checkIGPSO.selectedProperty().addListener(algoUpdater);
        checkWWO.selectedProperty().addListener(algoUpdater);
        checkBPSO.selectedProperty().addListener(algoUpdater);
        checkASO.selectedProperty().addListener(algoUpdater);
        checkPLS.selectedProperty().addListener(algoUpdater);
        checkSA.selectedProperty().addListener(algoUpdater);
        checkGWO.selectedProperty().addListener(algoUpdater);
        checkDE.selectedProperty().addListener(algoUpdater);

        selectAllAlgorithms.setOnAction(e -> {
            boolean s = selectAllAlgorithms.isSelected();
            checkGA.setSelected(s);
            checkIGPSO.setSelected(s);
            checkWWO.setSelected(s);
            checkBPSO.setSelected(s);
            checkASO.setSelected(s);
            checkPLS.setSelected(s);
            checkSA.setSelected(s);
            checkGWO.setSelected(s);
            checkDE.setSelected(s);
        });

        // Setup base models specific to ML currently in registry
        List<String> mlModels = List.of("Random Forest", "SVM", "KNN", "XGBoost", "Logistic Regression"); // hardcoded
                                                                                                          // for ML
                                                                                                          // specifically
        baseModelChoiceBox.setItems(FXCollections.observableArrayList(mlModels));
        if (appState.getSelectedBaseModelName() != null && mlModels.contains(appState.getSelectedBaseModelName())) {
            baseModelChoiceBox.getSelectionModel().select(appState.getSelectedBaseModelName());
        } else {
            baseModelChoiceBox.getSelectionModel().select(0);
        }

        baseModelChoiceBox.getSelectionModel().selectedItemProperty().addListener((obs, oldV, newV) -> {
            appState.setSelectedBaseModelName(newV);
            appState.setSelectedClassifier(
                    com.example.automl_prototype_1.registry.ClassifierRegistry.getInstance().getClassifier(newV));
        });
    }

    private void updateSelectedAlgorithms() {
        List<String> list = new java.util.ArrayList<>();
        if (checkGA.isSelected())
            list.add("GA");
        if (checkIGPSO.isSelected())
            list.add("IGPSO");
        if (checkWWO.isSelected())
            list.add("WWO");
        if (checkBPSO.isSelected())
            list.add("BPSO");
        if (checkASO.isSelected())
            list.add("ASO");
        if (checkPLS.isSelected())
            list.add("PLS");
        if (checkSA.isSelected())
            list.add("SA");
        if (checkGWO.isSelected())
            list.add("GWO");
        if (checkDE.isSelected())
            list.add("DE");
        appState.setSelectedAlgorithmNames(list);
    }

    private void setupExecutionSettings() {
        kFoldsTextField.setText(String.valueOf(appState.getKFolds()));
        kFoldsTextField.textProperty().addListener((obs, oldV, newV) -> {
            try {
                appState.setKFolds(Integer.parseInt(newV));
            } catch (NumberFormatException ignored) {
            }
        });

        enableTuningCheckBox.setSelected(appState.isTuningEnabled());
        enableTuningCheckBox.selectedProperty().addListener((obs, o, n) -> appState.setTuningEnabled(n));

        chkBalancedOpt.setSelected(appState.isBalancedOptimizationEnabled());
        chkBalancedOpt.selectedProperty().addListener((obs, o, n) -> appState.setEnableBalancedOptimization(n));

        enableLoggingCheckBox.setSelected(appState.isLoggingEnabled());
        enableLoggingCheckBox.selectedProperty().addListener((obs, o, n) -> appState.setLoggingEnabled(n));

        chkUseWsl.setSelected(appState.isUseWsl());
        chkUseWsl.selectedProperty().addListener((obs, o, n) -> appState.setUseWsl(n));

        // GPU Check
        new Thread(() -> {
            boolean isGpu = HardwareAcceleratorService.isGpuAvailable();
            Platform.runLater(() -> {
                chkUseGpu.setSelected(appState.isUseGpu());
                if (isGpu) {
                    chkUseGpu.setText("Use GPU");
                    cmbGpuSelector.getItems().setAll(HardwareAcceleratorService.getAvailableGpuNames());
                    if (!cmbGpuSelector.getItems().isEmpty())
                        cmbGpuSelector.getSelectionModel().select(0);
                    cmbGpuSelector.visibleProperty().bind(chkUseGpu.selectedProperty());
                    cmbGpuSelector.managedProperty().bind(chkUseGpu.selectedProperty());
                } else {
                    chkUseGpu.setText("Use GPU (Not Detected)");
                    chkUseGpu.setDisable(true);
                }
            });
        }).start();

        chkUseGpu.selectedProperty().addListener((obs, o, n) -> appState.setUseGpu(n));
    }

    private void handleAdvice() {
        adviceTextArea.setText("Querying Agentic Advisor...");
        new Thread(() -> {
            try (AgenticAdvisor advisor = new AgenticAdvisor()) {
                if (currentAnalysisDataset != null) {
                    String fileName = appState.getSelectedDatasetFile() != null
                            ? appState.getSelectedDatasetFile().getName()
                            : "Synthetic Dataset";
                    String report = advisor.generateStrategicReport(
                            currentAnalysisDataset, fileName, appState.getCurrentDomain());
                    Platform.runLater(() -> adviceTextArea.setText(report));
                } else {
                    String quickAdvice = advisor.quickAdvice(appState.getCurrentDomain());
                    Platform.runLater(() -> adviceTextArea.setText(
                            "No dataset loaded — general ML advice:\n\n" + quickAdvice));
                }
            } catch (Exception e) {
                Platform.runLater(() -> adviceTextArea.setText("Advisor error: " + e.getMessage()));
            }
        }).start();
    }

    private void handleQuickScan() {
        if (appState.getSelectedDatasetFile() == null) {
            quickScanResultArea.setText("Upload dataset first.");
            return;
        }
        quickScanResultArea.setText("Scanning data...");
        new Thread(() -> {
            try {
                Dataset ds = csvDataProvider.loadDataset(new FileReader(appState.getSelectedDatasetFile()));
                EmpiricalBenchmarker benchmarker = new EmpiricalBenchmarker();
                Map<String, Double> res = benchmarker.benchmark(ds);
                String resultStr = res.entrySet().stream()
                        .sorted((a, b) -> b.getValue().compareTo(a.getValue()))
                        .map(e -> e.getKey() + ": " + String.format("%.2f%%", e.getValue() * 100))
                        .collect(Collectors.joining("\n"));
                Platform.runLater(() -> quickScanResultArea.setText("Top Base Models:\n" + resultStr));
            } catch (Exception e) {
                Platform.runLater(() -> quickScanResultArea.setText("Error: " + e.getMessage()));
            }
        }).start();
    }

    @Override
    public void cleanup() {
        // Implement Cleanable
    }
}
