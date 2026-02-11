package com.example.automl_prototype_1.controller;

import com.example.automl_prototype_1.viewmodel.AlgoSelectViewModel;
import javafx.collections.FXCollections;
import javafx.fxml.FXML;

import javafx.scene.control.*;
import com.example.automl_prototype_1.gpu.HardwareAcceleratorService;
import com.example.automl_prototype_1.service.AppStateService;

public class AlgoSelectViewController {
    @FXML
    private TextArea adviceTextArea;
    @FXML
    private CheckBox checkGA, checkIGPSO, checkWWO, checkBPSO, checkASO, checkPLS, checkSA, checkGWO, checkDE;
    @FXML
    private ChoiceBox<String> baseModelChoiceBox;

    @FXML
    private TextField kFoldsTextField;

    @FXML
    private CheckBox enableTuningCheckBox;
    @FXML
    private CheckBox selectAllAlgorithms;
    @FXML
    private CheckBox chkUseGpu, chkUseWsl, chkBalancedOpt, enableLoggingCheckBox;
    @FXML
    private ComboBox<String> cmbGpuSelector;
    @FXML
    private Button recommendButton;
    @FXML
    private Button quickScanBtn;
    @FXML
    private TextArea quickScanResultArea;

    // Advanced Mode Controls
    @FXML
    private javafx.scene.layout.VBox experimentalCard;
    @FXML
    private TextField manualTrialsField;
    @FXML
    private TextField manualIterationsField;
    @FXML
    private CheckBox forceGpuCheck;

    private final AlgoSelectViewModel viewModel;

    public AlgoSelectViewController() {
        this.viewModel = new AlgoSelectViewModel();
    }

    @FXML
    private void initialize() {
        // Bind UI elements to ViewModel properties

        // Advice Text
        adviceTextArea.textProperty().bind(viewModel.adviceTextProperty());

        // Tuning Checkbox
        enableTuningCheckBox.selectedProperty().bindBidirectional(viewModel.tuningEnabledProperty());

        // K-Folds TextField
        kFoldsTextField.textProperty().bindBidirectional(viewModel.kFoldsProperty());

        // Base Model ChoiceBox
        baseModelChoiceBox.setItems(FXCollections.observableArrayList(viewModel.getAvailableClassifiers()));
        baseModelChoiceBox.valueProperty().bindBidirectional(viewModel.selectedBaseModelProperty());

        // Algorithm Checkboxes
        checkGA.selectedProperty().bindBidirectional(viewModel.gaSelectedProperty());
        checkIGPSO.selectedProperty().bindBidirectional(viewModel.igpsoSelectedProperty());
        checkWWO.selectedProperty().bindBidirectional(viewModel.wwoSelectedProperty());
        checkBPSO.selectedProperty().bindBidirectional(viewModel.bpsoSelectedProperty());
        checkASO.selectedProperty().bindBidirectional(viewModel.asoSelectedProperty());
        checkPLS.selectedProperty().bindBidirectional(viewModel.plsSelectedProperty());
        checkSA.selectedProperty().bindBidirectional(viewModel.saSelectedProperty());
        checkGWO.selectedProperty().bindBidirectional(viewModel.gwoSelectedProperty());
        checkDE.selectedProperty().bindBidirectional(viewModel.deSelectedProperty());

        // Button Action
        recommendButton.setText("Get Advice");
        recommendButton.setOnAction(event -> viewModel.generateAdvice());

        // Select All Logic
        selectAllAlgorithms.setOnAction(e -> {
            boolean selected = selectAllAlgorithms.isSelected();
            checkGA.setSelected(selected);
            checkIGPSO.setSelected(selected);
            checkWWO.setSelected(selected);
            checkBPSO.setSelected(selected);
            checkASO.setSelected(selected);
            checkPLS.setSelected(selected);
            checkSA.setSelected(selected);
            checkGWO.setSelected(selected);
            checkDE.setSelected(selected);
        });

        // Quick Scan Button
        if (quickScanBtn != null) {
            quickScanBtn.setOnAction(event -> handleQuickScan());
        }

        initExecutionSettings();
    }

    private void handleQuickScan() {
        AppStateService appState = AppStateService.getInstance();
        java.io.File datasetFile = appState.getSelectedDatasetFile();
        if (datasetFile == null) {
            if (quickScanResultArea != null) {
                quickScanResultArea.setText("❌ Please upload a dataset first!");
            }
            return;
        }

        quickScanBtn.setDisable(true);
        quickScanBtn.setText("Scanning...");
        if (quickScanResultArea != null) {
            quickScanResultArea.setText("Running empirical benchmark on data subsample...");
        }

        new Thread(() -> {
            try {
                com.example.automl_prototype_1.dataprovider.CsvDataProvider provider = new com.example.automl_prototype_1.dataprovider.CsvDataProvider();
                com.example.automl_prototype_1.model.Dataset dataset = provider
                        .loadDataset(new java.io.FileReader(datasetFile));

                com.example.automl_prototype_1.service.agentic.EmpiricalBenchmarker benchmarker = new com.example.automl_prototype_1.service.agentic.EmpiricalBenchmarker();
                java.util.Map<String, Double> results = benchmarker.benchmark(dataset);

                // Sort by accuracy descending
                StringBuilder sb = new StringBuilder();
                sb.append("⚡ Quick Scan Results (Ranked by Accuracy)\n");
                sb.append("══════════════════════════════════════\n\n");

                results.entrySet().stream()
                        .sorted((a, b) -> Double.compare(b.getValue(), a.getValue()))
                        .forEach(entry -> {
                            String bar = "█".repeat(Math.max(1, (int) (entry.getValue() * 20)));
                            sb.append(String.format("%-15s  %.2f%%  %s\n",
                                    entry.getKey(), entry.getValue() * 100, bar));
                        });

                sb.append("\n💡 Tip: Select the top-ranked classifier as your base model.");

                javafx.application.Platform.runLater(() -> {
                    if (quickScanResultArea != null) {
                        quickScanResultArea.setText(sb.toString());
                    }
                    quickScanBtn.setDisable(false);
                    quickScanBtn.setText("⚡ Quick Scan");
                });

            } catch (Exception ex) {
                javafx.application.Platform.runLater(() -> {
                    if (quickScanResultArea != null) {
                        quickScanResultArea.setText("❌ Benchmark failed: " + ex.getMessage());
                    }
                    quickScanBtn.setDisable(false);
                    quickScanBtn.setText("⚡ Quick Scan");
                });
            }
        }).start();
    }

    private void initExecutionSettings() {
        // GPU Settings
        if (chkUseGpu != null) {
            // Initial state: Disable until check completes
            chkUseGpu.setDisable(true);
            chkUseGpu.setText("Checking GPU...");

            javafx.concurrent.Task<Boolean> gpuCheckTask = new javafx.concurrent.Task<>() {
                @Override
                protected Boolean call() throws Exception {
                    return HardwareAcceleratorService.isGpuAvailable();
                }
            };

            gpuCheckTask.setOnSucceeded(e -> {
                boolean isGpu = gpuCheckTask.getValue();
                chkUseGpu.setSelected(AppStateService.getInstance().isUseGpu());
                chkUseGpu.setDisable(false); // Enable interaction

                if (isGpu) {
                    chkUseGpu.setText("Use GPU");
                    if (cmbGpuSelector != null) {
                        java.util.List<String> gpuNames = HardwareAcceleratorService.getAvailableGpuNames();
                        cmbGpuSelector.getItems().setAll(gpuNames);

                        String savedGpu = AppStateService.getInstance().getSelectedGpuName();
                        if (savedGpu != null && gpuNames.contains(savedGpu)) {
                            cmbGpuSelector.getSelectionModel().select(savedGpu);
                        } else if (!gpuNames.isEmpty()) {
                            cmbGpuSelector.getSelectionModel().select(0);
                            AppStateService.getInstance().setSelectedGpuName(gpuNames.get(0));
                        }
                        cmbGpuSelector.visibleProperty().bind(chkUseGpu.selectedProperty());
                        cmbGpuSelector.managedProperty().bind(chkUseGpu.selectedProperty());

                        cmbGpuSelector.getSelectionModel().selectedItemProperty().addListener((obs, old, newVal) -> {
                            if (newVal != null)
                                AppStateService.getInstance().setSelectedGpuName(newVal);
                        });
                    }
                } else {
                    chkUseGpu.setText("Use GPU (Not Detected)");
                    chkUseGpu.setDisable(true);
                }
            });

            gpuCheckTask.setOnFailed(e -> {
                chkUseGpu.setText("Use GPU (Error)");
                chkUseGpu.setDisable(true);
                com.example.automl_prototype_1.service.LogService.error("AlgoSelect",
                        "GPU Check Failed: " + gpuCheckTask.getException().getMessage());
            });

            new Thread(gpuCheckTask).start();

            chkUseGpu.selectedProperty().addListener((obs, old, val) -> AppStateService.getInstance().setUseGpu(val));
        }

        // WSL
        if (chkUseWsl != null) {
            chkUseWsl.setSelected(AppStateService.getInstance().isUseWsl());
            chkUseWsl.selectedProperty().addListener((obs, old, val) -> AppStateService.getInstance().setUseWsl(val));
        }

        // Balanced Opt
        if (chkBalancedOpt != null) {
            chkBalancedOpt.setSelected(AppStateService.getInstance().isBalancedOptimizationEnabled());
            chkBalancedOpt.selectedProperty()
                    .addListener((obs, old, val) -> AppStateService.getInstance().setEnableBalancedOptimization(val));
        }

        // Logging
        if (enableLoggingCheckBox != null) {
            enableLoggingCheckBox.setSelected(AppStateService.getInstance().isLoggingEnabled());
            enableLoggingCheckBox.selectedProperty()
                    .addListener((obs, old, val) -> AppStateService.getInstance().setLoggingEnabled(val));
        }

        // --- Experimental Controls (Advanced Mode) ---
        if (experimentalCard != null) {
            // Bind Visibility to ViewModel 'isAdvancedMode'
            experimentalCard.visibleProperty().bind(viewModel.isAdvancedModeProperty());
            experimentalCard.managedProperty().bind(viewModel.isAdvancedModeProperty());

            if (manualTrialsField != null) {
                manualTrialsField.textProperty().bindBidirectional(viewModel.manualTrialsProperty());
            }
            if (manualIterationsField != null) {
                manualIterationsField.textProperty().bindBidirectional(viewModel.manualIterationsProperty());
            }
            if (forceGpuCheck != null) {
                forceGpuCheck.selectedProperty().bindBidirectional(viewModel.forceGpuProperty());
            }

            // --- Gen AI Architecture (Phase 8) ---
            Label genAiLabel = new Label("Gen AI Architecture (BERT)");
            genAiLabel.setStyle("-fx-font-weight: bold; -fx-padding: 10 0 5 0;");

            CheckBox chkTrainFromScratch = new CheckBox("Train from Scratch");
            chkTrainFromScratch.selectedProperty().bindBidirectional(viewModel.bertTrainFromScratchProperty());

            TextField txtLayers = new TextField();
            txtLayers.setPromptText("L");
            txtLayers.setPrefWidth(50);
            txtLayers.textProperty().bindBidirectional(viewModel.bertNumLayersProperty());

            TextField txtHidden = new TextField();
            txtHidden.setPromptText("H");
            txtHidden.setPrefWidth(60);
            txtHidden.textProperty().bindBidirectional(viewModel.bertHiddenSizeProperty());

            TextField txtHeads = new TextField();
            txtHeads.setPromptText("A");
            txtHeads.setPrefWidth(50);
            txtHeads.textProperty().bindBidirectional(viewModel.bertNumHeadsProperty());

            javafx.scene.layout.HBox genAiConfig = new javafx.scene.layout.HBox(5,
                    new Label("Layers:"), txtLayers,
                    new Label("Hidden:"), txtHidden,
                    new Label("Heads:"), txtHeads);
            genAiConfig.setAlignment(javafx.geometry.Pos.CENTER_LEFT);

            // Visibility: Only if BERT is selected
            javafx.beans.binding.BooleanBinding isBert = javafx.beans.binding.Bindings.createBooleanBinding(
                    () -> {
                        String model = viewModel.selectedBaseModelProperty().get();
                        return model != null && model.contains("BERT");
                    },
                    viewModel.selectedBaseModelProperty());

            javafx.scene.layout.VBox genAiBox = new javafx.scene.layout.VBox(5, genAiLabel, chkTrainFromScratch,
                    genAiConfig);
            genAiBox.visibleProperty().bind(isBert);
            genAiBox.managedProperty().bind(isBert);

            // Config fields disabled if NOT training from scratch
            genAiConfig.disableProperty().bind(chkTrainFromScratch.selectedProperty().not());

            experimentalCard.getChildren().add(genAiBox);
        }
    }
}