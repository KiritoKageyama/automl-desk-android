package com.example.automl_prototype_1.controller;

import com.example.automl_prototype_1.viewmodel.AlgoSelectViewModel;
import javafx.collections.FXCollections;
import javafx.fxml.FXML;

import javafx.scene.control.*;
import com.example.automl_prototype_1.service.HardwareAcceleratorService;
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

        initExecutionSettings();
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
        }
    }
}