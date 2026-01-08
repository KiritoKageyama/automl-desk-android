package com.example.automl_prototype_1.controller;

import com.example.automl_prototype_1.DashboardApplication;
import com.example.automl_prototype_1.service.HardwareAcceleratorService;

import javafx.fxml.FXML;
import javafx.scene.control.Button;
import javafx.scene.control.CheckBox;

public class SettingsViewController {
    @FXML
    private Button darkThemeButton;
    @FXML
    private Button lightThemeButton;
    @FXML
    private CheckBox enableLoggingCheckBox;
    @FXML
    private CheckBox darkModeCheckBox;
    @FXML
    private CheckBox chkUseGpu;
    @FXML
    private CheckBox chkUseWsl;

    public SettingsViewController() {
        // No-arg constructor
    }

    @FXML
    private javafx.scene.control.ComboBox<String> cmbGpuSelector;

    @FXML
    private CheckBox chkStandardization;
    @FXML
    private CheckBox chkBalancedOpt;

    @FXML
    private void initialize() {
        // --- Call the static methods on DashboardApplication ---
        darkThemeButton.setOnAction(e -> DashboardApplication.setTheme("trading"));
        lightThemeButton.setOnAction(e -> DashboardApplication.setTheme("light"));

        // Initialize features from AppStateService
        if (chkStandardization != null) {
            chkStandardization.setSelected(
                    com.example.automl_prototype_1.service.AppStateService.getInstance().isStandardizationEnabled());
            chkStandardization.selectedProperty()
                    .addListener((obs, old, val) -> com.example.automl_prototype_1.service.AppStateService.getInstance()
                            .setEnableStandardization(val));
        }

        if (chkBalancedOpt != null) {
            chkBalancedOpt.setSelected(com.example.automl_prototype_1.service.AppStateService.getInstance()
                    .isBalancedOptimizationEnabled());
            chkBalancedOpt.selectedProperty()
                    .addListener((obs, old, val) -> com.example.automl_prototype_1.service.AppStateService.getInstance()
                            .setEnableBalancedOptimization(val));
        }

        if (enableLoggingCheckBox != null) {
            enableLoggingCheckBox.setSelected(
                    com.example.automl_prototype_1.service.AppStateService.getInstance().isLoggingEnabled());
            enableLoggingCheckBox.selectedProperty()
                    .addListener((obs, old, val) -> com.example.automl_prototype_1.service.AppStateService.getInstance()
                            .setLoggingEnabled(val));
        }

        // Initialize GPU Checkbox
        // Initialize GPU Checkbox
        if (chkUseGpu != null) {
            boolean isGpu = HardwareAcceleratorService.isGpuAvailable();
            chkUseGpu.setSelected(com.example.automl_prototype_1.service.AppStateService.getInstance().isUseGpu()); // Load
                                                                                                                    // saved
                                                                                                                    // pref

            if (isGpu) {
                chkUseGpu.setText("Use GPU Acceleration");

                // Populate Selector
                if (cmbGpuSelector != null) {
                    java.util.List<String> gpuNames = HardwareAcceleratorService.getAvailableGpuNames();
                    cmbGpuSelector.getItems().setAll(gpuNames);

                    // Restore selection or default
                    String savedGpu = com.example.automl_prototype_1.service.AppStateService.getInstance()
                            .getSelectedGpuName();
                    if (savedGpu != null && gpuNames.contains(savedGpu)) {
                        cmbGpuSelector.getSelectionModel().select(savedGpu);
                    } else if (!gpuNames.isEmpty()) {
                        cmbGpuSelector.getSelectionModel().select(0);
                        com.example.automl_prototype_1.service.AppStateService.getInstance()
                                .setSelectedGpuName(gpuNames.get(0));
                    }

                    // Bind visibility
                    cmbGpuSelector.visibleProperty().bind(chkUseGpu.selectedProperty());
                    cmbGpuSelector.managedProperty().bind(chkUseGpu.selectedProperty());

                    // Listener
                    cmbGpuSelector.getSelectionModel().selectedItemProperty().addListener((obs, oldVal, newVal) -> {
                        if (newVal != null) {
                            com.example.automl_prototype_1.service.AppStateService.getInstance()
                                    .setSelectedGpuName(newVal);
                        }
                    });
                }

            } else {
                chkUseGpu.setText("Use GPU Acceleration (Not Detected)");
                chkUseGpu.setDisable(true);
                if (cmbGpuSelector != null) {
                    cmbGpuSelector.setVisible(false);
                    cmbGpuSelector.setManaged(false);
                }
            }

            chkUseGpu.selectedProperty().addListener((obs, oldVal, newVal) -> {
                System.out.println("GPU Acceleration preference set to: " + newVal);
                com.example.automl_prototype_1.service.AppStateService.getInstance().setUseGpu(newVal);
            });
        }

        // Initialize WSL Checkbox
        if (chkUseWsl != null) {
            chkUseWsl.setSelected(com.example.automl_prototype_1.service.AppStateService.getInstance().isUseWsl());
            chkUseWsl.selectedProperty().addListener((obs, oldVal, newVal) -> {
                com.example.automl_prototype_1.service.AppStateService.getInstance().setUseWsl(newVal);
            });
        }
    }
}