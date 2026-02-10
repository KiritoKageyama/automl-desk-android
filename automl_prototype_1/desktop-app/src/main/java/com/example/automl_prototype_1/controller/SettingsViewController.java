package com.example.automl_prototype_1.controller;

import com.example.automl_prototype_1.DashboardApplication;

import javafx.fxml.FXML;
import javafx.scene.control.Button;
import javafx.scene.control.CheckBox;

public class SettingsViewController {
    @FXML
    private Button darkThemeButton;
    @FXML
    private Button lightThemeButton;
    @FXML
    private CheckBox darkModeCheckBox;

    @FXML
    private CheckBox gpuCheckBox;
    @FXML
    private CheckBox hybridCheckBox;
    @FXML
    private javafx.scene.layout.VBox hybridConfigBox;

    @FXML
    private CheckBox advancedModeCheckBox;
    @FXML
    private CheckBox ollamaEnabledCheckBox;
    @FXML
    private javafx.scene.control.TextField ollamaModelField;
    @FXML
    private javafx.scene.control.TextField ollamaUrlField;

    @FXML
    private javafx.scene.layout.VBox advancedPipelineBox;
    @FXML
    private javafx.scene.control.ComboBox<String> scalingComboBox;
    @FXML
    private CheckBox featureSelectionCheckBox;
    @FXML
    private javafx.scene.control.Spinner<Integer> gaPopSpinner;
    @FXML
    private javafx.scene.control.Slider gaCrossoverSlider;
    @FXML
    private javafx.scene.control.Slider gaMutationSlider;
    @FXML
    private javafx.scene.control.Spinner<Integer> gwoPopSpinner;

    public SettingsViewController() {
        // No-arg constructor
    }

    @FXML
    private void initialize() {
        com.example.automl_prototype_1.service.AppStateService appState = com.example.automl_prototype_1.service.AppStateService
                .getInstance();

        // --- Call the static methods on DashboardApplication ---
        if (darkThemeButton != null)
            darkThemeButton.setOnAction(e -> DashboardApplication.setTheme("trading"));
        if (lightThemeButton != null)
            lightThemeButton.setOnAction(e -> DashboardApplication.setTheme("light"));

        // --- Hardware / Performance Settings ---
        if (gpuCheckBox != null) {
            gpuCheckBox.setSelected(appState.isUseGpu());
            gpuCheckBox.selectedProperty().addListener((obs, oldV, newV) -> {
                appState.setUseGpu(newV);
                if (hybridConfigBox != null)
                    hybridConfigBox.setDisable(!newV);
            });
        }

        if (hybridCheckBox != null) {
            hybridCheckBox.setSelected(appState.isHybridExecutionEnabled());
            hybridCheckBox.selectedProperty().addListener((obs, oldV, newV) -> {
                appState.setHybridExecutionEnabled(newV);
            });
        }

        if (hybridConfigBox != null) {
            hybridConfigBox.setDisable(!appState.isUseGpu());
        }

        // --- Advanced Mode ---
        if (advancedModeCheckBox != null) {
            advancedModeCheckBox.setSelected(appState.isAdvancedMode());
            advancedModeCheckBox.selectedProperty().addListener((obs, oldV, newV) -> {
                appState.setAdvancedMode(newV);
                if (advancedPipelineBox != null) {
                    advancedPipelineBox.setDisable(!newV); // Enable pipeline controls
                }
            });
            if (advancedPipelineBox != null) {
                advancedPipelineBox.setDisable(!appState.isAdvancedMode());
            }
        }

        // --- Pipeline Composition (Phase 3) ---
        if (scalingComboBox != null) {
            scalingComboBox.getItems().addAll("Standardization", "Normalization", "None");
            scalingComboBox.setValue(appState.getScalingStrategy());
            scalingComboBox.valueProperty().addListener((obs, oldV, newV) -> appState.setScalingStrategy(newV));
        }

        if (featureSelectionCheckBox != null) {
            featureSelectionCheckBox.setSelected(appState.isFeatureSelectionEnabled());
            featureSelectionCheckBox.selectedProperty()
                    .addListener((obs, oldV, newV) -> appState.setFeatureSelectionEnabled(newV));
        }

        // --- Algorithm Hyperparameters (Phase 2) ---
        if (gaPopSpinner != null) {
            Double val = appState.getAlgorithmHyperparameter("GA", "populationSize");
            int initial = (val != null) ? val.intValue() : 50;
            gaPopSpinner.setValueFactory(
                    new javafx.scene.control.SpinnerValueFactory.IntegerSpinnerValueFactory(10, 500, initial));
            gaPopSpinner.valueProperty().addListener((obs, oldV, newV) -> appState.setAlgorithmHyperparameter("GA",
                    "populationSize", newV.doubleValue()));
        }

        if (gaCrossoverSlider != null) {
            Double val = appState.getAlgorithmHyperparameter("GA", "crossoverRate");
            gaCrossoverSlider.setValue((val != null) ? val : 0.8);
            gaCrossoverSlider.valueProperty().addListener((obs, oldV, newV) -> appState.setAlgorithmHyperparameter("GA",
                    "crossoverRate", newV.doubleValue()));
        }

        if (gaMutationSlider != null) {
            Double val = appState.getAlgorithmHyperparameter("GA", "mutationRate");
            gaMutationSlider.setValue((val != null) ? val : 0.05);
            gaMutationSlider.valueProperty().addListener(
                    (obs, oldV, newV) -> appState.setAlgorithmHyperparameter("GA", "mutationRate", newV.doubleValue()));
        }

        if (gwoPopSpinner != null) {
            Double val = appState.getAlgorithmHyperparameter("GWO", "populationSize");
            int initial = (val != null) ? val.intValue() : 30;
            gwoPopSpinner.setValueFactory(
                    new javafx.scene.control.SpinnerValueFactory.IntegerSpinnerValueFactory(10, 500, initial));
            gwoPopSpinner.valueProperty().addListener((obs, oldV, newV) -> appState.setAlgorithmHyperparameter("GWO",
                    "populationSize", newV.doubleValue()));
        }

        // --- Ollama Settings ---
        if (ollamaEnabledCheckBox != null) {
            ollamaEnabledCheckBox.setSelected(appState.isOllamaEnabled());
            ollamaEnabledCheckBox.selectedProperty().addListener((obs, oldV, newV) -> {
                appState.setOllamaEnabled(newV);
                if (ollamaModelField != null)
                    ollamaModelField.setDisable(!newV);
                if (ollamaUrlField != null)
                    ollamaUrlField.setDisable(!newV);
            });
        }

        if (ollamaModelField != null) {
            ollamaModelField.setText(appState.getOllamaModelName());
            ollamaModelField.textProperty().addListener((obs, oldV, newV) -> appState.setOllamaModelName(newV));
            ollamaModelField.setDisable(!appState.isOllamaEnabled());
        }

        if (ollamaUrlField != null) {
            ollamaUrlField.setText(appState.getOllamaUrl());
            ollamaUrlField.textProperty().addListener((obs, oldV, newV) -> appState.setOllamaUrl(newV));
            ollamaUrlField.setDisable(!appState.isOllamaEnabled());
        }
    }
}