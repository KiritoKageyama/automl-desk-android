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

    // Phase 4
    @FXML
    private CheckBox manualFeatSelCheckBox;
    @FXML
    private CheckBox enableMlpCheckBox;
    @FXML
    private javafx.scene.layout.HBox mlpConfigBox;
    @FXML
    private javafx.scene.control.TextField mrMlpLayersField;

    // Phase 5: Advanced Preprocessing
    @FXML
    private javafx.scene.control.ComboBox<String> transformationComboBox;
    @FXML
    private javafx.scene.control.ComboBox<String> featureCreationComboBox;
    @FXML
    private CheckBox removeOutliersCheckBox;
    @FXML
    private Button askAiPreprocessingBtn;

    // AI Safety Controls
    @FXML
    private Button emergencyStopBtn;
    @FXML
    private Button interruptAiBtn;
    @FXML
    private Button clearAiCacheBtn;
    @FXML
    private CheckBox strictModeCheckBox;
    @FXML
    private javafx.scene.control.Label safetyStatusLabel;

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

        // --- Manual Feature Selection (Phase 4) ---
        if (manualFeatSelCheckBox != null) {
            manualFeatSelCheckBox.setSelected(appState.isManualFeatureSelectionEnabled());
            manualFeatSelCheckBox.selectedProperty().addListener((obs, oldV, newV) -> {
                appState.setManualFeatureSelectionEnabled(newV);
                // Mutually exclusive with Auto-FS? Maybe not strict, dealing with logic in
                // ExecutionService or Dashboard
                if (newV && featureSelectionCheckBox != null) {
                    featureSelectionCheckBox.setSelected(false); // Auto-disable optimization if manual is picked
                }
            });
        }

        // --- Custom MLP ---
        if (enableMlpCheckBox != null) {
            // We use a specific hyperparameter/flag for this?
            // "MLP" isn't in the algo list by default.
            // DashboardController.getAlgorithmImplementation handles it?
            // Actually, we need to add "MLP" to the algo list dynamically or just have it.
            // For now, let's just store the config in AppState algorithmHyperparameters
            // "MLP"

            Double existing = appState.getAlgorithmHyperparameter("MLP", "enabled");
            boolean isEnabled = (existing != null && existing > 0.5);
            enableMlpCheckBox.setSelected(isEnabled);

            enableMlpCheckBox.selectedProperty().addListener((obs, oldV, newV) -> {
                appState.setAlgorithmHyperparameter("MLP", "enabled", newV ? 1.0 : 0.0);
                if (mlpConfigBox != null)
                    mlpConfigBox.setDisable(!newV);
            });

            if (mlpConfigBox != null)
                mlpConfigBox.setDisable(!isEnabled);
        }

        if (mrMlpLayersField != null) {
            String val = appState.getAlgorithmConfigString("MLP", "hidden_layer_sizes");
            if (val != null) {
                mrMlpLayersField.setText(val);
            }
            mrMlpLayersField.textProperty().addListener((obs, oldV, newV) -> {
                appState.setAlgorithmConfigString("MLP", "hidden_layer_sizes", newV);
            });

            Double enabledVal = appState.getAlgorithmHyperparameter("MLP", "enabled");
            boolean isEnabled = (enabledVal != null && enabledVal > 0.5);
            mrMlpLayersField.setDisable(!isEnabled);
        }

        // --- Phase 5: Advanced Preprocessing Bindings ---
        com.example.automl_prototype_1.model.PreprocessingConfig config = appState.getPreprocessingConfig();
        if (config == null) {
            config = new com.example.automl_prototype_1.model.PreprocessingConfig();
            appState.setPreprocessingConfig(config);
        }

        if (transformationComboBox != null) {
            transformationComboBox.getItems().addAll("None", "Log", "Sqrt", "Auto-Skew");
            transformationComboBox.setValue(config.getTransformationStrategy());
            transformationComboBox.valueProperty().addListener((obs, oldV, newV) -> {
                com.example.automl_prototype_1.model.PreprocessingConfig c = appState.getPreprocessingConfig();
                c.setTransformationStrategy(newV);
                appState.setPreprocessingConfig(c); // Trigger updates
            });
        }

        if (featureCreationComboBox != null) {
            featureCreationComboBox.getItems().addAll("None", "Interaction", "Polynomial");
            featureCreationComboBox.setValue(config.getFeatureCreationStrategy());
            featureCreationComboBox.valueProperty().addListener((obs, oldV, newV) -> {
                com.example.automl_prototype_1.model.PreprocessingConfig c = appState.getPreprocessingConfig();
                c.setFeatureCreationStrategy(newV);
                appState.setPreprocessingConfig(c);
            });
        }

        if (removeOutliersCheckBox != null) {
            removeOutliersCheckBox.setSelected(config.isRemoveOutliers());
            removeOutliersCheckBox.selectedProperty().addListener((obs, oldV, newV) -> {
                com.example.automl_prototype_1.model.PreprocessingConfig c = appState.getPreprocessingConfig();
                c.setRemoveOutliers(newV);
                appState.setPreprocessingConfig(c);
            });
        }

        if (askAiPreprocessingBtn != null) {
            askAiPreprocessingBtn.setOnAction(e -> handleAskAiPreprocessing());
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

        // --- AI Safety Controls ---
        initializeSafetyControls();
    }

    private void initializeSafetyControls() {
        if (strictModeCheckBox != null) {
            strictModeCheckBox.selectedProperty().addListener((obs, oldV, newV) -> {
                try {
                    com.example.automl_prototype_1.service.agentic.ConstitutionalService constitution = new com.example.automl_prototype_1.service.agentic.ConstitutionalService();
                    constitution.setStrictMode(newV);
                    updateSafetyStatus("Strict Mode " + (newV ? "ENABLED" : "DISABLED"));
                } catch (Exception e) {
                    updateSafetyStatus("Error: " + e.getMessage());
                }
            });
        }

        if (emergencyStopBtn != null) {
            emergencyStopBtn.setOnAction(e -> handleEmergencyStop());
        }
        if (interruptAiBtn != null) {
            interruptAiBtn.setOnAction(e -> handleInterruptAi());
        }
        if (clearAiCacheBtn != null) {
            clearAiCacheBtn.setOnAction(e -> handleClearAiCache());
        }
    }

    private void handleEmergencyStop() {
        emergencyStopBtn.setDisable(true);
        new Thread(() -> {
            try {
                com.example.automl_prototype_1.service.agentic.LocalLlmService llm = new com.example.automl_prototype_1.service.agentic.LocalLlmService();
                llm.emergencyStop();
                javafx.application.Platform.runLater(() -> {
                    updateSafetyStatus("🔴 EMERGENCY STOP executed. All AI processes killed.");
                    emergencyStopBtn.setDisable(false);
                });
            } catch (Exception ex) {
                javafx.application.Platform.runLater(() -> {
                    updateSafetyStatus("Error: " + ex.getMessage());
                    emergencyStopBtn.setDisable(false);
                });
            }
        }).start();
    }

    private void handleInterruptAi() {
        try {
            com.example.automl_prototype_1.service.agentic.LocalLlmService llm = new com.example.automl_prototype_1.service.agentic.LocalLlmService();
            llm.interrupt();
            updateSafetyStatus("⏸ Interrupt signal sent. Current AI generation will stop.");
        } catch (Exception ex) {
            updateSafetyStatus("Error: " + ex.getMessage());
        }
    }

    private void handleClearAiCache() {
        clearAiCacheBtn.setDisable(true);
        new Thread(() -> {
            try {
                com.example.automl_prototype_1.service.agentic.LocalLlmService llm = new com.example.automl_prototype_1.service.agentic.LocalLlmService();
                llm.clearCache();
                javafx.application.Platform.runLater(() -> {
                    updateSafetyStatus("🗑 AI Cache cleared. LLM context reset.");
                    clearAiCacheBtn.setDisable(false);
                });
            } catch (Exception ex) {
                javafx.application.Platform.runLater(() -> {
                    updateSafetyStatus("Error: " + ex.getMessage());
                    clearAiCacheBtn.setDisable(false);
                });
            }
        }).start();
    }

    private void updateSafetyStatus(String message) {
        if (safetyStatusLabel != null) {
            safetyStatusLabel.setText(message);
        }
    }

    private void handleAskAiPreprocessing() {
        askAiPreprocessingBtn.setText("Thinking...");
        askAiPreprocessingBtn.setDisable(true);

        new Thread(() -> {
            com.example.automl_prototype_1.service.AppStateService appState = com.example.automl_prototype_1.service.AppStateService
                    .getInstance();
            java.io.File file = appState.getSelectedDatasetFile();
            if (file == null) {
                javafx.application.Platform.runLater(() -> {
                    askAiPreprocessingBtn.setText("Select Dataset First!");
                    askAiPreprocessingBtn.setDisable(false);
                });
                return;
            }

            try (com.example.automl_prototype_1.service.agentic.AgenticAdvisor advisor = new com.example.automl_prototype_1.service.agentic.AgenticAdvisor();
                    java.io.FileReader reader = new java.io.FileReader(file)) {

                com.example.automl_prototype_1.model.Dataset dataset = new com.example.automl_prototype_1.dataprovider.CsvDataProvider()
                        .loadDataset(reader);
                String suggestion = advisor.suggestPreprocessingStrategy(dataset);

                // Parse suggestion (Simple heuristic parser for demo)
                // Expected format: "Transform: Log, Creation: Interaction, Outliers: True"
                final String lowerSuggestion = suggestion.toLowerCase();

                javafx.application.Platform.runLater(() -> {
                    if (lowerSuggestion.contains("log"))
                        transformationComboBox.setValue("Log");
                    else if (lowerSuggestion.contains("sqrt"))
                        transformationComboBox.setValue("Sqrt");
                    else if (lowerSuggestion.contains("auto"))
                        transformationComboBox.setValue("Auto-Skew");

                    if (lowerSuggestion.contains("interaction"))
                        featureCreationComboBox.setValue("Interaction");
                    else if (lowerSuggestion.contains("poly"))
                        featureCreationComboBox.setValue("Polynomial");

                    if (lowerSuggestion.contains("remove outliers") || lowerSuggestion.contains("iqr")) {
                        removeOutliersCheckBox.setSelected(true);
                    }

                    askAiPreprocessingBtn.setText("Strategy Applied! (Click to Retry)");
                    askAiPreprocessingBtn.setDisable(false);

                    // Show full reasoning in an alert? Or just existing patterns
                    javafx.scene.control.Alert alert = new javafx.scene.control.Alert(
                            javafx.scene.control.Alert.AlertType.INFORMATION);
                    alert.setTitle("AI Preprocessing Strategy");
                    alert.setHeaderText("Agentic Advisor Recommendation");
                    alert.setContentText(suggestion);
                    alert.showAndWait();
                });

            } catch (Exception e) {
                e.printStackTrace();
                javafx.application.Platform.runLater(() -> {
                    askAiPreprocessingBtn.setText("Error (See Log)");
                    askAiPreprocessingBtn.setDisable(false);
                });
            }
        }).start();
    }
}