package com.example.automl_prototype_1.viewmodel;

import com.example.automl_prototype_1.model.Dataset;

import com.example.automl_prototype_1.service.AppStateService;

import com.example.automl_prototype_1.service.IClassifier;

import com.example.automl_prototype_1.dataprovider.CsvDataProvider;
import javafx.beans.property.*;
import javafx.beans.value.ChangeListener;

import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;

public class AlgoSelectViewModel {

    private final AppStateService appStateService = AppStateService.getInstance();
    private final com.example.automl_prototype_1.service.agentic.AgenticAdvisor agenticAdvisor;

    // Properties for UI Binding
    private final StringProperty adviceText = new SimpleStringProperty("");
    private final BooleanProperty tuningEnabled = new SimpleBooleanProperty();
    private final StringProperty kFolds = new SimpleStringProperty();
    private final StringProperty scalingStrategy = new SimpleStringProperty();
    private final ObjectProperty<String> selectedBaseModel = new SimpleObjectProperty<>();

    // Checkbox properties
    private final BooleanProperty gaSelected = new SimpleBooleanProperty();
    private final BooleanProperty igpsoSelected = new SimpleBooleanProperty();
    private final BooleanProperty wwoSelected = new SimpleBooleanProperty();
    private final BooleanProperty bpsoSelected = new SimpleBooleanProperty();
    private final BooleanProperty asoSelected = new SimpleBooleanProperty();

    private final BooleanProperty plsSelected = new SimpleBooleanProperty();
    private final BooleanProperty saSelected = new SimpleBooleanProperty();
    private final BooleanProperty gwoSelected = new SimpleBooleanProperty();
    private final BooleanProperty deSelected = new SimpleBooleanProperty();

    public AlgoSelectViewModel() {
        this.agenticAdvisor = new com.example.automl_prototype_1.service.agentic.AgenticAdvisor();
        initializeProperties();
    }

    private void initializeProperties() {
        // Initialize from AppStateService
        tuningEnabled.set(appStateService.isTuningEnabled());
        kFolds.set(String.valueOf(appStateService.getKFolds()));
        scalingStrategy.set(appStateService.getScalingStrategy());
        selectedBaseModel.set(appStateService.getSelectedBaseModelName());

        List<String> selectedAlgos = appStateService.getSelectedAlgorithmNames();
        if (selectedAlgos != null) {
            gaSelected.set(selectedAlgos.contains("GA"));
            igpsoSelected.set(selectedAlgos.contains("IGPSO"));
            wwoSelected.set(selectedAlgos.contains("WWO"));
            bpsoSelected.set(selectedAlgos.contains("BPSO"));
            asoSelected.set(selectedAlgos.contains("ASO"));
            asoSelected.set(selectedAlgos.contains("ASO"));
            plsSelected.set(selectedAlgos.contains("PLS"));
            saSelected.set(selectedAlgos.contains("SA"));
            gwoSelected.set(selectedAlgos.contains("GWO"));
            deSelected.set(selectedAlgos.contains("DE"));
        }

        // Add listeners to update AppStateService when properties change
        tuningEnabled.addListener((obs, old, val) -> appStateService.setTuningEnabled(val));

        kFolds.addListener((obs, old, val) -> {
            try {
                int k = Integer.parseInt(val);
                if (k > 1 && k <= 100)
                    appStateService.setKFolds(k);
            } catch (NumberFormatException e) {
                /* Ignore */ }
        });

        scalingStrategy.addListener((obs, old, val) -> {
            if (val != null)
                appStateService.setScalingStrategy(val);
        });

        selectedBaseModel.addListener((obs, old, val) -> {
            if (val != null) {
                appStateService.setSelectedBaseModelName(val);
                appStateService.setSelectedClassifier(createClassifierFromString(val));
            }
        });

        // Listeners for checkboxes
        ChangeListener<Boolean> algoListener = (obs, old, val) -> updateSelectedAlgorithms();
        gaSelected.addListener(algoListener);
        igpsoSelected.addListener(algoListener);
        wwoSelected.addListener(algoListener);
        bpsoSelected.addListener(algoListener);
        asoSelected.addListener(algoListener);
        asoSelected.addListener(algoListener);
        plsSelected.addListener(algoListener);
        saSelected.addListener(algoListener);
        gwoSelected.addListener(algoListener);
        deSelected.addListener(algoListener);

        // --- Advanced Mode Bindings ---
        isAdvancedMode.set(appStateService.isAdvancedMode());
        forceGpu.set(appStateService.isForceGpu());
        if (appStateService.getManualTrialsOverride() != null)
            manualTrials.set(String.valueOf(appStateService.getManualTrialsOverride()));
        if (appStateService.getManualIterationsOverride() != null)
            manualIterations.set(String.valueOf(appStateService.getManualIterationsOverride()));

        manualTrials.addListener((obs, old, val) -> {
            try {
                if (val == null || val.trim().isEmpty()) {
                    appStateService.setManualTrialsOverride(null);
                } else {
                    appStateService.setManualTrialsOverride(Integer.parseInt(val.trim()));
                }
            } catch (NumberFormatException e) {
                /* Ignore invalid input */ }
        });

        manualIterations.addListener((obs, old, val) -> {
            try {
                if (val == null || val.trim().isEmpty()) {
                    appStateService.setManualIterationsOverride(null);
                } else {
                    appStateService.setManualIterationsOverride(Integer.parseInt(val.trim()));
                }
            } catch (NumberFormatException e) {
                /* Ignore invalid input */ }
        });

        forceGpu.addListener((obs, old, val) -> appStateService.setForceGpu(val));

        // --- BERT Listeners ---
        Double savedScratch = appStateService.getAlgorithmHyperparameter("BERT", "scratch");
        if (savedScratch != null)
            bertTrainFromScratch.set(savedScratch > 0.5);

        Double savedLayers = appStateService.getAlgorithmHyperparameter("BERT", "num_layers");
        if (savedLayers != null)
            bertNumLayers.set(String.valueOf(savedLayers.intValue()));

        Double savedHidden = appStateService.getAlgorithmHyperparameter("BERT", "hidden_size");
        if (savedHidden != null)
            bertHiddenSize.set(String.valueOf(savedHidden.intValue()));

        Double savedHeads = appStateService.getAlgorithmHyperparameter("BERT", "num_heads");
        if (savedHeads != null)
            bertNumHeads.set(String.valueOf(savedHeads.intValue()));

        bertTrainFromScratch.addListener(
                (obs, old, val) -> appStateService.setAlgorithmHyperparameter("BERT", "scratch", val ? 1.0 : 0.0));

        bertNumLayers.addListener((obs, old, val) -> {
            try {
                appStateService.setAlgorithmHyperparameter("BERT", "num_layers", (double) Integer.parseInt(val));
            } catch (Exception e) {
            }
        });
        bertHiddenSize.addListener((obs, old, val) -> {
            try {
                appStateService.setAlgorithmHyperparameter("BERT", "hidden_size", (double) Integer.parseInt(val));
            } catch (Exception e) {
            }
        });
        bertNumHeads.addListener((obs, old, val) -> {
            try {
                appStateService.setAlgorithmHyperparameter("BERT", "num_heads", (double) Integer.parseInt(val));
            } catch (Exception e) {
            }
        });
    }

    // --- Advanced Properties ---
    private final BooleanProperty isAdvancedMode = new SimpleBooleanProperty();
    private final StringProperty manualTrials = new SimpleStringProperty();
    private final StringProperty manualIterations = new SimpleStringProperty();
    private final BooleanProperty forceGpu = new SimpleBooleanProperty();

    public BooleanProperty isAdvancedModeProperty() {
        return isAdvancedMode;
    }

    public StringProperty manualTrialsProperty() {
        return manualTrials;
    }

    public StringProperty manualIterationsProperty() {
        return manualIterations;
    }

    public BooleanProperty forceGpuProperty() {
        return forceGpu;
    }

    private void updateSelectedAlgorithms() {
        List<String> selected = new ArrayList<>();
        if (gaSelected.get())
            selected.add("GA");
        if (igpsoSelected.get())
            selected.add("IGPSO");
        if (wwoSelected.get())
            selected.add("WWO");
        if (bpsoSelected.get())
            selected.add("BPSO");
        if (asoSelected.get())
            selected.add("ASO");
        if (plsSelected.get())
            selected.add("PLS");
        if (saSelected.get())
            selected.add("SA");
        if (gwoSelected.get())
            selected.add("GWO");
        if (deSelected.get())
            selected.add("DE");
        appStateService.setSelectedAlgorithmNames(selected);
    }

    public void generateAdvice() {
        File datasetFile = appStateService.getSelectedDatasetFile();
        if (datasetFile == null) {
            adviceText.set("Warning: Please select a dataset first on the Upload tab.");
            return;
        }

        // Run in background thread to avoid freezing UI during empirical benchmark
        adviceText.set(
                "Agentic AI is thinking...\n\n1. Analyzing dataset meta-features...\n2. Running internal empirical benchmarks (Quick Scan)...\n3. Searching external knowledge base for SOTA results...");

        new Thread(() -> {
            try (FileReader reader = new FileReader(datasetFile)) {
                Dataset tempDataset = new CsvDataProvider().loadDataset(reader);
                String datasetName = datasetFile.getName();

                // Use AgenticAdvisor to get suggestions (now includes empirical + SOTA)
                List<com.example.automl_prototype_1.service.agentic.PipelineSuggestion> suggestions = agenticAdvisor
                        .suggestPipelines(tempDataset, datasetName);

                javafx.application.Platform.runLater(() -> {
                    StringBuilder sb = new StringBuilder();
                    sb.append("--- Agentic AI Report ---\n");
                    sb.append("Analyzed Dataset: ").append(datasetName).append("\n\n");

                    if (suggestions.isEmpty()) {
                        sb.append("No specific recommendations found. Try standard algorithms like Random Forest.");
                    } else {
                        // Display SOTA from the first suggestion (they all share the same lookup)
                        String sota = suggestions.get(0).getSotaComparison();
                        sb.append(">> External Knowledge (SOTA): ").append(sota).append("\n\n");
                        sb.append(">> Internal Benchmark & Recommendations:\n");

                        for (int i = 0; i < suggestions.size(); i++) {
                            com.example.automl_prototype_1.service.agentic.PipelineSuggestion s = suggestions.get(i);
                            sb.append(
                                    String.format("%d. %s + %s\n", i + 1, s.getAlgorithmName(), s.getClassifierName()));
                            sb.append(String.format("   Confidence: %.0f%% | Empirical Score: %.2f\n",
                                    s.getConfidence() * 100, s.getEmpiricalScore()));
                            sb.append("   Reason: ").append(s.getReasoning()).append("\n\n");
                        }
                    }
                    adviceText.set(sb.toString());
                });

            } catch (Exception e) {
                javafx.application.Platform.runLater(() -> {
                    adviceText.set("Error: Could not load dataset for analysis.\n\n" + e.getMessage());
                });
                e.printStackTrace();
            }
        }).start();
    }

    // --- BERT Gen AI Properties (Phase 8) ---
    private final BooleanProperty bertTrainFromScratch = new SimpleBooleanProperty();
    private final StringProperty bertNumLayers = new SimpleStringProperty("6");
    private final StringProperty bertHiddenSize = new SimpleStringProperty("768");
    private final StringProperty bertNumHeads = new SimpleStringProperty("12");

    public BooleanProperty bertTrainFromScratchProperty() {
        return bertTrainFromScratch;
    }

    public StringProperty bertNumLayersProperty() {
        return bertNumLayers;
    }

    public StringProperty bertHiddenSizeProperty() {
        return bertHiddenSize;
    }

    public StringProperty bertNumHeadsProperty() {
        return bertNumHeads;
    }

    {
        // Block to init BERT listeners (in instance init or constructor)
        // We'll do it in initializeProperties called by constructor
    }

    // ... inside initializeProperties ...
    // See below for injection

    public IClassifier createClassifierFromString(String name) {
        return com.example.automl_prototype_1.registry.ClassifierRegistry.getInstance().getClassifier(name);
    }

    public List<String> getAvailableClassifiers() {
        return com.example.automl_prototype_1.registry.ClassifierRegistry.getInstance().getClassifierNames();
    }

    // Getters for properties
    public StringProperty adviceTextProperty() {
        return adviceText;
    }

    public BooleanProperty tuningEnabledProperty() {
        return tuningEnabled;
    }

    public StringProperty kFoldsProperty() {
        return kFolds;
    }

    public StringProperty scalingStrategyProperty() {
        return scalingStrategy;
    }

    public ObjectProperty<String> selectedBaseModelProperty() {
        return selectedBaseModel;
    }

    public BooleanProperty gaSelectedProperty() {
        return gaSelected;
    }

    public BooleanProperty igpsoSelectedProperty() {
        return igpsoSelected;
    }

    public BooleanProperty wwoSelectedProperty() {
        return wwoSelected;
    }

    public BooleanProperty bpsoSelectedProperty() {
        return bpsoSelected;
    }

    public BooleanProperty asoSelectedProperty() {
        return asoSelected;
    }

    public BooleanProperty plsSelectedProperty() {
        return plsSelected;
    }

    public BooleanProperty saSelectedProperty() {
        return saSelected;
    }

    public BooleanProperty gwoSelectedProperty() {
        return gwoSelected;
    }

    public BooleanProperty deSelectedProperty() {
        return deSelected;
    }
}
