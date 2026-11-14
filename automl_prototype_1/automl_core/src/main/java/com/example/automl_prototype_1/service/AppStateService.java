package com.example.automl_prototype_1.service;

import com.example.automl_prototype_1.model.Dataset;
import com.example.automl_prototype_1.model.ExecutionResult;
import java.io.File;
import java.util.List;
import java.util.Map;

public class AppStateService {

    private static AppStateService instance;

    private java.util.Map<com.example.automl_prototype_1.model.Domain, com.example.automl_prototype_1.model.DomainState> domainStates = new java.util.EnumMap<>(
            com.example.automl_prototype_1.model.Domain.class);
    private com.example.automl_prototype_1.model.Domain currentDomain = com.example.automl_prototype_1.model.Domain.ML;

    // --- Phase 2: UI Expansion Enums ---
    public enum DataType {
        NUMERICAL, IMAGE, USER_PATTERN, TEXT
    }

    public enum DataQuality {
        STRUCTURED, UNSTRUCTURED, SEMI_STRUCTURED
    }

    public enum TrainingMode {
        STATISTICAL, ML_ALGORITHMS, NEURAL_NETWORK_MLP,
        DEEP_LEARNING, CNN, VISION_TRANSFORMERS,
        GEN_AI, ATTENTION_NETWORKS, RNN_LSTM,
        AGENTIC_TRAINING, REINFORCEMENT_LEARNING
    }

    private DataType selectedDataType = DataType.NUMERICAL;
    private DataQuality selectedDataQuality = DataQuality.STRUCTURED;
    private boolean enableQualityCheck = false; // Default disabled as requested
    private TrainingMode selectedTrainingMode = TrainingMode.ML_ALGORITHMS;

    // Phase 13: Modular Visibility
    private boolean showMl = true;
    private boolean showDl = true;
    private boolean showCnn = true;
    private boolean showLlm = true;
    private boolean showGenAi = true;
    private boolean showRl = true;

    // Getters and Setters for Visibility
    public boolean isShowMl() {
        return showMl;
    }

    public void setShowMl(boolean show) {
        this.showMl = show;
    }

    public boolean isShowDl() {
        return showDl;
    }

    public void setShowDl(boolean show) {
        this.showDl = show;
    }

    public boolean isShowCnn() {
        return showCnn;
    }

    public void setShowCnn(boolean show) {
        this.showCnn = show;
    }

    public boolean isShowLlm() {
        return showLlm;
    }

    public void setShowLlm(boolean show) {
        this.showLlm = show;
    }

    public boolean isShowGenAi() {
        return showGenAi;
    }

    public void setShowGenAi(boolean show) {
        this.showGenAi = show;
    }

    public boolean isShowRl() {
        return showRl;
    }

    public void setShowRl(boolean show) {
        this.showRl = show;
    }

    private AppStateService() {
        for (com.example.automl_prototype_1.model.Domain d : com.example.automl_prototype_1.model.Domain.values()) {
            domainStates.put(d, new com.example.automl_prototype_1.model.DomainState());
        }
        LogService.info("AppStateService", "AppStateService Initialized (New Instance)");
    }

    public static synchronized AppStateService getInstance() {
        if (instance == null) {
            instance = new AppStateService();
        }
        return instance;
    }

    public static boolean hasInstance() {
        return instance != null;
    }

    // --- Getters and Setters ---

    public com.example.automl_prototype_1.model.Domain getCurrentDomain() {
        return currentDomain;
    }

    public void setCurrentDomain(com.example.automl_prototype_1.model.Domain currentDomain) {
        this.currentDomain = currentDomain;
    }

    public com.example.automl_prototype_1.model.DomainState getDomainState(
            com.example.automl_prototype_1.model.Domain domain) {
        return domainStates.get(domain);
    }

    public com.example.automl_prototype_1.model.DomainState getCurrentDomainState() {
        return domainStates.get(currentDomain);
    }

    public File getSelectedDatasetFile() {
        return getCurrentDomainState().getSelectedDatasetFile();
    }

    public void setSelectedDatasetFile(File selectedDatasetFile) {
        getCurrentDomainState().setSelectedDatasetFile(selectedDatasetFile);
    }

    public List<String> getDatasetHeaders() {
        return getCurrentDomainState().getDatasetHeaders();
    }

    public void setDatasetHeaders(List<String> datasetHeaders) {
        getCurrentDomainState().setDatasetHeaders(datasetHeaders);
    }

    public Dataset getCurrentDataset() {
        return getCurrentDomainState().getCurrentDataset();
    }

    public void setCurrentDataset(Dataset currentDataset) {
        getCurrentDomainState().setCurrentDataset(currentDataset);
        LogService.info("AppStateService", "Current dataset set: " +
                (currentDataset != null ? currentDataset.getHeaders().size() + " columns" : "null"));
    }

    public String getTargetColumnName() {
        return getCurrentDomainState().getTargetColumnName();
    }

    public void setTargetColumnName(String targetColumnName) {
        getCurrentDomainState().setTargetColumnName(targetColumnName);
    }

    public List<String> getSelectedAlgorithmNames() {
        return getCurrentDomainState().getSelectedAlgorithmNames();
    }

    public void setSelectedAlgorithmNames(List<String> selectedAlgorithmNames) {
        getCurrentDomainState().setSelectedAlgorithmNames(selectedAlgorithmNames);
    }

    public String getSelectedBaseModelName() {
        return getCurrentDomainState().getSelectedBaseModelName();
    }

    public void setSelectedBaseModelName(String selectedBaseModelName) {
        getCurrentDomainState().setSelectedBaseModelName(selectedBaseModelName);
    }

    public IClassifier getSelectedClassifier() {
        return getCurrentDomainState().getSelectedClassifier();
    }

    public void setSelectedClassifier(IClassifier selectedClassifier) {
        getCurrentDomainState().setSelectedClassifier(selectedClassifier);
    }

    public String getScalingStrategy() {
        return getCurrentDomainState().getScalingStrategy();
    }

    public void setScalingStrategy(String scalingStrategy) {
        getCurrentDomainState().setScalingStrategy(scalingStrategy);
    }

    public int getKFolds() {
        return getCurrentDomainState().getKFolds();
    }

    public void setKFolds(int kFolds) {
        getCurrentDomainState().setKFolds(kFolds);
    }

    public boolean isTuningEnabled() {
        return getCurrentDomainState().isTuningEnabled();
    }

    public void setTuningEnabled(boolean tuningEnabled) {
        getCurrentDomainState().setTuningEnabled(tuningEnabled);
    }

    public boolean isStandardizationEnabled() {
        return getCurrentDomainState().isStandardizationEnabled();
    }

    public void setEnableStandardization(boolean enableStandardization) {
        getCurrentDomainState().setEnableStandardization(enableStandardization);
    }

    public boolean isBalancedOptimizationEnabled() {
        return getCurrentDomainState().isBalancedOptimizationEnabled();
    }

    public void setEnableBalancedOptimization(boolean enableBalancedOptimization) {
        getCurrentDomainState().setEnableBalancedOptimization(enableBalancedOptimization);
    }

    public List<ExecutionResult> getExecutionResults() {
        return getCurrentDomainState().getExecutionResults();
    }

    public void setExecutionResults(List<ExecutionResult> executionResults) {
        getCurrentDomainState().setExecutionResults(executionResults);
    }

    private boolean useGpu = true;

    public boolean isUseGpu() {
        return useGpu;
    }

    public void setUseGpu(boolean useGpu) {
        this.useGpu = useGpu;
        LogService.info("AppStateService", "GPU Acceleration Preference updated: " + useGpu);
    }

    private String selectedGpuName = "NVIDIA"; // Default to NVIDIA as requested

    public String getSelectedGpuName() {
        return selectedGpuName;
    }

    public void setSelectedGpuName(String selectedGpuName) {
        this.selectedGpuName = selectedGpuName;
        LogService.info("AppStateService", "Selected GPU updated to: " + selectedGpuName);
    }

    private boolean loggingEnabled = true; // Default ON to support reporting

    public boolean isLoggingEnabled() {
        return loggingEnabled;
    }

    public void setLoggingEnabled(boolean loggingEnabled) {
        this.loggingEnabled = loggingEnabled;
    }

    // --- WSL Configuration ---
    private boolean useWsl = true;
    private String wslDistributionName = "Ubuntu"; // Default

    public boolean isUseWsl() {
        return useWsl;
    }

    public void setUseWsl(boolean useWsl) {
        this.useWsl = useWsl;
        LogService.info("AppStateService", "WSL Mode Preference updated: " + useWsl);
    }

    public String getWslDistributionName() {
        return wslDistributionName;
    }

    public void setWslDistributionName(String wslDistributionName) {
        this.wslDistributionName = wslDistributionName;
    }

    // --- Hybrid Execution (CPU + GPU) ---
    private volatile boolean hybridExecutionEnabled = true;
    private volatile double cpuOffloadRatio = 0.1; // 10% CPU, 90% GPU

    public boolean isHybridExecutionEnabled() {
        return hybridExecutionEnabled;
    }

    public void setHybridExecutionEnabled(boolean hybridExecutionEnabled) {
        this.hybridExecutionEnabled = hybridExecutionEnabled;
        LogService.info("AppStateService", "Hybrid Execution Preference updated: " + hybridExecutionEnabled);
    }

    public double getCpuOffloadRatio() {
        return cpuOffloadRatio;
    }

    public void setCpuOffloadRatio(double cpuOffloadRatio) {
        this.cpuOffloadRatio = Math.max(0.0, Math.min(1.0, cpuOffloadRatio));
        LogService.info("AppStateService", "Hybrid Execution Ratio updated: " + this.cpuOffloadRatio);
    }

    public com.example.automl_prototype_1.model.PreprocessingConfig getPreprocessingConfig() {
        return getCurrentDomainState().getPreprocessingConfig();
    }

    public void setPreprocessingConfig(com.example.automl_prototype_1.model.PreprocessingConfig preprocessingConfig) {
        getCurrentDomainState().setPreprocessingConfig(preprocessingConfig);
    }

    // --- Advanced Mode Settings ---
    private boolean isAdvancedMode = false;
    private Integer manualTrialsOverride = null;
    private Integer manualIterationsOverride = null;
    private boolean forceGpu = false;

    public boolean isAdvancedMode() {
        return isAdvancedMode;
    }

    public void setAdvancedMode(boolean advancedMode) {
        isAdvancedMode = advancedMode;
        LogService.info("AppStateService", "Advanced Mode: " + advancedMode);
    }

    public Integer getManualTrialsOverride() {
        return manualTrialsOverride;
    }

    public void setManualTrialsOverride(Integer manualTrialsOverride) {
        this.manualTrialsOverride = manualTrialsOverride;
    }

    public Integer getManualIterationsOverride() {
        return manualIterationsOverride;
    }

    public void setManualIterationsOverride(Integer manualIterationsOverride) {
        this.manualIterationsOverride = manualIterationsOverride;
    }

    public boolean isForceGpu() {
        return forceGpu;
    }

    public void setForceGpu(boolean forceGpu) {
        this.forceGpu = forceGpu;
        LogService.info("AppStateService", "Force GPU: " + forceGpu);
    }

    /** Convenience alias: disables GPU and switches to CPU-only mode. */
    public void setForceGpuOff() {
        setForceGpu(false);
        setUseGpu(false);
        LogService.warn("AppStateService", "GPU disabled by self-correction loop (CUDA OOM). CPU-only mode active.");
    }

    // --- Pipeline Composition Settings (Phase 3) ---
    private boolean featureSelectionEnabled = true;

    public boolean isFeatureSelectionEnabled() {
        return featureSelectionEnabled;
    }

    public void setFeatureSelectionEnabled(boolean featureSelectionEnabled) {
        this.featureSelectionEnabled = featureSelectionEnabled;
        LogService.info("AppStateService", "Feature Selection Enabled: " + featureSelectionEnabled);
    }

    // --- Manual Feature Selection (Phase 4) ---
    private boolean manualFeatureSelectionEnabled = false;
    private java.util.List<Integer> manualFeatureIndices = new java.util.ArrayList<>();

    public boolean isManualFeatureSelectionEnabled() {
        return manualFeatureSelectionEnabled;
    }

    public void setManualFeatureSelectionEnabled(boolean manualFeatureSelectionEnabled) {
        this.manualFeatureSelectionEnabled = manualFeatureSelectionEnabled;
        LogService.info("AppStateService", "Manual Feature Selection Enabled: " + manualFeatureSelectionEnabled);
    }

    public java.util.List<Integer> getManualFeatureIndices() {
        return manualFeatureIndices;
    }

    public void setManualFeatureIndices(java.util.List<Integer> manualFeatureIndices) {
        this.manualFeatureIndices = manualFeatureIndices;
        LogService.info("AppStateService", "Manual Feature Indices Updated: "
                + (manualFeatureIndices != null ? manualFeatureIndices.size() : "null"));
    }

    // --- Algorithm Hyperparameters (Phase 2) ---
    // Outer Key: Algorithm Name (e.g., "GA", "GWO")
    // Inner Key: Param Name (e.g., "crossoverRate")
    private Map<String, Map<String, Double>> algorithmHyperparameters = new java.util.HashMap<>();

    {
        // Initialize Defaults
        Map<String, Double> gaParams = new java.util.HashMap<>();
        gaParams.put("crossoverRate", 0.8);
        gaParams.put("mutationRate", 0.05);
        gaParams.put("populationSize", 50.0); // Stored as double for uniformity
        algorithmHyperparameters.put("GA", gaParams);

        Map<String, Double> gwoParams = new java.util.HashMap<>();
        gwoParams.put("populationSize", 30.0);
        algorithmHyperparameters.put("GWO", gwoParams);
    }

    public Map<String, Map<String, Double>> getAlgorithmHyperparameters() {
        return algorithmHyperparameters;
    }

    public void setAlgorithmHyperparameter(String algo, String param, Double value) {
        algorithmHyperparameters.computeIfAbsent(algo, k -> new java.util.HashMap<>()).put(param, value);
        LogService.info("AppStateService", "Updated Hyperparameter [" + algo + "." + param + "] = " + value);
    }

    public Double getAlgorithmHyperparameter(String algo, String param) {
        if (algorithmHyperparameters.containsKey(algo)) {
            return algorithmHyperparameters.get(algo).get(param);
        }
        return null;
    }

    // --- String Params (Phase 4) ---
    private Map<String, Map<String, String>> algorithmConfigStrings = new java.util.HashMap<>();

    public void setAlgorithmConfigString(String algo, String param, String value) {
        algorithmConfigStrings.computeIfAbsent(algo, k -> new java.util.HashMap<>()).put(param, value);
        LogService.info("AppStateService", "Updated Config String [" + algo + "." + param + "] = " + value);
    }

    public String getAlgorithmConfigString(String algo, String param) {
        if (algorithmConfigStrings.containsKey(algo)) {
            return algorithmConfigStrings.get(algo).get(param);
        }
        return null; // or default
    }

    // --- Ollama Settings ---
    private boolean isOllamaEnabled = true;
    private String ollamaModelName = "deepseek-r1:8b";
    private String ollamaUrl = "http://localhost:11434";

    public boolean isOllamaEnabled() {
        return isOllamaEnabled;
    }

    public void setOllamaEnabled(boolean ollamaEnabled) {
        isOllamaEnabled = ollamaEnabled;
    }

    public String getOllamaModelName() {
        return ollamaModelName;
    }

    public void setOllamaModelName(String ollamaModelName) {
        this.ollamaModelName = ollamaModelName;
    }

    public String getOllamaUrl() {
        return ollamaUrl;
    }

    public void setOllamaUrl(String ollamaUrl) {
        this.ollamaUrl = ollamaUrl;
    }

    // --- Phase 2: UI Expansion Getters/Setters ---

    public DataType getSelectedDataType() {
        return selectedDataType;
    }

    public void setSelectedDataType(DataType selectedDataType) {
        this.selectedDataType = selectedDataType;
        LogService.info("AppState", "Data Type Selected: " + selectedDataType);
    }

    public DataQuality getSelectedDataQuality() {
        return selectedDataQuality;
    }

    public void setSelectedDataQuality(DataQuality selectedDataQuality) {
        this.selectedDataQuality = selectedDataQuality;
        LogService.info("AppState", "Data Quality Selected: " + selectedDataQuality);
    }

    public boolean isEnableQualityCheck() {
        return enableQualityCheck;
    }

    public void setEnableQualityCheck(boolean enableQualityCheck) {
        this.enableQualityCheck = enableQualityCheck;
        LogService.info("AppState", "Quality Check Enabled: " + enableQualityCheck);
    }

    public TrainingMode getSelectedTrainingMode() {
        return selectedTrainingMode;
    }

    public void setSelectedTrainingMode(TrainingMode selectedTrainingMode) {
        this.selectedTrainingMode = selectedTrainingMode;
        LogService.info("AppState", "Training Mode Selected: " + selectedTrainingMode);
    }

    // --- Dynamic Models (Phase 9) ---
    private Map<String, String> dynamicModels = new java.util.HashMap<>();

    public void registerDynamicModel(String name, String hfModelId) {
        dynamicModels.put(name, hfModelId);
        LogService.info("AppState", "Registered Dynamic Model: " + name + " -> " + hfModelId);
    }

    public Map<String, String> getDynamicModels() {
        return dynamicModels;
    }
}