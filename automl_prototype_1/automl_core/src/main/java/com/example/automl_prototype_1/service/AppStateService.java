package com.example.automl_prototype_1.service;

import com.example.automl_prototype_1.model.Dataset;
import com.example.automl_prototype_1.model.ExecutionResult;
import java.io.File;
import java.util.List;
import java.util.Map;

public class AppStateService {

    private static AppStateService instance;

    private File selectedDatasetFile;
    private List<String> datasetHeaders;
    private String targetColumnName;
    private List<String> selectedAlgorithmNames;
    private String selectedBaseModelName;
    private IClassifier selectedClassifier;
    private String scalingStrategy = "Standardization";
    private int kFolds = 5;
    private boolean isTuningEnabled = false;
    private boolean enableStandardization = true; // Default ON
    private boolean enableBalancedOptimization = true; // Default ON
    private List<ExecutionResult> executionResults;
    private com.example.automl_prototype_1.model.PreprocessingConfig preprocessingConfig = new com.example.automl_prototype_1.model.PreprocessingConfig(); // Default
                                                                                                                                                           // config
    private Dataset currentDataset; // For synthetic or in-memory datasets

    private AppStateService() {
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

    public File getSelectedDatasetFile() {
        return selectedDatasetFile;
    }

    public void setSelectedDatasetFile(File selectedDatasetFile) {
        this.selectedDatasetFile = selectedDatasetFile;
    }

    public List<String> getDatasetHeaders() {
        return datasetHeaders;
    }

    public void setDatasetHeaders(List<String> datasetHeaders) {
        this.datasetHeaders = datasetHeaders;
    }

    public Dataset getCurrentDataset() {
        return currentDataset;
    }

    public void setCurrentDataset(Dataset currentDataset) {
        this.currentDataset = currentDataset;
        LogService.info("AppStateService", "Current dataset set: " +
                (currentDataset != null ? currentDataset.getHeaders().size() + " columns" : "null"));
    }

    public String getTargetColumnName() {
        return targetColumnName;
    }

    public void setTargetColumnName(String targetColumnName) {
        this.targetColumnName = targetColumnName;
    }

    public List<String> getSelectedAlgorithmNames() {
        return selectedAlgorithmNames;
    }

    public void setSelectedAlgorithmNames(List<String> selectedAlgorithmNames) {
        this.selectedAlgorithmNames = selectedAlgorithmNames;
    }

    public String getSelectedBaseModelName() {
        return selectedBaseModelName;
    }

    public void setSelectedBaseModelName(String selectedBaseModelName) {
        this.selectedBaseModelName = selectedBaseModelName;
    }

    public IClassifier getSelectedClassifier() {
        return selectedClassifier;
    }

    public void setSelectedClassifier(IClassifier selectedClassifier) {
        this.selectedClassifier = selectedClassifier;
    }

    public String getScalingStrategy() {
        return scalingStrategy;
    }

    public void setScalingStrategy(String scalingStrategy) {
        this.scalingStrategy = scalingStrategy;
    }

    public int getKFolds() {
        return kFolds;
    }

    public void setKFolds(int kFolds) {
        this.kFolds = kFolds;
    }

    public boolean isTuningEnabled() {
        return isTuningEnabled;
    }

    public void setTuningEnabled(boolean tuningEnabled) {
        isTuningEnabled = tuningEnabled;
    }

    public boolean isStandardizationEnabled() {
        return enableStandardization;
    }

    public void setEnableStandardization(boolean enableStandardization) {
        this.enableStandardization = enableStandardization;
    }

    public boolean isBalancedOptimizationEnabled() {
        return enableBalancedOptimization;
    }

    public void setEnableBalancedOptimization(boolean enableBalancedOptimization) {
        this.enableBalancedOptimization = enableBalancedOptimization;
    }

    public List<ExecutionResult> getExecutionResults() {
        return executionResults;
    }

    public void setExecutionResults(List<ExecutionResult> executionResults) {
        this.executionResults = executionResults;
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
        return preprocessingConfig;
    }

    public void setPreprocessingConfig(com.example.automl_prototype_1.model.PreprocessingConfig preprocessingConfig) {
        this.preprocessingConfig = preprocessingConfig;
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

    // --- Pipeline Composition Settings (Phase 3) ---
    private boolean featureSelectionEnabled = true;

    public boolean isFeatureSelectionEnabled() {
        return featureSelectionEnabled;
    }

    public void setFeatureSelectionEnabled(boolean featureSelectionEnabled) {
        this.featureSelectionEnabled = featureSelectionEnabled;
        LogService.info("AppStateService", "Feature Selection Enabled: " + featureSelectionEnabled);
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
}