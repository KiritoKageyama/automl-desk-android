package com.example.automl_prototype_1.service;

import com.example.automl_prototype_1.model.ExecutionResult;
import java.io.File;
import java.util.List;

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
}