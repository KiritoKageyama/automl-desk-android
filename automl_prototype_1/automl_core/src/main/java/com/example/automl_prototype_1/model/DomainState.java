package com.example.automl_prototype_1.model;

import com.example.automl_prototype_1.service.IClassifier;
import java.io.File;
import java.util.ArrayList;
import java.util.List;

/**
 * Holds the specific state for a single Workspace Domain.
 * Ensures that jumping between LLM and ML keeps configurations independent.
 */
public class DomainState {
    private File selectedDatasetFile;
    private List<String> datasetHeaders;
    private String targetColumnName;
    private List<String> selectedAlgorithmNames = new ArrayList<>();
    private String selectedBaseModelName;
    private IClassifier selectedClassifier;
    private String scalingStrategy = "Standardization";
    private int kFolds = 5;
    private boolean isTuningEnabled = false;
    private boolean enableStandardization = true;
    private boolean enableBalancedOptimization = true;
    private List<ExecutionResult> executionResults = new ArrayList<>();
    private PreprocessingConfig preprocessingConfig;
    private Dataset currentDataset;
    private ExecutionResult lastExecutionResult;

    public DomainState() {
        this.preprocessingConfig = new PreprocessingConfig();
    }

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

    public void setTuningEnabled(boolean isTuningEnabled) {
        this.isTuningEnabled = isTuningEnabled;
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

    public PreprocessingConfig getPreprocessingConfig() {
        return preprocessingConfig;
    }

    public void setPreprocessingConfig(PreprocessingConfig preprocessingConfig) {
        this.preprocessingConfig = preprocessingConfig;
    }

    public Dataset getCurrentDataset() {
        return currentDataset;
    }

    public void setCurrentDataset(Dataset currentDataset) {
        this.currentDataset = currentDataset;
    }

    public ExecutionResult getLastExecutionResult() {
        return lastExecutionResult;
    }

    public void setLastExecutionResult(ExecutionResult lastExecutionResult) {
        this.lastExecutionResult = lastExecutionResult;
    }
}
