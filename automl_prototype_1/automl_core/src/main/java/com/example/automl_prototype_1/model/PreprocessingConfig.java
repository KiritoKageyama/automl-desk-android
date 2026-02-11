package com.example.automl_prototype_1.model;

/**
 * Configuration options for the Preprocessor Service.
 * Allows the user to toggle specific data cleaning steps.
 */
public class PreprocessingConfig {

    private boolean autoRemoveIdColumns = true;
    private boolean imputeMissingValues = true;
    private boolean removeOutliers = false; // Default off as it can be destructive
    private String scalingStrategy = "Standardization"; // Options: "Standardization", "Normalization", "None"

    // Phase 5: Advanced Preprocessing
    private String transformationStrategy = "None"; // Options: "None", "Log", "Sqrt", "Box-Cox", "Auto-Skew"
    private String featureCreationStrategy = "None"; // Options: "None", "Polynomial", "Interaction", "Agentic"

    public PreprocessingConfig() {
    }

    public PreprocessingConfig(boolean autoRemoveIdColumns, boolean imputeMissingValues, boolean removeOutliers,
            String scalingStrategy) {
        this.autoRemoveIdColumns = autoRemoveIdColumns;
        this.imputeMissingValues = imputeMissingValues;
        this.removeOutliers = removeOutliers;
        this.scalingStrategy = scalingStrategy;
    }

    public boolean isAutoRemoveIdColumns() {
        return autoRemoveIdColumns;
    }

    public void setAutoRemoveIdColumns(boolean autoRemoveIdColumns) {
        this.autoRemoveIdColumns = autoRemoveIdColumns;
    }

    public boolean isImputeMissingValues() {
        return imputeMissingValues;
    }

    public void setImputeMissingValues(boolean imputeMissingValues) {
        this.imputeMissingValues = imputeMissingValues;
    }

    public boolean isRemoveOutliers() {
        return removeOutliers;
    }

    public void setRemoveOutliers(boolean removeOutliers) {
        this.removeOutliers = removeOutliers;
    }

    public String getScalingStrategy() {
        return scalingStrategy;
    }

    public void setScalingStrategy(String scalingStrategy) {
        this.scalingStrategy = scalingStrategy;
    }

    public String getTransformationStrategy() {
        return transformationStrategy;
    }

    public void setTransformationStrategy(String transformationStrategy) {
        this.transformationStrategy = transformationStrategy;
    }

    public String getFeatureCreationStrategy() {
        return featureCreationStrategy;
    }

    public void setFeatureCreationStrategy(String featureCreationStrategy) {
        this.featureCreationStrategy = featureCreationStrategy;
    }

    @Override
    public String toString() {
        return "PreprocessingConfig{" +
                "removeIds=" + autoRemoveIdColumns +
                ", impute=" + imputeMissingValues +
                ", outliers=" + removeOutliers +
                ", scaling='" + scalingStrategy + '\'' +
                ", transform='" + transformationStrategy + '\'' +
                ", creation='" + featureCreationStrategy + '\'' +
                '}';
    }
}
