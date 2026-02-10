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

    @Override
    public String toString() {
        return "PreprocessingConfig{" +
                "removeIds=" + autoRemoveIdColumns +
                ", impute=" + imputeMissingValues +
                ", outliers=" + removeOutliers +
                ", scaling='" + scalingStrategy + '\'' +
                '}';
    }
}
