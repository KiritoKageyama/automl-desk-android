package com.example.automl_prototype_1.model;

import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

/**
 * Represents a loaded dataset using efficient primitive arrays.
 * Stores features as double[][] and labels as int[] (if a target is specified).
 */
public class Dataset {
    private final File sourceFile;
    private final List<String> headers;

    // Core data storage - Primitive arrays for efficiency
    private final double[][] features;
    private final int[] labels; // Can be null if no target is set yet or for inference
    private String targetColumnName;

    public enum DataType {
        TABULAR,
        IMAGE,
        TEXT,
        TIME_SERIES
    }

    private final DataType dataType;

    /**
     * Constructor for the Dataset.
     * 
     * @param sourceFile       The original source file (optional).
     * @param headers          The list of header names.
     * @param features         The feature matrix (rows x columns).
     * @param labels           The label vector (optional, can be null).
     * @param targetColumnName The name of the target column (optional).
     */
    public Dataset(File sourceFile, List<String> headers, double[][] features, int[] labels, String targetColumnName) {
        this.sourceFile = sourceFile;
        this.headers = (headers != null) ? new ArrayList<>(headers) : new ArrayList<>();
        this.features = features;
        this.labels = labels;
        this.targetColumnName = targetColumnName;
        this.dataType = DataType.TABULAR; // Default
    }

    /**
     * Compatibility constructor for legacy code using List of Maps.
     * Converts List<Map> to double[][] features.
     *
     * @param sourceFile The source file.
     * @param headers    The list of headers.
     * @param records    The list of records as maps.
     */
    public Dataset(File sourceFile, List<String> headers, List<Map<String, String>> records) {
        this.sourceFile = sourceFile;
        this.headers = (headers != null) ? new ArrayList<>(headers) : new ArrayList<>();
        this.targetColumnName = null;
        this.labels = null;
        this.dataType = DataType.TABULAR; // Default for legacy constructor

        if (records == null || records.isEmpty()) {
            this.features = new double[0][0];
        } else {
            int numRows = records.size();
            int numCols = this.headers.size();
            this.features = new double[numRows][numCols];

            for (int i = 0; i < numRows; i++) {
                Map<String, String> record = records.get(i);
                for (int j = 0; j < numCols; j++) {
                    String header = this.headers.get(j);
                    String valStr = record.get(header);
                    try {
                        this.features[i][j] = Double.parseDouble(valStr);
                    } catch (NumberFormatException | NullPointerException e) {
                        this.features[i][j] = Double.NaN; // Use NaN for missing/invalid to allow proper imputation
                    }
                }
            }
        }
    }

    // --- Getters ---

    public File getSourceFile() {
        return sourceFile;
    }

    public List<String> getHeaders() {
        return Collections.unmodifiableList(headers);
    }

    public int getRecordCount() {
        return features != null ? features.length : 0;
    }

    public int getFeatureCount() {
        return features != null && features.length > 0 ? features[0].length : 0;
    }

    public double[][] getFeatures() {
        return features;
    }

    public int[] getLabels() {
        return labels;
    }

    public DataType getDataType() {
        return dataType;
    }

    public String getTargetColumnName() {
        return targetColumnName;
    }

    public void setTargetColumnName(String targetColumnName) {
        this.targetColumnName = targetColumnName;
    }

    /**
     * Returns the feature matrix.
     * If the dataset was created with a target column separated, this returns the
     * features as is.
     * If not, it might return all columns.
     * 
     * @return double[][] representing the features.
     */
    public double[][] getFeatureMatrix() {
        return features;
    }

    /**
     * Returns the label vector.
     * 
     * @return int[] representing the labels, or null if not set.
     */
    public int[] getLabelVector() {
        return labels;
    }

    @Override
    public String toString() {
        String sourceFileName = (sourceFile != null) ? sourceFile.getName() : "Stream/Reader Input";
        return "Dataset{" +
                "source=" + sourceFileName +
                ", records=" + getRecordCount() +
                ", features=" + getFeatureCount() +
                ", target='" + targetColumnName + '\'' +
                '}';
    }

    /**
     * Extracts the feature matrix by excluding the specified target column.
     * This creates a copy of the data.
     *
     * @param targetColumnName The name of the column to be treated as the label.
     * @return A double[][] array representing the feature matrix.
     */
    public double[][] getFeatureMatrix(String targetColumnName) {
        int targetIndex = headers.indexOf(targetColumnName);
        if (targetIndex == -1) {
            // If target not found, we MUST NOT return all features as that would cause data
            // leakage.
            // Throwing an exception is safer to alert the caller.
            throw new IllegalArgumentException(
                    "Target column '" + targetColumnName + "' not found in dataset headers: " + headers);
        }

        int numRows = features.length;
        int numCols = features[0].length;
        double[][] X = new double[numRows][numCols - 1];

        for (int i = 0; i < numRows; i++) {
            int k = 0;
            for (int j = 0; j < numCols; j++) {
                if (j == targetIndex)
                    continue;
                X[i][k++] = features[i][j];
            }
        }
        return X;
    }

    /**
     * Extracts the label vector for the specified target column.
     *
     * @param targetColumnName The name of the column to be treated as the label.
     * @return An int[] array representing the label vector.
     */
    public int[] getLabelVector(String targetColumnName) {
        int targetIndex = headers.indexOf(targetColumnName);
        if (targetIndex == -1) {
            return new int[0];
        }

        int numRows = features.length;
        int[] y = new int[numRows];

        for (int i = 0; i < numRows; i++) {
            y[i] = (int) features[i][targetIndex];
        }
        return y;
    }

    /**
     * Standardizes the features (Z-score scaling) in-place.
     * x' = (x - mean) / std_dev
     * Handles constant columns by setting them to 0.
     */
    /**
     * Standardizes the features (Z-score scaling) in-place.
     * x' = (x - mean) / std_dev
     * Handles constant columns by setting them to 0.
     * Skips the target column if specified.
     */
    public void standardize() {
        standardize(this.targetColumnName);
    }

    public void standardize(String excludeColumnName) {
        if (features == null || features.length == 0)
            return;

        int numRows = features.length;
        int numCols = features[0].length;
        int excludeIndex = (excludeColumnName != null) ? headers.indexOf(excludeColumnName) : -1;

        for (int j = 0; j < numCols; j++) {
            if (j == excludeIndex)
                continue; // Skip target column

            // 1. Calculate Mean
            double sum = 0.0;
            for (int i = 0; i < numRows; i++) {
                sum += features[i][j];
            }
            double mean = sum / numRows;

            // 2. Calculate Std Dev
            double sumSqDiff = 0.0;
            for (int i = 0; i < numRows; i++) {
                double diff = features[i][j] - mean;
                sumSqDiff += diff * diff;
            }
            double stdDev = Math.sqrt(sumSqDiff / numRows);

            // 3. Apply Transformation
            for (int i = 0; i < numRows; i++) {
                if (stdDev > 1e-9) {
                    features[i][j] = (features[i][j] - mean) / stdDev;
                } else {
                    features[i][j] = 0.0; // Constant column
                }
            }
        }
        System.out.println("Dataset standardized (Z-Score scaling applied, skipped: " + excludeColumnName + ").");
    }

    /**
     * Normalizes the features (Min-Max scaling) in-place.
     * x' = (x - min) / (max - min)
     * Rescales features to range [0, 1].
     */
    public void normalize() {
        normalize(this.targetColumnName);
    }

    public void normalize(String excludeColumnName) {
        if (features == null || features.length == 0)
            return;

        int numRows = features.length;
        int numCols = features[0].length;
        int excludeIndex = (excludeColumnName != null) ? headers.indexOf(excludeColumnName) : -1;

        for (int j = 0; j < numCols; j++) {
            if (j == excludeIndex)
                continue; // Skip target column

            // 1. Find Min/Max
            double min = Double.MAX_VALUE;
            double max = -Double.MAX_VALUE;
            for (int i = 0; i < numRows; i++) {
                double val = features[i][j];
                if (val < min)
                    min = val;
                if (val > max)
                    max = val;
            }

            // 2. Apply Transformation
            double range = max - min;
            for (int i = 0; i < numRows; i++) {
                if (range > 1e-9) {
                    features[i][j] = (features[i][j] - min) / range;
                } else {
                    features[i][j] = 0.0; // Constant column
                }
            }
        }
        System.out.println("Dataset normalized (Min-Max scaling applied, skipped: " + excludeColumnName + ").");
    }

    /**
     * Imputes missing values (NaN) with the mean of the column.
     * Skips the target column if specified.
     */
    public void impute() {
        impute(this.targetColumnName);
    }

    public void impute(String excludeColumnName) {
        if (features == null || features.length == 0)
            return;

        int numRows = features.length;
        int numCols = features[0].length;
        int excludeIndex = (excludeColumnName != null) ? headers.indexOf(excludeColumnName) : -1;

        for (int j = 0; j < numCols; j++) {
            if (j == excludeIndex)
                continue;

            // 1. Calculate Mean (ignoring NaNs)
            double sum = 0.0;
            int count = 0;
            for (int i = 0; i < numRows; i++) {
                if (!Double.isNaN(features[i][j])) {
                    sum += features[i][j];
                    count++;
                }
            }

            double mean = (count > 0) ? sum / count : 0.0; // Default to 0 if all NaN

            // 2. Replace NaNs
            for (int i = 0; i < numRows; i++) {
                if (Double.isNaN(features[i][j])) {
                    features[i][j] = mean;
                }
            }
        }
        System.out.println("Dataset imputed (Mean substitution applied, skipped: " + excludeColumnName + ").");
    }

    /**
     * Compatibility method to retrieve records as a List of Maps.
     * This reconstructs the List<Map> structure from the primitive arrays.
     * WARNING: This is expensive and should be avoided in performance-critical
     * code.
     *
     * @return List of Maps where keys are headers and values are strings.
     */
    public List<Map<String, String>> getRecords() {
        List<Map<String, String>> records = new ArrayList<>();
        if (features == null)
            return records;

        int numRows = features.length;
        int numCols = features[0].length;

        for (int i = 0; i < numRows; i++) {
            Map<String, String> record = new java.util.HashMap<>();
            for (int j = 0; j < numCols; j++) {
                String header = (j < headers.size()) ? headers.get(j) : "Col" + j;
                double val = features[i][j];
                // Convert back to string. Note: precision loss possible.
                record.put(header, String.valueOf(val));
            }
            records.add(record);
        }
        return records;
    }

    /**
     * Get the first N rows of features.
     * 
     * @param limit Number of rows to return.
     * @return double[][] subset.
     */
    public double[][] getFeatures(int limit) {
        if (features == null)
            return new double[0][0];
        int rows = Math.min(limit, features.length);
        int cols = features[0].length;
        double[][] subset = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            System.arraycopy(features[i], 0, subset[i], 0, cols);
        }
        return subset;
    }

    /**
     * Get the number of unique labels in the specified target column.
     * 
     * @param targetCol Name of the target column.
     * @return count of unique values.
     */
    public int getUniqueLabelCount(String targetCol) {
        int[] y = getLabelVector(targetCol);
        if (y.length == 0)
            return 0;

        java.util.Set<Integer> unique = new java.util.HashSet<>();
        for (int val : y)
            unique.add(val);
        return unique.size();
    }
}