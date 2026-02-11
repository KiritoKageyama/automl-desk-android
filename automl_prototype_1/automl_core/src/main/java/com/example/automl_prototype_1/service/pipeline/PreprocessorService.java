// in .../service/PreprocessorService.java
package com.example.automl_prototype_1.service.pipeline;

import com.example.automl_prototype_1.service.LogService;

import com.example.automl_prototype_1.model.Dataset;

import java.util.*;
import java.util.stream.Collectors;

/**
 * A service for performing a pipeline of general-purpose preprocessing steps.
 */
public class PreprocessorService {

    /**
     * The main entry point for preprocessing. Runs a pipeline of cleaning and
     * transformation steps.
     * 
     * @param rawDataset       The unprocessed dataset.
     * @param targetColumnName The name of the column to be treated as the target
     *                         label.
     * @return A new, cleaned, and normalized Dataset object.
     */
    public Dataset runPreprocessingPipeline(Dataset rawDataset, String targetColumnName,
            com.example.automl_prototype_1.model.PreprocessingConfig config) {
        LogService.info("Preprocessor", "--- Starting Preprocessing Pipeline with Config: " + config + " ---");

        Dataset currentDataset = rawDataset;

        if (config.isAutoRemoveIdColumns()) {
            currentDataset = removeIdentifierColumns(currentDataset, targetColumnName);
        } else {
            LogService.info("Preprocessor", "Skipping ID Column Removal (User Request).");
        }

        if (config.isImputeMissingValues()) {
            // Impute logic
            LogService.info("Preprocessor", "Applying Mean Imputation...");
            currentDataset.impute(currentDataset.getTargetColumnName());
        } else {
            LogService.info("Preprocessor", "Skipping Imputation (User Request).");
        }

        currentDataset = convertTargetToNumeric(currentDataset, targetColumnName);

        // Ensure standardizeTargetColumn is always called as it sets up the "Class"
        // column required by classifiers
        currentDataset = standardizeTargetColumn(currentDataset, targetColumnName);

        // --- Phase 5: Advanced Preprocessing ---

        // 1. Outlier Removal (IQR)
        if (config.isRemoveOutliers()) {
            currentDataset = removeOutliersIQR(currentDataset, targetColumnName);
        }

        // 2. Feature Transformation
        currentDataset = applyTransformation(currentDataset, config.getTransformationStrategy(), targetColumnName);

        // 3. Feature Creation
        currentDataset = applyFeatureCreation(currentDataset, config.getFeatureCreationStrategy(), targetColumnName);

        // --- End Phase 5 ---

        // --- THIS IS THE DYNAMIC PART (Scaling) ---
        Dataset finalDataset = switch (config.getScalingStrategy()) {
            case "Standardization" -> standardize(currentDataset);
            case "Normalization" -> normalize(currentDataset);
            default -> { // "None" or any other value
                LogService.info("Preprocessor", "Skipping feature scaling.");
                yield currentDataset;
            }
        };

        LogService.info("Preprocessor", "--- Preprocessing Pipeline Finished ---");
        return finalDataset;
    }

    // --- Phase 5 Implementation ---

    private Dataset removeOutliersIQR(Dataset dataset, String targetCol) {
        LogService.info("Preprocessor", "Applying IQR Outlier Removal...");
        double[][] X = dataset.getFeatures();
        int[] y = dataset.getLabelVector(); // Assuming labels exist
        if (X == null || X.length == 0)
            return dataset;

        int numRows = X.length;
        int numCols = X[0].length;
        boolean[] keepRow = new boolean[numRows];
        Arrays.fill(keepRow, true);

        // Calculate bounds for each column
        for (int j = 0; j < numCols; j++) {
            List<Double> values = new ArrayList<>();
            for (int i = 0; i < numRows; i++)
                if (!Double.isNaN(X[i][j]))
                    values.add(X[i][j]);
            Collections.sort(values);

            if (values.isEmpty())
                continue;

            double q1 = values.get(values.size() / 4);
            double q3 = values.get(values.size() * 3 / 4);
            double iqr = q3 - q1;
            double lower = q1 - 1.5 * iqr;
            double upper = q3 + 1.5 * iqr;

            for (int i = 0; i < numRows; i++) {
                if (X[i][j] < lower || X[i][j] > upper) {
                    keepRow[i] = false; // Mark row for deletion
                }
            }
        }

        // Rebuild Dataset
        List<double[]> newXList = new ArrayList<>();
        List<Integer> newYList = new ArrayList<>();
        int removedCount = 0;

        for (int i = 0; i < numRows; i++) {
            if (keepRow[i]) {
                newXList.add(X[i]);
                if (y != null)
                    newYList.add(y[i]);
            } else {
                removedCount++;
            }
        }

        if (removedCount == 0)
            return dataset;

        LogService.info("Preprocessor", "Removed " + removedCount + " outlier rows.");

        double[][] newX = newXList.toArray(new double[0][]);
        int[] newY = (y != null) ? newYList.stream().mapToInt(i -> i).toArray() : null;

        return new Dataset(dataset.getSourceFile(), dataset.getHeaders(), newX, newY, dataset.getTargetColumnName());
    }

    private Dataset applyTransformation(Dataset dataset, String strategy, String targetCol) {
        if (strategy == null || strategy.equalsIgnoreCase("None"))
            return dataset;

        LogService.info("Preprocessor", "Applying Transformation Strategy: " + strategy);
        double[][] X = dataset.getFeatures();
        int numRows = X.length;
        int numCols = X[0].length;

        // Create deep copy to avoid mutating original if needed (Dataset is effectively
        // immutable struct logic here though)
        // But features array is shared. Let's modify in-place for efficiency as we are
        // in a pipeline.

        for (int j = 0; j < numCols; j++) {
            // Apply logic column-wise
            // Simple skew check for Auto
            boolean apply = true;
            if (strategy.equalsIgnoreCase("Auto-Skew")) {
                apply = calculateSkewness(X, j) > 1.0; // Apply Log if highly skewed
            }

            if (!apply)
                continue;

            for (int i = 0; i < numRows; i++) {
                double val = X[i][j];
                if (Double.isNaN(val))
                    continue;

                if (strategy.equalsIgnoreCase("Log") || (strategy.equalsIgnoreCase("Auto-Skew") && apply)) {
                    // Log(x+1) to handle zeros. If negative, shift.
                    // Simple approach: Log(x - min + 1) if min < 0
                    if (val <= 0)
                        val = 0.001; // Naive clamp
                    X[i][j] = Math.log(val + 1.0);
                } else if (strategy.equalsIgnoreCase("Sqrt")) {
                    if (val < 0)
                        val = 0;
                    X[i][j] = Math.sqrt(val);
                }
            }
        }
        return dataset; // In-place modification
    }

    private double calculateSkewness(double[][] X, int col) {
        // Simple Pearson's moment coefficient of skewness
        int n = X.length;
        double sum = 0, sumSq = 0, sumCub = 0;
        for (int i = 0; i < n; i++) {
            double v = X[i][col];
            sum += v;
            sumSq += v * v;
            sumCub += v * v * v;
        }
        double mean = sum / n;
        double std = Math.sqrt((sumSq / n) - mean * mean);
        if (std < 1e-9)
            return 0.0;

        double m3 = (sumCub / n) - 3 * mean * (sumSq / n) + 2 * mean * mean * mean;
        return m3 / (std * std * std);
    }

    private Dataset applyFeatureCreation(Dataset dataset, String strategy, String targetCol) {
        if (strategy == null || strategy.equalsIgnoreCase("None"))
            return dataset;
        LogService.info("Preprocessor", "Applying Feature Creation: " + strategy);

        // Interaction Terms
        if (strategy.equalsIgnoreCase("Interaction") || strategy.equalsIgnoreCase("Polynomial")) {
            // Add x_i * x_j for all pairs (Limiting to top 10 features to avoid explosion?)
            // For now, let's just do it for first 10 columns as a heuristic to disable
            // explosion
            double[][] X = dataset.getFeatures(); // N x D
            List<String> headers = new ArrayList<>(dataset.getHeaders());
            int D = headers.size();
            int limit = Math.min(D, 10); // Safety limit

            int newCount = (limit * (limit - 1)) / 2;
            int totalCols = D + newCount;

            double[][] newX = new double[X.length][totalCols];
            List<String> newHeaders = new ArrayList<>(headers);

            // Copy original
            for (int i = 0; i < X.length; i++) {
                System.arraycopy(X[i], 0, newX[i], 0, D);
            }

            // Create Interactions
            int colIdx = D;
            for (int i = 0; i < limit; i++) {
                for (int j = i + 1; j < limit; j++) {
                    newHeaders.add(headers.get(i) + "*" + headers.get(j));
                    for (int r = 0; r < X.length; r++) {
                        newX[r][colIdx] = X[r][i] * X[r][j];
                    }
                    colIdx++;
                }
            }

            return new Dataset(dataset.getSourceFile(), newHeaders, newX, dataset.getLabelVector(), targetCol);
        }

        return dataset;
    }

    private Dataset removeIdentifierColumns(Dataset dataset, String targetColumnName) {
        List<String> headers = new ArrayList<>(dataset.getHeaders());
        List<String> columnsToRemove = new ArrayList<>();

        for (String header : headers) {
            // Skip the target column, we never want to remove it
            if (header.equalsIgnoreCase(targetColumnName)) {
                continue;
            }
            // Heuristic: If a column has all unique values and there are many rows, it's
            // likely an ID.
            long totalCount = dataset.getRecordCount();
            long distinctCount = dataset.getRecords().stream().map(row -> row.get(header)).distinct().count();
            if (totalCount > 20 && totalCount == distinctCount) {
                columnsToRemove.add(header);
            }
        }

        if (columnsToRemove.isEmpty())
            return dataset;

        LogService.info("Preprocessor", "Removing likely ID columns: " + columnsToRemove);
        headers.removeAll(columnsToRemove);

        List<Map<String, String>> newRecords = dataset.getRecords().stream()
                .map(originalRecord -> {
                    Map<String, String> newRecord = new HashMap<>(originalRecord);
                    columnsToRemove.forEach(newRecord::remove);
                    return newRecord;
                }).collect(Collectors.toList());

        return new Dataset(dataset.getSourceFile(), headers, newRecords);
    }

    private Dataset convertTargetToNumeric(Dataset dataset, String targetColumnName) {
        Set<String> uniqueLabels = dataset.getRecords().stream()
                .map(row -> row.get(targetColumnName))
                .filter(Objects::nonNull)
                .collect(Collectors.toSet());

        // Handle both binary and multi-class cases
        if (uniqueLabels.size() >= 2) {
            LogService.info("Preprocessor",
                    "Detected categorical target '" + targetColumnName + "' with " + uniqueLabels.size() + " labels.");

            List<String> sortedLabels = new ArrayList<>(uniqueLabels);
            Collections.sort(sortedLabels); // Ensure deterministic ordering (0=ClassA, 1=ClassB...)

            Map<String, String> labelMap = new HashMap<>();
            for (int i = 0; i < sortedLabels.size(); i++) {
                labelMap.put(sortedLabels.get(i), String.valueOf(i));
            }

            LogService.info("Preprocessor", "Applying Label Encoding: " + labelMap);

            List<Map<String, String>> newRecords = dataset.getRecords().stream()
                    .map(originalRecord -> {
                        Map<String, String> newRecord = new HashMap<>(originalRecord);
                        String originalLabel = newRecord.get(targetColumnName);
                        // Map using the lookup, default to "0" if null (shouldn't happen with filter)
                        String numericLabel = labelMap.getOrDefault(originalLabel, "0");
                        newRecord.put(targetColumnName, numericLabel);
                        return newRecord;
                    }).collect(Collectors.toList());

            return new Dataset(dataset.getSourceFile(), dataset.getHeaders(), newRecords);
        }

        return dataset;
    }

    private Dataset standardizeTargetColumn(Dataset dataset, String originalTargetName) {
        final String newTargetName = "Class";
        if (originalTargetName.equals(newTargetName))
            return dataset;

        List<String> headers = new ArrayList<>(dataset.getHeaders());
        headers.remove(originalTargetName);
        headers.add(newTargetName);

        List<Map<String, String>> newRecords = dataset.getRecords().stream()
                .map(originalRecord -> {
                    Map<String, String> newRecord = new HashMap<>(originalRecord);
                    String value = newRecord.remove(originalTargetName);
                    newRecord.put(newTargetName, value);
                    return newRecord;
                }).collect(Collectors.toList());

        LogService.info("Preprocessor",
                "Standardized target column '" + originalTargetName + "' to '" + newTargetName + "'.");

        // Use the List<Map> constructor to correctly re-order features based on the new
        // headers
        Dataset temp = new Dataset(dataset.getSourceFile(), headers, newRecords);

        // Wrap it to set the target column name
        return new Dataset(temp.getSourceFile(), temp.getHeaders(), temp.getFeatures(), temp.getLabels(),
                newTargetName);
    }

    // Helper to rebuild dataset if needed, but the above is safer if we trust
    // features/labels are in sync
    // However, the previous code constructed it from 'newRecords' using the List
    // constructor which drops the target
    // So we must stick to the List constructor BUT we need a way to set the target.
    // Dataset doesn't have a setter for targetColumnName (it's final).
    // So we must use a constructor that accepts it, but we only have Map records.

    // Correction: Dataset has a constructor: Dataset(source, headers, features,
    // labels, targetName)
    // But we are transforming records in List<Map> form here.
    // The previous implementation used `new Dataset(dataset.getSourceFile(),
    // headers, newRecords);`
    // We need to use:
    // `new Dataset(dataset.getSourceFile(), headers, newRecords, newTargetName)` ->
    // DOES NOT EXIST.

    // OPTION: We should modify Dataset to add that constructor OR use the existing
    // one and convert maps.
    // But wait, Dataset(File, List, List<Map>) parses the maps into
    // features/labels.
    // So we can just add a constructor to Dataset that takes `targetColumnName`
    // with List<Map>.

    // ALTERNATIVELY (Easier here):
    // Parse the maps ourselves or let the current constructor do it, then RE-WRAP
    // it.
    /*
     * Dataset temp = new Dataset(dataset.getSourceFile(), headers, newRecords);
     * return new Dataset(temp.getSourceFile(), temp.getHeaders(),
     * temp.getFeatures(), temp.getLabels(), newTargetName);
     */

    // Let's do that. It's safe and effective.

    public Dataset normalize(Dataset dataset) {
        LogService.info("Preprocessor", "Normalizing dataset via efficient Dataset method...");
        // Efficient in-place normalization similar to standardization
        dataset.normalize();
        return dataset;
    }

    public Dataset standardize(Dataset dataset) {
        LogService.info("Preprocessor", "Standardizing dataset (Z-score) via efficient Dataset method...");
        // Use the efficient in-place method on the Dataset object
        // Note: dataset.standardize() modifies the dataset in-place.
        dataset.standardize();
        return dataset;
    }

    /**
     * Validates if a target column appears to be categorical and suitable for
     * classification.
     * 
     * @param dataset          The dataset to check.
     * @param targetColumnName The name of the column to validate.
     * @return true if the column is likely a class label, false otherwise.
     */
    public boolean isTargetColumnSuitableForClassification(Dataset dataset, String targetColumnName) {
        if (dataset == null || targetColumnName == null || dataset.getRecords().isEmpty()) {
            return false; // Cannot validate empty or invalid input
        }

        // Count the number of unique, non-null values in the target column
        long uniqueValueCount = dataset.getRecords().stream()
                .map(row -> row.get(targetColumnName))
                .filter(Objects::nonNull)
                .distinct()
                .count();

        // Heuristic: A classification target should not have too many unique values.
        // And it must have more than one class to be classifiable.
        // We'll set a reasonable threshold, e.g., 25.
        final int MAX_CLASSES_THRESHOLD = 25;

        boolean isSuitable = uniqueValueCount > 1 && uniqueValueCount <= MAX_CLASSES_THRESHOLD;

        if (!isSuitable) {
            LogService.error("PreprocessorService",
                    "VALIDATION FAILED: Target column '" + targetColumnName + "' has " + uniqueValueCount +
                            " unique values. This is not suitable for classification (Threshold is <= "
                            + MAX_CLASSES_THRESHOLD
                            + ").");
        }

        return isSuitable;
    }
}