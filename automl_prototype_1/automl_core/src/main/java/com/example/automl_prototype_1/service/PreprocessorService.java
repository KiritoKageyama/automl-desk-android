// in .../service/PreprocessorService.java
package com.example.automl_prototype_1.service;

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
    public Dataset runPreprocessingPipeline(Dataset rawDataset, String targetColumnName, String scalingStrategy) {
        LogService.info("Preprocessor", "--- Starting Preprocessing Pipeline ---");

        Dataset datasetWithoutIds = removeIdentifierColumns(rawDataset, targetColumnName);
        Dataset datasetWithNumericTarget = convertTargetToNumeric(datasetWithoutIds, targetColumnName);
        Dataset datasetWithStandardizedTarget = standardizeTargetColumn(datasetWithNumericTarget, targetColumnName);

        // --- THIS IS THE DYNAMIC PART ---
        Dataset finalDataset = switch (scalingStrategy) {
            case "Standardization" -> standardize(datasetWithStandardizedTarget);
            case "Normalization" -> normalize(datasetWithStandardizedTarget);
            default -> { // "None" or any other value
                LogService.info("Preprocessor", "Skipping feature scaling.");
                yield datasetWithStandardizedTarget;
            }
        };

        LogService.info("Preprocessor", "--- Preprocessing Pipeline Finished ---");
        return finalDataset;
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

        if (uniqueLabels.size() == 2) {
            LogService.info("Preprocessor",
                    "Detected binary target '" + targetColumnName + "' with labels " + uniqueLabels);
            List<String> sortedLabels = new ArrayList<>(uniqueLabels);
            Collections.sort(sortedLabels);
            String firstLabel = sortedLabels.get(0);
            LogService.info("Preprocessor",
                    "Mapping '" + firstLabel + "' to -1, and '" + sortedLabels.get(1) + "' to 1.");

            List<Map<String, String>> newRecords = dataset.getRecords().stream()
                    .map(originalRecord -> {
                        Map<String, String> newRecord = new HashMap<>(originalRecord);
                        String originalLabel = newRecord.get(targetColumnName);
                        String numericLabel = originalLabel.equals(firstLabel) ? "-1" : "1";
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
        return new Dataset(dataset.getSourceFile(), headers, newRecords);
    }

    public Dataset normalize(Dataset dataset) {
        LogService.info("Preprocessor", "Normalizing dataset...");
        final String TARGET_COLUMN = "Class";
        double[][] featureMatrix = dataset.getFeatureMatrix(TARGET_COLUMN);
        if (featureMatrix.length == 0)
            return dataset;
        int numFeatures = featureMatrix[0].length;
        double[] minValues = new double[numFeatures];
        double[] maxValues = new double[numFeatures];
        Arrays.fill(minValues, Double.POSITIVE_INFINITY);
        Arrays.fill(maxValues, Double.NEGATIVE_INFINITY);

        for (double[] row : featureMatrix) {
            for (int j = 0; j < numFeatures; j++) {
                if (row[j] < minValues[j])
                    minValues[j] = row[j];
                if (row[j] > maxValues[j])
                    maxValues[j] = row[j];
            }
        }
        for (double[] row : featureMatrix) {
            for (int j = 0; j < numFeatures; j++) {
                double range = maxValues[j] - minValues[j];
                row[j] = (range > 1e-9) ? (row[j] - minValues[j]) / range : 0.0;
            }
        }

        List<Map<String, String>> newRecords = new ArrayList<>();
        List<String> featureHeaders = new ArrayList<>(dataset.getHeaders());
        featureHeaders.remove(TARGET_COLUMN);
        List<Map<String, String>> originalRecords = dataset.getRecords();

        for (int i = 0; i < featureMatrix.length; i++) {
            Map<String, String> newRecord = new HashMap<>();
            for (int j = 0; j < featureHeaders.size(); j++) {
                newRecord.put(featureHeaders.get(j), String.valueOf(featureMatrix[i][j]));
            }
            newRecord.put(TARGET_COLUMN, originalRecords.get(i).get(TARGET_COLUMN));
            newRecords.add(newRecord);
        }
        LogService.info("Preprocessor", "Normalization complete.");
        return new Dataset(dataset.getSourceFile(), dataset.getHeaders(), newRecords);
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