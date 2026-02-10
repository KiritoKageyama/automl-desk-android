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

        // --- THIS IS THE DYNAMIC PART ---
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