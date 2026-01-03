package com.example.automl_prototype_1.model;

/**
 * A data-holder class representing the extracted characteristics of a Dataset.
 *
 * @param instanceCount          The number of rows (records).
 * @param featureCount           The number of columns (features), excluding the
 *                               target.
 * @param classCount             The number of unique classes in the target
 *                               variable.
 * @param imbalanceRatio         The ratio of the largest class to the smallest
 *                               class. (e.g., 1.0 is perfectly balanced).
 * @param featureToInstanceRatio The ratio of features to instances.
 */
public record DatasetMetaFeatures(
                long instanceCount,
                long featureCount,
                int classCount,
                double imbalanceRatio,
                double featureToInstanceRatio) {
}