package com.example.automl_prototype_1.model;

/**
 * Represents a detailed statistical summary of a single attribute (column).
 */
public record AttributeSummary(
                String attributeName,
                String dataType,
                long uniqueValues,
                long missingValues,
                String mean, // "N/A" for categorical
                String stdDev, // "N/A" for categorical
                String min, // "N/A" for categorical
                String max // "N/A" for categorical
) {
}